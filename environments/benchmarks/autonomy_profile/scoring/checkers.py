"""Deterministic checkers and setup actions for the Autonomy Profile benchmark.

Each *checker* is a callable that takes a ``ToolContext`` and an args dict and
returns a :class:`CheckResult`. The benchmark records per-check outcomes for
diagnostics, but a task is considered passed only when every check passes.

Each *setup action* mutates the sandbox before the agent starts (e.g. writing
fixture files). Setup actions return a :class:`CheckResult` so failures abort
the task with a clear reason recorded in the JSONL output.

Both registries are intentionally tiny -- the MVP scope is "what can we verify
with file/terminal tools alone." Adding a new checker means writing one
function and registering it in :data:`CHECKER_REGISTRY`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from environments.tool_context import ToolContext


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check or setup action."""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


CheckerFn = Callable[[ToolContext, Mapping[str, Any]], CheckResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(args: Mapping[str, Any], key: str) -> Any:
    if key not in args:
        raise KeyError(f"missing required argument: {key}")
    return args[key]


def _read_text(ctx: ToolContext, path: str) -> Dict[str, Any]:
    """Read a sandbox file. Returns a dict with ``content`` on success, ``error`` on miss."""
    return ctx.read_file(path)


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


def _file_exists(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    result = _read_text(ctx, path)
    if "content" in result:
        return CheckResult("file_exists", True, f"{path}: present")
    return CheckResult("file_exists", False, f"{path}: missing ({result.get('error', 'unknown')})")


def _file_contains(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    pattern = _require(args, "pattern")
    flags = re.MULTILINE | (re.IGNORECASE if args.get("ignore_case") else 0)
    result = _read_text(ctx, path)
    content = result.get("content")
    if content is None:
        return CheckResult("file_contains", False, f"{path}: unreadable ({result.get('error', '')})")
    if re.search(pattern, content, flags):
        return CheckResult("file_contains", True, f"{path}: matched /{pattern}/")
    return CheckResult("file_contains", False, f"{path}: no match for /{pattern}/")


def _file_has_sections(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    sections = list(_require(args, "sections"))
    level = int(args.get("level", 2))
    if level < 1 or level > 6:
        raise ValueError("markdown header level must be 1..6")
    prefix = "#" * level + " "
    result = _read_text(ctx, path)
    content = result.get("content")
    if content is None:
        return CheckResult("file_has_sections", False, f"{path}: unreadable")
    header_lines = {
        line[len(prefix):].strip().lower()
        for line in content.splitlines()
        if line.startswith(prefix)
    }
    missing = [s for s in sections if s.strip().lower() not in header_lines]
    if missing:
        return CheckResult("file_has_sections", False, f"{path}: missing sections {missing}")
    return CheckResult("file_has_sections", True, f"{path}: all {len(sections)} sections present")


def _terminal_exit_code(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    command = _require(args, "command")
    expected = int(args.get("expected", 0))
    timeout = int(args.get("timeout", 60))
    result = ctx.terminal(command, timeout=timeout)
    actual = int(result.get("exit_code", -1))
    if actual == expected:
        return CheckResult("terminal_exit_code", True, f"{command!r}: exit={actual}")
    return CheckResult(
        "terminal_exit_code",
        False,
        f"{command!r}: exit={actual} expected={expected}",
    )


def _terminal_stdout_equals(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    command = _require(args, "command")
    expected = _require(args, "expected")
    timeout = int(args.get("timeout", 60))
    strip = bool(args.get("strip", False))
    result = ctx.terminal(command, timeout=timeout)
    output = result.get("output", "")
    if strip:
        output_cmp = output.strip()
        expected_cmp = expected.strip()
    else:
        output_cmp = output
        expected_cmp = expected
    if output_cmp == expected_cmp:
        return CheckResult("terminal_stdout_equals", True, f"{command!r}: stdout matches")
    return CheckResult(
        "terminal_stdout_equals",
        False,
        f"{command!r}: stdout={output_cmp!r} expected={expected_cmp!r}",
    )


def _terminal_stdout_matches(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    command = _require(args, "command")
    pattern = _require(args, "pattern")
    timeout = int(args.get("timeout", 60))
    flags = re.MULTILINE | (re.IGNORECASE if args.get("ignore_case") else 0)
    result = ctx.terminal(command, timeout=timeout)
    output = result.get("output", "")
    if re.search(pattern, output, flags):
        return CheckResult("terminal_stdout_matches", True, f"{command!r}: matched /{pattern}/")
    return CheckResult(
        "terminal_stdout_matches",
        False,
        f"{command!r}: no match for /{pattern}/ in {output!r}",
    )


def _json_valid(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    result = _read_text(ctx, path)
    content = result.get("content")
    if content is None:
        return CheckResult("json_valid", False, f"{path}: unreadable")
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return CheckResult("json_valid", False, f"{path}: invalid JSON ({exc})")
    return CheckResult("json_valid", True, f"{path}: parses as JSON")


def _resolve_dot_path(obj: Any, dotted: str) -> Any:
    """Resolve ``a.b.c`` against nested dicts/lists. Returns sentinel ``_MISSING`` on failure."""
    cursor: Any = obj
    for part in dotted.split("."):
        if isinstance(cursor, list):
            try:
                cursor = cursor[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        elif isinstance(cursor, dict):
            if part not in cursor:
                return _MISSING
            cursor = cursor[part]
        else:
            return _MISSING
    return cursor


_MISSING = object()


def _json_has_keys(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    keys = list(_require(args, "keys"))
    result = _read_text(ctx, path)
    content = result.get("content")
    if content is None:
        return CheckResult("json_has_keys", False, f"{path}: unreadable")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return CheckResult("json_has_keys", False, f"{path}: invalid JSON ({exc})")
    missing = [k for k in keys if _resolve_dot_path(parsed, k) is _MISSING]
    if missing:
        return CheckResult("json_has_keys", False, f"{path}: missing keys {missing}")
    return CheckResult("json_has_keys", True, f"{path}: all {len(keys)} keys present")


def _numeric_close(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    """Read a single numeric value (JSON or plain text) and compare to expected within tolerance."""
    path = _require(args, "path")
    expected = float(_require(args, "expected"))
    tolerance = float(args.get("tolerance", 1e-6))
    key = args.get("key")
    result = _read_text(ctx, path)
    content = result.get("content")
    if content is None:
        return CheckResult("numeric_close", False, f"{path}: unreadable")
    actual: float
    try:
        if key:
            parsed = json.loads(content)
            value = _resolve_dot_path(parsed, key)
            if value is _MISSING:
                return CheckResult("numeric_close", False, f"{path}: key {key!r} missing")
            actual = float(value)
        else:
            actual = float(content.strip())
    except (ValueError, json.JSONDecodeError) as exc:
        return CheckResult("numeric_close", False, f"{path}: not numeric ({exc})")
    if math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
        return CheckResult("numeric_close", True, f"{path}: {actual} ~ {expected} (tol={tolerance})")
    return CheckResult(
        "numeric_close",
        False,
        f"{path}: {actual} != {expected} (tol={tolerance})",
    )


CHECKER_REGISTRY: Dict[str, CheckerFn] = {
    "file_exists": _file_exists,
    "file_contains": _file_contains,
    "file_has_sections": _file_has_sections,
    "terminal_exit_code": _terminal_exit_code,
    "terminal_stdout_equals": _terminal_stdout_equals,
    "terminal_stdout_matches": _terminal_stdout_matches,
    "json_valid": _json_valid,
    "json_has_keys": _json_has_keys,
    "numeric_close": _numeric_close,
}


# ---------------------------------------------------------------------------
# Setup actions
# ---------------------------------------------------------------------------


def _setup_write_file(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    path = _require(args, "path")
    content = _require(args, "content")
    result = ctx.write_file(path, content)
    if result.get("error"):
        return CheckResult("write_file", False, f"{path}: {result['error']}")
    return CheckResult("write_file", True, f"{path}: wrote {len(content)} chars")


def _setup_terminal_run(ctx: ToolContext, args: Mapping[str, Any]) -> CheckResult:
    command = _require(args, "command")
    timeout = int(args.get("timeout", 60))
    expected = int(args.get("expected_exit", 0))
    result = ctx.terminal(command, timeout=timeout)
    actual = int(result.get("exit_code", -1))
    if actual == expected:
        return CheckResult("terminal_run", True, f"{command!r}: exit={actual}")
    return CheckResult(
        "terminal_run",
        False,
        f"{command!r}: exit={actual} expected={expected}",
    )


SETUP_REGISTRY: Dict[str, CheckerFn] = {
    "write_file": _setup_write_file,
    "terminal_run": _setup_terminal_run,
}


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def run_check(ctx: ToolContext, spec: Mapping[str, Any]) -> CheckResult:
    """Run a single check defined as ``{"check": <name>, "args": {...}}``."""
    name = _require(spec, "check")
    args = spec.get("args", {})
    fn = CHECKER_REGISTRY.get(name)
    if fn is None:
        return CheckResult(name, False, f"unknown checker: {name}")
    try:
        return fn(ctx, args)
    except Exception as exc:  # noqa: BLE001 -- record any failure as a non-passing check
        return CheckResult(name, False, f"exception: {exc!r}")


def run_setup_action(ctx: ToolContext, action: Mapping[str, Any]) -> CheckResult:
    """Run a single setup action defined as ``{"action": <name>, ...}``."""
    name = _require(action, "action")
    fn = SETUP_REGISTRY.get(name)
    if fn is None:
        return CheckResult(name, False, f"unknown setup action: {name}")
    args = {k: v for k, v in action.items() if k != "action"}
    try:
        return fn(ctx, args)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, False, f"exception: {exc!r}")
