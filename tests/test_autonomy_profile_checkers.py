"""Unit tests for autonomy-profile checkers using a stub ToolContext.

The real ``ToolContext`` talks to a Modal/local terminal sandbox; in tests
we substitute a small fake that records calls and returns canned data. This
keeps the unit tests hermetic and ms-fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from environments.benchmarks.autonomy_profile.scoring.checkers import (
    CHECKER_REGISTRY,
    SETUP_REGISTRY,
    run_check,
    run_setup_action,
)


@dataclass
class _StubContext:
    """Drop-in stand-in for ToolContext that serves canned files / command output."""

    files: Dict[str, str] = field(default_factory=dict)
    commands: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    commands_run: List[str] = field(default_factory=list)
    writes: List[Dict[str, Any]] = field(default_factory=list)

    def read_file(self, path: str) -> Dict[str, Any]:
        if path in self.files:
            return {"content": self.files[path]}
        return {"error": "not found"}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        self.writes.append({"path": path, "content": content})
        self.files[path] = content
        return {"success": True}

    def terminal(self, command: str, timeout: int = 60) -> Dict[str, Any]:  # noqa: ARG002
        self.commands_run.append(command)
        return self.commands.get(command, {"exit_code": 0, "output": ""})


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_lists_all_documented_checkers() -> None:
    expected = {
        "file_exists",
        "file_contains",
        "file_has_sections",
        "terminal_exit_code",
        "terminal_stdout_equals",
        "terminal_stdout_matches",
        "json_valid",
        "json_has_keys",
        "numeric_close",
    }
    assert expected.issubset(CHECKER_REGISTRY)


def test_setup_registry_has_minimum_set() -> None:
    assert {"write_file", "terminal_run"}.issubset(SETUP_REGISTRY)


def test_unknown_check_returns_failure() -> None:
    ctx = _StubContext()
    result = run_check(ctx, {"check": "does_not_exist"})
    assert result.passed is False
    assert "unknown" in result.detail.lower()


def test_check_missing_required_arg_returns_failure() -> None:
    ctx = _StubContext()
    result = run_check(ctx, {"check": "file_exists", "args": {}})
    assert result.passed is False
    assert "missing" in result.detail.lower()


# ---------------------------------------------------------------------------
# file_exists / file_contains
# ---------------------------------------------------------------------------


def test_file_exists_passes_when_file_present() -> None:
    ctx = _StubContext(files={"/x.txt": "hi"})
    assert run_check(ctx, {"check": "file_exists", "args": {"path": "/x.txt"}}).passed


def test_file_exists_fails_when_missing() -> None:
    assert run_check(_StubContext(), {"check": "file_exists", "args": {"path": "/missing"}}).passed is False


def test_file_contains_regex_match() -> None:
    ctx = _StubContext(files={"/x.txt": "hello world\nsecond line\n"})
    spec = {"check": "file_contains", "args": {"path": "/x.txt", "pattern": "^second"}}
    assert run_check(ctx, spec).passed


def test_file_contains_no_match() -> None:
    ctx = _StubContext(files={"/x.txt": "nope"})
    spec = {"check": "file_contains", "args": {"path": "/x.txt", "pattern": "xyz"}}
    assert run_check(ctx, spec).passed is False


# ---------------------------------------------------------------------------
# file_has_sections
# ---------------------------------------------------------------------------


def test_file_has_sections_passes_with_all_headers() -> None:
    ctx = _StubContext(files={"/doc.md": "# Title\n## Alpha\nbody\n## Beta\nbody\n"})
    spec = {"check": "file_has_sections", "args": {"path": "/doc.md", "sections": ["Alpha", "Beta"], "level": 2}}
    assert run_check(ctx, spec).passed


def test_file_has_sections_fails_if_one_missing() -> None:
    ctx = _StubContext(files={"/doc.md": "## Alpha\nbody\n"})
    spec = {"check": "file_has_sections", "args": {"path": "/doc.md", "sections": ["Alpha", "Beta"], "level": 2}}
    assert run_check(ctx, spec).passed is False


def test_file_has_sections_level_one_supported() -> None:
    ctx = _StubContext(files={"/doc.md": "# Vision\n"})
    spec = {"check": "file_has_sections", "args": {"path": "/doc.md", "sections": ["Vision"], "level": 1}}
    assert run_check(ctx, spec).passed


# ---------------------------------------------------------------------------
# terminal_*
# ---------------------------------------------------------------------------


def test_terminal_exit_code_matches() -> None:
    ctx = _StubContext(commands={"true": {"exit_code": 0, "output": ""}})
    spec = {"check": "terminal_exit_code", "args": {"command": "true", "expected": 0}}
    assert run_check(ctx, spec).passed


def test_terminal_exit_code_mismatch_fails() -> None:
    ctx = _StubContext(commands={"false": {"exit_code": 1, "output": ""}})
    spec = {"check": "terminal_exit_code", "args": {"command": "false", "expected": 0}}
    assert run_check(ctx, spec).passed is False


def test_terminal_stdout_equals() -> None:
    ctx = _StubContext(commands={"echo hi": {"exit_code": 0, "output": "hi\n"}})
    spec = {"check": "terminal_stdout_equals", "args": {"command": "echo hi", "expected": "hi\n"}}
    assert run_check(ctx, spec).passed


def test_terminal_stdout_matches_regex() -> None:
    ctx = _StubContext(commands={"date": {"exit_code": 0, "output": "Tue May 12 09:00:00 2026"}})
    spec = {"check": "terminal_stdout_matches", "args": {"command": "date", "pattern": "20\\d\\d"}}
    assert run_check(ctx, spec).passed


# ---------------------------------------------------------------------------
# json_*
# ---------------------------------------------------------------------------


def test_json_valid_passes() -> None:
    ctx = _StubContext(files={"/x.json": '{"a": 1}'})
    assert run_check(ctx, {"check": "json_valid", "args": {"path": "/x.json"}}).passed


def test_json_valid_fails_on_garbage() -> None:
    ctx = _StubContext(files={"/x.json": "not json"})
    assert run_check(ctx, {"check": "json_valid", "args": {"path": "/x.json"}}).passed is False


def test_json_has_keys_dot_path() -> None:
    ctx = _StubContext(files={"/x.json": '{"a": {"b": [1, 2, 3]}}'})
    spec = {"check": "json_has_keys", "args": {"path": "/x.json", "keys": ["a.b.2"]}}
    assert run_check(ctx, spec).passed


def test_json_has_keys_reports_missing() -> None:
    ctx = _StubContext(files={"/x.json": '{"a": 1}'})
    spec = {"check": "json_has_keys", "args": {"path": "/x.json", "keys": ["a", "b"]}}
    result = run_check(ctx, spec)
    assert result.passed is False
    assert "b" in result.detail


# ---------------------------------------------------------------------------
# numeric_close
# ---------------------------------------------------------------------------


def test_numeric_close_plain_text() -> None:
    ctx = _StubContext(files={"/n.txt": "42\n"})
    spec = {"check": "numeric_close", "args": {"path": "/n.txt", "expected": 42, "tolerance": 0}}
    assert run_check(ctx, spec).passed


def test_numeric_close_within_tolerance() -> None:
    ctx = _StubContext(files={"/n.txt": "3.14159\n"})
    spec = {"check": "numeric_close", "args": {"path": "/n.txt", "expected": 3.14, "tolerance": 0.01}}
    assert run_check(ctx, spec).passed


def test_numeric_close_outside_tolerance_fails() -> None:
    ctx = _StubContext(files={"/n.txt": "10\n"})
    spec = {"check": "numeric_close", "args": {"path": "/n.txt", "expected": 1, "tolerance": 0.01}}
    assert run_check(ctx, spec).passed is False


def test_numeric_close_with_json_key() -> None:
    ctx = _StubContext(files={"/n.json": '{"result": 99.5}'})
    spec = {"check": "numeric_close", "args": {"path": "/n.json", "expected": 100, "tolerance": 1, "key": "result"}}
    assert run_check(ctx, spec).passed


# ---------------------------------------------------------------------------
# Setup actions
# ---------------------------------------------------------------------------


def test_setup_write_file_writes() -> None:
    ctx = _StubContext()
    action = {"action": "write_file", "path": "/x.txt", "content": "hi"}
    result = run_setup_action(ctx, action)
    assert result.passed
    assert ctx.writes == [{"path": "/x.txt", "content": "hi"}]
    assert ctx.files["/x.txt"] == "hi"


def test_setup_terminal_run_records_command() -> None:
    ctx = _StubContext()
    action = {"action": "terminal_run", "command": "echo hi"}
    result = run_setup_action(ctx, action)
    assert result.passed
    assert ctx.commands_run == ["echo hi"]


def test_setup_terminal_run_expected_exit_mismatch_fails() -> None:
    ctx = _StubContext(commands={"false": {"exit_code": 1, "output": ""}})
    action = {"action": "terminal_run", "command": "false", "expected_exit": 0}
    assert run_setup_action(ctx, action).passed is False


def test_setup_unknown_action_returns_failure() -> None:
    result = run_setup_action(_StubContext(), {"action": "nope"})
    assert result.passed is False
