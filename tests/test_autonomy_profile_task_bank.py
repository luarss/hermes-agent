"""Schema lint for the bundled Autonomy Profile task bank.

Catches typos in JSONL records before they reach the runtime loader: bad
JSON, unknown taxonomy references, missing required fields, or check
specifications that name a checker that doesn't exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.benchmarks.autonomy_profile.loader import (
    load_task_bank,
    load_taxonomy,
)
from environments.benchmarks.autonomy_profile.scoring.checkers import (
    CHECKER_REGISTRY,
    SETUP_REGISTRY,
)

_BENCH_DIR = Path(__file__).resolve().parent.parent / "environments" / "benchmarks" / "autonomy_profile"
_TASK_FILES = sorted((_BENCH_DIR / "tasks").glob("*.jsonl"))


def _load_taxonomy() -> tuple[set[str], set[str]]:
    domains = json.loads((_BENCH_DIR / "taxonomy" / "onet_domains.json").read_text())
    skills = json.loads((_BENCH_DIR / "taxonomy" / "onet_skills.json").read_text())
    return (
        {d["name"] for d in domains["domains"]},
        {s["name"] for s in skills["skills"]},
    )


def _iter_records():
    for path in _TASK_FILES:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield path, line_no, json.loads(stripped)


def test_task_files_exist() -> None:
    assert _TASK_FILES, f"no JSONL task files under {_BENCH_DIR / 'tasks'}"


def test_every_record_is_valid_json() -> None:
    # _iter_records() already json.loads — if any line is malformed, this raises.
    records = list(_iter_records())
    assert records, "task bank loaded zero records"


def test_required_fields_present() -> None:
    required = {"id", "domain", "complexity", "instruction", "evaluation"}
    for path, line_no, rec in _iter_records():
        missing = required - rec.keys()
        assert not missing, f"{path.name}:{line_no} missing fields {missing}"


def test_ids_are_unique() -> None:
    seen: dict[str, str] = {}
    for path, line_no, rec in _iter_records():
        loc = f"{path.name}:{line_no}"
        if rec["id"] in seen:
            pytest.fail(f"duplicate task id {rec['id']!r} at {loc} (previously {seen[rec['id']]})")
        seen[rec["id"]] = loc


def test_domain_and_skills_reference_taxonomy() -> None:
    known_domains, known_skills = _load_taxonomy()
    for path, line_no, rec in _iter_records():
        loc = f"{path.name}:{line_no}"
        assert rec["domain"] in known_domains, f"{loc}: unknown domain {rec['domain']!r}"
        for skill in rec.get("skills", []):
            assert skill in known_skills, f"{loc}: unknown skill {skill!r}"


def test_complexity_within_supported_range() -> None:
    for path, line_no, rec in _iter_records():
        loc = f"{path.name}:{line_no}"
        c = rec["complexity"]
        assert isinstance(c, int), f"{loc}: complexity must be int (got {type(c).__name__})"
        assert 1 <= c <= 7, f"{loc}: complexity {c} outside MVP range [1,7]"


def test_evaluation_uses_known_checkers() -> None:
    for path, line_no, rec in _iter_records():
        loc = f"{path.name}:{line_no}"
        evaluation = rec["evaluation"]
        assert evaluation.get("type") == "checks", f"{loc}: evaluation.type must be 'checks'"
        checks = evaluation.get("checks") or []
        assert checks, f"{loc}: evaluation.checks must be non-empty"
        for spec in checks:
            name = spec.get("check")
            assert name in CHECKER_REGISTRY, f"{loc}: unknown checker {name!r}"


def test_setup_actions_use_known_registry() -> None:
    for path, line_no, rec in _iter_records():
        loc = f"{path.name}:{line_no}"
        for action in rec.get("setup", []):
            name = action.get("action")
            assert name in SETUP_REGISTRY, f"{loc}: unknown setup action {name!r}"


def test_every_task_file_has_minimum_volume() -> None:
    """Each domain file should ship at least 25 tasks to be useful for SR statistics."""
    for path in _TASK_FILES:
        non_empty = [
            line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
        ]
        assert len(non_empty) >= 25, f"{path.name} has only {len(non_empty)} tasks; expected >= 25"


def test_each_complexity_level_present_per_domain() -> None:
    """The MVP design calls for at least one task per complexity level in each domain."""
    by_domain: dict[str, set[int]] = {}
    for _path, _line, rec in _iter_records():
        by_domain.setdefault(rec["domain"], set()).add(rec["complexity"])
    for domain, levels in by_domain.items():
        missing = set(range(1, 8)) - levels
        assert not missing, f"{domain} missing complexity levels {sorted(missing)}"


def test_loader_succeeds_end_to_end() -> None:
    """The shared loader must accept the bundled task bank without raising."""
    domains, skills = load_taxonomy()
    tasks = load_task_bank(
        [
            "tasks/computer.jsonl",
            "tasks/office_admin.jsonl",
            "tasks/business_financial.jsonl",
            "tasks/management.jsonl",
            "tasks/data_analysis.jsonl",
        ],
        domains,
        skills,
    )
    assert len(tasks) >= 150, len(tasks)
    assert {t["domain"] for t in tasks} == set(domains.keys())
