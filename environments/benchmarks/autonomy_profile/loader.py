"""Pure-Python loaders for the Autonomy Profile taxonomy and task bank.

Kept independent of ``atroposlib`` so the JSON/JSONL schema can be lint-checked
without the heavyweight env dependency tree. The env class delegates to these
helpers in :meth:`AutonomyProfileEvalEnv.setup`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BENCH_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    """Absolute paths pass through, relative paths resolve against the benchmark dir."""
    path = Path(p)
    return path if path.is_absolute() else (_BENCH_DIR / path)


def load_taxonomy(
    taxonomy_dir: str = "taxonomy",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load O*NET subset. Returns ``(domains_by_name, skills_by_name)``."""
    base = resolve_path(taxonomy_dir)
    domains_payload = json.loads((base / "onet_domains.json").read_text(encoding="utf-8"))
    skills_payload = json.loads((base / "onet_skills.json").read_text(encoding="utf-8"))
    domains_by_name = {d["name"]: d for d in domains_payload["domains"]}
    skills_by_name = {s["name"]: s for s in skills_payload["skills"]}
    return domains_by_name, skills_by_name


def load_task_bank(
    task_files: List[str],
    domains_by_name: Dict[str, Dict[str, Any]],
    skills_by_name: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load and validate every record across the configured JSONL files."""
    tasks: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for rel in task_files:
        path = resolve_path(rel)
        if not path.exists():
            raise FileNotFoundError(f"task bank file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
                validate_task_record(record, path, line_no, domains_by_name, skills_by_name)
                if record["id"] in seen_ids:
                    raise ValueError(f"{path}:{line_no}: duplicate task id {record['id']!r}")
                seen_ids.add(record["id"])
                record["_source"] = f"{path.name}:{line_no}"
                tasks.append(record)
    return tasks


def validate_task_record(
    record: Dict[str, Any],
    path: Path,
    line_no: int,
    domains_by_name: Dict[str, Dict[str, Any]],
    skills_by_name: Dict[str, Dict[str, Any]],
) -> None:
    """Raise ``ValueError`` if a task record is malformed."""
    where = f"{path.name}:{line_no}"
    for field in ("id", "domain", "complexity", "instruction", "evaluation"):
        if field not in record:
            raise ValueError(f"{where}: missing required field {field!r}")
    if record["domain"] not in domains_by_name:
        raise ValueError(f"{where}: unknown domain {record['domain']!r}")
    if not isinstance(record["complexity"], int) or record["complexity"] < 1:
        raise ValueError(f"{where}: complexity must be a positive int")
    skills = record.get("skills", [])
    if not isinstance(skills, list):
        raise ValueError(f"{where}: skills must be a list")
    for skill in skills:
        if skill not in skills_by_name:
            raise ValueError(f"{where}: unknown skill {skill!r}")
    evaluation = record["evaluation"]
    if evaluation.get("type") != "checks":
        raise ValueError(f"{where}: evaluation.type must be 'checks' in MVP")
    checks = evaluation.get("checks") or []
    if not checks:
        raise ValueError(f"{where}: evaluation.checks must be non-empty")
    for spec in checks:
        if "check" not in spec:
            raise ValueError(f"{where}: each check must have a 'check' name")
    setup = record.get("setup", [])
    if not isinstance(setup, list):
        raise ValueError(f"{where}: setup must be a list (got {type(setup).__name__})")
    for action in setup:
        if "action" not in action:
            raise ValueError(f"{where}: each setup item must have an 'action' name")


def slug(text: str) -> str:
    """Normalise a domain or skill name for use in wandb metric keys."""
    cleaned = "".join(c if c.isalnum() else "_" for c in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()
