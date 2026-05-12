"""Scoring components for the Autonomy Profile benchmark."""

from .checkers import (
    CHECKER_REGISTRY,
    SETUP_REGISTRY,
    CheckResult,
    run_check,
    run_setup_action,
)
from .frontier import (
    TaskResult,
    autonomy_frontier,
    per_domain_frontiers,
    per_cell_success_rate,
)

__all__ = [
    "CHECKER_REGISTRY",
    "SETUP_REGISTRY",
    "CheckResult",
    "run_check",
    "run_setup_action",
    "TaskResult",
    "autonomy_frontier",
    "per_domain_frontiers",
    "per_cell_success_rate",
]
