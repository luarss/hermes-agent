"""Autonomy-frontier computation and per-domain aggregation.

The "autonomy frontier" for a slice of results is the maximum complexity level
:math:`k` at which the success rate stays at or above a threshold :math:`H`:

.. math::

    \\text{Autonomy} = \\max\\{k \\mid \\text{SR}(k) \\ge H\\}

Levels with no observations contribute nothing. If no level qualifies, the
frontier is reported as 0 (the agent fails even the easiest tasks at the
configured threshold).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task rollout for frontier calculation."""

    task_id: str
    domain: str
    complexity: int
    passed: bool


def autonomy_frontier(results: Iterable[TaskResult], threshold: float) -> int:
    """Return ``max{k | SR(k) >= threshold}`` across the given results.

    Returns ``0`` if no complexity level qualifies. ``threshold`` is interpreted
    as an inclusive lower bound on the success rate.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.0, 1.0]")

    by_level: Dict[int, List[bool]] = defaultdict(list)
    for r in results:
        by_level[r.complexity].append(r.passed)

    qualifying = [
        k for k, outcomes in by_level.items()
        if outcomes and (sum(outcomes) / len(outcomes)) >= threshold
    ]
    return max(qualifying, default=0)


def per_domain_frontiers(
    results: Iterable[TaskResult],
    threshold: float,
) -> Dict[str, int]:
    """Compute the autonomy frontier separately for each domain present in ``results``."""
    by_domain: Dict[str, List[TaskResult]] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)

    return {
        domain: autonomy_frontier(items, threshold)
        for domain, items in by_domain.items()
    }


def per_cell_success_rate(
    results: Iterable[TaskResult],
) -> Dict[Tuple[str, int], float]:
    """Success rate per ``(domain, complexity)`` cell. Empty cells are omitted."""
    by_cell: Dict[Tuple[str, int], List[bool]] = defaultdict(list)
    for r in results:
        by_cell[(r.domain, r.complexity)].append(r.passed)
    return {
        cell: (sum(outcomes) / len(outcomes))
        for cell, outcomes in by_cell.items()
        if outcomes
    }
