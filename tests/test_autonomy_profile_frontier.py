"""Unit tests for the autonomy-frontier math.

These tests run without any external dependencies (no Modal, no LLM). They
exist mainly to make sure the public formula ``max{k | SR(k) >= H}`` keeps
behaving correctly across refactors -- frontier math is the single piece of
this benchmark that we cannot eyeball from a JSONL trace.
"""

from __future__ import annotations

import pytest

from environments.benchmarks.autonomy_profile.scoring.frontier import (
    TaskResult,
    autonomy_frontier,
    per_cell_success_rate,
    per_domain_frontiers,
)


def _r(domain: str, level: int, passed: bool) -> TaskResult:
    return TaskResult(task_id=f"{domain}-L{level}-{passed}", domain=domain, complexity=level, passed=passed)


def test_frontier_empty_results_is_zero() -> None:
    assert autonomy_frontier([], threshold=0.8) == 0


def test_frontier_below_threshold_at_every_level_is_zero() -> None:
    results = [_r("A", 1, False), _r("A", 1, False), _r("A", 2, False)]
    assert autonomy_frontier(results, threshold=0.5) == 0


def test_frontier_picks_highest_qualifying_level() -> None:
    results = [
        _r("A", 1, True), _r("A", 1, True),
        _r("A", 2, True), _r("A", 2, True),
        _r("A", 3, True), _r("A", 3, False),  # SR=0.5
        _r("A", 4, False), _r("A", 4, False),
    ]
    assert autonomy_frontier(results, threshold=0.8) == 2
    assert autonomy_frontier(results, threshold=0.5) == 3


def test_frontier_threshold_is_inclusive() -> None:
    """Levels whose SR equals the threshold exactly must qualify."""
    results = [
        _r("A", 1, True), _r("A", 1, True), _r("A", 1, True), _r("A", 1, True),
        _r("A", 2, True), _r("A", 2, True), _r("A", 2, True), _r("A", 2, False),  # SR=0.75
    ]
    assert autonomy_frontier(results, threshold=0.75) == 2
    assert autonomy_frontier(results, threshold=0.76) == 1


def test_frontier_ignores_levels_with_no_observations() -> None:
    results = [_r("A", 1, True), _r("A", 5, True)]
    # Both observed levels have SR=1.0, frontier is the higher.
    assert autonomy_frontier(results, threshold=0.8) == 5


def test_frontier_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        autonomy_frontier([], threshold=1.5)
    with pytest.raises(ValueError):
        autonomy_frontier([], threshold=-0.1)


def test_per_domain_frontiers_separates_correctly() -> None:
    results = [
        # A: 1.0 at L1, 0.5 at L2 -> frontier 1 at H=0.8
        _r("A", 1, True), _r("A", 1, True),
        _r("A", 2, True), _r("A", 2, False),
        # B: 1.0 at L1 and L3 -> frontier 3 at H=0.8
        _r("B", 1, True), _r("B", 3, True),
    ]
    assert per_domain_frontiers(results, threshold=0.8) == {"A": 1, "B": 3}


def test_per_cell_success_rate_omits_empty_cells() -> None:
    results = [
        _r("A", 1, True), _r("A", 1, False),
        _r("A", 2, True), _r("A", 2, True),
    ]
    cells = per_cell_success_rate(results)
    assert cells == {("A", 1): 0.5, ("A", 2): 1.0}
