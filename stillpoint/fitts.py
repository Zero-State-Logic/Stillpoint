"""ISO 9241-411 multidirectional Fitts task: layouts and metrics.

Throughput follows Soukoreff and MacKenzie (2004):

* endpoint deviation ``dx`` is projected on the task axis (from the previous
  target centre to the current one);
* effective width ``We = 4.133 * SD(dx)``;
* effective distance ``De`` is the mean movement distance, start point to
  endpoint, projected on the same axis;
* ``IDe = log2(De / We + 1)``, throughput ``TP = IDe / MT`` per condition;
* a participant's throughput is the mean over conditions, and the reported
  figure is the mean over participants ("mean of means").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WE_FACTOR = 4.133  # sqrt(2 * pi * e): 96 % of a normal distribution


@dataclass(frozen=True)
class Condition:
    """One amplitude / width pair (screen pixels)."""

    distance: float
    width: float

    def __post_init__(self) -> None:
        if self.distance <= 0 or self.width <= 0:
            raise ValueError("distance and width must be positive")

    @property
    def nominal_id(self) -> float:
        return math.log2(self.distance / self.width + 1.0)


def circle_targets(n: int, distance: float, center=(960.0, 540.0)) -> np.ndarray:
    """``n`` target centres on a circle of diameter ``distance`` (n odd)."""
    if n < 3 or n % 2 == 0:
        raise ValueError("ISO layouts use an odd number of targets >= 3")
    angles = -np.pi / 2 + 2 * np.pi * np.arange(n) / n
    r = distance / 2.0
    return np.column_stack([center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)])


def selection_order(n: int) -> list[int]:
    """Standard alternating order: each target sits almost opposite the last.

    Starting at target 0 and stepping by (n + 1) / 2 visits every target once
    and returns to 0 after n moves.
    """
    step = (n + 1) // 2
    return [(i * step) % n for i in range(n + 1)]


@dataclass(frozen=True)
class Trial:
    """One selection. Points are screen pixels; times are seconds."""

    condition: Condition
    origin: tuple[float, float]  # previous target centre
    target: tuple[float, float]
    start: tuple[float, float] | None  # where the movement began (last click)
    endpoint: tuple[float, float] | None  # click position; None = no click
    movement_time: float | None

    @property
    def axis(self) -> np.ndarray:
        vec = np.subtract(self.target, self.origin)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("origin and target coincide")
        return vec / norm

    @property
    def hit(self) -> bool:
        if self.endpoint is None:
            return False
        return float(np.linalg.norm(np.subtract(self.endpoint, self.target))) <= self.condition.width / 2.0

    @property
    def deviation(self) -> float:
        """Signed endpoint offset along the task axis (+ = overshoot)."""
        if self.endpoint is None:
            raise ValueError("trial has no endpoint")
        return float(np.dot(np.subtract(self.endpoint, self.target), self.axis))

    @property
    def effective_distance(self) -> float:
        if self.endpoint is None:
            raise ValueError("trial has no endpoint")
        start = self.start if self.start is not None else self.origin
        return float(np.dot(np.subtract(self.endpoint, start), self.axis))


@dataclass(frozen=True)
class ConditionSummary:
    condition: Condition
    n_trials: int
    n_used: int
    error_rate: float
    movement_time: float
    we: float
    de: float
    ide: float
    throughput: float


def summarize_condition(trials: list[Trial]) -> ConditionSummary:
    """Effective-width metrics for one participant x condition."""
    if not trials:
        raise ValueError("no trials")
    condition = trials[0].condition
    if any(t.condition != condition for t in trials):
        raise ValueError("mixed conditions")
    used = [t for t in trials if t.endpoint is not None and t.movement_time is not None]
    errors = sum(1 for t in trials if not t.hit)
    if len(used) < 2:
        nan = float("nan")
        return ConditionSummary(condition, len(trials), len(used), errors / len(trials), nan, nan, nan, nan, nan)
    dx = np.array([t.deviation for t in used])
    we = WE_FACTOR * float(np.std(dx, ddof=1))
    de = float(np.mean([t.effective_distance for t in used]))
    mt = float(np.mean([t.movement_time for t in used]))
    ide = math.log2(de / we + 1.0) if we > 0 else float("nan")
    tp = ide / mt if mt > 0 else float("nan")
    return ConditionSummary(condition, len(trials), len(used), errors / len(trials), mt, we, de, ide, tp)


def mean_of_means(per_participant: dict) -> float:
    """Mean over participants of each participant's mean over conditions.

    ``per_participant`` maps a participant id to a list of
    ``ConditionSummary``; conditions without a finite throughput are skipped.
    """
    means = []
    for summaries in per_participant.values():
        values = [s.throughput for s in summaries if math.isfinite(s.throughput)]
        if values:
            means.append(float(np.mean(values)))
    return float(np.mean(means)) if means else float("nan")
