"""Live Fitts-task logic and its CSV log, kept free of camera and GUI code.

``FittsTask`` sequences the ISO multidirectional task (circles of targets,
one circle per condition) and turns click events into logged trials.
``fitts_app`` drives it from the camera; the tests drive it with scripted
clicks; ``analyze`` reads the CSV back.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fitts import Condition, Trial, circle_targets, selection_order

FIELDS = (
    "participant", "variant", "block", "sequence", "trial",
    "distance", "width", "nominal_id",
    "origin_x", "origin_y", "target_x", "target_y",
    "start_x", "start_y", "click_x", "click_y",
    "t_start", "t_click", "movement_time", "hit",
)


@dataclass(frozen=True)
class Target:
    x: float
    y: float
    radius: float


class TrialLogger:
    """Append trial rows to a CSV file (header written once, flushed per row)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        if new:
            self._writer.writeheader()

    def write(self, row: dict) -> None:
        missing = set(FIELDS) - set(row)
        if missing:
            raise ValueError(f"trial row missing fields: {sorted(missing)}")
        self._writer.writerow({k: row[k] for k in FIELDS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TrialLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FittsTask:
    """Sequence of circles; the first click of each circle only starts it."""

    def __init__(self, conditions, participant: str, variant: str, block: int = 0,
                 n_targets: int = 13, center=(640.0, 360.0), logger: TrialLogger | None = None) -> None:
        if not conditions:
            raise ValueError("need at least one condition")
        self.conditions = list(conditions)
        self.participant, self.variant, self.block = participant, variant, block
        self.n_targets = n_targets
        self.center = center
        self.logger = logger
        self.order = selection_order(n_targets)
        self.sequence = 0
        self.position = 0  # index into self.order
        self.last: tuple[float, float, float] | None = None  # t, x, y of last click
        self.rows: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.sequence >= len(self.conditions)

    @property
    def condition(self) -> Condition:
        return self.conditions[self.sequence]

    def targets(self) -> np.ndarray:
        return circle_targets(self.n_targets, self.condition.distance, self.center)

    def current_target(self) -> Target | None:
        if self.finished:
            return None
        x, y = self.targets()[self.order[self.position]]
        return Target(float(x), float(y), self.condition.width / 2.0)

    def on_click(self, t: float, x: float, y: float) -> dict | None:
        """Register a click. Returns the logged row, or None for a start click."""
        if self.finished:
            return None
        row = None
        if self.position > 0:
            row = self._row(t, x, y)
            self.rows.append(row)
            if self.logger is not None:
                self.logger.write(row)
        if not (np.isfinite(x) and np.isfinite(y)):  # skipped target: move on from its centre
            x, y = self.targets()[self.order[self.position]]
        self.last = (t, float(x), float(y))
        self.position += 1
        if self.position >= len(self.order):
            self.sequence += 1
            self.position = 0
            self.last = None
        return row

    def _row(self, t: float, x: float, y: float) -> dict:
        targets = self.targets()
        origin = targets[self.order[self.position - 1]]
        target = targets[self.order[self.position]]
        start = (self.last[1], self.last[2]) if self.last else (origin[0], origin[1])
        t_start = self.last[0] if self.last else float("nan")
        hit = float(np.hypot(x - target[0], y - target[1])) <= self.condition.width / 2.0
        return {
            "participant": self.participant, "variant": self.variant, "block": self.block,
            "sequence": self.sequence, "trial": self.position,
            "distance": self.condition.distance, "width": self.condition.width,
            "nominal_id": round(self.condition.nominal_id, 4),
            "origin_x": float(origin[0]), "origin_y": float(origin[1]),
            "target_x": float(target[0]), "target_y": float(target[1]),
            "start_x": float(start[0]), "start_y": float(start[1]),
            "click_x": float(x), "click_y": float(y),
            "t_start": t_start, "t_click": float(t), "movement_time": float(t - t_start),
            "hit": int(hit),
        }


def read_rows(paths) -> list[dict]:
    """Read every row of one or more trial CSV files."""
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or set(FIELDS) - set(reader.fieldnames):
                raise ValueError(f"{path}: not a STILLPOINT trial log (missing columns)")
            rows.extend(reader)
    return rows


def row_to_trial(row: dict) -> Trial:
    """Convert one CSV row back into a ``fitts.Trial``."""
    f = {k: float(row[k]) for k in ("distance", "width", "origin_x", "origin_y", "target_x", "target_y",
                                     "start_x", "start_y", "click_x", "click_y", "movement_time")}
    clicked = np.isfinite(f["click_x"]) and np.isfinite(f["click_y"])
    mt = f["movement_time"] if clicked and np.isfinite(f["movement_time"]) else None
    return Trial(
        Condition(f["distance"], f["width"]),
        (f["origin_x"], f["origin_y"]), (f["target_x"], f["target_y"]),
        (f["start_x"], f["start_y"]), (f["click_x"], f["click_y"]) if clicked else None, mt,
    )
