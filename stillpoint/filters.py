"""Cursor filters.

* ``OneEuroFilter``: the 1-euro filter of Casiez, Roussel and Vogel (CHI 2012).
  A first-order low-pass whose cutoff rises with speed: heavy smoothing when
  the hand is still (kills jitter), light smoothing when it moves (kills lag).
  Works on scalars or vectors; for a vector the cutoff follows the norm of the
  velocity so both axes are smoothed together and the path keeps its shape.
* ``ExponentialSmoother``: the original controller's fixed smoothing,
  ``x += (target - x) / SMOOTHING`` once per camera frame, kept as the baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def smoothing_factor(cutoff_hz: float, dt: float) -> float:
    """Exponential-smoothing weight of a first-order low-pass at ``cutoff_hz``."""
    if cutoff_hz <= 0 or dt <= 0:
        raise ValueError("cutoff and dt must be positive")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass(frozen=True)
class OneEuroParams:
    """Tuning knobs. Units: Hz for cutoffs, 1/(units per second) for beta."""

    min_cutoff: float = 1.0
    beta: float = 0.01
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0 or self.d_cutoff <= 0 or self.beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0, beta >= 0")


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez et al., 2012)."""

    def __init__(self, params: OneEuroParams | None = None) -> None:
        self.params = params or OneEuroParams()
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._x is None else self._x.copy()

    def __call__(self, x, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds). Returns a new array."""
        sample = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x = sample.copy()
            self._dx = np.zeros_like(sample)
            self._t = t
            return self._x.copy()
        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        raw_velocity = (sample - self._x) / dt
        a_d = smoothing_factor(self.params.d_cutoff, dt)
        self._dx = self._dx + a_d * (raw_velocity - self._dx)
        speed = float(np.linalg.norm(self._dx))
        cutoff = self.params.min_cutoff + self.params.beta * speed
        a = smoothing_factor(cutoff, dt)
        self._x = self._x + a * (sample - self._x)
        self._t = t
        return self._x.copy()


class ExponentialSmoother:
    """The original fixed smoother: ``x += (target - x) / smoothing`` per frame.

    It starts at ``initial`` (the original started at the screen origin) and
    ignores time stamps, exactly like the original loop.
    """

    def __init__(self, smoothing: float = 5.0, initial=(0.0, 0.0)) -> None:
        if smoothing < 1:
            raise ValueError("smoothing must be >= 1")
        self.smoothing = smoothing
        self._initial = np.asarray(initial, dtype=float)
        self.reset()

    def reset(self) -> None:
        self._x = self._initial.copy()

    @property
    def value(self) -> np.ndarray:
        return self._x.copy()

    def __call__(self, x, t: float = 0.0) -> np.ndarray:
        target = np.asarray(x, dtype=float)
        self._x = self._x + (target - self._x) / self.smoothing
        return self._x.copy()
