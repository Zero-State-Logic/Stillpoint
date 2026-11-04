"""Simulated participants: who they are and how they move.

Every number here is a modelling assumption, listed in the README's
simulation table. Participants are drawn from a seeded generator, so the same
seed always gives the same people.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitts import Condition

DEFAULT_CONDITIONS = tuple(
    Condition(d, w) for d in (300.0, 600.0) for w in (40.0, 80.0, 120.0)
)


@dataclass(frozen=True)
class SimParams:
    """Experiment-wide knobs (the ones the sensitivity sweep varies)."""

    fps: float = 30.0
    slip: float = 0.20  # fraction of the pinch closing distance the index tip travels
    jitter_px: float = 1.5  # per-landmark tracking noise SD, camera pixels
    drop_prob: float = 0.02  # chance a frame has no hand
    fitts_a: float = 0.20  # s, reach time intercept
    fitts_b: float = 0.22  # s/bit, reach time slope
    n_targets: int = 13
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS
    hover_s: float = 6.0  # hand wanders without clicking after each circle
    warmup_s: float = 1.0
    settle_speed: float = 60.0  # screen px/s: cursor "has arrived" below this
    max_settle_s: float = 0.6
    track_modes: bool = False  # also run both mode classifiers on the stream

    def __post_init__(self) -> None:
        if not 0.0 <= self.slip <= 1.0:
            raise ValueError("slip must be in [0, 1]")
        if self.jitter_px < 0 or not 0.0 <= self.drop_prob < 1.0:
            raise ValueError("jitter must be >= 0 and drop_prob in [0, 1)")


@dataclass(frozen=True)
class Participant:
    pid: int
    hand_px: float  # palm length (wrist to middle knuckle) in camera pixels
    speed: float  # multiplies the Fitts a and b
    accuracy: float  # multiplies the endpoint spread
    roll: float  # degrees, mean hand tilt about the camera axis
    pitch: float
    yaw: float
    rest_aperture: float  # palm units, thumb-index gap while pointing
    hover_low: float  # palm units, smallest relaxed gap while not clicking
    closed_aperture: float  # palm units, gap at full pinch (tip landmarks)
    close_s: float
    hold_s: float
    open_s: float
    verify_s: float  # pause between the cursor arriving and the pinch
    slip_bias: float  # this person's offset to the global slip fraction
    tremor_hz: float
    tremor_px: float
    tremor_phase: tuple[float, float]


def make_participants(n: int, seed: int) -> list[Participant]:
    """Draw ``n`` participants. Same (n, seed) -> same people."""
    if n < 1:
        raise ValueError("need at least one participant")
    rng = np.random.default_rng([seed, 7919])
    people = []
    for pid in range(n):
        u = rng.uniform
        people.append(
            Participant(
                pid=pid,
                hand_px=u(100.0, 170.0),
                speed=u(0.85, 1.15),
                accuracy=u(0.85, 1.15),
                roll=u(-20.0, 20.0),
                pitch=u(-15.0, 15.0),
                yaw=u(-15.0, 15.0),
                rest_aperture=u(0.60, 1.00),
                hover_low=u(0.30, 0.45),
                closed_aperture=u(0.08, 0.16),
                close_s=u(0.12, 0.22),
                hold_s=u(0.08, 0.20),
                open_s=u(0.12, 0.22),
                verify_s=u(0.10, 0.20),
                slip_bias=float(np.clip(rng.normal(0.0, 0.05), -0.15, 0.15)),
                tremor_hz=u(8.0, 12.0),
                tremor_px=u(0.3, 0.8),
                tremor_phase=(u(0, 2 * np.pi), u(0, 2 * np.pi)),
            )
        )
    return people


def minimum_jerk(p0, p1, tau: float) -> np.ndarray:
    """Minimum-jerk interpolation from p0 to p1 at normalised time tau."""
    s = float(np.clip(tau, 0.0, 1.0))
    shape = 10 * s**3 - 15 * s**4 + 6 * s**5
    return np.asarray(p0, dtype=float) + (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)) * shape


def reach_time(p: Participant, params: SimParams, distance: float, width: float) -> float:
    """Fitts-law reach duration for this participant (s)."""
    index = np.log2(max(distance, 1e-6) / width + 1.0)
    return p.speed * (params.fitts_a + params.fitts_b * index)


def endpoint_sd(p: Participant, width: float) -> float:
    """Per-axis endpoint SD (screen px). Motor noise alone misses about 4 %."""
    return p.accuracy * width / 5.08
