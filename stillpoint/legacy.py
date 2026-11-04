"""The original controller's rules, kept verbatim as the baseline.

These reproduce ``hand_tracker.fingers_up`` and ``gesture_controller``'s mode
logic from the first version of this project, so the evaluation compares
against what actually shipped, not a straw man.
"""

from __future__ import annotations

import numpy as np

TIP_IDS = (4, 8, 12, 16, 20)
MODE_SWITCH_COOLDOWN = 0.5  # seconds, as in the original


def to_int_pixels(landmarks) -> np.ndarray:
    """The original stored ``int(lm.x * w), int(lm.y * h)`` (truncation)."""
    arr = np.asarray(landmarks, dtype=float)[:, :2]
    return np.trunc(arr).astype(int)


def legacy_fingers_up(landmarks) -> list[int]:
    """Original rule: thumb tip x > thumb IP x; other tips above their PIP joint."""
    lms = to_int_pixels(landmarks)
    if len(lms) < 21:
        return [0, 0, 0, 0, 0]
    fingers = [1 if lms[TIP_IDS[0]][0] > lms[TIP_IDS[0] - 1][0] else 0]
    for i in range(1, 5):
        fingers.append(1 if lms[TIP_IDS[i]][1] < lms[TIP_IDS[i] - 2][1] else 0)
    return fingers


def fingers_to_mode(fingers) -> str | None:
    """Original dispatcher: exact finger patterns to modes."""
    pattern = list(fingers)
    if pattern == [0, 1, 0, 0, 0]:
        return "mouse"
    if pattern == [1, 1, 0, 0, 0]:
        return "volume"
    if pattern == [1, 1, 1, 1, 1]:
        return "draw"
    if pattern == [0, 1, 1, 0, 0]:
        return "tabs"
    return None


class LegacyModeSwitcher:
    """Original switching: any recognised pattern wins after a 0.5 s cooldown."""

    def __init__(self, initial: str = "idle", cooldown: float = MODE_SWITCH_COOLDOWN) -> None:
        self.mode = initial
        self.cooldown = cooldown
        self._last_switch = -np.inf

    def update(self, fingers, t: float) -> str:
        requested = fingers_to_mode(fingers)
        if requested and requested != self.mode and (t - self._last_switch) > self.cooldown:
            self.mode = requested
            self._last_switch = t
        return self.mode
