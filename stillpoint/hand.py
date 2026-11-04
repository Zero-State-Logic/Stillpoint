"""Hand geometry on MediaPipe's 21-landmark layout.

Landmarks are a ``(21, 2)`` or ``(21, 3)`` array in camera pixels (z, when
present, in the same pixel scale as x). Everything here is a pure function of
that array, and every quantity is either a ratio of lengths or an angle, so it
does not change when the hand moves, turns, or comes closer to the camera.

* ``palm_scale``: a hand-size yardstick (mean of wrist-to-middle-knuckle and
  index-to-pinky knuckle span).
* ``normalized_aperture``: thumb-tip to index-tip distance in palm units, so a
  pinch threshold means the same thing at 30 cm and at 1 m from the camera.
* ``finger_curls`` / ``thumb_spread``: joint-angle features. Angles between
  bone vectors do not change under rotation, and the thumb is measured along
  the hand's own pinky-to-index axis, so the same rule works for a left or a
  right hand, upright or tilted (the original compared raw x and y values).
* ``FingerStateTracker``: per-finger hysteresis on those features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (MCP, PIP, DIP, TIP) for the four long fingers: index, middle, ring, pinky.
FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

NUM_LANDMARKS = 21
_EPS = 1e-9


def as_landmarks(landmarks) -> np.ndarray:
    """Validate and return landmarks as a float array of shape (21, 2|3)."""
    arr = np.asarray(landmarks, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != NUM_LANDMARKS or arr.shape[1] not in (2, 3):
        raise ValueError(f"expected (21, 2) or (21, 3) landmarks, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN or inf")
    return arr


def palm_scale(landmarks) -> float:
    """Hand size in pixels: mean of |wrist-middle MCP| and |index MCP-pinky MCP|.

    Uses the image plane only (x, y) because MediaPipe's z is a relative,
    noisier estimate. Raises ``ValueError`` for a degenerate (zero-size) hand.
    """
    lm = as_landmarks(landmarks)[:, :2]
    length = np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])
    span = np.linalg.norm(lm[INDEX_MCP] - lm[PINKY_MCP])
    scale = 0.5 * (length + span)
    if scale < _EPS:
        raise ValueError("degenerate hand: zero palm size")
    return float(scale)


def pinch_distance_px(landmarks) -> float:
    """Raw thumb-tip to index-tip distance in image pixels (the original metric)."""
    lm = as_landmarks(landmarks)
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2]))


def normalized_aperture(landmarks) -> float:
    """Thumb-tip to index-tip distance divided by ``palm_scale`` (image plane)."""
    return pinch_distance_px(landmarks) / palm_scale(landmarks)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors (0 when parallel)."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < _EPS or nv < _EPS:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def finger_curls(landmarks) -> np.ndarray:
    """Curl of index, middle, ring, pinky in degrees (PIP flexion + DIP flexion).

    A straight finger reads near 0; a fist reads 150 or more. Knuckle (MCP)
    flexion is left out on purpose, so a finger pointing at the camera still
    counts as extended.
    """
    lm = as_landmarks(landmarks)
    curls = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        proximal = lm[pip] - lm[mcp]
        middle = lm[dip] - lm[pip]
        distal = lm[tip] - lm[dip]
        curls.append(_angle_between(proximal, middle) + _angle_between(middle, distal))
    return np.array(curls)


def thumb_spread(landmarks) -> float:
    """How far the thumb tip sticks out past the index knuckle, in palm units.

    Measured along the hand's own lateral axis (pinky MCP -> index MCP), so it
    is the same number for a left hand, a right hand, a tilted hand, or a hand
    seen from the back. A thumb tucked against the palm or resting on the index
    finger reads near 0 or below; an "L" thumb reads about 0.6 or more.
    Uses z when it is present.
    """
    lm = as_landmarks(landmarks)
    lateral = lm[INDEX_MCP] - lm[PINKY_MCP]
    norm = np.linalg.norm(lateral)
    if norm < _EPS:
        raise ValueError("degenerate hand: index and pinky knuckles coincide")
    offset = float(np.dot(lm[THUMB_TIP] - lm[INDEX_MCP], lateral / norm))
    # Same yardstick as palm_scale but in every available dimension, so the
    # ratio is unchanged by any 3-D rotation of the hand.
    size = 0.5 * (np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]) + norm)
    return offset / float(size)


@dataclass(frozen=True)
class FingerThresholds:
    """Hysteresis bands. A finger flips state only when it leaves its band."""

    curl_extended_below: float = 60.0  # degrees: becomes "extended" below this
    curl_folded_above: float = 95.0  # degrees: becomes "folded" above this
    thumb_out_above: float = 0.50  # palm units: thumb becomes "extended"
    thumb_in_below: float = 0.32  # palm units: thumb becomes "folded"

    def __post_init__(self) -> None:
        if not self.curl_extended_below < self.curl_folded_above:
            raise ValueError("curl band is empty")
        if not self.thumb_in_below < self.thumb_out_above:
            raise ValueError("thumb band is empty")


def finger_states(landmarks, thresholds: FingerThresholds | None = None) -> tuple[int, ...]:
    """Stateless [thumb, index, middle, ring, pinky] (1 = extended).

    Uses the middle of each hysteresis band. ``FingerStateTracker`` adds the
    hysteresis for live use.
    """
    th = thresholds or FingerThresholds()
    curl_mid = 0.5 * (th.curl_extended_below + th.curl_folded_above)
    thumb_mid = 0.5 * (th.thumb_in_below + th.thumb_out_above)
    thumb = int(thumb_spread(landmarks) > thumb_mid)
    fingers = tuple(int(c < curl_mid) for c in finger_curls(landmarks))
    return (thumb,) + fingers


class FingerStateTracker:
    """Finger states with per-finger hysteresis, so noise near a threshold
    cannot make a finger flicker between extended and folded."""

    def __init__(self, thresholds: FingerThresholds | None = None) -> None:
        self.thresholds = thresholds or FingerThresholds()
        self._state: tuple[int, ...] | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, landmarks) -> tuple[int, ...]:
        th = self.thresholds
        if self._state is None:
            self._state = finger_states(landmarks, th)
            return self._state
        spread = thumb_spread(landmarks)
        thumb = _band(self._state[0], spread, th.thumb_out_above, th.thumb_in_below, higher_is_on=True)
        fingers = tuple(
            _band(prev, curl, th.curl_extended_below, th.curl_folded_above, higher_is_on=False)
            for prev, curl in zip(self._state[1:], finger_curls(landmarks))
        )
        self._state = (thumb,) + fingers
        return self._state


def _band(prev: int, value: float, on_edge: float, off_edge: float, higher_is_on: bool) -> int:
    """Two-threshold switch: turn on past ``on_edge``, off past ``off_edge``."""
    if higher_is_on:
        if value > on_edge:
            return 1
        if value < off_edge:
            return 0
        return prev
    if value < on_edge:
        return 1
    if value > off_edge:
        return 0
    return prev
