"""Mode state machine: which of the four modes the hand is asking for.

The same four poses as the original (index = mouse, thumb + index = volume,
open palm = draw, index + middle = tabs), with three changes that stop
accidental switches:

1. Finger states come from ``hand.FingerStateTracker`` (joint angles with
   hysteresis), not raw x/y comparisons.
2. Dwell: a new pose must be held for ``dwell_s`` without a break before the
   mode changes. A one-frame misread cannot switch modes.
3. Pinch lock: while thumb and index are pinched or closing, the mode is
   frozen. A pinch moves the thumb, and that must never turn a click into a
   mode change.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("mouse", "volume", "draw", "tabs")

_PATTERNS = {
    (0, 1, 0, 0, 0): "mouse",
    (1, 1, 0, 0, 0): "volume",
    (1, 1, 1, 1, 1): "draw",
    (0, 1, 1, 0, 0): "tabs",
}


def pattern_to_mode(states) -> str | None:
    """Map [thumb, index, middle, ring, pinky] to a mode name, or None."""
    return _PATTERNS.get(tuple(int(s) for s in states))


@dataclass(frozen=True)
class ModeEvent:
    t: float
    previous: str
    mode: str


class ModeMachine:
    """Dwell-and-lock mode switching. Call ``update`` once per frame."""

    def __init__(self, initial: str = "idle", dwell_s: float = 0.20) -> None:
        if dwell_s < 0:
            raise ValueError("dwell_s must be >= 0")
        self.mode = initial
        self.dwell_s = dwell_s
        self._candidate: str | None = None
        self._since: float | None = None

    def reset(self, mode: str = "idle") -> None:
        self.mode = mode
        self._candidate = None
        self._since = None

    def update(self, states, t: float, pinch_locked: bool = False) -> ModeEvent | None:
        """Feed one frame of finger states. Returns a ``ModeEvent`` on a switch."""
        if pinch_locked:
            self._candidate = None
            self._since = None
            return None
        requested = pattern_to_mode(states)
        if requested is None or requested == self.mode:
            self._candidate = None
            self._since = None
            return None
        if requested != self._candidate:
            self._candidate = requested
            self._since = t
        if self._since is not None and t - self._since >= self.dwell_s:
            event = ModeEvent(t=t, previous=self.mode, mode=requested)
            self.mode = requested
            self._candidate = None
            self._since = None
            return event
        return None

    def lost_hand(self) -> None:
        """No hand this frame: forget any half-held candidate, keep the mode."""
        self._candidate = None
        self._since = None
