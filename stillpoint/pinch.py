"""Pinch detection and the onset rewind.

Three pieces, all working on the scale-normalised aperture (thumb-index
distance in palm units, see ``hand.normalized_aperture``):

* ``HysteresisPinch``: press below one threshold, release above a higher
  one. Noise near a single threshold makes a detector chatter (click, click,
  click); a gap between the two thresholds wider than the noise makes that
  impossible.
* ``ApertureVelocity``: a lightly smoothed rate of change of the aperture.
  Fingers starting to close show up here a few frames before the press.
* ``RewindBuffer``: the last second of (time, cursor, aperture) samples. On a
  press it walks back through the closing run to the last sample before the
  aperture started to fall, which is where the cursor was before the click
  gesture began to drag the index fingertip.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PinchParams:
    """Thresholds in palm units (aperture / palm size)."""

    press_below: float = 0.22
    release_above: float = 0.32

    def __post_init__(self) -> None:
        if not 0 < self.press_below < self.release_above:
            raise ValueError("need 0 < press_below < release_above")


class HysteresisPinch:
    """Two-threshold pinch switch. ``update`` returns 'press', 'release' or None."""

    def __init__(self, params: PinchParams | None = None) -> None:
        self.params = params or PinchParams()
        self.pressed = False

    def reset(self) -> None:
        self.pressed = False

    def update(self, aperture: float) -> str | None:
        if not self.pressed and aperture < self.params.press_below:
            self.pressed = True
            return "press"
        if self.pressed and aperture > self.params.release_above:
            self.pressed = False
            return "release"
        return None


class SingleThresholdPinch:
    """One threshold, no hysteresis. Only used to show why hysteresis matters."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.pressed = False

    def update(self, aperture: float) -> str | None:
        now = aperture < self.threshold
        if now == self.pressed:
            return None
        self.pressed = now
        return "press" if now else "release"


class ApertureVelocity:
    """Exponentially smoothed d(aperture)/dt in palm units per second."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._last: tuple[float, float] | None = None
        self.value = 0.0

    def update(self, t: float, aperture: float) -> float:
        if self._last is not None and t > self._last[0]:
            raw = (aperture - self._last[1]) / (t - self._last[0])
            self.value += self.alpha * (raw - self.value)
        self._last = (t, aperture)
        return self.value


@dataclass(frozen=True)
class CursorSample:
    t: float
    x: float
    y: float
    aperture: float


class RewindBuffer:
    """Ring buffer of recent cursor samples with a closing-onset search."""

    def __init__(self, horizon_s: float = 1.0) -> None:
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        self.horizon_s = horizon_s
        self._samples: deque[CursorSample] = deque()

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def push(self, sample: CursorSample) -> None:
        self._samples.append(sample)
        while self._samples and sample.t - self._samples[0].t > self.horizon_s:
            self._samples.popleft()

    def latest(self) -> CursorSample | None:
        return self._samples[-1] if self._samples else None

    def closing_start(self, still_speed: float, max_lookback_s: float) -> CursorSample | None:
        """Last sample before the current closing run began.

        Walks back from the newest sample while the aperture was falling
        faster than ``still_speed`` (palm units/s), never further back than
        ``max_lookback_s``. Returns None when the buffer is empty.
        """
        samples = list(self._samples)
        if not samples:
            return None
        newest_t = samples[-1].t
        k = len(samples) - 1
        while k >= 1:
            prev, cur = samples[k - 1], samples[k]
            dt = cur.t - prev.t
            if dt <= 0 or newest_t - prev.t > max_lookback_s:
                break
            if (cur.aperture - prev.aperture) / dt >= -still_speed:
                break
            k -= 1
        return samples[k]
