"""The pointer pipeline: landmarks in, cursor and clicks out.

    landmarks -> active-region mapping -> filter -> pinch logic -> events

Four variants form an ablation ladder; each adds one idea to the one before:

``baseline``
    The original controller, exactly: fixed smoothing ``x += (t - x) / 5``
    starting at the screen origin, click when thumb and index tips are closer
    than 30 camera pixels, click at the current cursor, then a 0.25 s sleep
    during which camera frames are lost (a pinch still closed after the sleep
    clicks again).
``oneeuro``
    Baseline with the One Euro filter instead of the fixed smoother.
``oneeuro+hysteresis``
    Plus the scale-normalised pinch with hysteresis: one click per pinch on
    the press edge, no sleep, same behaviour at any distance from the camera.
``stillpoint``
    Plus pinch-onset rewind: when the aperture starts to fall fast the
    cursor freezes, and the click lands where the cursor was just before the
    fingers began to close. The freeze lasts until the pinch is released.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import VARIANTS
from .filters import ExponentialSmoother, OneEuroFilter, OneEuroParams
from .hand import INDEX_TIP, as_landmarks, normalized_aperture, pinch_distance_px
from .pinch import ApertureVelocity, CursorSample, HysteresisPinch, PinchParams, RewindBuffer


@dataclass(frozen=True)
class PointerConfig:
    """Geometry and tuning shared by every variant."""

    cam_w: int = 1280
    cam_h: int = 720
    screen_w: int = 1920
    screen_h: int = 1080
    frame_reduction: int = 100  # px dead band at the camera edges (original)
    smoothing: float = 5.0  # original fixed smoother
    click_px: float = 30.0  # original absolute pinch threshold
    click_sleep_s: float = 0.25  # original post-click sleep
    one_euro: OneEuroParams = field(default_factory=lambda: OneEuroParams(min_cutoff=0.6, beta=0.012, d_cutoff=1.0))
    pinch: PinchParams = field(default_factory=PinchParams)
    onset_speed: float = 2.0  # palm units/s of closing that freezes the cursor
    still_speed: float = 0.6  # closing slower than this is "not closing yet"
    max_rewind_s: float = 0.40  # never rewind further back than this
    cancel_after_s: float = 0.40  # a freeze with no press by then is dropped
    buffer_s: float = 1.0
    lost_hand_reset_s: float = 0.5

    def __post_init__(self) -> None:
        if 2 * self.frame_reduction >= min(self.cam_w, self.cam_h):
            raise ValueError("frame_reduction leaves no active region")


@dataclass(frozen=True)
class Click:
    t: float
    x: float
    y: float
    onset_t: float | None = None  # when the closing run began (stillpoint only)


@dataclass(frozen=True)
class PointerFrame:
    t: float
    cursor: tuple[float, float]
    clicks: tuple[Click, ...] = ()
    pinch: str = "open"  # open | closing | pressed
    aperture: float | None = None
    frozen: bool = False
    skipped: bool = False  # frame lost (no hand, or inside the original's sleep)


def map_to_screen(point_px, cfg: PointerConfig) -> np.ndarray:
    """Original active-region mapping (``np.interp`` clamps at the edges)."""
    fr = cfg.frame_reduction
    x = np.interp(point_px[0], (fr, cfg.cam_w - fr), (0, cfg.screen_w))
    y = np.interp(point_px[1], (fr, cfg.cam_h - fr), (0, cfg.screen_h))
    return np.array([x, y], dtype=float)


def screen_to_camera(point_screen, cfg: PointerConfig) -> np.ndarray:
    """Inverse of ``map_to_screen`` inside the active region (no clamping)."""
    fr = cfg.frame_reduction
    x = fr + point_screen[0] * (cfg.cam_w - 2 * fr) / cfg.screen_w
    y = fr + point_screen[1] * (cfg.cam_h - 2 * fr) / cfg.screen_h
    return np.array([x, y], dtype=float)


class _LegacyClicker:
    """Original click rule: absolute 30 px threshold, then a blocking sleep."""

    def __init__(self, cfg: PointerConfig) -> None:
        self.cfg = cfg
        self.blocked_until = -np.inf

    def blocked(self, t: float) -> bool:
        return t < self.blocked_until

    def step(self, t, landmarks, cursor) -> tuple[tuple[Click, ...], str, bool]:
        if pinch_distance_px(landmarks) < self.cfg.click_px:
            self.blocked_until = t + self.cfg.click_sleep_s
            return (Click(t, float(cursor[0]), float(cursor[1])),), "pressed", False
        return (), "open", False

    def frozen_cursor(self) -> tuple[float, float] | None:
        return None

    def lost_hand(self) -> None:
        pass


class _HysteresisClicker:
    """Normalised aperture, two thresholds, click on the press edge."""

    def __init__(self, cfg: PointerConfig) -> None:
        self.cfg = cfg
        self.switch = HysteresisPinch(cfg.pinch)

    def blocked(self, t: float) -> bool:
        return False

    def step(self, t, landmarks, cursor) -> tuple[tuple[Click, ...], str, bool]:
        edge = self.switch.update(normalized_aperture(landmarks))
        clicks = (Click(t, float(cursor[0]), float(cursor[1])),) if edge == "press" else ()
        return clicks, "pressed" if self.switch.pressed else "open", False

    def frozen_cursor(self) -> tuple[float, float] | None:
        return None

    def lost_hand(self) -> None:
        self.switch.reset()


class _RewindClicker(_HysteresisClicker):
    """Hysteresis pinch plus onset freeze and rewind to the pre-closing cursor."""

    def __init__(self, cfg: PointerConfig) -> None:
        super().__init__(cfg)
        self.velocity = ApertureVelocity()
        self.buffer = RewindBuffer(cfg.buffer_s)
        self.anchor: CursorSample | None = None  # frozen cursor, if any
        self.anchor_since = 0.0

    def step(self, t, landmarks, cursor) -> tuple[tuple[Click, ...], str, bool]:
        aperture = normalized_aperture(landmarks)
        speed = self.velocity.update(t, aperture)
        self.buffer.push(CursorSample(t, float(cursor[0]), float(cursor[1]), aperture))
        edge = self.switch.update(aperture)
        clicks: tuple[Click, ...] = ()
        if edge == "press":
            anchor = self.anchor or self._find_anchor()
            clicks = (Click(t, anchor.x, anchor.y, onset_t=anchor.t),)
            self.anchor = anchor
        elif edge == "release":
            self.anchor = None
        elif not self.switch.pressed:
            self._update_freeze(t, speed)
        if self.switch.pressed:
            state = "pressed"
        else:
            state = "closing" if self.anchor is not None else "open"
        return clicks, state, self.anchor is not None

    def _find_anchor(self) -> CursorSample:
        found = self.buffer.closing_start(self.cfg.still_speed, self.cfg.max_rewind_s)
        if found is None:
            raise RuntimeError("rewind buffer is empty")
        return found

    def _update_freeze(self, t: float, speed: float) -> None:
        if self.anchor is None:
            if speed < -self.cfg.onset_speed:
                self.anchor = self._find_anchor()
                self.anchor_since = t
            return
        timed_out = t - self.anchor_since > self.cfg.cancel_after_s
        if timed_out or speed > 0:
            self.anchor = None  # fingers stopped or reopened: not a click

    def frozen_cursor(self) -> tuple[float, float] | None:
        return None if self.anchor is None else (self.anchor.x, self.anchor.y)

    def lost_hand(self) -> None:
        super().lost_hand()
        self.velocity.reset()
        self.buffer.clear()
        self.anchor = None


class PointerPipeline:
    """Run one variant frame by frame. ``update(None, t)`` means no hand."""

    def __init__(self, variant: str = "stillpoint", config: PointerConfig | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        self.variant = variant
        self.cfg = config or PointerConfig()
        if variant == "baseline":
            self.filter = ExponentialSmoother(self.cfg.smoothing, initial=(0.0, 0.0))
        else:
            self.filter = OneEuroFilter(self.cfg.one_euro)
        if variant in ("baseline", "oneeuro"):
            self.clicker = _LegacyClicker(self.cfg)
        elif variant == "oneeuro+hysteresis":
            self.clicker = _HysteresisClicker(self.cfg)
        else:
            self.clicker = _RewindClicker(self.cfg)
        self.cursor = (0.0, 0.0)
        self._last_seen = None

    def update(self, landmarks, t: float) -> PointerFrame:
        if self.clicker.blocked(t):
            return PointerFrame(t, self.cursor, skipped=True, pinch="pressed")
        if landmarks is None:
            return self._no_hand(t)
        lm = as_landmarks(landmarks)
        self._last_seen = t
        target = map_to_screen(lm[INDEX_TIP, :2], self.cfg)
        filtered = self.filter(target, t)
        clicks, state, frozen = self.clicker.step(t, lm, filtered)
        shown = self.clicker.frozen_cursor() if frozen else None
        self.cursor = shown if shown is not None else (float(filtered[0]), float(filtered[1]))
        aperture = normalized_aperture(lm)
        return PointerFrame(t, self.cursor, clicks, state, aperture, frozen)

    def _no_hand(self, t: float) -> PointerFrame:
        if self._last_seen is not None and t - self._last_seen > self.cfg.lost_hand_reset_s:
            self.clicker.lost_hand()
        return PointerFrame(t, self.cursor, skipped=True)
