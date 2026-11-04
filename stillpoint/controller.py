"""The live controller's decisions, with no camera, GUI or OS calls.

``Controller.step(landmarks, t)`` returns a list of actions (move the cursor,
click, set the volume, switch tab, draw a stroke). ``app`` executes them with
PyAutoGUI, pycaw and OpenCV. Keeping the decisions pure makes every mode
testable headless with synthetic hands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import OneEuroFilter, OneEuroParams
from .gestures import ModeMachine
from .hand import INDEX_TIP, MIDDLE_TIP, FingerStateTracker, as_landmarks, normalized_aperture, palm_scale
from .pointer import PointerConfig, PointerPipeline


@dataclass(frozen=True)
class MoveCursor:
    x: float
    y: float


@dataclass(frozen=True)
class Click:
    x: float
    y: float


@dataclass(frozen=True)
class SetVolume:
    level: float  # 0..1


@dataclass(frozen=True)
class SwitchTab:
    step: int  # +1 next tab, -1 previous tab


@dataclass(frozen=True)
class DrawStroke:
    start: tuple[int, int]  # camera pixels
    end: tuple[int, int]


@dataclass(frozen=True)
class ModeChanged:
    mode: str


@dataclass(frozen=True)
class ControllerConfig:
    variant: str = "stillpoint"
    pointer: PointerConfig = field(default_factory=PointerConfig)
    dwell_s: float = 0.20
    volume_aperture: tuple[float, float] = (0.15, 1.20)  # palm units -> 0 % .. 100 %
    tab_swipe_palms: float = 0.6  # hand travel, in palm sizes, for one tab
    tab_cooldown_s: float = 0.8
    draw_filter: OneEuroParams = field(default_factory=lambda: OneEuroParams(min_cutoff=1.0, beta=0.02))


class Controller:
    """Mode machine plus one handler per mode."""

    def __init__(self, cfg: ControllerConfig | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self.fingers = FingerStateTracker()
        self.modes = ModeMachine(initial="idle", dwell_s=self.cfg.dwell_s)
        self.pointer = PointerPipeline(self.cfg.variant, self.cfg.pointer)
        self.volume_filter = OneEuroFilter(OneEuroParams(min_cutoff=1.0, beta=0.5))
        self.draw_filter = OneEuroFilter(self.cfg.draw_filter)
        self.pinch_state = "open"
        self.volume: float | None = None
        self._reset_transient()

    @property
    def mode(self) -> str:
        return self.modes.mode

    def _reset_transient(self) -> None:
        self.draw_prev: tuple[int, int] | None = None
        self.tab_anchor: float | None = None
        self.last_tab_t = -np.inf
        self.draw_filter.reset()

    def step(self, landmarks, t: float) -> list:
        if landmarks is None:
            self.modes.lost_hand()
            self.draw_prev, self.tab_anchor = None, None
            if self.mode == "mouse":
                self.pointer.update(None, t)
            return []
        lm = as_landmarks(landmarks)
        actions: list = []
        aperture = normalized_aperture(lm)
        locked = aperture < self.cfg.pointer.pinch.release_above or self.pinch_state != "open"
        event = self.modes.update(self.fingers.update(lm), t, pinch_locked=locked)
        if event is not None:
            self._reset_transient()
            self.pinch_state = "open"
            actions.append(ModeChanged(event.mode))
        handler = {"mouse": self._mouse, "volume": self._volume, "draw": self._draw, "tabs": self._tabs}.get(self.mode)
        if handler is not None:
            actions.extend(handler(lm, t, aperture))
        return actions

    def _mouse(self, lm, t, aperture) -> list:
        frame = self.pointer.update(lm, t)
        self.pinch_state = frame.pinch
        if frame.skipped:
            return []
        actions: list = [MoveCursor(*frame.cursor)]
        actions.extend(Click(c.x, c.y) for c in frame.clicks)
        return actions

    def _volume(self, lm, t, aperture) -> list:
        lo, hi = self.cfg.volume_aperture
        raw = float(np.interp(aperture, (lo, hi), (0.0, 1.0)))
        level = float(np.clip(self.volume_filter(raw, t), 0.0, 1.0))
        if self.volume is not None and abs(level - self.volume) < 0.01:
            return []
        self.volume = level
        return [SetVolume(level)]

    def _draw(self, lm, t, aperture) -> list:
        tip = self.draw_filter(lm[INDEX_TIP, :2], t)
        point = (int(round(tip[0])), int(round(tip[1])))
        prev = self.draw_prev or point
        self.draw_prev = point
        return [DrawStroke(prev, point)]

    def _tabs(self, lm, t, aperture) -> list:
        hand_x = 0.5 * (lm[INDEX_TIP, 0] + lm[MIDDLE_TIP, 0])
        if self.tab_anchor is None:
            self.tab_anchor = hand_x
            return []
        travel = (hand_x - self.tab_anchor) / palm_scale(lm)
        if t - self.last_tab_t <= self.cfg.tab_cooldown_s or abs(travel) <= self.cfg.tab_swipe_palms:
            return []
        self.last_tab_t = t
        self.tab_anchor = hand_x
        return [SwitchTab(1 if travel > 0 else -1)]
