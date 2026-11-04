"""Closed-loop simulation of a participant doing the ISO pointing task.

Per 30 fps frame the simulated hand is rendered as 21 MediaPipe-style
landmarks (``handmodel``), corrupted like a webcam tracker would corrupt it,
and fed to a ``PointerPipeline``. The simulated user watches the pipeline's
cursor, so a laggy filter really does make them wait longer.

One selection:

1. Reach: minimum-jerk movement of the index fingertip to an aimed endpoint
   (target centre + motor noise), lasting a + b * ID seconds (Fitts' law).
2. Settle: wait at least ``verify_s`` and until the cursor has stopped
   (speed below ``settle_speed``), at most ``max_settle_s``.
3. Pinch: the thumb-index aperture closes (minimum jerk), holds, reopens.
   While it closes the index tip travels ``slip`` of the closing distance
   toward the thumb. That drag is the slip this project removes.

After each circle of targets the hand wanders for ``hover_s`` seconds with the
thumb drifting toward the index and the hand moving nearer and farther from
the camera, but no intended click; every click there is a false click.

Noise: per-landmark Gaussian jitter (half shared by the whole hand, half per
landmark, z twice as noisy), 8-12 Hz physiological tremor, and frames where
the tracker returns no hand. All randomness is drawn from generators keyed by
(seed, participant, block), with a fixed number of draws per frame, so every
pointer variant sees the same people, the same aims and the same noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import handmodel as hm
from .fitts import Condition, Trial, circle_targets, selection_order
from .participants import Participant, SimParams, endpoint_sd, minimum_jerk, reach_time
from .pointer import PointerConfig, PointerPipeline, screen_to_camera
from .simmodes import ModeTracker

CLICK_WINDOW_PAD = (0.05, 0.15)  # s before onset / after reopening


@dataclass(frozen=True)
class Selection:
    """Ground truth for one intended pinch."""

    measured: bool
    condition: Condition
    origin: tuple[float, float]
    target: tuple[float, float]
    onset_t: float
    window: tuple[float, float]


@dataclass(frozen=True)
class SessionResult:
    pid: int
    block: int
    variant: str
    trials: tuple[Trial, ...]
    slips: tuple[float, ...]
    selections: int
    misses: int
    duplicate_clicks: int
    stray_clicks: int
    duration_s: float
    modes: dict | None = None

    @property
    def false_clicks(self) -> int:
        return self.duplicate_clicks + self.stray_clicks


@dataclass(frozen=True)
class _Command:
    """What the simulated hand does this frame."""

    index_cam: np.ndarray  # intended index fingertip, camera px (before slip)
    aperture: float  # palm units
    slip: float
    scale: float = 1.0  # multiplies hand size (closer / farther)


class _HandRenderer:
    """Turns commands into noisy camera-pixel landmarks for one participant."""

    def __init__(self, p: Participant, params: SimParams) -> None:
        self.p = p
        self.params = params
        self.base = hm.pose_landmarks("mouse", p.rest_aperture)
        self.rotation = hm.rotation_matrix(p.roll, p.pitch, p.yaw)

    def clean(self, cmd: _Command) -> np.ndarray:
        local = self.base
        if cmd.aperture < self.p.rest_aperture - 1e-9:
            local = hm.pinch_landmarks(self.base, self.p.rest_aperture, cmd.aperture, cmd.slip)
        size = self.p.hand_px * cmd.scale
        pts = (local @ self.rotation.T) * size
        anchor = (self.base[hm.INDEX_TIP] @ self.rotation.T) * size
        pts[:, :2] += cmd.index_cam - anchor[:2]
        return pts

    def noisy(self, cmd: _Command, t: float, normals: np.ndarray, uniform: float) -> np.ndarray | None:
        """Add tremor and jitter; None when the tracker drops this frame."""
        pts = self.clean(cmd)
        phase = 2 * np.pi * self.p.tremor_hz * t
        pts[:, 0] += self.p.tremor_px * np.sin(phase + self.p.tremor_phase[0])
        pts[:, 1] += self.p.tremor_px * np.sin(phase + self.p.tremor_phase[1])
        sd = self.params.jitter_px / np.sqrt(2.0)
        noise = sd * (normals[:21] + normals[21])  # per-landmark + shared
        noise[:, 2] *= 2.0
        pts += noise
        return None if uniform < self.params.drop_prob else pts


class SessionRunner:
    """One participant, one block, one pointer variant."""

    def __init__(self, p: Participant, variant: str, params: SimParams, seed: int, block: int = 0, config: PointerConfig | None = None) -> None:
        self.p = p
        self.params = params
        self.variant = variant
        self.block = block
        self.cfg = config or PointerConfig()
        self.pipeline = PointerPipeline(variant, self.cfg)
        self.renderer = _HandRenderer(p, params)
        self.trial_rng = np.random.default_rng([seed, p.pid, block, 1])
        self.noise_rng = np.random.default_rng([seed, p.pid, block, 2])
        self.modes = ModeTracker(self.cfg) if params.track_modes else None
        self.k = 0
        self.times: list[float] = []
        self.cursors: list[tuple[float, float]] = []
        self.clicks: list[tuple[float, float, float]] = []
        self.selections: list[Selection] = []
        self.hand_cam = screen_to_camera((960.0, 540.0), self.cfg)
        self.release: tuple[float, float, float, float] | None = None  # start, duration, from, slip

    # ---- time and aperture ------------------------------------------------
    @property
    def t(self) -> float:
        return self.k / self.params.fps

    def _resting(self, t: float) -> tuple[float, float]:
        """(aperture, slip) when not pinching: rest, or still reopening."""
        rest = self.p.rest_aperture
        if self.release is None:
            return rest, 0.0
        start, duration, low, slip = self.release
        if t >= start + duration:
            self.release = None
            return rest, 0.0
        return float(minimum_jerk(low, rest, (t - start) / duration)), slip

    def _idle(self, scale: float = 1.0, aperture_cap: float | None = None) -> _Command:
        aperture, slip = self._resting(self.t)
        if aperture_cap is not None and self.release is None:
            aperture = min(aperture, aperture_cap)
        return _Command(self.hand_cam, aperture, slip, scale)

    def _slip(self) -> float:
        noise = float(self.trial_rng.normal(0.0, 0.03))
        return float(np.clip(self.params.slip + self.p.slip_bias + noise, 0.0, 1.0))

    def _cursor_speed(self, frames: int = 3) -> float:
        if len(self.cursors) <= frames:
            return float("inf")
        step = np.subtract(self.cursors[-1], self.cursors[-1 - frames])
        return float(np.linalg.norm(step)) * self.params.fps / frames

    # ---- the script (one yield = one frame) --------------------------------
    def _hold(self, seconds: float):
        end = self.t + seconds
        while self.t < end:
            yield self._idle()

    def _reach(self, goal_cam: np.ndarray, duration: float):
        start = self.hand_cam.copy()
        t0 = self.t
        while self.t < t0 + duration:
            self.hand_cam = minimum_jerk(start, goal_cam, (self.t - t0) / duration)
            yield self._idle()
        self.hand_cam = np.asarray(goal_cam, dtype=float)

    def _select(self, target, origin, condition: Condition, measured: bool):
        p, cfg = self.p, self.cfg
        slip = self._slip()
        aim = np.asarray(target) + self.trial_rng.normal(0.0, endpoint_sd(p, condition.width), 2)
        jitter = self.trial_rng.uniform(0.85, 1.15, 3)  # this trial's pinch timing
        close_s, hold_s, open_s = p.close_s * jitter[0], p.hold_s * jitter[1], p.open_s * jitter[2]
        here_screen = _camera_to_screen(self.hand_cam, cfg)
        duration = reach_time(p, self.params, float(np.linalg.norm(np.subtract(target, here_screen))), condition.width)
        yield from self._reach(screen_to_camera(aim, cfg), duration)
        settle_start = self.t
        while True:
            waited = self.t - settle_start
            arrived = waited >= p.verify_s and self._cursor_speed() < self.params.settle_speed
            if arrived or waited >= self.params.max_settle_s:
                break
            yield self._idle()
        onset = self.t - 1.0 / self.params.fps  # last frame drawn at rest
        rest, closed = p.rest_aperture, p.closed_aperture
        while self.t < onset + close_s:
            a = float(minimum_jerk(rest, closed, (self.t - onset) / close_s))
            yield _Command(self.hand_cam, a, slip)
        while self.t < onset + close_s + hold_s:
            yield _Command(self.hand_cam, closed, slip)
        release_start = self.t
        self.release = (release_start, open_s, closed, slip)
        window = (onset - CLICK_WINDOW_PAD[0], release_start + open_s + CLICK_WINDOW_PAD[1])
        self.selections.append(Selection(measured, condition, tuple(map(float, origin)), tuple(map(float, target)), onset, window))

    def _hover(self, seconds: float):
        rng = self.trial_rng
        t0 = self.t
        cycles = int(rng.integers(2, 5))
        rest, low = self.p.rest_aperture, self.p.hover_low
        while self.t < t0 + seconds:
            goal_screen = np.array([960.0, 540.0]) + rng.uniform(-0.35, 0.35, 2) * (1920.0, 1080.0)
            duration = float(rng.uniform(0.6, 1.4))
            start = self.hand_cam.copy()
            ts = self.t
            goal = screen_to_camera(goal_screen, self.cfg)
            while self.t < ts + duration and self.t < t0 + seconds:
                tau = (self.t - t0) / seconds
                self.hand_cam = minimum_jerk(start, goal, (self.t - ts) / duration)
                dip = 0.5 * (1.0 - np.cos(2 * np.pi * cycles * tau))
                scale = 1.0 + 0.2 * np.sin(2 * np.pi * tau)
                yield self._idle(scale, aperture_cap=rest - (rest - low) * dip)

    def _script(self):
        params = self.params
        order_rng = np.random.default_rng([self.trial_rng.integers(1 << 30), 3])
        conditions = [params.conditions[i] for i in order_rng.permutation(len(params.conditions))]
        first = circle_targets(params.n_targets, conditions[0].distance)[0]
        self.hand_cam = screen_to_camera(first, self.cfg)
        yield from self._hold(params.warmup_s)
        for condition in conditions:
            targets = circle_targets(params.n_targets, condition.distance)
            order = selection_order(params.n_targets)
            for j, ti in enumerate(order):
                origin = targets[order[j - 1]] if j > 0 else targets[ti]
                yield from self._select(targets[ti], origin, condition, measured=j > 0)
            yield from self._hover(params.hover_s)
        yield from self._hold(0.5)

    # ---- main loop ----------------------------------------------------------
    def run(self) -> SessionResult:
        for cmd in self._script():
            t = self.t
            normals = self.noise_rng.standard_normal((22, 3))
            uniform = float(self.noise_rng.uniform())
            landmarks = self.renderer.noisy(cmd, t, normals, uniform)
            frame = self.pipeline.update(landmarks, t)
            self.times.append(t)
            self.cursors.append(frame.cursor)
            self.clicks.extend((c.t, c.x, c.y) for c in frame.clicks)
            if self.modes is not None and landmarks is not None:
                self.modes.update(landmarks, t)
            self.k += 1
        return self._score()

    def _score(self) -> SessionResult:
        times = np.array(self.times)
        cursors = np.array(self.cursors)
        clicks = sorted(self.clicks)
        used = [False] * len(clicks)
        trials, slips = [], []
        misses = duplicates = 0
        prev_click: tuple[float, float, float] | None = None
        for sel in self.selections:
            inside = [i for i, c in enumerate(clicks) if not used[i] and sel.window[0] <= c[0] <= sel.window[1]]
            for i in inside:
                used[i] = True
            duplicates += max(len(inside) - 1, 0)
            click = clicks[inside[0]] if inside else None
            if click is None:
                misses += 1
            else:
                k = max(int(np.searchsorted(times, sel.onset_t + 1e-9)) - 1, 0)
                slips.append(float(np.linalg.norm(np.subtract(click[1:], cursors[k]))))
            if sel.measured:
                start = None if prev_click is None else prev_click[1:]
                mt = None if (click is None or prev_click is None) else click[0] - prev_click[0]
                trials.append(Trial(sel.condition, sel.origin, sel.target, start, None if click is None else click[1:], mt))
            prev_click = click
        stray = used.count(False)
        modes = None if self.modes is None else self.modes.summary(self.t)
        return SessionResult(self.p.pid, self.block, self.variant, tuple(trials), tuple(slips), len(self.selections), misses, duplicates, stray, self.t, modes)


def _camera_to_screen(point_cam, cfg: PointerConfig) -> np.ndarray:
    fr = cfg.frame_reduction
    x = (point_cam[0] - fr) * cfg.screen_w / (cfg.cam_w - 2 * fr)
    y = (point_cam[1] - fr) * cfg.screen_h / (cfg.cam_h - 2 * fr)
    return np.array([x, y])


def simulate_session(p: Participant, variant: str, params: SimParams, seed: int, block: int = 0, config: PointerConfig | None = None) -> SessionResult:
    """Run one block for one participant with one pointer variant."""
    return SessionRunner(p, variant, params, seed, block, config).run()
