"""Mode-stability bookkeeping for simulated sessions.

During the pointing task the right answer is always "mouse". This runs the
original classifier (x/y comparisons + 0.5 s cooldown) and STILLPOINT's
(joint angles with hysteresis + dwell + pinch lock) on the same landmark
stream, for the participant's right hand and for its mirror image (the same
movements made with a left hand), and counts how often each one leaves mouse
mode.
"""

from __future__ import annotations

import numpy as np

from .gestures import ModeMachine
from .hand import FingerStateTracker, normalized_aperture
from .legacy import LegacyModeSwitcher, legacy_fingers_up
from .pointer import PointerConfig

EXPECTED = "mouse"


class _Counter:
    def __init__(self) -> None:
        self.frames = 0
        self.wrong = 0
        self.switches = 0
        self.last = EXPECTED

    def add(self, mode: str) -> None:
        self.frames += 1
        self.wrong += int(mode != EXPECTED)
        self.switches += int(mode != self.last)
        self.last = mode

    def summary(self, minutes: float) -> dict:
        return {
            "frames": self.frames,
            "wrong_fraction": self.wrong / self.frames if self.frames else float("nan"),
            "switches": self.switches,
            "switches_per_min": self.switches / minutes if minutes > 0 else float("nan"),
        }


class ModeTracker:
    """Runs both classifiers on both hands and counts mode errors."""

    def __init__(self, cfg: PointerConfig) -> None:
        self.cfg = cfg
        self.legacy = {h: LegacyModeSwitcher(initial=EXPECTED) for h in ("right", "left")}
        self.trackers = {h: FingerStateTracker() for h in ("right", "left")}
        self.machines = {h: ModeMachine(initial=EXPECTED) for h in ("right", "left")}
        self.counts = {(c, h): _Counter() for c in ("original", "stillpoint") for h in ("right", "left")}

    def update(self, landmarks: np.ndarray, t: float) -> None:
        mirrored = np.array(landmarks, copy=True)
        mirrored[:, 0] = self.cfg.cam_w - mirrored[:, 0]
        for hand, pts in (("right", landmarks), ("left", mirrored)):
            old = self.legacy[hand].update(legacy_fingers_up(pts), t)
            self.counts[("original", hand)].add(old)
            states = self.trackers[hand].update(pts)
            locked = normalized_aperture(pts) < self.cfg.pinch.release_above
            self.machines[hand].update(states, t, pinch_locked=locked)
            self.counts[("stillpoint", hand)].add(self.machines[hand].mode)

    def summary(self, duration_s: float) -> dict:
        minutes = duration_s / 60.0
        return {f"{c}/{h}": counter.summary(minutes) for (c, h), counter in self.counts.items()}


def roll_sweep(seed: int, angles=tuple(range(-90, 91, 15)), samples: int = 40, jitter_px: float = 1.5) -> dict:
    """Single-frame pose recognition accuracy versus hand roll.

    For each roll angle: the four poses, random pitch/yaw (+-15 deg), hand
    size and thumb rest gap, tracker-like jitter, for a right hand and its
    mirror (left hand). Returns the fraction recognised correctly by the
    original rules and by STILLPOINT's joint-angle rules.
    """
    from . import handmodel as hm
    from .gestures import pattern_to_mode
    from .hand import finger_states
    from .legacy import fingers_to_mode

    rng = np.random.default_rng([seed, 4242])
    out = {f"{c}/{h}": [] for c in ("original", "stillpoint") for h in ("right", "left")}
    for angle in angles:
        hits = {key: 0 for key in out}
        total = 0
        for pose in hm.POSES:
            for _ in range(samples):
                local = hm.pose_landmarks(pose, rng.uniform(0.6, 1.0))
                rot = hm.rotation_matrix(angle, rng.uniform(-15, 15), rng.uniform(-15, 15))
                size = rng.uniform(100, 170)
                noise = jitter_px / np.sqrt(2.0) * (rng.standard_normal((21, 3)) + rng.standard_normal(3))
                noise[:, 2] *= 2.0
                total += 1
                for hand_name, pts_local in (("right", local), ("left", hm.mirror(local))):
                    pts = hm.to_camera(pts_local, rot, size, (640.0, 360.0)) + noise
                    hits[f"original/{hand_name}"] += fingers_to_mode(legacy_fingers_up(pts)) == pose
                    hits[f"stillpoint/{hand_name}"] += pattern_to_mode(finger_states(pts)) == pose
        for key in out:
            out[key].append(hits[key] / total)
    return {"angles": list(angles), "accuracy": out, "samples_per_angle": len(hm.POSES) * samples}
