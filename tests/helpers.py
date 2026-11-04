"""Shared test helpers: synthetic hands in camera pixels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stillpoint import handmodel as hm  # noqa: E402


def make_hand(pose: str = "mouse", aperture: float = 0.8, at=(640.0, 360.0), size: float = 120.0,
              roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, left: bool = False) -> np.ndarray:
    """A clean hand in camera pixels with its index tip at ``at``."""
    local = hm.pose_landmarks(pose, aperture)
    if left:
        local = hm.mirror(local)
    return hm.to_camera(local, hm.rotation_matrix(roll, pitch, yaw), size, at)


def pinch_hand(rest: float, aperture: float, slip: float, at=(640.0, 360.0), size: float = 120.0) -> np.ndarray:
    """A mouse-pose hand part-way through a pinch; ``at`` is the resting index tip."""
    base = hm.pose_landmarks("mouse", rest)
    local = hm.pinch_landmarks(base, rest, aperture, slip)
    rot = hm.rotation_matrix()
    anchor = (base[hm.INDEX_TIP] @ rot.T) * size
    pts = (local @ rot.T) * size
    pts[:, :2] += np.asarray(at) - anchor[:2]
    return pts
