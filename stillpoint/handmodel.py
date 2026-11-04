"""A small kinematic hand model that writes MediaPipe-style landmarks.

The simulator and the tests need realistic 21-point hands without a camera.
This module builds them from joint angles:

* Hand frame: wrist at the origin, palm length (wrist to middle knuckle) = 1,
  fingers point up the image (-y), the index side is +x, and fingers curl
  toward +z (away from the camera). This is a right hand seen in the mirrored
  (selfie) feed with the back of the hand to the camera, which is the
  orientation the original controller's thumb rule expects; ``mirror`` gives
  the left hand.
* Long fingers: three bones whose directions rotate toward +z by the
  cumulative knuckle flexion, so PIP + DIP flexion equals the curl that
  ``hand.finger_curls`` measures.
* Thumb: fixed base joint, then a two-link inverse-kinematics solve that puts
  the tip where the pose asks (resting on the index, sticking out, or meeting
  the index tip during a pinch).
"""

from __future__ import annotations

import numpy as np

from .hand import (
    FINGER_CHAINS,
    INDEX_DIP,
    INDEX_PIP,
    INDEX_TIP,
    NUM_LANDMARKS,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    palm_scale,
)

# Knuckle (MCP) positions in the hand frame: index, middle, ring, pinky.
_MCP_POS = np.array(
    [[0.30, -0.95, 0.0], [0.0, -1.0, 0.0], [-0.27, -0.93, 0.0], [-0.50, -0.83, 0.0]]
)
_SPLAY_DEG = (5.0, 0.0, -5.0, -12.0)  # positive leans toward the thumb side
_BONES = np.array(
    [[0.42, 0.25, 0.20], [0.46, 0.28, 0.21], [0.43, 0.26, 0.20], [0.34, 0.20, 0.18]]
)
_THUMB_CMC = np.array([0.22, -0.25, 0.0])
_THUMB_BONES = (0.42, 0.34, 0.28)  # metacarpal, proximal, distal
_THUMB_AXIS = np.array([0.6, -0.8, 0.1]) / np.linalg.norm([0.6, -0.8, 0.1])
_THUMB_BULGE = np.array([1.0, 0.0, -0.3])  # the IP joint bows outward, this way
# Direction from the index tip to a resting (not pinching) thumb tip: down the
# finger and tucked across the palm, slightly in front of it.
REST_DIRECTION = np.array([-0.45, 1.0, 0.3]) / np.linalg.norm([-0.45, 1.0, 0.3])

EXTENDED = (8.0, 6.0, 4.0)  # MCP, PIP, DIP flexion in degrees
CURLED = (70.0, 95.0, 60.0)
THUMB_OUT_TIP = np.array([0.95, -0.55, -0.05])

POSES = {
    "mouse": ((EXTENDED, CURLED, CURLED, CURLED), "rest"),
    "volume": ((EXTENDED, CURLED, CURLED, CURLED), "out"),
    "draw": ((EXTENDED, EXTENDED, EXTENDED, EXTENDED), "out"),
    "tabs": ((EXTENDED, EXTENDED, CURLED, CURLED), "rest"),
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def finger_points(index: int, flex) -> np.ndarray:
    """PIP, DIP and TIP of long finger ``index`` (0 = index ... 3 = pinky)."""
    mcp_flex, pip_flex, dip_flex = flex
    splay = np.radians(_SPLAY_DEG[index])
    base = np.array([np.sin(splay), -np.cos(splay), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    points = []
    joint = _MCP_POS[index]
    angle = 0.0
    for bone, add in zip(_BONES[index], (mcp_flex, pip_flex, dip_flex)):
        angle += np.radians(add)
        joint = joint + bone * (np.cos(angle) * base + np.sin(angle) * up)
        points.append(joint)
    return np.array(points)


def thumb_chain(tip: np.ndarray) -> np.ndarray:
    """CMC, MCP, IP, TIP for a thumb whose tip sits at ``tip`` (hand frame)."""
    tip = np.asarray(tip, dtype=float)
    meta, prox, dist = _THUMB_BONES
    toward_tip = _unit(tip - _THUMB_CMC)
    mcp = _THUMB_CMC + meta * _unit(toward_tip + _THUMB_AXIS)
    span_vec = tip - mcp
    span = float(np.linalg.norm(span_vec))
    if span < 1e-9:
        raise ValueError("thumb tip coincides with its MCP joint")
    axis = span_vec / span
    if span >= prox + dist or span <= abs(prox - dist):
        ip = mcp + axis * span * prox / (prox + dist)  # straight (stretched) chain
    else:
        along = (prox**2 - dist**2 + span**2) / (2.0 * span)
        height = np.sqrt(max(prox**2 - along**2, 0.0))
        bulge = _THUMB_BULGE - np.dot(_THUMB_BULGE, axis) * axis
        ip = mcp + along * axis + height * _unit(bulge)
    return np.array([_THUMB_CMC, mcp, ip, tip])


def hand_landmarks(finger_flex, thumb_tip) -> np.ndarray:
    """Assemble all 21 landmarks in the hand frame from joint angles."""
    lm = np.zeros((NUM_LANDMARKS, 3))
    lm[WRIST] = 0.0
    lm[[THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]] = thumb_chain(thumb_tip)
    for i, (mcp, pip, dip, tip) in enumerate(FINGER_CHAINS):
        lm[mcp] = _MCP_POS[i]
        lm[[pip, dip, tip]] = finger_points(i, finger_flex[i])
    return lm


PALM_LOCAL = palm_scale(hand_landmarks((EXTENDED,) * 4, THUMB_OUT_TIP))


def rest_thumb_tip(index_tip: np.ndarray, aperture: float) -> np.ndarray:
    """Thumb tip ``aperture`` palm units away from the index tip, at rest."""
    return np.asarray(index_tip, dtype=float) + aperture * PALM_LOCAL * REST_DIRECTION


def pose_landmarks(name: str, aperture: float = 0.8) -> np.ndarray:
    """Hand-frame landmarks for a named pose (mouse, volume, draw, tabs)."""
    if name not in POSES:
        raise KeyError(f"unknown pose {name!r}; choose from {sorted(POSES)}")
    flex, thumb = POSES[name]
    index_tip = finger_points(0, flex[0])[-1]
    tip = rest_thumb_tip(index_tip, aperture) if thumb == "rest" else THUMB_OUT_TIP
    return hand_landmarks(flex, tip)


def pinch_landmarks(base: np.ndarray, rest_aperture: float, aperture: float, slip: float) -> np.ndarray:
    """Close a resting hand toward a pinch (all apertures in palm units).

    The thumb and index tips approach each other along the line between them.
    The index tip covers ``slip`` of the closing distance, the thumb the rest,
    and the index PIP/DIP follow part of the way. ``slip`` is the mechanism
    that drags the cursor when you click.
    """
    if not 0.0 <= slip <= 1.0:
        raise ValueError("slip must be in [0, 1]")
    lm = base.copy()
    gap = lm[THUMB_TIP] - lm[INDEX_TIP]
    direction = _unit(gap)
    shift = slip * (rest_aperture - aperture) * PALM_LOCAL * direction
    lm[INDEX_TIP] = lm[INDEX_TIP] + shift
    lm[INDEX_DIP] = lm[INDEX_DIP] + 0.6 * shift
    lm[INDEX_PIP] = lm[INDEX_PIP] + 0.25 * shift
    new_tip = lm[INDEX_TIP] + aperture * PALM_LOCAL * direction
    lm[[THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]] = thumb_chain(new_tip)
    return lm


def rotation_matrix(roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation: roll about the camera axis (z), pitch about x, yaw about y."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry


def mirror(local: np.ndarray) -> np.ndarray:
    """The other hand: reflect the hand frame across its long axis (x -> -x)."""
    out = np.array(local, dtype=float, copy=True)
    out[:, 0] = -out[:, 0]
    return out


def to_camera(local: np.ndarray, rotation: np.ndarray, hand_px: float, anchor_px, anchor: int = INDEX_TIP) -> np.ndarray:
    """Place a hand-frame hand in the image: rotate, scale to ``hand_px``
    pixels per palm length, and move landmark ``anchor`` to ``anchor_px``."""
    pts = (np.asarray(local, dtype=float) @ rotation.T) * hand_px
    offset = np.zeros(3)
    offset[:2] = np.asarray(anchor_px, dtype=float) - pts[anchor, :2]
    return pts + offset
