"""MediaPipe hand tracking behind one small interface.

Two backends, picked automatically:

* ``SolutionsBackend``: the classic ``mediapipe.solutions.hands`` API, which
  ships with older MediaPipe releases and needs no model file.
* ``TasksBackend``: the ``HandLandmarker`` task, the only hand API in recent
  MediaPipe releases (no ``mp.solutions``). It needs the
  ``hand_landmarker.task`` model file, which you download once from
  MediaPipe's model page and pass with ``--model``.

MediaPipe is imported only when a backend is opened, so the rest of the
package (and the tests) run without it. Both backends return landmarks in
camera pixels with z scaled like x, and the handedness label MediaPipe
reports for the (mirrored) frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path("models") / "hand_landmarker.task"
MODEL_HELP = (
    "This MediaPipe version has no mp.solutions, so STILLPOINT uses the HandLandmarker task, "
    "which needs its model file. Download hand_landmarker.task from the MediaPipe hand landmarker "
    "page (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and save it as "
    f"{DEFAULT_MODEL} or pass --model PATH."
)


class BackendUnavailable(RuntimeError):
    """No usable MediaPipe hand backend (not installed, or model file missing)."""


@dataclass(frozen=True)
class HandObservation:
    landmarks: np.ndarray  # (21, 3) camera pixels
    handedness: str  # "Left", "Right" or "Unknown"
    score: float


def normalized_to_pixels(points, width: int, height: int) -> np.ndarray:
    """MediaPipe normalised landmarks (objects with x, y, z) to pixel array."""
    arr = np.array([[p.x * width, p.y * height, p.z * width] for p in points], dtype=float)
    if arr.shape != (21, 3):
        raise ValueError(f"expected 21 landmarks, got {arr.shape[0]}")
    return arr


def _label(category) -> tuple[str, float]:
    name = getattr(category, "category_name", None) or getattr(category, "label", None) or "Unknown"
    return str(name), float(getattr(category, "score", 0.0))


def build_observations(hands, handedness, width: int, height: int) -> list[HandObservation]:
    """Pair landmark lists with their handedness categories."""
    out = []
    for i, points in enumerate(hands or []):
        cats = handedness[i] if handedness and i < len(handedness) else []
        name, score = _label(cats[0]) if cats else ("Unknown", 0.0)
        out.append(HandObservation(normalized_to_pixels(points, width, height), name, score))
    return out


def select_hand(observations: list[HandObservation], preferred: str = "any") -> HandObservation | None:
    """Highest-scoring hand, restricted to ``preferred`` ("left"/"right") if set."""
    if preferred not in ("any", "left", "right"):
        raise ValueError("preferred must be 'any', 'left' or 'right'")
    pool = observations
    if preferred != "any":
        pool = [o for o in observations if o.handedness.lower() == preferred]
    return max(pool, key=lambda o: o.score) if pool else None


class SolutionsBackend:
    def __init__(self, mp, max_hands: int, detection: float, tracking: float) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=max_hands,
            min_detection_confidence=detection, min_tracking_confidence=tracking,
        )

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        result = self._hands.process(rgb)
        hands = [h.landmark for h in (result.multi_hand_landmarks or [])]
        sides = [h.classification for h in (result.multi_handedness or [])]
        return build_observations(hands, sides, rgb.shape[1], rgb.shape[0])

    def close(self) -> None:
        self._hands.close()


class TasksBackend:
    def __init__(self, mp, model_path: Path, max_hands: int, detection: float, tracking: float) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection,
            min_hand_presence_confidence=detection,
            min_tracking_confidence=tracking,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        stamp = max(int(timestamp_ms), self._last_ms + 1)  # VIDEO mode needs increasing stamps
        self._last_ms = stamp
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, stamp)
        return build_observations(result.hand_landmarks, result.handedness, rgb.shape[1], rgb.shape[0])

    def close(self) -> None:
        self._landmarker.close()


def open_tracker(model_path=None, max_hands: int = 1, detection: float = 0.7, tracking: float = 0.7):
    """Open the best available backend or raise ``BackendUnavailable``."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise BackendUnavailable("mediapipe is not installed: pip install -r requirements.txt") from exc
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands"):
        return SolutionsBackend(mp, max_hands, detection, tracking)
    path = Path(model_path) if model_path else DEFAULT_MODEL
    if not path.is_file():
        raise BackendUnavailable(MODEL_HELP)
    return TasksBackend(mp, path, max_hands, detection, tracking)
