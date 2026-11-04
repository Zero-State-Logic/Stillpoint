"""STILLPOINT live app: webcam in, mouse / volume / drawing / tab control out.

    python -m stillpoint.app                      # full system
    python -m stillpoint.app --variant baseline   # the original behaviour
    python -m stillpoint.app --dry-run            # watch it decide, touch nothing

Keys in the preview window: q quits, c clears the drawing.
OpenCV, MediaPipe, PyAutoGUI and pycaw are imported here, at start-up, so the
rest of the package stays importable without them.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import VARIANTS
from .controller import Click, Controller, ControllerConfig, DrawStroke, ModeChanged, MoveCursor, SetVolume, SwitchTab
from .pointer import PointerConfig
from .tracking import BackendUnavailable, open_tracker, select_hand

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
VIOLET = (240, 76, 108)  # BGR of #6C4CF0
AMBER = (0, 162, 232)  # BGR of #E8A200


class OsActuator:
    """Executes controller actions with PyAutoGUI and pycaw (Windows volume)."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self._gui = None
        self._volume = None
        if not dry_run:
            import pyautogui

            pyautogui.FAILSAFE = False  # corners must not raise mid-gesture
            self._gui = pyautogui
            self._volume = _open_volume()

    def screen_size(self) -> tuple[int, int]:
        return tuple(self._gui.size()) if self._gui else (1920, 1080)

    def apply(self, action) -> None:
        if self.dry_run:
            return
        if isinstance(action, MoveCursor):
            self._gui.moveTo(action.x, action.y)
        elif isinstance(action, Click):
            self._gui.click(x=action.x, y=action.y)
        elif isinstance(action, SwitchTab):
            keys = ("ctrl", "tab") if action.step > 0 else ("ctrl", "shift", "tab")
            self._gui.hotkey(*keys)
        elif isinstance(action, SetVolume) and self._volume is not None:
            self._volume.SetMasterVolumeLevelScalar(float(action.level), None)


def _open_volume():
    """System volume endpoint via pycaw, or None (non-Windows, no audio device)."""
    try:
        from comtypes import COMError
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError:
        print("pycaw not available: volume mode will show the level but not change it", file=sys.stderr)
        return None
    try:
        speakers = AudioUtilities.GetSpeakers()
        endpoint = getattr(speakers, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL

        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except (OSError, AttributeError, COMError) as exc:
        print(f"could not open the audio endpoint ({exc}): volume mode disabled", file=sys.stderr)
        return None


def _draw_overlay(cv2, frame, landmarks, controller: Controller, variant: str) -> None:
    if landmarks is not None:
        pts = landmarks[:, :2].astype(int)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (220, 220, 220), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, VIOLET, -1)
        color = AMBER if controller.pinch_state != "open" else VIOLET
        cv2.circle(frame, tuple(pts[8]), 12, color, -1)
    cv2.rectangle(frame, (0, 0), (380, 78), (20, 23, 26), -1)
    cv2.putText(frame, f"MODE: {controller.mode.upper()}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    detail = f"{variant}  pinch: {controller.pinch_state}"
    if controller.mode == "volume" and controller.volume is not None:
        detail = f"volume: {100 * controller.volume:.0f}%"
    cv2.putText(frame, detail, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def _composite(cv2, frame, canvas):
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 20
    out = frame.copy()
    out[mask] = canvas[mask]
    return out


def run(args) -> int:
    import cv2

    try:
        tracker = open_tracker(args.model)
    except BackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    actuator = OsActuator(args.dry_run)
    screen_w, screen_h = actuator.screen_size()
    pointer = PointerConfig(cam_w=args.width, cam_h=args.height, screen_w=screen_w, screen_h=screen_h)
    controller = Controller(ControllerConfig(variant=args.variant, pointer=pointer))
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        tracker.close()
        return 2
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped delivering frames", file=sys.stderr)
                break
            frame = cv2.resize(cv2.flip(frame, 1), (args.width, args.height))
            t = time.perf_counter() - t0
            hand = select_hand(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000)), args.hand)
            landmarks = None if hand is None else hand.landmarks
            for action in controller.step(landmarks, t):
                if isinstance(action, DrawStroke):
                    cv2.line(canvas, action.start, action.end, VIOLET, 8)
                elif not isinstance(action, ModeChanged):
                    actuator.apply(action)
            frame = _composite(cv2, frame, canvas)
            _draw_overlay(cv2, frame, landmarks, controller, args.variant)
            cv2.imshow("STILLPOINT", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                canvas[:] = 0
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Mid-air hand control for your laptop.")
    ap.add_argument("--variant", choices=VARIANTS, default="stillpoint")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hand", choices=("any", "left", "right"), default="any", help="which hand to follow")
    ap.add_argument("--model", help="hand_landmarker.task (only for MediaPipe builds without mp.solutions)")
    ap.add_argument("--dry-run", action="store_true", help="show decisions without moving the mouse or keys")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
