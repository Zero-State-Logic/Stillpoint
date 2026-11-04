"""Real-user ISO 9241-411 test, driven by the live pointer pipeline.

    python -m stillpoint.fitts_app --participant P01 --variant stillpoint
    python -m stillpoint.analyze logs/*.csv

Targets are drawn in an OpenCV window; the cursor there is the pipeline's
output, so the test never moves the OS cursor. Pinch to select. The first
pinch on each circle starts it and is not logged; the rest are written to
``logs/<participant>_<variant>_<time>.csv`` as they happen. Run each variant
in a counterbalanced order per participant. Keys: q aborts (rows so far are
kept), space skips a target you cannot reach.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import VARIANTS
from .fitts import Condition
from .fittstask import FittsTask, TrialLogger
from .participants import DEFAULT_CONDITIONS
from .pointer import PointerConfig, PointerPipeline
from .tracking import BackendUnavailable, open_tracker, select_hand

BG = (249, 248, 247)
INK = (26, 23, 20)
VIOLET = (240, 76, 108)
AMBER = (0, 162, 232)
GRID = (219, 215, 211)


def scaled_conditions(height: int) -> list[Condition]:
    """The simulation's conditions scaled from a 1080-px to this window height."""
    k = height / 1080.0
    return [Condition(c.distance * k, c.width * k) for c in DEFAULT_CONDITIONS]


def _draw(cv2, task: FittsTask, cursor, pinch: str, size) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    if task.finished:
        cv2.putText(img, "Done. Thank you!", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, INK, 2)
        return img
    current = task.current_target()
    for x, y in task.targets():
        cv2.circle(img, (int(x), int(y)), int(current.radius), GRID, 2)
    cv2.circle(img, (int(current.x), int(current.y)), int(current.radius), VIOLET, -1)
    color = AMBER if pinch != "open" else INK
    cv2.circle(img, (int(cursor[0]), int(cursor[1])), 7, color, -1)
    label = f"circle {task.sequence + 1}/{len(task.conditions)}   target {task.position}/{task.n_targets}"
    cv2.putText(img, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 2)
    return img


def run(args) -> int:
    import cv2

    try:
        tracker = open_tracker(args.model)
    except BackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    size = (args.window_width, args.window_height)
    cfg = PointerConfig(cam_w=args.width, cam_h=args.height, screen_w=size[0], screen_h=size[1])
    pipeline = PointerPipeline(args.variant, cfg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_dir) / f"{args.participant}_{args.variant}_{stamp}.csv"
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        tracker.close()
        return 2
    t0 = time.perf_counter()
    with TrialLogger(log_path) as logger:
        task = FittsTask(scaled_conditions(size[1]), args.participant, args.variant, args.block,
                         center=(size[0] / 2, size[1] / 2), logger=logger)
        try:
            while not task.finished:
                ok, frame = cap.read()
                if not ok:
                    print("camera stopped delivering frames", file=sys.stderr)
                    break
                frame = cv2.resize(cv2.flip(frame, 1), (args.width, args.height))
                t = time.perf_counter() - t0
                hand = select_hand(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000)), args.hand)
                out = pipeline.update(None if hand is None else hand.landmarks, t)
                for click in out.clicks:
                    task.on_click(click.t, click.x, click.y)
                cv2.imshow("STILLPOINT Fitts test", _draw(cv2, task, out.cursor, out.pinch, size))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    task.on_click(t, float("nan"), float("nan"))  # logged as a miss
        finally:
            cap.release()
            tracker.close()
            cv2.destroyAllWindows()
    print(f"{len(task.rows)} trials written to {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Live ISO 9241-411 pointing test for one variant.")
    ap.add_argument("--participant", required=True, help="anonymous id, e.g. P01")
    ap.add_argument("--variant", choices=VARIANTS, default="stillpoint")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280, help="camera width")
    ap.add_argument("--height", type=int, default=720, help="camera height")
    ap.add_argument("--window-width", type=int, default=1280)
    ap.add_argument("--window-height", type=int, default=720)
    ap.add_argument("--hand", choices=("any", "left", "right"), default="any")
    ap.add_argument("--model", help="hand_landmarker.task (only for MediaPipe builds without mp.solutions)")
    ap.add_argument("--log-dir", default="logs")
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
