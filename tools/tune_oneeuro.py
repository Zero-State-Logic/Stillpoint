"""Pick One Euro parameters the way Casiez et al. suggest, on a held-out seed.

Procedure: (1) for each min_cutoff, measure cursor jitter with the hand still
and keep the cutoffs whose jitter is no worse than the original fixed
smoother's; (2) among those, raise beta until the lag during a fast reach
stops improving. Uses its own seed (not the evaluation seed), so the
evaluation does not grade the filter on the data it was tuned on.

    python tools/tune_oneeuro.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stillpoint.filters import ExponentialSmoother, OneEuroFilter, OneEuroParams  # noqa: E402
from stillpoint.participants import minimum_jerk  # noqa: E402
from stillpoint.pointer import PointerConfig, map_to_screen, screen_to_camera  # noqa: E402

TUNING_SEED = 99
FPS = 30.0
JITTER_PX = 1.5
REPEATS = 40


def _trace(rng: np.random.Generator, cfg: PointerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Hold 1 s, reach 600 screen px in 0.8 s, hold 1 s. Returns (true, noisy) in screen px."""
    start = screen_to_camera((660.0, 540.0), cfg)
    goal = screen_to_camera((1260.0, 540.0), cfg)
    t = np.arange(0.0, 2.8, 1.0 / FPS)
    truth_cam = np.array([minimum_jerk(start, goal, (ti - 1.0) / 0.8) for ti in t])
    noisy_cam = truth_cam + rng.normal(0.0, JITTER_PX, truth_cam.shape)
    truth = np.array([map_to_screen(p, cfg) for p in truth_cam])
    noisy = np.array([map_to_screen(p, cfg) for p in noisy_cam])
    return truth, noisy


def _score(make_filter, cfg: PointerConfig) -> tuple[float, float]:
    """(jitter SD at rest, mean error during the reach), screen px."""
    rng = np.random.default_rng(TUNING_SEED)
    rest_sd, reach_err = [], []
    times = np.arange(0.0, 2.8, 1.0 / FPS)
    for _ in range(REPEATS):
        truth, noisy = _trace(rng, cfg)
        filt = make_filter(truth[0])
        out = np.array([filt(x, t) for x, t in zip(noisy, times)])
        rest = (times >= 0.4) & (times < 1.0)
        reach = (times >= 1.0) & (times < 1.8)
        rest_sd.append(float(np.std(out[rest] - truth[rest], axis=0).mean()))
        reach_err.append(float(np.linalg.norm(out[reach] - truth[reach], axis=1).mean()))
    return float(np.mean(rest_sd)), float(np.mean(reach_err))


def main() -> int:
    cfg = PointerConfig()
    base_sd, base_err = _score(lambda x0: ExponentialSmoother(cfg.smoothing, initial=x0), cfg)
    print(f"original smoother: rest jitter {base_sd:.2f} px, reach error {base_err:.1f} px")
    rows = []
    for min_cutoff, beta in itertools.product((0.4, 0.6, 0.8, 1.0, 1.5), (0.0, 0.004, 0.008, 0.012, 0.02, 0.03)):
        params = OneEuroParams(min_cutoff=min_cutoff, beta=beta)
        sd, err = _score(lambda x0, p=params: OneEuroFilter(p), cfg)
        rows.append((min_cutoff, beta, sd, err))
        print(f"min_cutoff {min_cutoff:4.1f}  beta {beta:5.3f}  rest jitter {sd:5.2f} px  reach error {err:6.1f} px")
    still_enough = [r for r in rows if r[2] <= base_sd]
    if not still_enough:
        print("no setting is as still as the original smoother")
        return 1
    best = min(still_enough, key=lambda r: r[3])
    print(f"\nchosen: min_cutoff={best[0]}, beta={best[1]} "
          f"(rest jitter {best[2]:.2f} <= {base_sd:.2f} px, reach error {best[3]:.1f} vs {base_err:.1f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
