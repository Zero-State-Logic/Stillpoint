"""Small paired statistics with numpy only (no scipy needed)."""

from __future__ import annotations

import itertools

import numpy as np

EXACT_LIMIT = 16  # enumerate all sign patterns up to this many pairs


def sign_flip_p(diffs, seed: int = 0, draws: int = 20000) -> float:
    """Two-sided paired permutation (sign-flip) test of mean(diffs) == 0.

    Exact for up to ``EXACT_LIMIT`` pairs, Monte Carlo beyond that.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    observed = abs(d.mean())
    if d.size <= EXACT_LIMIT:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(draws, d.size))
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


def bootstrap_ci(values, seed: int = 0, draws: int = 10000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(draws, v.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


def paired_summary(before, after, seed: int = 0) -> dict:
    """Mean of each arm, mean paired difference (after - before), 95 % CI, p."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    diff = b - a
    lo, hi = bootstrap_ci(diff, seed=seed)
    return {
        "before_mean": float(np.nanmean(a)),
        "after_mean": float(np.nanmean(b)),
        "mean_diff": float(np.nanmean(diff)),
        "ci95": [lo, hi],
        "p_sign_flip": sign_flip_p(diff, seed=seed),
        "n": int(np.isfinite(diff).sum()),
    }
