"""Analyse real-user Fitts logs written by ``fitts_app``.

    python -m stillpoint.analyze logs/*.csv
    python -m stillpoint.analyze logs/*.csv --json results/user_study.json

Globs are expanded here too, so the command works in shells that do not
expand them (PowerShell, cmd). Throughput is the mean of means: per
participant x variant, the mean over conditions of IDe / MT.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

from .fitts import mean_of_means, summarize_condition
from .fittstask import read_rows, row_to_trial


def expand(patterns) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        paths.extend(Path(m) for m in (matches or [pattern]))
    return paths


def analyze_rows(rows: list[dict]) -> dict:
    """Per-variant throughput, error rate and movement time from trial rows."""
    grouped: dict = {}
    for row in rows:
        key = (row["variant"], row["participant"])
        grouped.setdefault(key, []).append(row_to_trial(row))
    variants: dict = {}
    for (variant, participant), trials in sorted(grouped.items()):
        by_cond: dict = {}
        for t in trials:
            by_cond.setdefault(t.condition, []).append(t)
        summaries = [summarize_condition(ts) for ts in by_cond.values()]
        entry = variants.setdefault(variant, {"per_participant": {}, "trials": []})
        entry["per_participant"][participant] = summaries
        entry["trials"].extend(trials)
    out = {}
    for variant, entry in variants.items():
        trials = entry["trials"]
        mts = [t.movement_time for t in trials if t.movement_time is not None]
        per = {
            p: float(np.mean([s.throughput for s in ss if math.isfinite(s.throughput)] or [float("nan")]))
            for p, ss in entry["per_participant"].items()
        }
        out[variant] = {
            "participants": len(entry["per_participant"]),
            "trials": len(trials),
            "throughput_bps": mean_of_means(entry["per_participant"]),
            "error_rate": float(np.mean([not t.hit for t in trials])),
            "movement_time_s": float(np.mean(mts)) if mts else float("nan"),
            "per_participant_throughput": per,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Throughput and errors per variant from fitts_app logs.")
    ap.add_argument("logs", nargs="+", help="CSV files or glob patterns")
    ap.add_argument("--json", type=Path, help="also write the summary here")
    args = ap.parse_args(argv)
    paths = expand(args.logs)
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"no such file: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    try:
        summary = analyze_rows(read_rows(paths))
    except (ValueError, KeyError) as exc:
        print(f"could not read logs: {exc}", file=sys.stderr)
        return 2
    print(f"{'variant':20s} {'people':>6s} {'trials':>6s} {'TP bit/s':>9s} {'error %':>8s} {'MT s':>6s}")
    for variant, s in summary.items():
        print(f"{variant:20s} {s['participants']:6d} {s['trials']:6d} {s['throughput_bps']:9.2f} "
              f"{100 * s['error_rate']:8.1f} {s['movement_time_s']:6.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
