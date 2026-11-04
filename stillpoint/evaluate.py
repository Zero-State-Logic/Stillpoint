"""Run the simulated ISO 9241-411 study and write the results and charts.

    python -m stillpoint.evaluate                # full run (a few minutes)
    python -m stillpoint.evaluate --quick        # small smoke run

Outputs ``results/fitts_sim.json`` and three SVG charts in ``docs/assets/``.
Every number in the README comes from that JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import VARIANTS, __version__
from .fitts import mean_of_means, summarize_condition
from .participants import SimParams, make_participants
from .pointer import PointerConfig
from .sim import SessionResult, simulate_session
from .simmodes import roll_sweep
from .stats import paired_summary

EVAL_SEED = 1
ROOT = Path(__file__).resolve().parents[1]


# ---- running ---------------------------------------------------------------
def _job(args) -> SessionResult:
    person, variant, params, seed, block = args
    return simulate_session(person, variant, params, seed, block)


def run_sessions(n: int, blocks: int, seed: int, params: SimParams, workers: int, variants=VARIANTS) -> dict:
    """Every participant x block x variant. Returns {variant: [SessionResult]}."""
    people = make_participants(n, seed)
    jobs = [(p, v, params, seed, b) for v in variants for p in people for b in range(blocks)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs, chunksize=1))
    else:
        results = [_job(j) for j in jobs]
    out: dict = {v: [] for v in variants}
    for r in results:
        out[r.variant].append(r)
    return out


# ---- summarising -----------------------------------------------------------
def _participant_metrics(results: list[SessionResult]) -> dict:
    """pid -> throughput (mean of conditions), median slip, error rate."""
    by_pid: dict = {}
    for r in results:
        by_pid.setdefault(r.pid, []).append(r)
    metrics = {}
    for pid, sessions in sorted(by_pid.items()):
        trials = [t for s in sessions for t in s.trials]
        slips = [x for s in sessions for x in s.slips]
        by_cond: dict = {}
        for t in trials:
            by_cond.setdefault(t.condition, []).append(t)
        summaries = [summarize_condition(ts) for ts in by_cond.values()]
        metrics[pid] = {
            "summaries": summaries,
            "throughput": mean_of_means({pid: summaries}),
            "median_slip_px": float(np.median(slips)) if slips else float("nan"),
            "error_rate": float(np.mean([not t.hit for t in trials])),
        }
    return metrics


def summarize_variant(results: list[SessionResult]) -> dict:
    trials = [t for r in results for t in r.trials]
    slips = np.array([x for r in results for x in r.slips])
    minutes = sum(r.duration_s for r in results) / 60.0
    per = _participant_metrics(results)
    tps = [m["throughput"] for m in per.values()]
    mts = [t.movement_time for t in trials if t.movement_time is not None]
    return {
        "median_slip_px": float(np.median(slips)),
        "mean_slip_px": float(np.mean(slips)),
        "p90_slip_px": float(np.percentile(slips, 90)),
        "error_rate": float(np.mean([not t.hit for t in trials])),
        "throughput_bps": mean_of_means({pid: m["summaries"] for pid, m in per.items()}),
        "throughput_sd": float(np.std(tps, ddof=1)) if len(tps) > 1 else 0.0,
        "movement_time_s": float(np.mean(mts)),
        "false_clicks_per_min": sum(r.false_clicks for r in results) / minutes,
        "duplicate_clicks_per_min": sum(r.duplicate_clicks for r in results) / minutes,
        "stray_clicks_per_min": sum(r.stray_clicks for r in results) / minutes,
        "missed_pinches": int(sum(r.misses for r in results)),
        "selections": int(sum(r.selections for r in results)),
        "measured_trials": len(trials),
        "simulated_minutes": minutes,
        "per_participant": [
            {"pid": pid, "throughput_bps": m["throughput"], "median_slip_px": m["median_slip_px"], "error_rate": m["error_rate"]}
            for pid, m in per.items()
        ],
    }


def by_condition(results: list[SessionResult]) -> list[dict]:
    per = _participant_metrics(results)
    rows: dict = {}
    for m in per.values():
        for s in m["summaries"]:
            rows.setdefault(s.condition, []).append(s)
    out = []
    for cond in sorted(rows, key=lambda c: c.nominal_id):
        ss = rows[cond]
        out.append({
            "distance": cond.distance, "width": cond.width, "id": cond.nominal_id,
            "error_rate": float(np.mean([s.error_rate for s in ss])),
            "throughput_bps": float(np.nanmean([s.throughput for s in ss])),
            "we": float(np.nanmean([s.we for s in ss])),
            "movement_time_s": float(np.nanmean([s.movement_time for s in ss])),
        })
    return out


def paired_tests(summary: dict, a: str = "baseline", b: str = "stillpoint") -> dict:
    pa = {row["pid"]: row for row in summary[a]["per_participant"]}
    pb = {row["pid"]: row for row in summary[b]["per_participant"]}
    pids = sorted(set(pa) & set(pb))
    return {
        key: paired_summary([pa[p][key] for p in pids], [pb[p][key] for p in pids], seed=EVAL_SEED)
        for key in ("throughput_bps", "median_slip_px", "error_rate")
    }


def mode_stability(results: list[SessionResult]) -> dict:
    """Pool the per-session mode counters of the given sessions."""
    pooled: dict = {}
    minutes = sum(r.duration_s for r in results) / 60.0
    for r in results:
        for key, c in (r.modes or {}).items():
            acc = pooled.setdefault(key, {"frames": 0, "wrong": 0.0, "switches": 0})
            acc["frames"] += c["frames"]
            acc["wrong"] += c["wrong_fraction"] * c["frames"]
            acc["switches"] += c["switches"]
    return {
        key: {"wrong_fraction": v["wrong"] / v["frames"], "switches_per_min": v["switches"] / minutes, "frames": v["frames"]}
        for key, v in pooled.items()
    }


# ---- experiments -----------------------------------------------------------
def run_main(n: int, blocks: int, seed: int, workers: int, params: SimParams | None = None) -> dict:
    params = params or SimParams(track_modes=True)
    sessions = run_sessions(n, blocks, seed, params, workers)
    summary = {v: summarize_variant(rs) for v, rs in sessions.items()}
    result = {
        "variants": summary,
        "by_condition": {v: by_condition(rs) for v, rs in sessions.items()},
        "paired_stillpoint_vs_baseline": paired_tests(summary),
    }
    if params.track_modes:
        result["mode_stability"] = mode_stability(sessions["stillpoint"])
    return result


def run_sweep(field: str, values, n: int, seed: int, workers: int, base: SimParams) -> dict:
    """Re-run the study with one SimParams field swept over ``values``."""
    out = {"field": field, "values": list(values), "participants": n, "metrics": {}}
    keys = ("median_slip_px", "error_rate", "throughput_bps", "false_clicks_per_min")
    table = {k: {v: [] for v in VARIANTS} for k in keys}
    for value in values:
        params = dataclasses.replace(base, **{field: value}, track_modes=False)
        sessions = run_sessions(n, 1, seed, params, workers)
        for variant, rs in sessions.items():
            s = summarize_variant(rs)
            for k in keys:
                table[k][variant].append(s[k])
    out["metrics"] = table
    return out


def _meta(args, params: SimParams, elapsed: float) -> dict:
    return {
        "generator": "python -m stillpoint.evaluate",
        "stillpoint_version": __version__,
        "note": "Simulated participants (a kinematic hand model with tracker noise). Not a user study.",
        "seed": args.seed,
        "participants": args.participants,
        "blocks": args.blocks,
        "targets_per_circle": params.n_targets,
        "conditions": [{"distance": c.distance, "width": c.width, "id": c.nominal_id} for c in params.conditions],
        "sim_params": {k: v for k, v in dataclasses.asdict(params).items() if k != "conditions"},
        "pointer_config": json.loads(json.dumps(dataclasses.asdict(PointerConfig()))),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_s": round(elapsed, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Simulated ISO 9241-411 evaluation of the pointer variants.")
    ap.add_argument("--participants", type=int, default=12)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sweep-participants", type=int, default=6)
    ap.add_argument("--no-sweep", action="store_true", help="skip the sensitivity sweeps")
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "fitts_sim.json")
    ap.add_argument("--charts", type=Path, default=ROOT / "docs" / "assets")
    args = ap.parse_args(argv)
    if args.quick:
        args.participants, args.blocks, args.sweep_participants = 2, 1, 2
    t0 = time.time()
    params = SimParams(track_modes=True)
    print(f"main study: {args.participants} participants x {args.blocks} blocks x {len(params.conditions)} conditions x 4 variants")
    result = run_main(args.participants, args.blocks, args.seed, args.workers, params)
    if not args.no_sweep:
        print("sensitivity sweeps: slip fraction, tracker jitter")
        base = SimParams()
        result["sensitivity"] = {
            "slip": run_sweep("slip", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), args.sweep_participants, args.seed, args.workers, base),
            "jitter_px": run_sweep("jitter_px", (0.5, 1.0, 1.5, 2.0, 3.0, 4.0), args.sweep_participants, args.seed, args.workers, base),
        }
    result["roll_sweep"] = roll_sweep(args.seed, samples=10 if args.quick else 40)
    result = {"meta": _meta(args, params, time.time() - t0), **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    from .charts import write_charts

    written = write_charts(result, args.charts)
    _print_summary(result)
    print(f"wrote {args.out}")
    for path in written:
        print(f"wrote {path}")
    return 0


def _print_summary(result: dict) -> None:
    print(f"\n{'variant':20s} {'slip med px':>11s} {'error %':>8s} {'TP bit/s':>9s} {'false/min':>10s}")
    for v, s in result["variants"].items():
        print(f"{v:20s} {s['median_slip_px']:11.1f} {100 * s['error_rate']:8.1f} {s['throughput_bps']:9.2f} {s['false_clicks_per_min']:10.2f}")


if __name__ == "__main__":
    sys.exit(main())
