"""Build the README charts from the evaluation result dictionary."""

from __future__ import annotations

from pathlib import Path

from .svgchart import bar_table, line_panels

# Ordinal violet ramp (validated light -> dark): each step adds one idea.
COLORS = {
    "baseline": "#ACA0F4",
    "oneeuro": "#8F79F2",
    "oneeuro+hysteresis": "#6C4CF0",
    "stillpoint": "#4E2DC4",
}
LABELS = {
    "baseline": "Original",
    "oneeuro": "+ One Euro",
    "oneeuro+hysteresis": "+ hysteresis pinch",
    "stillpoint": "STILLPOINT (+ rewind)",
}
ORIGINAL_COLOR = COLORS["baseline"]
NEW_COLOR = COLORS["stillpoint"]


def _px(v: float) -> str:
    return f"{v:.1f} px" if v < 10 else f"{v:.0f} px"


def results_chart(result: dict, path: Path) -> None:
    variants = list(result["variants"])
    s = result["variants"]
    meta = result["meta"]
    rows = [(LABELS[v], COLORS[v]) for v in variants]
    columns = [
        {"title": "Click slip (median)", "hint": "lower is better", "values": [s[v]["median_slip_px"] for v in variants], "fmt": _px},
        {"title": "Error rate", "hint": "lower is better", "values": [100 * s[v]["error_rate"] for v in variants], "fmt": lambda x: f"{x:.0f} %" if x >= 10 or x == 0 else f"{x:.1f} %"},
        {"title": "Effective throughput", "hint": "higher is better", "values": [s[v]["throughput_bps"] for v in variants], "fmt": lambda x: f"{x:.2f}" if x % 1 else f"{x:.0f}"},
        {"title": "False clicks / min", "hint": "lower is better", "values": [s[v]["false_clicks_per_min"] for v in variants], "fmt": lambda x: f"{x:.1f}" if x % 1 else f"{x:.0f}"},
    ]
    blocks = f"{meta['blocks']} block" + ("s" if meta["blocks"] != 1 else "")
    subtitle = (f"Simulated ISO 9241-411 task: {meta['participants']} participants x {blocks} x "
                f"{len(meta['conditions'])} ID conditions x {meta['targets_per_circle']} targets. Throughput in bit/s.")
    bar_table(path, "Pointer variants, one idea added per row", subtitle, rows, columns)


def sensitivity_chart(result: dict, path: Path) -> None:
    sens = result["sensitivity"]
    series = [(LABELS[v], COLORS[v]) for v in COLORS]

    def ys(sweep: dict, key: str, scale: float = 1.0) -> dict:
        return {LABELS[v]: [scale * x for x in vals] for v, vals in sweep["metrics"][key].items()}

    panels = [
        {"title": "Click slip vs slip fraction", "xlabel": "slip fraction (share of the pinch the index tip travels)",
         "x": sens["slip"]["values"], "ys": ys(sens["slip"], "median_slip_px"), "fmt": lambda v: f"{v:g}"},
        {"title": "Error rate (%) vs slip fraction", "xlabel": "slip fraction",
         "x": sens["slip"]["values"], "ys": ys(sens["slip"], "error_rate", 100), "fmt": lambda v: f"{v:g}"},
        {"title": "Error rate (%) vs tracker jitter", "xlabel": "landmark jitter SD (camera px)",
         "x": sens["jitter_px"]["values"], "ys": ys(sens["jitter_px"], "error_rate", 100), "fmt": lambda v: f"{v:g}"},
    ]
    n = sens["slip"]["participants"]
    line_panels(path, "Sensitivity: how the gap grows with slip and noise",
                f"Simulated, {n} participants x 1 block per point. Median slip in screen px.", panels, series)


def modes_chart(result: dict, path: Path) -> None:
    roll = result["roll_sweep"]
    acc = roll["accuracy"]
    series = [("Original rules", ORIGINAL_COLOR), ("STILLPOINT joint angles", NEW_COLOR)]
    panels = [
        {"title": f"{hand.capitalize()} hand: poses recognised (%)", "xlabel": "hand roll (degrees, 0 = upright)",
         "x": roll["angles"], "ymax": 100, "fmt": lambda v: f"{v:g}",
         "ys": {"Original rules": [100 * a for a in acc[f"original/{hand}"]],
                "STILLPOINT joint angles": [100 * a for a in acc[f"stillpoint/{hand}"]]}}
        for hand in ("right", "left")
    ]
    line_panels(path, "Pose recognition vs hand roll",
                f"Simulated single frames: 4 poses, tracker jitter, pitch and yaw within 15 degrees, {roll['samples_per_angle']} frames per angle.",
                panels, series)


def write_charts(result: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = out_dir / "results.svg"
    results_chart(result, path)
    written.append(path)
    if "sensitivity" in result:
        path = out_dir / "sensitivity.svg"
        sensitivity_chart(result, path)
        written.append(path)
    if "roll_sweep" in result:
        path = out_dir / "modes.svg"
        modes_chart(result, path)
        written.append(path)
    return written
