"""Tiny dependency-free SVG chart writer for the README figures.

Two layouts: a bar table (rows = pointer variants, columns = metrics, one
axis per column) and line panels (one axis per panel). Light instrument-bench
panel, hairline grid, values in ink colours, series colours only on marks.
Every mark carries a <title> so hovering the opened SVG shows its value.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

INK = "#14171A"
INK_2 = "#4A5157"
MUTED = "#828A90"
GRID = "#E3E6E9"
SURFACE = "#FCFCFB"
BORDER = "#D3D7DB"
SANS = "'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif"
MONO = "'JetBrains Mono','Cascadia Mono',Consolas,'Courier New',monospace"


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400", family=SANS) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}">{escape(str(s))}</text>')


def nice_ticks(vmax: float, count: int = 4) -> list[float]:
    """Round tick values from 0 up to at least ``vmax``."""
    if not math.isfinite(vmax) or vmax <= 0:
        return [0.0, 1.0]
    raw = vmax / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    n = math.ceil(vmax / step - 1e-9)
    return [round(i * step, 10) for i in range(n + 1)]


def _frame(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="ct cd">',
        f'<title id="ct">{escape(title)}</title><desc id="cd">{escape(subtitle)}</desc>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="12" fill="{SURFACE}" stroke="{BORDER}" stroke-width="1.5"/>',
        _text(28, 38, title, 18, INK, weight="700"),
        _text(28, 60, subtitle, 12, INK_2),
    ]


def _legend(x: float, y: float, items: list[tuple[str, str]], line: bool = False) -> list[str]:
    out = []
    for label, color in items:
        if line:
            out.append(f'<path d="M{x:.1f} {y - 4:.1f}h16" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(_text(x + 20, y, label, 12, INK_2))
        x += 28 + 7.0 * len(label)
    return out


def bar_table(path, title: str, subtitle: str, rows: list[tuple[str, str]], columns: list[dict], width: int = 1200) -> None:
    """Rows = (label, colour). Each column: {title, hint, values: [..], fmt}.

    Row labels carry a colour swatch, so no separate legend is needed.
    """
    label_w, top, row_h, gap = 190, 112, 46, 26
    height = top + row_h * len(rows) + 44
    col_w = (width - 28 - label_w - gap * (len(columns) - 1) - 24) / len(columns)
    svg = _frame(width, height, title, subtitle)
    for i, (label, color) in enumerate(rows):
        yc = top + i * row_h + row_h / 2
        svg.append(f'<rect x="28" y="{yc - 6:.1f}" width="12" height="12" rx="3" fill="{color}"/>')
        svg.append(_text(48, yc + 4, label, 13, INK, weight="600"))
    for c, col in enumerate(columns):
        x0 = 28 + label_w + c * (col_w + gap)
        values = col["values"]
        finite = [v for v in values if math.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        ticks = nice_ticks(vmax)
        scale = (col_w - 64) / ticks[-1]
        svg.append(_text(x0, top - 26, col["title"], 13, INK, weight="700"))
        svg.append(_text(x0, top - 10, col.get("hint", ""), 11, MUTED))
        bottom = top + row_h * len(rows)
        for tk in ticks:
            xx = x0 + tk * scale
            svg.append(f'<path d="M{xx:.1f} {top:.1f}V{bottom:.1f}" stroke="{GRID}" stroke-width="1"/>')
            svg.append(_text(xx, bottom + 16, f"{tk:g}", 10, MUTED, "middle", family=MONO))
        for i, (v, (label, color)) in enumerate(zip(values, rows)):
            yc = top + i * row_h + row_h / 2
            if not math.isfinite(v):
                svg.append(_text(x0 + 4, yc + 4, "n/a", 11, MUTED))
                continue
            w = max(v * scale, 0.0)
            svg.append(f'<path d="{_hbar(x0, yc - 11, w, 22)}" fill="{color}"><title>{escape(label)}: {escape(col["fmt"](v))}</title></path>')
            svg.append(_text(x0 + w + 6, yc + 4, col["fmt"](v), 12, INK, weight="600", family=MONO))
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def _hbar(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Horizontal bar, square at the baseline, rounded at the data end."""
    if w <= 0:
        return f"M{x:.1f} {y:.1f}h0.5v{h:.1f}h-0.5Z"
    r = min(r, w, h / 2)
    return (f"M{x:.1f} {y:.1f}H{x + w - r:.1f}Q{x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f}"
            f"V{y + h - r:.1f}Q{x + w:.1f} {y + h:.1f} {x + w - r:.1f} {y + h:.1f}H{x:.1f}Z")


def line_panels(path, title: str, subtitle: str, panels: list[dict], series: list[tuple[str, str]], width: int = 1200, height: int = 400) -> None:
    """Panels: {title, xlabel, x: [..], ys: {series_label: [..]}, fmt, ymax?}."""
    top, bottom_pad, gap = 96, 52, 40
    pw = (width - 56 - gap * (len(panels) - 1)) / len(panels)
    svg = _frame(width, height, title, subtitle)
    legend_w = sum(28 + 7.0 * len(label) for label, _ in series)
    svg += _legend(width - 28 - legend_w, 38, series, line=True)
    for k, panel in enumerate(panels):
        x0 = 28 + k * (pw + gap) + 44
        x1 = 28 + k * (pw + gap) + pw - 10
        y0, y1 = height - bottom_pad, top + 22
        xs = panel["x"]
        allv = [v for ys in panel["ys"].values() for v in ys if math.isfinite(v)]
        ticks = nice_ticks(panel.get("ymax") or (max(allv) if allv else 1.0))
        xmin, xmax = min(xs), max(xs)

        def px(xv, a=x0, b=x1, lo=xmin, hi=xmax):
            return a + (xv - lo) / (hi - lo) * (b - a) if hi > lo else (a + b) / 2

        def py(yv, t=ticks):
            return y0 - yv / t[-1] * (y0 - y1)

        svg.append(_text(x0 - 44, top, panel["title"], 13, INK, weight="700"))
        for tk in ticks:
            svg.append(f'<path d="M{x0:.1f} {py(tk):.1f}H{x1:.1f}" stroke="{GRID}" stroke-width="1"/>')
            svg.append(_text(x0 - 8, py(tk) + 4, panel["fmt"](tk), 10, MUTED, "end", family=MONO))
        for xv in xs:
            svg.append(_text(px(xv), y0 + 16, f"{xv:g}", 10, MUTED, "middle", family=MONO))
        svg.append(_text((x0 + x1) / 2, y0 + 36, panel["xlabel"], 11, INK_2, "middle"))
        for label, color in series:
            ys = panel["ys"].get(label)
            if not ys:
                continue
            pts = [(px(xv), py(yv), xv, yv) for xv, yv in zip(xs, ys) if math.isfinite(yv)]
            d = "M" + "L".join(f"{a:.1f} {b:.1f}" for a, b, _, _ in pts)
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for a, b, xv, yv in pts:
                svg.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2">'
                           f'<title>{escape(label)} @ {xv:g}: {escape(panel["fmt"](yv))}</title></circle>')
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")
