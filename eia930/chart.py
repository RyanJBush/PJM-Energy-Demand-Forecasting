"""Failure-case chart for one forecast origin, drawn as a dependency-free SVG from paired_predictions.csv.

Usage:
  python -m eia930.chart --predictions outputs/real/PJM/simulation/paired_predictions.csv \\
      --origin 2024-12-25T00:00:00Z --out docs/worst_day_2024-12-25.svg

Without --origin the fully scored origin with the highest ridge MAE is used. The output is deterministic
(same input -> same bytes). Exit codes: 0 written; 2 bad input (missing file, origin not fully scored).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

ISO = "%Y-%m-%dT%H:%M:%SZ"
COLORS = {"actual": "#111111", "baseline": "#0072B2", "ridge": "#D55E00"}
DATA_NOTES = {"real": "real EIA-930 snapshot (PJM)", "synthetic": "SYNTHETIC test data, not real EIA data"}


class ChartError(ValueError):
    pass


def load_origin(path: str, origin: str | None) -> tuple:
    """Return (origin, data_label, rows) with 24 fully scored rows for the origin, ordered by horizon."""
    by_origin: dict = {}
    labels, run_types = set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels.add(r["data_label"])
            run_types.add(r["run_type"])
            if r["in_common_set"] == "1":
                by_origin.setdefault(r["origin_utc"], []).append(r)
    if len(labels) != 1 or len(run_types) != 1:
        raise ChartError("predictions file mixes data labels or run types")
    if run_types != {"retrospective_simulation"}:
        raise ChartError(f"unexpected run_type {sorted(run_types)}")
    full = {o: sorted(rs, key=lambda r: int(r["horizon_hours"])) for o, rs in by_origin.items() if len(rs) == 24}
    if origin is None:
        if not full:
            raise ChartError("no origin has all 24 rows scored")
        origin = max(sorted(full), key=lambda o: _mae([float(r["ridge_forecast_mw"]) - float(r["actual_mw"])
                                                       for r in full[o]]))
    if origin not in full:
        n = len(by_origin.get(origin, []))
        raise ChartError(f"origin {origin} is not fully scored ({n} of 24 rows in the common set)")
    return origin, labels.pop(), full[origin]


def _mae(errs):
    return sum(abs(e) for e in errs) / len(errs)


def _nice_axis(lo: float, hi: float, step: float = 5000.0):
    a = step * int(lo // step)
    b = step * (int(hi // step) + 1)
    return a, b, [a + i * step for i in range(int((b - a) / step) + 1)]


def render_svg(origin: str, label: str, rows: list) -> str:
    actual = [float(r["actual_mw"]) for r in rows]
    base = [float(r["baseline_forecast_mw"]) for r in rows]
    ridge = [float(r["ridge_forecast_mw"]) for r in rows]
    mae_b = _mae([b - a for b, a in zip(base, actual)])
    mae_r = _mae([r - a for r, a in zip(ridge, actual)])
    over_r = sum(1 for r, a in zip(ridge, actual) if r > a)
    over_b = sum(1 for b, a in zip(base, actual) if b > a)
    targets = [datetime.strptime(r["target_utc"], ISO) for r in rows]
    if [r["horizon_hours"] for r in rows] != [str(i) for i in range(1, 25)]:
        raise ChartError("expected horizons 1..24")

    W, H = 1000, 600
    L, R_, T, B = 95, 30, 145, 100
    pw, ph = W - L - R_, H - T - B
    y0, y1, ticks = _nice_axis(min(actual + base + ridge), max(actual + base + ridge))
    x = lambda i: L + pw * i / 23
    y = lambda v: T + ph * (1 - (v - y0) / (y1 - y0))
    o_dt = datetime.strptime(origin, ISO)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Helvetica, Arial, sans-serif">',
           f'<title>PJM origin {origin}: actual vs weekly baseline vs ridge (retrospective simulation)</title>',
           f'<desc>24 hourly targets, horizon 1 to 24, hour-ending UTC. Day MAE: weekly baseline {mae_b:,.2f} MW, '
           f'ridge {mae_r:,.2f} MW. Ridge over-forecast {over_r} of 24 hours.</desc>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L}" y="30" font-size="20" font-weight="bold" fill="#111">'
           f'PJM demand, forecast origin {origin[:10]} 00:00 UTC (24 targets, horizon 1-24 h)</text>',
           f'<text x="{L}" y="54" font-size="14" font-weight="bold" fill="#B00020">'
           f'RETROSPECTIVE SIMULATION - historical replay, not a live forecast - {escape(DATA_NOTES.get(label, label))}'
           f'</text>',
           f'<text x="{L}" y="78" font-size="14" fill="#111">Day MAE (this origin): '
           f'<tspan fill="{COLORS["baseline"]}" font-weight="bold">weekly baseline {mae_b:,.2f} MW</tspan>'
           f'   |   <tspan fill="{COLORS["ridge"]}" font-weight="bold">ridge {mae_r:,.2f} MW</tspan></text>',
           f'<text x="{L}" y="98" font-size="13" fill="#444">Ridge forecast above actual in {over_r} of 24 hours; '
           f'weekly baseline above actual in {over_b} of 24.</text>']
    # grid and y axis
    for t in ticks:
        out.append(f'<line x1="{L}" y1="{y(t):.1f}" x2="{L + pw}" y2="{y(t):.1f}" stroke="#dddddd" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y(t) + 4:.1f}" font-size="12" text-anchor="end" fill="#333">{t:,.0f}</text>')
    out.append(f'<text transform="translate(22 {T + ph / 2:.0f}) rotate(-90)" font-size="14" text-anchor="middle" '
               f'fill="#111">Demand (MW)</text>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" stroke="#333" stroke-width="1.5"/>')
    out.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="#333" stroke-width="1.5"/>')
    # x axis: hour-ending UTC for each target
    for i, t in enumerate(targets):
        out.append(f'<line x1="{x(i):.1f}" y1="{T + ph}" x2="{x(i):.1f}" y2="{T + ph + 5}" stroke="#333"/>')
        out.append(f'<text x="{x(i):.1f}" y="{T + ph + 20}" font-size="11" text-anchor="middle" fill="#333">'
                   f'{t:%H}</text>')
    out.append(f'<text x="{x(0):.1f}" y="{T + ph + 36}" font-size="11" text-anchor="middle" fill="#333">'
               f'{targets[0]:%b %d}</text>')
    out.append(f'<text x="{x(23):.1f}" y="{T + ph + 36}" font-size="11" text-anchor="middle" fill="#333">'
               f'{targets[23]:%b %d}</text>')
    out.append(f'<text x="{L + pw / 2:.0f}" y="{T + ph + 62}" font-size="14" text-anchor="middle" fill="#111">'
               f'Target hour, hour-ending UTC (h1 = {targets[0]:%H:%M}Z {targets[0]:%b %d}, '
               f'h24 = {targets[23]:%H:%M}Z {targets[23]:%b %d}, the following midnight)</text>')
    # series
    for name, vals, dash, width in (("actual", actual, "", 3.0), ("baseline", base, "7 4", 2.2),
                                    ("ridge", ridge, "", 2.2)):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[name]}" stroke-width="{width}"{d}/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{COLORS[name]}"/>')
    # legend: one row above the plot so it never covers a data line
    lx = L
    for name, text, dash, width in (("actual", "Actual demand", "", 230),
                                    ("baseline", "Weekly baseline (demand 168 h earlier)", "7 4", 330),
                                    ("ridge", "Ridge (frozen, alpha 100)", "", 260)):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" y1="124" x2="{lx + 28}" y2="124" stroke="{COLORS[name]}" stroke-width="2.5"{d}/>')
        out.append(f'<text x="{lx + 36}" y="128" font-size="12" fill="#111">{escape(text)}</text>')
        lx += width
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eia930.chart", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="paired_predictions.csv from eia930.simulate")
    ap.add_argument("--origin", help="origin like 2024-12-25T00:00:00Z (default: worst fully scored ridge origin)")
    ap.add_argument("--out", required=True, help="SVG path to write")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        origin, label, rows = load_origin(a.predictions, a.origin)
        svg = render_svg(origin, label, rows)
    except (OSError, ChartError, KeyError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")
    print(f"wrote {out} (origin {origin}, data label {label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
