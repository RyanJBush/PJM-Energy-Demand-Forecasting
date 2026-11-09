"""CLI: python -m eia930 --csv PATH --ba PJM --data-label real --origin-start 2024-11-01 --origin-end 2024-12-31

Exit codes: 0 scored at least one row; 1 nothing scored; 2 input error (missing file or column,
unknown BA, duplicate UTC keys, checksum mismatch, bad arguments).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from .baseline import Config, run_baseline
from .ingest import IngestError, read_observations

ISO = "%Y-%m-%dT%H:%M:%SZ"
BANNERS = {
    "real": "DATA: REAL SNAPSHOT",
    "synthetic": "DATA: SYNTHETIC FIXTURE (hand-set test values, not real EIA data)",
}


def _date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def write_outputs(result, obs, out_dir: Path, label: str) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = out_dir / "forecasts.csv"
    with open(fc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_label", "origin_utc", "cutoff_utc", "target_utc", "horizon_hours",
                    "lag_source_utc", "forecast_mw", "actual_mw", "error_mw", "status", "reason"])
        for r in result.rows:
            s = r.slot
            w.writerow([label, f"{s.origin:{ISO}}", f"{s.cutoff:{ISO}}", f"{s.target:{ISO}}",
                        s.horizon, f"{r.source:{ISO}}",
                        "" if r.forecast is None else r.forecast,
                        "" if r.actual is None else r.actual,
                        "" if r.error is None else r.error, r.status, r.reason])
    sk = out_dir / "ingest_skipped.csv"
    with open(sk, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_label", "csv_line", "ba", "raw_utc_timestamp", "reason"])
        for s in obs.skipped:
            w.writerow([label, s.line, s.ba, s.raw_timestamp, s.reason])
    sm = out_dir / "summary.json"
    sm.write_text(json.dumps({"data_label": label, **result.summary}, indent=2) + "\n")
    return [fc, sk, sm]


def print_summary(s: dict, label: str) -> None:
    print(BANNERS[label])
    print(f"BA: {s['ba']}   snapshot sha256: {s['snapshot_sha256']}")
    c = s["config"]
    print(f"config: lag={c['lag_hours']}h  delay={c['delay_hours']}h (assumed; cutoff = origin - delay)  "
          f"horizons=1..{c['horizons']}")
    print(f"origins: {s['origins']}  ({s['origin_first_utc']} .. {s['origin_last_utc']}, daily 00:00 UTC)")
    print(f"targets: {s['target_first_utc']} .. {s['target_last_utc']}")
    print(f"csv rows read: {s['csv_rows_read']}  rows for BA: {s['csv_rows_for_ba']}  "
          f"ingest-skipped rows: {s['ingest_skipped_rows']}")
    print(f"scheduled: {s['scheduled']}  forecasts generated: {s['forecasts_generated']}  "
          f"scored: {s['scored']}  not scored: {s['not_scored']}")
    print(f"forecasts without an actual: {s['forecasts_without_actual']}   "
          f"lags used after cutoff: {s['lags_after_cutoff_used']}")
    print(f"scored target range: {s['scored_target_first_utc']} .. {s['scored_target_last_utc']}")
    print(f"coverage of scheduled: {s['coverage_of_scheduled']:.4%}")
    for reason, n in s["not_scored_by_reason"].items():
        print(f"  not scored {n:>5}  {reason}")
    if s["scored"]:
        print(f"MAE {s['mae_mw']:.2f} MW  RMSE {s['rmse_mw']:.2f} MW  bias {s['bias_mw']:+.2f} MW  "
              f"MAPE {s['mape_pct']:.3f}% (n={s['mape_n']})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eia930", description=__doc__)
    p.add_argument("--csv", required=True, help="EIA-930 balance CSV")
    p.add_argument("--ba", required=True, help="balancing authority code, e.g. PJM")
    p.add_argument("--data-label", required=True, choices=["real", "synthetic"],
                   help="stamped on every output row, the summary, the banner and the default folder")
    p.add_argument("--origin-start", required=True, type=_date, help="first origin date (00:00 UTC)")
    p.add_argument("--origin-end", required=True, type=_date, help="last origin date, inclusive")
    p.add_argument("--out-dir", default=None, help="default: outputs/<data-label>/<BA>")
    p.add_argument("--lag-hours", type=int, default=168)
    p.add_argument("--delay-hours", type=int, default=2, help="assumed publication delay")
    p.add_argument("--expect-sha256", help="fail if the file checksum differs")
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        obs = read_observations(a.csv, a.ba)
        if a.expect_sha256 and obs.sha256.lower() != a.expect_sha256.lower():
            print(f"ERROR: checksum mismatch: got {obs.sha256}", file=sys.stderr)
            return 2
        cfg = Config(a.origin_start, a.origin_end, a.lag_hours, a.delay_hours)
        result = run_baseline(obs, cfg)
    except (IngestError, OSError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out_dir = Path(a.out_dir) if a.out_dir else Path("outputs") / a.data_label / a.ba
    files = write_outputs(result, obs, out_dir, a.data_label)
    print_summary(result.summary, a.data_label)
    print("wrote:", ", ".join(str(f) for f in files))
    return 0 if result.summary["scored"] else 1


if __name__ == "__main__":
    sys.exit(main())
