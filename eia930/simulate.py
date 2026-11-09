"""Retrospective simulation: weekly baseline vs one frozen ridge model on the same test rows.

Protocol (fixed in code, see PERIODS):
  fit    origins 2024-07-09 .. 2024-09-29  (daily 00:00 UTC, 24 targets each)
  tune   origins 2024-10-01 .. 2024-10-30  (pick ridge alpha by MAE; model fit on the fit block only)
  refit  fit + tune blocks with the chosen alpha, then FREEZE (digest written to frozen_model.json)
  test   origins 2024-11-01 .. 2024-12-31  (cases are constructed and scored only after the freeze;
         baseline and ridge are compared on the same rows)

Every lag is checked against its origin's shared cutoff (origin - delay). A forecast exists whenever its
lags are usable, whether or not the actual exists. This is a historical replay on a fixed snapshot, not a
live forecast, and published values may have been revised after the fact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .baseline import Config, Slot, build_schedule, run_baseline
from .ingest import IngestError, Observations, read_observations
from .ridge import Prepared, RidgeModel, interval_start_ny, raw_features

RUN_TYPE = "retrospective_simulation"
RUN_BANNER = "RETROSPECTIVE SIMULATION (historical replay on a fixed snapshot; not a live forecast)"
DATA_BANNERS = {"real": "DATA: REAL SNAPSHOT",
                "synthetic": "DATA: SYNTHETIC (generated test data, not real EIA data)"}
ISO = "%Y-%m-%dT%H:%M:%SZ"
PERIODS = {"fit": (datetime(2024, 7, 9), datetime(2024, 9, 29)),
           "tune": (datetime(2024, 10, 1), datetime(2024, 10, 30)),
           "test": (datetime(2024, 11, 1), datetime(2024, 12, 31))}
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
HOUR = timedelta(hours=1)


# ------------------------------------------------------------------ cases
@dataclass(frozen=True)
class Case:
    slot: Slot
    src48: datetime
    src168: datetime
    lag48: float | None
    lag168: float | None
    r48: str
    r168: str
    actual: float | None
    r_actual: str

    @property
    def forecastable(self) -> bool:  # both lags usable, independent of the actual
        return self.lag48 is not None and self.lag168 is not None

    @property
    def reason(self) -> str:  # first failing check: lag168, then lag48, then the actual
        return self.r168 or self.r48 or self.r_actual


def _lag(obs: Observations, src: datetime, slot: Slot, first_obs: datetime, tag: int):
    if src > slot.cutoff:  # every lag is checked against its origin's shared cutoff
        return None, f"lag{tag}_after_cutoff"
    if src not in obs.values:
        return None, f"lag{tag}_" + ("before_history_start" if src < first_obs else "no_row")
    value, why = obs.values[src]
    return (value, "") if value is not None else (None, f"lag{tag}_{why}")


def build_cases(obs: Observations, start: datetime, end: datetime, delay_hours: int = 2) -> list:
    first_obs = min(obs.values)
    out = []
    for slot in build_schedule(Config(start, end, delay_hours=delay_hours)):
        s48, s168 = slot.target - timedelta(hours=48), slot.target - timedelta(hours=168)
        l48, r48 = _lag(obs, s48, slot, first_obs, 48)
        l168, r168 = _lag(obs, s168, slot, first_obs, 168)
        tv = obs.values.get(slot.target)
        if tv is None:
            actual, ra = None, "target_no_row"
        elif tv[0] is None:
            actual, ra = None, f"target_{tv[1]}"
        else:
            actual, ra = tv[0], ""
        out.append(Case(slot, s48, s168, l48, l168, r48, r168, actual, ra))
    return out


def usable(cases: list) -> list:
    return [c for c in cases if c.forecastable and c.actual is not None]


def features(c: Case) -> list:
    return raw_features(c.lag48, c.lag168, c.slot.horizon, c.slot.target)


def assert_no_test_leakage(cases: list) -> None:
    """Nothing used for fitting or tuning may reach into the test period."""
    boundary = PERIODS["test"][0]
    for c in cases:
        assert c.slot.target <= boundary and c.src48 < boundary and c.src168 < boundary, \
            f"leakage: {c.slot.target} reaches the test period"


# ------------------------------------------------------------------ frozen model
@dataclass(frozen=True)
class FrozenModel:
    model: RidgeModel
    digest: str
    alpha: float
    fit_rows: int
    tune_rows: int

    def verify(self) -> None:
        if self.model.digest() != self.digest:
            raise RuntimeError("frozen model was modified after freezing")

    def to_json(self) -> dict:
        return {"run_type": RUN_TYPE, "digest_sha256": self.digest, "alpha": self.alpha,
                "fit_rows": self.fit_rows, "tune_rows": self.tune_rows,
                "fit_origins": [f"{d:%Y-%m-%d}" for d in PERIODS["fit"]],
                "tune_origins": [f"{d:%Y-%m-%d}" for d in PERIODS["tune"]],
                "model": self.model.params()}


class NoCommonRowsError(ValueError):
    """Fitting and tuning worked, but no test hour has both forecasts and an actual."""


# ------------------------------------------------------------------ metrics
def mae(errs):
    return sum(abs(e) for e in errs) / len(errs) if errs else None


def rmse(errs):
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None


@dataclass(frozen=True)
class Paired:
    case: Case
    baseline: float | None
    ridge: float | None
    in_common: bool


@dataclass
class Simulation:
    obs: Observations
    delay_hours: int
    validation: list            # [(alpha, mae)]
    validation_baseline_mae: float
    frozen: FrozenModel
    fit_cases: int
    tune_cases: int
    paired: list
    summary: dict


def run_simulation(obs: Observations, delay_hours: int = 2, on_freeze=None,
                   expect_common: int | None = None) -> Simulation:
    fit_all = build_cases(obs, *PERIODS["fit"], delay_hours)
    tune_all = build_cases(obs, *PERIODS["tune"], delay_hours)
    fit_rows, tune_rows = usable(fit_all), usable(tune_all)
    assert_no_test_leakage(fit_rows + tune_rows)
    if not fit_rows or not tune_rows:
        raise ValueError("no usable fit or tune rows")

    # ---- tune alpha: fit on the fit block only, score MAE on the tune block
    prep_fit = Prepared([features(c) for c in fit_rows], [c.actual for c in fit_rows])
    validation = []
    for alpha in ALPHAS:
        m = prep_fit.fit(alpha)
        validation.append((alpha, mae([m.predict(features(c)) - c.actual for c in tune_rows])))
    best_alpha = min(validation, key=lambda t: t[1])[0]
    val_base = mae([c.lag168 - c.actual for c in tune_rows])  # context only

    # ---- refit on both blocks with the chosen alpha, then freeze BEFORE reading any test row
    both = fit_rows + tune_rows
    model = Prepared([features(c) for c in both], [c.actual for c in both]).fit(best_alpha)
    frozen = FrozenModel(model, model.digest(), best_alpha, len(fit_rows), len(tune_rows))
    if on_freeze:
        on_freeze(frozen)

    # ---- test: baseline and ridge on the same rows
    frozen.verify()
    t0, t1 = PERIODS["test"]
    cases = build_cases(obs, t0, t1, delay_hours)
    base = run_baseline(obs, Config(t0, t1, delay_hours=delay_hours))
    paired = []
    for c, brow in zip(cases, base.rows):
        assert brow.slot == c.slot
        assert brow.forecast == (c.lag168 if not c.r168 else None), "baseline lookup disagrees with lag168"
        assert brow.actual == c.actual
        ridge = frozen.model.predict(features(c)) if c.forecastable else None  # no peeking at actual
        paired.append(Paired(c, brow.forecast, ridge, c.forecastable and c.actual is not None))
    if not any(x.in_common for x in paired):
        reasons = Counter(x.case.reason for x in paired)
        raise NoCommonRowsError(
            "no test hour could be scored for both forecasts. Fitting and tuning succeeded (frozen model: "
            f"alpha {frozen.alpha:g}, digest {frozen.digest}), but of {len(paired)} scheduled test "
            f"hours: baseline forecasts {sum(1 for x in paired if x.baseline is not None)}, "
            f"ridge forecasts {sum(1 for x in paired if x.ridge is not None)}, "
            f"actuals present {sum(1 for x in paired if x.case.actual is not None)}. "
            "Skip reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) +
            ". Check that the CSV covers the test period (2024-11-01 .. 2025-01-01) and has demand values there. "
            "No output files were written or changed.")
    sim = Simulation(obs, delay_hours, validation, val_base, frozen, len(fit_rows), len(tune_rows),
                     paired, {})
    sim.summary = summarize(sim, expect_common)
    return sim


def summarize(sim: Simulation, expect_common: int | None) -> dict:
    p = sim.paired
    common = [x for x in p if x.in_common]
    be = [x.baseline - x.case.actual for x in common]
    re_ = [x.ridge - x.case.actual for x in common]
    by_h = {}
    for h in range(1, 25):
        rows = [x for x in common if x.case.slot.horizon == h]
        b = [x.baseline - x.case.actual for x in rows]
        r = [x.ridge - x.case.actual for x in rows]
        by_h[h] = {"n": len(rows), "baseline_mae": mae(b), "ridge_mae": mae(r),
                   "baseline_rmse": rmse(b), "ridge_rmse": rmse(r)}
    days = defaultdict(list)
    for x in common:
        days[x.case.slot.origin].append(x)
    full = {o: rows for o, rows in days.items() if len(rows) == 24}

    def day_stats(o):
        rows = full[o]
        return {"origin_utc": f"{o:{ISO}}",
                "ridge_mae": mae([x.ridge - x.case.actual for x in rows]),
                "baseline_mae": mae([x.baseline - x.case.actual for x in rows])}
    worst = max(full, key=lambda o: day_stats(o)["ridge_mae"]) if full else None
    worst_base = max(full, key=lambda o: day_stats(o)["baseline_mae"]) if full else None
    wins = sum(1 for b, r in zip(be, re_) if abs(r) < abs(b))
    losses = sum(1 for b, r in zip(be, re_) if abs(r) > abs(b))
    base_own = [x for x in p if x.baseline is not None and x.case.actual is not None]
    reasons_common = Counter(x.case.reason for x in p if not x.in_common)
    reasons_base = Counter((x.case.r168 or x.case.r_actual) for x in p
                           if not (x.baseline is not None and x.case.actual is not None))
    n = len(p)
    return {
        "run_type": RUN_TYPE,
        "ba": sim.obs.ba, "snapshot_sha256": sim.obs.sha256, "delay_hours": sim.delay_hours,
        "periods": {k: [f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for k, (a, b) in PERIODS.items()},
        "fit_rows": sim.fit_cases, "tune_rows": sim.tune_cases,
        "alpha_grid": list(ALPHAS), "chosen_alpha": sim.frozen.alpha,
        "frozen_digest_sha256": sim.frozen.digest,
        "validation_mae": [{"alpha": a, "mae_mw": m} for a, m in sim.validation],
        "validation_baseline_mae_mw": sim.validation_baseline_mae,
        "test_scheduled": n,
        "test_baseline_forecasts": sum(1 for x in p if x.baseline is not None),
        "test_ridge_forecasts": sum(1 for x in p if x.ridge is not None),
        "test_actuals_present": sum(1 for x in p if x.case.actual is not None),
        "test_baseline_scored_own_set": len(base_own),
        "test_common_rows": len(common),
        "expected_common_rows": expect_common,
        "common_rows_match_expected": None if expect_common is None else len(common) == expect_common,
        "lags_after_cutoff_used": sum(1 for x in p if x.ridge is not None and
                                      (x.case.src48 > x.case.slot.cutoff or x.case.src168 > x.case.slot.cutoff)),
        "not_in_common_by_reason": dict(sorted(reasons_common.items())),
        "baseline_not_scored_by_reason": dict(sorted(reasons_base.items())),
        "coverage_common_of_scheduled": len(common) / n if n else 0.0,
        "overall": {"n": len(common),
                    "baseline": {"mae_mw": mae(be), "rmse_mw": rmse(be)},
                    "ridge": {"mae_mw": mae(re_), "rmse_mw": rmse(re_)},
                    "ridge_wins": wins, "ridge_losses": losses},
        "by_horizon": by_h,
        "fully_scored_days": len(full),
        "highest_mae_day_ridge": day_stats(worst) if worst else None,
        "highest_mae_day_baseline": day_stats(worst_base) if worst_base else None,
    }


# ------------------------------------------------------------------ outputs
def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def render_report(sim: Simulation, label: str) -> str:
    s = sim.summary
    o = s["overall"]
    L = [f"# Paired report: weekly baseline vs frozen ridge, PJM",
         "", f"**{RUN_BANNER}**", f"**{DATA_BANNERS[label]}**", "",
         f"- Snapshot SHA-256: `{s['snapshot_sha256']}`",
         f"- Cutoff = origin - {s['delay_hours']} h (assumed); every lag source (48 h and 168 h before the target) must be at or before it.",
         f"- Fit origins {s['periods']['fit'][0]} .. {s['periods']['fit'][1]} ({s['fit_rows']} usable rows); "
         f"tune origins {s['periods']['tune'][0]} .. {s['periods']['tune'][1]} ({s['tune_rows']} usable rows); "
         f"test origins {s['periods']['test'][0]} .. {s['periods']['test'][1]}.",
         f"- Ridge alpha chosen by MAE on the tune block: **{s['chosen_alpha']:g}**; refit on fit + tune; "
         f"frozen digest `{s['frozen_digest_sha256']}` (the model was frozen before test cases were constructed or scored).",
         "- Features: demand 48 h and 168 h before the target, horizon (1..24), hour-of-day and day-of-week of the "
         "target interval's start in America/New_York (UTC labels are stamped as UTC first, then converted).",
         "", "## Tuning (tune block, model fit on the fit block only)", "",
         "| alpha | tune MAE (MW) |", "|---:|---:|"]
    for v in s["validation_mae"]:
        star = " (chosen)" if v["alpha"] == s["chosen_alpha"] else ""
        L.append(f"| {v['alpha']:g}{star} | {_fmt(v['mae_mw'])} |")
    L += [f"", f"Weekly baseline on the same tune rows (context): {_fmt(s['validation_baseline_mae_mw'])} MW.",
          "", "## Coverage on the test block", "",
          "| quantity | rows |", "|---|---:|",
          f"| scheduled (61 origins x 24) | {s['test_scheduled']:,} |",
          f"| baseline forecasts generated | {s['test_baseline_forecasts']:,} |",
          f"| ridge forecasts generated | {s['test_ridge_forecasts']:,} |",
          f"| actuals present | {s['test_actuals_present']:,} |",
          f"| baseline scored on its own (CHECKPOINT2 figure) | {s['test_baseline_scored_own_set']:,} |",
          f"| **common rows (both forecasts and the actual)** | **{s['test_common_rows']:,}** |",
          f"| expected common rows | {s['expected_common_rows'] if s['expected_common_rows'] is not None else 'not set'}"
          f" ({'match' if s['common_rows_match_expected'] else 'MISMATCH' if s['common_rows_match_expected'] is False else 'n/a'}) |",
          f"| coverage of scheduled | {s['coverage_common_of_scheduled']:.4%} |",
          f"| lags used after cutoff | {s['lags_after_cutoff_used']} |",
          "", "Skip reasons (first failing check: lag168, then lag48, then the actual):", "",
          "| reason | baseline's own set | common set |", "|---|---:|---:|"]
    reasons = sorted(set(s["baseline_not_scored_by_reason"]) | set(s["not_in_common_by_reason"]))
    for r in reasons:
        L.append(f"| {r} | {s['baseline_not_scored_by_reason'].get(r, 0)} | {s['not_in_common_by_reason'].get(r, 0)} |")
    L.append(f"| **total not scored** | **{sum(s['baseline_not_scored_by_reason'].values())}** | "
             f"**{sum(s['not_in_common_by_reason'].values())}** |")
    imp = (o["baseline"]["mae_mw"] - o["ridge"]["mae_mw"]) / o["baseline"]["mae_mw"] * 100
    L += ["", f"## Overall, recomputed on the {o['n']:,} common rows", "",
          "| model | MAE (MW) | RMSE (MW) |", "|---|---:|---:|",
          f"| weekly baseline (168 h) | {_fmt(o['baseline']['mae_mw'])} | {_fmt(o['baseline']['rmse_mw'])} |",
          f"| frozen ridge | {_fmt(o['ridge']['mae_mw'])} | {_fmt(o['ridge']['rmse_mw'])} |",
          "", f"Ridge MAE change vs baseline: {-imp:+.2f}% (negative = better). Ridge has the smaller absolute error on "
          f"{o['ridge_wins']:,} rows, the larger on {o['ridge_losses']:,}.",
          "", "## By horizon (common rows)", "",
          "| horizon (h) | n | baseline MAE | ridge MAE | baseline RMSE | ridge RMSE |",
          "|---:|---:|---:|---:|---:|---:|"]
    for h, v in s["by_horizon"].items():
        L.append(f"| {h} | {v['n']} | {_fmt(v['baseline_mae'])} | {_fmt(v['ridge_mae'])} | "
                 f"{_fmt(v['baseline_rmse'])} | {_fmt(v['ridge_rmse'])} |")
    w = s["highest_mae_day_ridge"]
    L += ["", f"## Highest-MAE fully scored day ({s['fully_scored_days']} origins have all 24 rows in the common set)", ""]
    if w:
        L += [f"By ridge MAE: origin **{w['origin_utc']}** (targets {w['origin_utc'][:10]} 01:00Z to the next midnight): "
              f"ridge MAE {_fmt(w['ridge_mae'])} MW, baseline MAE {_fmt(w['baseline_mae'])} MW.",
              f"Highest by baseline MAE: origin {s['highest_mae_day_baseline']['origin_utc']} "
              f"(baseline {_fmt(s['highest_mae_day_baseline']['baseline_mae'])} MW, "
              f"ridge {_fmt(s['highest_mae_day_baseline']['ridge_mae'])} MW).", "",
              "| h | target UTC | interval start (NY) | actual | baseline | ridge | baseline err | ridge err |",
              "|---:|---|---|---:|---:|---:|---:|---:|"]
        for x in sim.paired:
            if f"{x.case.slot.origin:{ISO}}" == w["origin_utc"]:
                c = x.case
                L.append(f"| {c.slot.horizon} | {c.slot.target:{ISO}} | {interval_start_ny(c.slot.target):%Y-%m-%d %H:%M %Z} | "
                         f"{c.actual:,.0f} | {x.baseline:,.0f} | {x.ridge:,.0f} | {x.baseline - c.actual:+,.0f} | "
                         f"{x.ridge - c.actual:+,.0f} |")
    else:
        L.append("No origin has all 24 rows scored.")
    L += ["", "## Caveats", "",
          "- Retrospective replay on one fixed snapshot: published values may have been revised after first release, "
          "so real-time errors can differ.",
          "- The two-hour publication delay and the cutoff definition are assumptions.",
          "- One balancing authority, one two-month test block; alpha was tuned on one month.",
          "- The model was frozen before test cases were constructed or scored; no test value influenced fitting or "
          "tuning. The CLI loads the full CSV into memory up front, so test rows are in memory before the freeze; "
          "the guarantee is about what the code reads, and a test checks that no test-period timestamp is looked "
          "up before the freeze.",
          "- The two-hour publication delay (and so the cutoff) is an assumption. The snapshot may contain observations "
          "that were revised after first publication, so a real-time forecast would have seen different inputs.",
          "- The source publication date of the snapshot is unverified (download date 2026-11-08)."]
    return "\n".join(L) + "\n"


def write_outputs(sim: Simulation, out_dir: Path, label: str) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"pred": out_dir / "paired_predictions.csv", "val": out_dir / "validation.csv",
             "sum": out_dir / "summary.json", "rep": out_dir / "report.md"}
    with open(paths["pred"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_type", "data_label", "origin_utc", "cutoff_utc", "target_utc", "horizon_hours",
                    "interval_start_ny", "lag48_source_utc", "lag48_mw", "lag168_source_utc", "lag168_mw",
                    "baseline_forecast_mw", "ridge_forecast_mw", "actual_mw", "in_common_set", "reason"])
        for x in sim.paired:
            c, s = x.case, x.case.slot
            blank = lambda v: "" if v is None else v
            w.writerow([RUN_TYPE, label, f"{s.origin:{ISO}}", f"{s.cutoff:{ISO}}", f"{s.target:{ISO}}",
                        s.horizon, f"{interval_start_ny(s.target):%Y-%m-%d %H:%M %Z}",
                        f"{c.src48:{ISO}}", blank(c.lag48), f"{c.src168:{ISO}}", blank(c.lag168),
                        blank(x.baseline), blank(x.ridge), blank(c.actual),
                        int(x.in_common), c.reason])
    with open(paths["val"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_type", "data_label", "alpha", "tune_mae_mw", "chosen"])
        for a, m in sim.validation:
            w.writerow([RUN_TYPE, label, a, m, int(a == sim.frozen.alpha)])
    paths["sum"].write_text(json.dumps({"data_label": label, **sim.summary}, indent=2) + "\n")
    paths["rep"].write_text(render_report(sim, label), encoding="utf-8")
    return list(paths.values())


def write_frozen(frozen: FrozenModel, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "frozen_model.json"
    p.write_text(json.dumps({"data_label": label, **frozen.to_json()}, indent=2) + "\n")
    return p


class PublishError(OSError):
    """Publishing failed. `retained` is the staging directory kept because rollback also failed (else None)."""

    retained = None


def _move(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def _mkdirs(path: Path) -> list:
    """mkdir -p, returning the directories this call created (innermost first)."""
    missing, p = [], path
    while not p.exists():
        missing.append(p)
        p = p.parent
    path.mkdir(parents=True, exist_ok=True)
    return missing


def _new_staging(out_dir: Path) -> Path:
    """A staging directory unique to this invocation, beside out_dir (same volume, so moves are renames)."""
    return Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))


def publish(new_dir: Path, backup_dir: Path, out_dir: Path) -> list:
    """Final step: move the complete staged set into out_dir, all or nothing.

    Phase 1 moves any existing same-named files into backup_dir; phase 2 moves the new files in. If anything
    fails, the new files already placed are removed and the backed-up originals are moved back, so the previous
    artifacts are left byte-for-byte intact. If the rollback itself fails, the originals stay in backup_dir and
    the raised PublishError says where.
    """
    names = sorted(os.listdir(new_dir), key=lambda n: (n != "frozen_model.json", n))
    out_existed = out_dir.exists()
    backed, placed = [], []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            if (out_dir / n).exists():
                _move(out_dir / n, backup_dir / n)
                backed.append(n)
        for n in names:
            _move(new_dir / n, out_dir / n)
            placed.append(n)
    except BaseException as exc:
        try:
            for n in reversed(placed):
                (out_dir / n).unlink()
            for n in reversed(backed):
                _move(backup_dir / n, out_dir / n)
            if not out_existed:
                try:
                    out_dir.rmdir()
                except OSError:
                    pass
        except BaseException as rollback_exc:
            err = PublishError(f"publishing failed ({exc}) and the rollback failed ({rollback_exc}); the previous "
                               f"files that were moved aside are preserved in {backup_dir}")
            err.retained = backup_dir.parent
            raise err from exc
        raise
    return [out_dir / n for n in names]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eia930.simulate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ba", required=True)
    ap.add_argument("--data-label", required=True, choices=["real", "synthetic"])
    ap.add_argument("--out-dir", default=None, help="default: outputs/<data-label>/<BA>/simulation")
    ap.add_argument("--delay-hours", type=int, default=2, help="assumed publication delay")
    ap.add_argument("--expect-sha256")
    ap.add_argument("--expect-common-rows", type=int, help="exit 1 if the common test rows differ")
    # exit codes: 0 ok; 1 expectation mismatch or no test hour scorable by both forecasts; 2 input error
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    out_dir = Path(a.out_dir) if a.out_dir else Path("outputs") / a.data_label / a.ba / "simulation"
    # Publication is the final step. Everything is built in a staging directory that belongs to this invocation
    # alone and is moved into out_dir only after every check has passed (checksum, fit, tune, freeze, scoring,
    # --expect-common-rows). Any earlier failure (exit 1 or 2) leaves an existing output set untouched.
    created, staging, keep_staging, published = [], None, False, False
    try:
        obs = read_observations(a.csv, a.ba)
        if a.expect_sha256 and obs.sha256.lower() != a.expect_sha256.lower():
            print(f"ERROR: checksum mismatch: got {obs.sha256}", file=sys.stderr)
            return 2
        created = _mkdirs(out_dir.parent)
        staging = _new_staging(out_dir)
        new_dir, backup_dir = staging / "new", staging / "backup"
        sim = run_simulation(obs, a.delay_hours, on_freeze=lambda f: write_frozen(f, new_dir, a.data_label),
                             expect_common=a.expect_common_rows)
        if sim.summary["common_rows_match_expected"] is False:
            print(f"MISMATCH: expected {a.expect_common_rows} common rows, got {sim.summary['test_common_rows']}. "
                  "No output files were written or changed.", file=sys.stderr)
            return 1
        write_outputs(sim, new_dir, a.data_label)
        files = publish(new_dir, backup_dir, out_dir)
        published = True
    except NoCommonRowsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except PublishError as e:
        keep_staging = e.retained is not None
        print(f"ERROR: PublishError: {e}", file=sys.stderr)
        return 2
    except (IngestError, OSError, ValueError, AssertionError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        if staging is not None and not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        if not published and not keep_staging:
            for d in created:  # directories this run created, removed again if still empty
                try:
                    d.rmdir()
                except OSError:
                    pass
    s, o = sim.summary, sim.summary["overall"]
    print(RUN_BANNER)
    print(DATA_BANNERS[a.data_label])
    print(f"BA: {s['ba']}   snapshot sha256: {s['snapshot_sha256']}")
    print(f"fit {s['periods']['fit']} rows={s['fit_rows']}   tune {s['periods']['tune']} rows={s['tune_rows']}   "
          f"test {s['periods']['test']}")
    print(f"chosen alpha: {s['chosen_alpha']:g}   frozen digest: {s['frozen_digest_sha256']}")
    print(f"test: scheduled {s['test_scheduled']}  baseline scored (own) {s['test_baseline_scored_own_set']}  "
          f"common rows {s['test_common_rows']}  expected {s['expected_common_rows']}  "
          f"lags after cutoff {s['lags_after_cutoff_used']}")
    for name in ("baseline", "ridge"):
        print(f"  {name:<8} MAE {o[name]['mae_mw']:,.2f} MW   RMSE {o[name]['rmse_mw']:,.2f} MW   (n={o['n']})")
    w = s["highest_mae_day_ridge"]
    if w:
        print(f"highest-MAE fully scored day (ridge): origin {w['origin_utc']}  ridge {w['ridge_mae']:,.2f}  "
              f"baseline {w['baseline_mae']:,.2f}")
    print("wrote:", ", ".join(str(p) for p in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
