"""Ridge and retrospective-simulation tests. All data here is SYNTHETIC (generated), never real EIA data.

Synthetic series (BA "SYN"): demand(t) = 1000 + 5*hour_NY + 20*dow_NY + 0.05*i, where hour_NY/dow_NY come from
the interval START (t - 1 h) read as UTC and converted to America/New_York, and i = hours since 2024-07-01 06:00Z.
Weekly baseline error is a constant 0.05*168 = 8.4 MW and a ridge on lag168 + intercept can be ~exact, except in
the week after the 2024-11-03 DST change when the 168 h lag source sits in the other UTC offset.

Hand-set blank hours (each removes the rows where it is the target, the 48 h lag source, or the 168 h lag source):
  test  : Nov 10 12:00, Nov 20 03:00, Dec 5 18:00   -> 3 hours x 3 rows = 9 rows leave the common set
  fit   : Aug 10 12:00                                -> 3 fit rows lost (1992 -> 1989)
  tune  : Oct 12 04:00                                -> 3 tune rows lost (720 -> 717)
Expected test counts: scheduled 1464; baseline forecasts 1461, own scored 1458 (lag168 + actual);
ridge forecasts 1458 (lag48 + lag168); actuals 1461; common 1455.
"""
import csv
import dataclasses
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eia930.ingest import Observations, read_observations  # noqa: E402
from eia930.ridge import (FEATURES, Prepared, RidgeModel, cholesky_solve,  # noqa: E402
                          interval_start_ny, raw_features)
from eia930.simulate import (ALPHAS, PERIODS, Case, FrozenModel, assert_no_test_leakage,  # noqa: E402
                             build_cases, main, run_simulation, usable)

HOUR = timedelta(hours=1)
NY = ZoneInfo("America/New_York")
START = datetime(2024, 7, 1, 6)
END = datetime(2025, 1, 1, 0)
BLANKS = [datetime(2024, 11, 10, 12), datetime(2024, 11, 20, 3), datetime(2024, 12, 5, 18),
          datetime(2024, 8, 10, 12), datetime(2024, 10, 12, 4)]


# ------------------------------------------------------------------ synthetic data
def demand(t: datetime) -> float:
    s = (t - HOUR).replace(tzinfo=timezone.utc).astimezone(NY)
    return 1000 + 5 * s.hour + 20 * s.weekday() + 0.05 * ((t - START) / HOUR)


def write_series(path, blanks=BLANKS, overrides=None, test_scale=None, end=END, blank_from=None, shift=0.0):
    overrides = overrides or {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Balancing Authority", "UTC Time at End of Hour", "Demand (MW)"])
        t = START
        while t <= end:
            if t in blanks or (blank_from is not None and t >= blank_from):
                v = ""
            elif t in overrides:
                v = overrides[t]
            else:
                v = demand(t) + shift
                if test_scale is not None and t >= datetime(2024, 11, 1, 1):
                    v = test_scale
            w.writerow(["SYN", f"{t:%m/%d/%Y %I:%M:%S %p}", v])
            t += HOUR


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    p = tmp_path_factory.mktemp("synth") / "synth.csv"
    write_series(p)
    return str(p)


@pytest.fixture(scope="module")
def sim(synth):
    frozen_seen = []
    s = run_simulation(read_observations(synth, "SYN"), on_freeze=frozen_seen.append, expect_common=1455)
    s.frozen_seen = frozen_seen
    return s


# ------------------------------------------------------------------ time zone handling
def test_label_is_stamped_utc_then_converted_using_interval_start():
    # label 2024-07-08 06:00Z ends the hour that STARTS 05:00Z = 01:00 EDT, Monday
    s = interval_start_ny(datetime(2024, 7, 8, 6))
    assert (s.hour, s.weekday(), s.utcoffset()) == (1, 0, timedelta(hours=-4))
    # one hour earlier label: interval starts 04:00Z -> 00:00 EDT Monday; if the label were used as the start
    # (or read as Eastern) these would differ
    assert (interval_start_ny(datetime(2024, 7, 8, 5)).hour, interval_start_ny(datetime(2024, 7, 8, 5)).weekday()) == (0, 0)
    # label 04:00Z -> interval starts 03:00Z = 23:00 EDT SUNDAY: the day of week flips with the interval start
    sun = interval_start_ny(datetime(2024, 7, 8, 4))
    assert (sun.hour, sun.weekday()) == (23, 6)
    assert interval_start_ny(datetime(2024, 7, 8, 6)).hour != 6  # not read as local time


def test_dst_transitions():
    # fall back 2024-11-03: 05:00Z start is 01:00 EDT, 06:00Z start is 01:00 EST (same local hour, offset differs)
    a, b = interval_start_ny(datetime(2024, 11, 3, 6)), interval_start_ny(datetime(2024, 11, 3, 7))
    assert (a.hour, a.utcoffset()) == (1, timedelta(hours=-4))
    assert (b.hour, b.utcoffset()) == (1, timedelta(hours=-5))
    # spring forward 2024-03-10: 06:00Z start is 01:00 EST, 07:00Z start is 03:00 EDT (02:00 never occurs)
    assert interval_start_ny(datetime(2024, 3, 10, 7)).hour == 1
    assert interval_start_ny(datetime(2024, 3, 10, 8)).hour == 3
    # winter: Dec 25 2024 17:00Z label -> start 16:00Z = 11:00 EST, Wednesday
    w = interval_start_ny(datetime(2024, 12, 25, 17))
    assert (w.hour, w.weekday()) == (11, 2)


def test_aware_datetime_rejected():
    with pytest.raises(ValueError):
        interval_start_ny(datetime(2024, 7, 8, 6, tzinfo=timezone.utc))


def test_feature_vector_layout():
    x = raw_features(111.0, 222.0, 5, datetime(2024, 7, 8, 6))
    assert len(x) == len(FEATURES) == 3 + 24 + 7
    assert x[:3] == [111.0, 222.0, 5.0]
    assert x[3 + 1] == 1.0 and sum(x[3:27]) == 1.0          # hour_ny_01
    assert x[27 + 0] == 1.0 and sum(x[27:]) == 1.0          # dow_ny_0 (Monday)
    assert FEATURES[:3] == ("lag48_mw", "lag168_mw", "horizon_hours")


# ------------------------------------------------------------------ ridge numerics
def test_cholesky_solves_known_system():
    a = [[4.0, 2.0, 0.6], [2.0, 2.0, 0.5], [0.6, 0.5, 3.0]]
    x_true = [1.0, -2.0, 3.0]
    b = [sum(a[i][j] * x_true[j] for j in range(3)) for i in range(3)]
    assert all(math.isclose(g, t, abs_tol=1e-12) for g, t in zip(cholesky_solve(a, b), x_true))
    with pytest.raises(ValueError):
        cholesky_solve([[1.0, 2.0], [2.0, 1.0]], [1.0, 1.0])  # indefinite


def _toy(n=300):
    rows, y = [], []
    for i in range(n):
        t = datetime(2024, 7, 9) + i * HOUR
        l48, l168, h = 1000 + 7 * (i % 11), 2000 + 13 * (i % 17), 1 + i % 24
        rows.append(raw_features(l48, l168, h, t))
        s = interval_start_ny(t)
        y.append(50 + 0.3 * l48 + 0.6 * l168 + 4.0 * h + 9 * s.hour + 11 * s.weekday())
    return rows, y


def test_exactly_linear_data_is_recovered_with_tiny_alpha():
    rows, y = _toy()
    m = Prepared(rows, y).fit(1e-9)
    assert max(abs(m.predict(r) - v) for r, v in zip(rows, y)) < 1e-4


def test_intercept_is_not_penalized_and_scale_is_standardized():
    rows, y = _toy()
    for alpha in (0.1, 100.0, 1e6):
        m0 = Prepared(rows, y).fit(alpha)
        m1 = Prepared(rows, [v + 12345.0 for v in y]).fit(alpha)
        assert all(abs((m1.predict(r) - m0.predict(r)) - 12345.0) < 1e-6 for r in rows[:20])
        scaled = [[r[0] * 1000, r[1] * 1000] + r[2:] for r in rows]  # change units of the lag columns
        m2 = Prepared(scaled, y).fit(alpha)
        assert all(abs(m2.predict(s) - m0.predict(r)) < 1e-5 for s, r in zip(scaled[:20], rows[:20]))


def test_larger_alpha_shrinks_coefficients_and_converges_to_mean():
    rows, y = _toy()
    p = Prepared(rows, y)
    norms = [math.sqrt(sum(c * c for c in p.fit(a).coef)) for a in (0.01, 1.0, 100.0, 1e4)]
    assert norms == sorted(norms, reverse=True)
    huge = p.fit(1e12)
    assert all(abs(huge.predict(r) - sum(y) / len(y)) < 1e-3 for r in rows[:10])


def test_alpha_must_be_positive():
    rows, y = _toy(50)
    with pytest.raises(ValueError):
        Prepared(rows, y).fit(0.0)


# ------------------------------------------------------------------ cases and cutoff
def dense_obs(origin, lo=-400, hi=30):
    return Observations(ba="MEM", values={origin + i * HOUR: (float(i), None) for i in range(lo, hi + 1)})


def case_by_h(cases, h):
    return next(c for c in cases if c.slot.horizon == h)


def test_both_lags_are_checked_against_the_shared_cutoff():
    # delay 30 h: cutoff = origin - 30 h; lag48 source = origin + h - 48 is allowed only for h <= 18
    o = datetime(2024, 11, 1)
    cases = build_cases(dense_obs(o), o, o, delay_hours=30)
    for c in cases:
        h = c.slot.horizon
        assert c.r48 == ("" if h <= 18 else "lag48_after_cutoff"), h
        assert c.r168 == "" and c.src168 <= c.slot.cutoff
        assert c.forecastable == (h <= 18)
        assert c.src48 <= c.slot.cutoff or c.r48 == "lag48_after_cutoff"
    # delay 150 h: lag168 allowed only for h <= 18, lag48 never; lag168 is reported first
    cases = build_cases(dense_obs(o), o, o, delay_hours=150)
    assert {c.reason for c in cases if c.slot.horizon <= 18} == {"lag48_after_cutoff"}
    assert {c.reason for c in cases if c.slot.horizon > 18} == {"lag168_after_cutoff"}
    assert not any(c.forecastable for c in cases)


def test_lag_value_after_cutoff_is_never_used():
    o = datetime(2024, 11, 1)
    c = case_by_h(build_cases(dense_obs(o), o, o, delay_hours=30), 24)
    assert c.lag48 is None and c.src48 > c.slot.cutoff  # the value exists in the data but is refused


def test_default_delay_never_blocks_the_two_lags():
    o = datetime(2024, 11, 1)
    cases = build_cases(dense_obs(o), o, o)
    assert all(c.forecastable and c.src48 <= c.slot.cutoff and c.src168 <= c.slot.cutoff for c in cases)
    assert [c.lag48 for c in cases] == [float(h - 48) for h in range(1, 25)]
    assert [c.lag168 for c in cases] == [float(h - 168) for h in range(1, 25)]


def test_forecastable_does_not_depend_on_the_actual():
    o = datetime(2024, 11, 1)
    cases = build_cases(dense_obs(o, hi=0), o, o)  # no data after the origin: no actuals
    assert all(c.forecastable and c.actual is None and c.r_actual == "target_no_row" for c in cases)
    assert usable(cases) == []


def test_leakage_guard_rejects_test_period_cases():
    o = datetime(2024, 11, 1)
    good = build_cases(dense_obs(datetime(2024, 10, 30)), datetime(2024, 10, 30), datetime(2024, 10, 30))
    assert_no_test_leakage(good)  # target of h24 is Oct 31 00:00: allowed
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_test_leakage(build_cases(dense_obs(o), o, o))


# ------------------------------------------------------------------ end-to-end on synthetic data
def test_periods_are_the_requested_ones():
    assert PERIODS["fit"] == (datetime(2024, 7, 9), datetime(2024, 9, 29))
    assert PERIODS["tune"] == (datetime(2024, 10, 1), datetime(2024, 10, 30))
    assert PERIODS["test"] == (datetime(2024, 11, 1), datetime(2024, 12, 31))


def test_hand_computed_counts(sim):
    s = sim.summary
    assert (s["fit_rows"], s["tune_rows"]) == (1989, 717)
    assert (s["test_scheduled"], s["test_baseline_forecasts"], s["test_baseline_scored_own_set"]) == (1464, 1461, 1458)
    assert (s["test_ridge_forecasts"], s["test_actuals_present"], s["test_common_rows"]) == (1458, 1461, 1455)
    assert s["common_rows_match_expected"] is True and s["lags_after_cutoff_used"] == 0
    assert s["not_in_common_by_reason"] == {"lag168_blank_demand": 3, "lag48_blank_demand": 3,
                                            "target_blank_demand": 3}
    assert s["baseline_not_scored_by_reason"] == {"lag168_blank_demand": 3, "target_blank_demand": 3}
    assert s["run_type"] == "retrospective_simulation"


def test_ridge_and_baseline_scored_on_exactly_the_same_rows(sim):
    common = [x for x in sim.paired if x.in_common]
    assert len(common) == 1455
    assert all(x.baseline is not None and x.ridge is not None and x.case.actual is not None for x in common)
    # forecasts exist outside the common set when only the actual is missing
    forecast_no_actual = [x for x in sim.paired if x.ridge is not None and x.case.actual is None]
    assert len(forecast_no_actual) == 3 and not any(x.in_common for x in forecast_no_actual)
    assert sim.summary["overall"]["n"] == 1455
    assert sum(v["n"] for v in sim.summary["by_horizon"].values()) == 1455


def test_metrics_recomputed_on_common_rows_and_ridge_learns_the_trend(sim):
    o = sim.summary["overall"]
    common = [x for x in sim.paired if x.in_common]
    # The week after the 2024-11-03 DST change is hard for both models: the 168 h lag source sits in the other
    # UTC offset, so its New York hour differs. From 2024-11-10 06:00Z on, the lag is again in the same regime
    # and the baseline error is exactly the 8.4 MW trend while the ridge is essentially exact.
    late = [x for x in common if x.case.slot.target > datetime(2024, 11, 10, 6)]
    assert len(late) > 1200
    assert all(math.isclose(abs(x.baseline - x.case.actual), 8.4, abs_tol=1e-6) for x in late)
    assert max(abs(x.ridge - x.case.actual) for x in late) < 1e-3
    assert o["ridge"]["mae_mw"] < o["baseline"]["mae_mw"] and o["ridge"]["rmse_mw"] < o["baseline"]["rmse_mw"]
    assert o["ridge_wins"] > o["ridge_losses"]
    manual = sum(abs(x.ridge - x.case.actual) for x in common) / len(common)
    assert math.isclose(manual, o["ridge"]["mae_mw"])
    h24 = sim.summary["by_horizon"][24]
    assert h24["n"] == sum(1 for x in common if x.case.slot.horizon == 24)


def test_frozen_before_test_rows_are_read(synth):
    class Spy(dict):
        touched = None

        def _note(self, k):
            if k >= datetime(2024, 11, 1, 1):
                self.touched.add(k)

        def get(self, k, d=None):
            self._note(k)
            return super().get(k, d)

        def __getitem__(self, k):
            self._note(k)
            return super().__getitem__(k)

        def __contains__(self, k):
            self._note(k)
            return super().__contains__(k)

    obs = read_observations(synth, "SYN")
    spy = Spy(obs.values)
    spy.touched = set()
    obs.values = spy
    at_freeze = []
    run_simulation(obs, on_freeze=lambda f: at_freeze.append(set(spy.touched)))
    assert at_freeze == [set()]          # nothing from the test period was read before the freeze
    assert spy.touched                    # and the test block was read afterwards


def test_test_period_actuals_cannot_change_the_frozen_model(synth, tmp_path):
    other = tmp_path / "other.csv"
    write_series(other, test_scale=987654321.0)  # every test-period demand value replaced by garbage
    a = run_simulation(read_observations(synth, "SYN"))
    b = run_simulation(read_observations(str(other), "SYN"))
    assert a.frozen.digest == b.frozen.digest
    assert a.frozen.model.coef == b.frozen.model.coef
    assert a.summary["overall"]["ridge"]["mae_mw"] != b.summary["overall"]["ridge"]["mae_mw"]


def test_chosen_alpha_is_the_tune_mae_minimizer(sim):
    best = min(sim.validation, key=lambda t: t[1])
    assert sim.frozen.alpha == best[0] and [a for a, _ in sim.validation] == list(ALPHAS)
    assert sim.frozen_seen == [sim.frozen]


def test_frozen_model_detects_tampering(sim):
    sim.frozen.verify()
    tampered = dataclasses.replace(sim.frozen.model, intercept=sim.frozen.model.intercept + 1.0)
    with pytest.raises(RuntimeError, match="modified after freezing"):
        FrozenModel(tampered, sim.frozen.digest, 1.0, 0, 0).verify()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sim.frozen.model.intercept = 0.0


def test_highest_mae_fully_scored_day(synth, tmp_path):
    day = datetime(2024, 12, 30)  # its targets feed no later scheduled lag, so only this day is disturbed
    bump = {day + h * HOUR: demand(day + h * HOUR) + 8000 for h in range(1, 25)}
    p = tmp_path / "bump.csv"
    write_series(p, overrides=bump)
    s = run_simulation(read_observations(str(p), "SYN")).summary
    w = s["highest_mae_day_ridge"]
    assert w["origin_utc"] == "2024-12-30T00:00:00Z" and math.isclose(w["ridge_mae"], 8000, rel_tol=1e-3)
    full_a = s["fully_scored_days"]
    # blank one target of that day: it is no longer fully scored and must not be selected
    p2 = tmp_path / "bump_blank.csv"
    write_series(p2, overrides=bump, blanks=BLANKS + [day + 12 * HOUR])
    s2 = run_simulation(read_observations(str(p2), "SYN")).summary
    assert s2["fully_scored_days"] == full_a - 1
    assert s2["highest_mae_day_ridge"]["origin_utc"] != "2024-12-30T00:00:00Z"
    assert s2["highest_mae_day_ridge"]["ridge_mae"] < 100  # the DST-week day, not the 8000 MW day


def test_fully_scored_day_count(sim):
    # 61 origins; the 9 lost rows touch origins: Nov 10 (h12), Nov 12 (h12), Nov 17 (h12), Nov 20 (h3),
    # Nov 22 (h3), Nov 27 (h3), Dec 5 (h18), Dec 7 (h18), Dec 12 (h18) -> 9 distinct origins affected
    assert sim.summary["fully_scored_days"] == 61 - 9


# ------------------------------------------------------------------ CLI and labels
def test_cli_outputs_are_labeled_and_consistent(synth, tmp_path, capsys):
    out = tmp_path / "sim"
    code = main(["--csv", synth, "--ba", "SYN", "--data-label", "synthetic", "--out-dir", str(out),
                 "--expect-common-rows", "1455"])
    text = capsys.readouterr().out
    assert code == 0
    assert text.splitlines()[0].startswith("RETROSPECTIVE SIMULATION")
    assert text.splitlines()[1].startswith("DATA: SYNTHETIC (generated test data, not real EIA data)")
    summ = json.loads((out / "summary.json").read_text())
    frozen = json.loads((out / "frozen_model.json").read_text())
    assert summ["data_label"] == frozen["data_label"] == "synthetic"
    assert summ["run_type"] == frozen["run_type"] == "retrospective_simulation"
    assert frozen["digest_sha256"] == summ["frozen_digest_sha256"]
    rows = list(csv.DictReader(open(out / "paired_predictions.csv", newline="")))
    assert len(rows) == 1464 and {r["run_type"] for r in rows} == {"retrospective_simulation"}
    assert {r["data_label"] for r in rows} == {"synthetic"}
    assert sum(r["in_common_set"] == "1" for r in rows) == 1455
    assert rows[0]["interval_start_ny"].endswith(("EDT", "EST")) and rows[0]["origin_utc"] == "2024-11-01T00:00:00Z"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "RETROSPECTIVE SIMULATION" in report and "SYNTHETIC (generated test data" in report
    for needle in ("## By horizon", "## Highest-MAE fully scored day", "| frozen ridge |", "skip reasons",
                   "lag168_blank_demand", "(chosen)"):
        assert needle.lower() in report.lower(), needle


def test_cli_exit_codes(synth, tmp_path, capsys):
    base = ["--csv", synth, "--ba", "SYN", "--data-label", "synthetic", "--out-dir", str(tmp_path / "o")]
    assert main(base + ["--expect-common-rows", "1389"]) == 1
    assert "MISMATCH" in capsys.readouterr().err
    assert main(base + ["--expect-sha256", "0" * 64]) == 2
    assert main(["--csv", synth, "--ba", "NOPE", "--data-label", "synthetic"]) == 2
    assert main(["--csv", synth, "--ba", "SYN"]) == 2  # --data-label required


# ------------------------------------------------------------------ real snapshot (opt-in)
@pytest.mark.skipif(not __import__("os").environ.get("EIA930_SNAPSHOT_CSV"),
                    reason="set EIA930_SNAPSHOT_CSV to the real snapshot to run this check")
def test_real_pjm_simulation_counts():
    s = run_simulation(read_observations(__import__("os").environ["EIA930_SNAPSHOT_CSV"], "PJM"),
                       expect_common=1389).summary
    assert (s["test_scheduled"], s["test_baseline_scored_own_set"], s["test_common_rows"]) == (1464, 1414, 1389)
    assert s["common_rows_match_expected"] is True and s["lags_after_cutoff_used"] == 0
