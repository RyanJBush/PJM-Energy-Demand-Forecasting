"""Tests use only hand-set SYNTHETIC fixtures in tests/fixtures (BA "SYN"), never real EIA data.
The single exception is test_real_pjm_schedule_counts, which runs only if EIA930_SNAPSHOT_CSV is set.

Fixture map (SYN, UTC hour-ending), forecast = demand 168 h earlier, error = forecast - actual:
  Jul 1: 06:00=100 07:00=110 08:00=blank 10:00=200 11:00=210 13:00=220 (09:00, 12:00 absent)
  Jul 2: 00:00=148
  Jul 8: 06:00=103 07:00=108 08:00=120 09:00=130 10:00=blank 12:00=140 13:00="n/a" (11:00 absent)
  Jul 9: 00:00=150
  one SYN row has an unparseable timestamp; BA OTH has two rows that must never be used.
Origin Jul 8 00:00 (cutoff Jul 7 22:00), horizons 1..24:
  h1-h5   no forecast (lag before history), no actual           -> lag_before_history_start
  h6      forecast 100, actual 103 -> scored, error -3
  h7      forecast 110, actual 108 -> scored, error +2
  h8      lag blank, actual 120                                  -> lag_blank_demand
  h9      lag absent, actual 130                                 -> no_lag_row
  h10     forecast 200, actual blank                             -> target_blank_demand
  h11     forecast 210, target row absent                        -> no_target_row
  h12     lag absent, actual 140                                 -> no_lag_row
  h13     forecast 220, actual "n/a"                             -> target_unparseable_demand
  h14-h23 lag absent, no actual                                  -> no_lag_row
  h24     target Jul 9 00:00 (belongs to THIS origin): forecast 148, actual 150 -> scored, error -2
Origin Jul 9 00:00: 24 rows, every lag absent                    -> no_lag_row
"""
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eia930.__main__ import main  # noqa: E402
from eia930.baseline import Config, build_schedule, run_baseline  # noqa: E402
from eia930.ingest import DuplicateKeyError, IngestError, Observations, read_observations  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
SYN = str(FIX / "synthetic_eia930.csv")
DUP = str(FIX / "synthetic_duplicate.csv")
DUP_OTHER = str(FIX / "synthetic_duplicate_other_ba.csv")
HOUR = timedelta(hours=1)
J8 = datetime(2024, 7, 8)
J9 = datetime(2024, 7, 9)


def cfg(**kw):
    kw.setdefault("origin_start", J8)
    kw.setdefault("origin_end", J9)
    return Config(**kw)


def run(**kw):
    return run_baseline(read_observations(SYN, "SYN"), cfg(**kw))


def row(result, target):
    return next(r for r in result.rows if r.slot.target == target)


def at(day, hour, month=7):
    return datetime(2024, month, day, hour)


def dense(origin, lo=-400, hi=30, skip=(), blank=()):
    """In-memory observations with a distinct value per hour (value = hour offset)."""
    vals = {}
    for i in range(lo, hi + 1):
        t = origin + i * HOUR
        if t in skip:
            continue
        vals[t] = (None, "blank_demand") if t in blank else (float(i), None)
    return Observations(ba="MEM", values=vals)


# ------------------------------------------------------------------ schedule
def test_schedule_nov_dec_has_61_origins_and_1464_unique_contiguous_targets():
    s = build_schedule(Config(datetime(2024, 11, 1), datetime(2024, 12, 31)))
    assert len(s) == 61 * 24 == 1464
    assert len({x.origin for x in s}) == 61
    targets = [x.target for x in s]
    assert len(set(targets)) == 1464 and targets == sorted(targets)
    assert all(b - a == HOUR for a, b in zip(targets, targets[1:]))  # no gaps, no overlaps
    assert targets[0] == datetime(2024, 11, 1, 1) and targets[-1] == datetime(2025, 1, 1, 0)


def test_horizon_24_belongs_to_the_prior_origin():
    s = build_schedule(Config(datetime(2024, 11, 30), datetime(2024, 12, 1)))
    midnight = [x for x in s if x.target == datetime(2024, 12, 1)]
    assert len(midnight) == 1
    assert (midnight[0].origin, midnight[0].horizon) == (datetime(2024, 11, 30), 24)
    first_of_dec1 = [x for x in s if x.origin == datetime(2024, 12, 1)][0]
    assert (first_of_dec1.target, first_of_dec1.horizon) == (datetime(2024, 12, 1, 1), 1)


def test_slot_fields_and_shared_cutoff():
    s = build_schedule(cfg())
    assert len(s) == 48
    o8 = [x for x in s if x.origin == J8]
    assert [x.horizon for x in o8] == list(range(1, 25))
    assert {x.cutoff for x in o8} == {datetime(2024, 7, 7, 22)}  # one cutoff shared by 24 targets
    assert [x.target for x in o8] == [J8 + h * HOUR for h in range(1, 25)]


@pytest.mark.parametrize("kw", [dict(origin_end=datetime(2024, 7, 7)), dict(lag_hours=0),
                                dict(horizons=0), dict(delay_hours=-1),
                                dict(origin_start=datetime(2024, 7, 8, 1))])
def test_invalid_schedule_rejected(kw):
    with pytest.raises(ValueError):
        build_schedule(cfg(**kw))


# ------------------------------------------------------------------ hand-computed fixture results
def test_weekly_lookup_values():
    r = run()
    a, b, c = row(r, at(8, 6)), row(r, at(8, 7)), row(r, at(9, 0))
    assert (a.status, a.source, a.forecast, a.actual, a.error) == ("scored", at(1, 6), 100, 103, -3)
    assert (b.status, b.source, b.forecast, b.actual, b.error) == ("scored", at(1, 7), 110, 108, 2)
    assert (c.status, c.source, c.forecast, c.actual, c.error) == ("scored", at(2, 0), 148, 150, -2)
    assert (c.slot.origin, c.slot.horizon) == (J8, 24)  # midnight target owned by the Jul 8 origin


@pytest.mark.parametrize("target,reason", [
    (at(8, 1), "lag_before_history_start"), (at(8, 5), "lag_before_history_start"),
    (at(8, 8), "lag_blank_demand"),
    (at(8, 9), "no_lag_row"),
    (at(8, 10), "target_blank_demand"),
    (at(8, 11), "no_target_row"),
    (at(8, 12), "no_lag_row"),
    (at(8, 13), "target_unparseable_demand"),
    (at(8, 14), "no_lag_row"), (at(8, 23), "no_lag_row"),
    (at(9, 1), "no_lag_row"), (at(10, 0), "no_lag_row"),
])
def test_every_not_scored_row_has_its_reason(target, reason):
    r = row(run(), target)
    assert (r.status, r.reason) == ("not_scored", reason)


def test_forecast_is_generated_without_an_actual():
    r = run()
    for target, fc, why in [(at(8, 10), 200, "target_blank_demand"), (at(8, 11), 210, "no_target_row"),
                            (at(8, 13), 220, "target_unparseable_demand")]:
        x = row(r, target)
        assert (x.forecast, x.actual, x.error, x.status, x.reason) == (fc, None, None, "not_scored", why)
    # and the reverse: an actual with no forecast stays visible, not scored
    y = row(r, at(8, 8))
    assert (y.forecast, y.actual, y.status) == (None, 120, "not_scored")


def test_hand_computed_summary():
    s = run().summary
    assert (s["scheduled"], s["origins"]) == (48, 2)
    assert (s["forecasts_generated"], s["forecasts_without_actual"], s["scored"], s["not_scored"]) == (6, 3, 3, 45)
    assert s["not_scored_by_reason"] == {
        "lag_before_history_start": 5, "lag_blank_demand": 1, "no_lag_row": 36,
        "target_blank_demand": 1, "no_target_row": 1, "target_unparseable_demand": 1}
    assert sum(s["not_scored_by_reason"].values()) == s["not_scored"]
    assert (s["scored_target_first_utc"], s["scored_target_last_utc"]) == (
        "2024-07-08T06:00:00Z", "2024-07-09T00:00:00Z")
    assert (s["target_first_utc"], s["target_last_utc"]) == ("2024-07-08T01:00:00Z", "2024-07-10T00:00:00Z")
    assert math.isclose(s["mae_mw"], 7 / 3) and s["bias_mw"] == -1.0
    assert math.isclose(s["rmse_mw"], math.sqrt(17 / 3))
    assert math.isclose(s["mape_pct"], (3 / 103 + 2 / 108 + 2 / 150) / 3 * 100)
    assert math.isclose(s["coverage_of_scheduled"], 3 / 48)
    assert s["lags_after_cutoff_used"] == 0


def test_every_scheduled_slot_is_reported_once():
    r = run()
    assert len(r.rows) == 48
    assert [x.slot for x in r.rows] == build_schedule(cfg())
    assert all(x.reason for x in r.rows if x.status == "not_scored")
    assert all(x.reason == "" for x in r.rows if x.status == "scored")


# ------------------------------------------------------------------ cutoff (in-memory, hand-set)
@pytest.mark.parametrize("lag,delay,blocked", [
    (168, 2, set()),                    # defaults never bind for horizons 1..24
    (26, 2, set()), (25, 2, {24}),      # source <= origin - 2h: equality allowed, one hour later blocked
    (27, 3, set()), (26, 3, {24}),
    (24, 0, set()), (23, 0, {24}),
    (24, 1, {24}),
    (22, 2, {21, 22, 23, 24}),
])
def test_lag_checked_against_shared_cutoff(lag, delay, blocked):
    origin = datetime(2024, 11, 1)
    obs = dense(origin)
    r = run_baseline(obs, Config(origin, origin, lag_hours=lag, delay_hours=delay))
    got = {x.slot.horizon for x in r.rows if x.reason == "lag_after_cutoff"}
    assert got == blocked
    for x in r.rows:
        if x.slot.horizon in blocked:
            assert x.forecast is None and x.status == "not_scored"
        else:
            assert x.forecast is not None and x.source <= x.slot.cutoff
    assert r.summary["lags_after_cutoff_used"] == 0


def test_cutoff_applies_even_when_the_blocked_lag_has_a_value():
    origin = datetime(2024, 11, 1)
    r = run_baseline(dense(origin), Config(origin, origin, lag_hours=25, delay_hours=2))
    x = next(x for x in r.rows if x.slot.horizon == 24)
    assert x.source == origin - HOUR and x.source > x.slot.cutoff
    assert (x.forecast, x.reason) == (None, "lag_after_cutoff")


def test_forecast_values_do_not_depend_on_actuals():
    origin = datetime(2024, 11, 1)
    full = run_baseline(dense(origin), Config(origin, origin))
    no_actuals = run_baseline(dense(origin, hi=0), Config(origin, origin))  # nothing after origin
    assert [x.forecast for x in full.rows] == [x.forecast for x in no_actuals.rows]
    assert all(x.forecast is not None for x in no_actuals.rows)
    assert all(x.actual is None and x.reason == "no_target_row" for x in no_actuals.rows)
    assert no_actuals.summary["scored"] == 0 and no_actuals.summary["forecasts_generated"] == 24
    blanked = run_baseline(dense(origin, blank={origin + h * HOUR for h in range(1, 25)}),
                           Config(origin, origin))
    assert [x.forecast for x in blanked.rows] == [x.forecast for x in full.rows]
    assert blanked.summary["forecasts_without_actual"] == 24


# ------------------------------------------------------------------ ingestion
def test_other_balancing_authority_is_never_used():
    obs = read_observations(SYN, "SYN")
    assert max(v for v, _ in obs.values.values() if v is not None) == 220  # OTH's 900/999 absent
    assert (obs.rows_read, obs.rows_for_ba, obs.bas_in_file) == (18, 16, 2)


def test_ingest_skipped_rows_carry_reasons_and_lines():
    got = sorted((s.line, s.reason) for s in read_observations(SYN, "SYN").skipped)
    assert got == [(4, "blank_demand"), (13, "blank_demand"), (15, "unparseable_demand"),
                   (17, "unparseable_timestamp")]


def test_duplicate_utc_keys_fail():
    with pytest.raises(DuplicateKeyError, match="duplicate"):
        read_observations(DUP, "SYN")
    with pytest.raises(DuplicateKeyError):  # a duplicate in another BA also fails the snapshot
        read_observations(DUP_OTHER, "SYN")


def test_unknown_ba_and_missing_column(tmp_path):
    with pytest.raises(IngestError, match="no usable rows"):
        read_observations(SYN, "NOPE")
    bad = tmp_path / "bad.csv"
    bad.write_text('"Balancing Authority","UTC Time at End of Hour"\n"SYN","07/01/2024 6:00:00 AM"\n')
    with pytest.raises(IngestError, match="missing required column"):
        read_observations(str(bad), "SYN")


# ------------------------------------------------------------------ CLI and labeling
BASE = ["--csv", SYN, "--ba", "SYN", "--data-label", "synthetic",
        "--origin-start", "2024-07-08", "--origin-end", "2024-07-09"]


def test_cli_exports_labeled_synthetic_outputs(tmp_path, capsys):
    assert main(BASE + ["--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("DATA: SYNTHETIC FIXTURE") and "not real EIA data" in out
    assert "scheduled: 48  forecasts generated: 6  scored: 3  not scored: 45" in out
    lines = (tmp_path / "forecasts.csv").read_text().splitlines()
    assert lines[0] == ("data_label,origin_utc,cutoff_utc,target_utc,horizon_hours,lag_source_utc,"
                        "forecast_mw,actual_mw,error_mw,status,reason")
    assert len(lines) == 49
    assert all(ln.startswith("synthetic,") for ln in lines[1:])
    assert lines[1].startswith("synthetic,2024-07-08T00:00:00Z,2024-07-07T22:00:00Z,2024-07-08T01:00:00Z,1,")
    assert any(ln.startswith("synthetic,2024-07-08T00:00:00Z,2024-07-07T22:00:00Z,2024-07-09T00:00:00Z,24,"
                             "2024-07-02T00:00:00Z,148.0,150.0,-2.0,scored,") for ln in lines)
    import json
    assert json.loads((tmp_path / "summary.json").read_text())["data_label"] == "synthetic"
    assert all(ln.split(",")[0] in ("data_label", "synthetic")
               for ln in (tmp_path / "ingest_skipped.csv").read_text().splitlines())


def test_cli_default_output_folder_carries_the_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(BASE) == 0
    assert (tmp_path / "outputs" / "synthetic" / "SYN" / "forecasts.csv").exists()
    assert not (tmp_path / "outputs" / "real").exists()


def test_cli_exit_codes(tmp_path, capsys):
    assert main(BASE[:-2] + ["--origin-start", "2024-07-20", "--origin-end", "2024-07-20",
                             "--out-dir", str(tmp_path)]) == 1  # nothing scored
    dup = ["--csv", DUP, "--ba", "SYN", "--data-label", "synthetic", "--origin-start", "2024-07-08",
           "--origin-end", "2024-07-08", "--out-dir", str(tmp_path)]
    assert main(dup) == 2 and "duplicate" in capsys.readouterr().err
    assert main(BASE[:3] + ["NOPE"] + BASE[4:] + ["--out-dir", str(tmp_path)]) == 2
    assert main(BASE + ["--out-dir", str(tmp_path), "--expect-sha256", "0" * 64]) == 2
    assert main(BASE[:-1] + ["2024-07-01", "--out-dir", str(tmp_path)]) == 2  # end before start
    assert main(["--csv", str(tmp_path / "missing.csv"), "--ba", "SYN", "--data-label", "synthetic",
                 "--origin-start", "2024-07-08", "--origin-end", "2024-07-09"]) == 2
    no_label = [a for i, a in enumerate(BASE) if i not in (4, 5)]
    assert main(no_label) == 2  # --data-label is required


# ------------------------------------------------------------------ real snapshot (opt-in)
@pytest.mark.skipif(not os.environ.get("EIA930_SNAPSHOT_CSV"),
                    reason="set EIA930_SNAPSHOT_CSV to the real snapshot to run this check")
def test_real_pjm_schedule_counts(tmp_path):
    obs = read_observations(os.environ["EIA930_SNAPSHOT_CSV"], "PJM")
    r = run_baseline(obs, Config(datetime(2024, 11, 1), datetime(2024, 12, 31)))
    s = r.summary
    assert (s["origins"], s["scheduled"], s["scored"]) == (61, 1464, 1414)
    assert s["lags_after_cutoff_used"] == 0
