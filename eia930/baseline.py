"""168-hour seasonal-naive baseline scored on a fixed schedule of daily forecast origins.

Schedule: one origin per day at 00:00 UTC; each origin owns 24 targets at horizons 1..24 hours
(origin + 1h ... origin + 24h). Horizon 24 lands on the following midnight and belongs to the origin
that issued it, so no target is scheduled twice. The schedule is built first, then forecasts and
actuals are joined to it, so missing rows and blank values stay visible with a reason.

Forecast generation never looks at the actual: a forecast exists whenever its lag source passes the
cutoff and has a value. A row is scored only if both a forecast and an actual exist.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from .ingest import Observations

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


@dataclass(frozen=True)
class Config:
    origin_start: datetime          # first origin (00:00 UTC), inclusive
    origin_end: datetime            # last origin (00:00 UTC), inclusive
    lag_hours: int = 168            # forecast for target t is the demand observed at t - lag_hours
    delay_hours: int = 2            # ASSUMED publication delay; shared cutoff = origin - delay
    horizons: int = 24              # targets per origin (horizon_hours 1..horizons)


@dataclass(frozen=True)
class Slot:
    origin: datetime
    cutoff: datetime                # latest hour-ending label usable at this origin (shared by its targets)
    target: datetime
    horizon: int


@dataclass(frozen=True)
class Row:
    slot: Slot
    source: datetime                # target - lag_hours
    forecast: float | None
    actual: float | None
    error: float | None             # forecast - actual
    status: str                     # "scored" | "not_scored"
    reason: str                     # "" when scored


@dataclass
class Result:
    config: Config
    ba: str
    rows: list
    summary: dict


def build_schedule(cfg: Config) -> list:
    if cfg.origin_end < cfg.origin_start:
        raise ValueError("origin_end is before origin_start")
    if cfg.lag_hours <= 0 or cfg.delay_hours < 0 or cfg.horizons <= 0:
        raise ValueError("lag_hours and horizons must be > 0; delay_hours must be >= 0")
    for d in (cfg.origin_start, cfg.origin_end):
        if (d.hour, d.minute, d.second, d.microsecond) != (0, 0, 0, 0):
            raise ValueError("origins must be at 00:00 UTC")
    slots = []
    origin = cfg.origin_start
    while origin <= cfg.origin_end:
        cutoff = origin - timedelta(hours=cfg.delay_hours)
        for h in range(1, cfg.horizons + 1):
            slots.append(Slot(origin, cutoff, origin + h * HOUR, h))
        origin += DAY
    return slots


def run_baseline(obs: Observations, cfg: Config) -> Result:
    schedule = build_schedule(cfg)  # built BEFORE joining observations
    first_obs = min(obs.values)
    lag = timedelta(hours=cfg.lag_hours)
    rows = []
    for slot in schedule:
        src = slot.target - lag
        # ---- forecast side: uses only the lag source and the shared cutoff, never the actual
        forecast, f_reason = None, ""
        if src > slot.cutoff:
            f_reason = "lag_after_cutoff"
        elif src not in obs.values:
            f_reason = "lag_before_history_start" if src < first_obs else "no_lag_row"
        elif obs.values[src][0] is None:
            f_reason = f"lag_{obs.values[src][1]}"
        else:
            forecast = obs.values[src][0]
        # ---- actual side
        actual, a_reason = None, ""
        tv = obs.values.get(slot.target)
        if tv is None:
            a_reason = "no_target_row"
        elif tv[0] is None:
            a_reason = f"target_{tv[1]}"
        else:
            actual = tv[0]
        scored = forecast is not None and actual is not None
        rows.append(Row(slot, src, forecast, actual,
                        forecast - actual if scored else None,
                        "scored" if scored else "not_scored",
                        "" if scored else (f_reason or a_reason)))
    return Result(cfg, obs.ba, rows, summarize(rows, obs, cfg))


def _iso(t: datetime | None) -> str | None:
    return f"{t:%Y-%m-%dT%H:%M:%SZ}" if t else None


def summarize(rows: list, obs: Observations, cfg: Config) -> dict:
    scored = [r for r in rows if r.status == "scored"]
    forecasts = [r for r in rows if r.forecast is not None]
    reasons = Counter(r.reason for r in rows if r.status == "not_scored")
    errs = [r.error for r in scored]
    pct = [abs(r.error) / r.actual * 100 for r in scored if r.actual > 0]
    origins = sorted({r.slot.origin for r in rows})
    n = len(rows)
    return {
        "ba": obs.ba,
        "snapshot_sha256": obs.sha256,
        "config": {"lag_hours": cfg.lag_hours, "delay_hours": cfg.delay_hours,
                   "horizons": cfg.horizons},
        "origins": len(origins),
        "origin_first_utc": _iso(origins[0]),
        "origin_last_utc": _iso(origins[-1]),
        "scheduled": n,
        "target_first_utc": _iso(min(r.slot.target for r in rows)),
        "target_last_utc": _iso(max(r.slot.target for r in rows)),
        "csv_rows_read": obs.rows_read,
        "csv_rows_for_ba": obs.rows_for_ba,
        "ingest_skipped_rows": len(obs.skipped),
        "forecasts_generated": len(forecasts),
        "forecasts_without_actual": len(forecasts) - len(scored),
        "lags_after_cutoff_used": sum(1 for r in forecasts if r.source > r.slot.cutoff),
        "scored": len(scored),
        "not_scored": n - len(scored),
        "not_scored_by_reason": dict(sorted(reasons.items())),
        "coverage_of_scheduled": len(scored) / n if n else 0.0,
        "scored_target_first_utc": _iso(min((r.slot.target for r in scored), default=None)),
        "scored_target_last_utc": _iso(max((r.slot.target for r in scored), default=None)),
        "mae_mw": sum(abs(e) for e in errs) / len(errs) if errs else None,
        "rmse_mw": math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None,
        "bias_mw": sum(errs) / len(errs) if errs else None,
        "mape_pct": sum(pct) / len(pct) if pct else None,
        "mape_n": len(pct),
    }
