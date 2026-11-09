"""EIA-930 balance CSV ingestion (standard library only).

Reads one balancing authority's `Demand (MW)` keyed by the UTC end-of-hour timestamp.
Nothing is dropped silently: rows that cannot be used are recorded with a reason, and blank or
unparseable demand values are kept as `None` plus a reason so the grid join can still show them.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime

BA_COL = "Balancing Authority"
TS_COL = "UTC Time at End of Hour"
DEMAND_COL = "Demand (MW)"
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # e.g. 07/01/2024 6:00:00 AM (UTC, hour-ending)
REQUIRED_COLUMNS = (BA_COL, TS_COL, DEMAND_COL)


class IngestError(Exception):
    """The file cannot be used as-is (missing columns, no rows for the BA)."""


class DuplicateKeyError(IngestError):
    """The same (balancing authority, UTC timestamp) key appears more than once."""


@dataclass(frozen=True)
class SkippedRow:
    line: int
    ba: str
    raw_timestamp: str
    reason: str


@dataclass
class Observations:
    ba: str
    # timestamp -> (demand or None, reason or None). Key present means the row exists in the file.
    values: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)  # SkippedRow, for rows that could not be keyed/used
    rows_read: int = 0
    rows_for_ba: int = 0
    bas_in_file: int = 0
    sha256: str = ""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamp(text: str) -> datetime | None:
    try:
        return datetime.strptime(text.strip(), TS_FORMAT)
    except (ValueError, AttributeError):
        return None


def parse_demand(text: str) -> tuple[float | None, str | None]:
    """(value, None) on success; (None, reason) for blank or unusable values."""
    s = (text or "").strip()
    if s == "":
        return None, "blank_demand"
    try:
        v = float(s)
    except ValueError:
        return None, "unparseable_demand"
    if not math.isfinite(v):
        return None, "unparseable_demand"
    return v, None


def read_observations(path: str, ba: str) -> Observations:
    """Scan the whole file: fail on any duplicate (BA, UTC) key, collect rows for `ba`."""
    obs = Observations(ba=ba, sha256=sha256_file(path))
    seen: dict = {}
    duplicates: list = []
    bas: set = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise IngestError(f"missing required column(s): {missing}")
        for row in reader:
            obs.rows_read += 1
            line = reader.line_num
            row_ba = row[BA_COL]
            bas.add(row_ba)
            ts = parse_timestamp(row[TS_COL])
            if ts is None:
                if row_ba == ba:
                    obs.rows_for_ba += 1
                    obs.skipped.append(SkippedRow(line, row_ba, row[TS_COL], "unparseable_timestamp"))
                continue
            key = (row_ba, ts)
            if key in seen:
                duplicates.append((row_ba, ts, seen[key], line))
                continue
            seen[key] = line
            if row_ba != ba:
                continue
            obs.rows_for_ba += 1
            value, reason = parse_demand(row[DEMAND_COL])
            obs.values[ts] = (value, reason)
            if reason:
                obs.skipped.append(SkippedRow(line, row_ba, row[TS_COL], reason))
    if duplicates:
        shown = "; ".join(f"{b} {t:%Y-%m-%d %H:%M} lines {a},{c}" for b, t, a, c in duplicates[:5])
        raise DuplicateKeyError(f"{len(duplicates)} duplicate (BA, UTC) key(s): {shown}")
    obs.bas_in_file = len(bas)
    if not obs.values:
        raise IngestError(f"no usable rows for balancing authority {ba!r} "
                          f"(file has {len(bas)} BAs, e.g. {sorted(bas)[:5]})")
    return obs
