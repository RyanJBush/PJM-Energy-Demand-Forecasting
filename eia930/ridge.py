"""One ridge regression model, standard library only.

Features per forecast row: demand 48 h and 168 h before the target, the forecast horizon, and
hour-of-day / day-of-week of the TARGET INTERVAL'S START in America/New_York.

Timestamps in the EIA-930 file are UTC hour-ending labels. They are naive datetimes in this code and
are explicitly stamped as UTC before conversion to Eastern time (see `interval_start_ny`).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
NUMERIC = ("lag48_mw", "lag168_mw", "horizon_hours")
FEATURES = (NUMERIC
            + tuple(f"hour_ny_{h:02d}" for h in range(24))
            + tuple(f"dow_ny_{d}" for d in range(7)))  # Monday = 0
N_NUM = len(NUMERIC)


def interval_start_ny(target_label_utc: datetime) -> datetime:
    """Start of the target hour in America/New_York.

    `target_label_utc` is the naive UTC hour-ENDING label from the CSV. The interval it ends starts
    one hour earlier. The value is stamped as UTC first, then converted; it is never read as local time.
    """
    if target_label_utc.tzinfo is not None:
        raise ValueError("expected a naive datetime holding a UTC value")
    start_utc = (target_label_utc - timedelta(hours=1)).replace(tzinfo=timezone.utc)
    return start_utc.astimezone(NY)


def raw_features(lag48: float, lag168: float, horizon: int, target_label_utc: datetime) -> list:
    s = interval_start_ny(target_label_utc)
    return ([float(lag48), float(lag168), float(horizon)]
            + [1.0 if s.hour == h else 0.0 for h in range(24)]
            + [1.0 if s.weekday() == d else 0.0 for d in range(7)])


# ------------------------------------------------------------------ linear algebra
def cholesky_solve(a: list, b: list) -> list:
    """Solve a x = b for symmetric positive-definite `a` (Cholesky; raises if not SPD)."""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                l[i][j] = math.sqrt(s)
            else:
                l[i][j] = s / l[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(l[i][k] * y[k] for k in range(i))) / l[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(l[k][i] * x[k] for k in range(i + 1, n))) / l[i][i]
    return x


# ------------------------------------------------------------------ model
@dataclass(frozen=True)
class RidgeModel:
    alpha: float
    feature_names: tuple
    num_means: tuple      # training means of the numeric features
    num_sds: tuple        # training standard deviations (1.0 if constant)
    coef: tuple           # on the standardized feature vector
    intercept: float

    def predict(self, raw: list) -> float:
        z = list(raw)
        for i in range(N_NUM):
            z[i] = (raw[i] - self.num_means[i]) / self.num_sds[i]
        return self.intercept + sum(c * v for c, v in zip(self.coef, z))

    def params(self) -> dict:
        return {"alpha": self.alpha, "feature_names": list(self.feature_names),
                "num_means": list(self.num_means), "num_sds": list(self.num_sds),
                "coef": list(self.coef), "intercept": self.intercept}

    def digest(self) -> str:
        blob = json.dumps(self.params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


class Prepared:
    """Standardized training matrix with its Gram matrix, reusable across alphas.

    Objective: sum (y - b0 - x.b)^2 + alpha * |b|^2 on standardized numeric features and raw
    one-hot columns; the intercept is not penalized.
    """

    def __init__(self, rows: list, y: list):
        if not rows or len(rows) != len(y):
            raise ValueError("need matching non-empty rows and targets")
        n, p = len(rows), len(FEATURES)
        self.n = n
        self.num_means = tuple(sum(r[i] for r in rows) / n for i in range(N_NUM))
        sds = []
        for i in range(N_NUM):
            var = sum((r[i] - self.num_means[i]) ** 2 for r in rows) / n
            sds.append(math.sqrt(var) if var > 0 else 1.0)
        self.num_sds = tuple(sds)
        z = []
        for r in rows:
            zr = list(r)
            for i in range(N_NUM):
                zr[i] = (r[i] - self.num_means[i]) / self.num_sds[i]
            z.append(zr)
        self.col_means = [sum(zr[j] for zr in z) / n for j in range(p)]
        self.y_mean = sum(y) / n
        yc = [v - self.y_mean for v in y]
        gram = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for zr, yv in zip(z, yc):
            nz = [(j, v) for j, v in enumerate(zr) if v != 0.0]
            for j, vj in nz:
                xty[j] += vj * yv
                gj = gram[j]
                for k, vk in nz:
                    if k >= j:
                        gj[k] += vj * vk
        for j in range(p):  # centre: G_c = Z'Z - n m m'; Z_c'yc = Z'yc because sum(yc) = 0
            for k in range(j, p):
                gram[j][k] -= n * self.col_means[j] * self.col_means[k]
                gram[k][j] = gram[j][k]
        self.gram, self.xty = gram, xty

    def fit(self, alpha: float) -> RidgeModel:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        p = len(FEATURES)
        a = [row[:] for row in self.gram]
        for j in range(p):
            a[j][j] += alpha
        beta = cholesky_solve(a, self.xty)
        intercept = self.y_mean - sum(b * m for b, m in zip(beta, self.col_means))
        return RidgeModel(alpha, FEATURES, self.num_means, self.num_sds, tuple(beta), intercept)
