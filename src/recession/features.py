"""Leakage-safe feature engineering.

Every series is represented as a frame with columns:

* ``period`` — the calendar period the value describes (FRED convention),
* ``value`` — the observation (first-release value in vintage mode),
* ``avail`` — the first calendar date on which a forecaster could know it.

Transforms operate on the period axis using **trailing windows only**. The
availability of a transformed observation is the running maximum of the
availability of every input observation used, so a feature can never be seen
before its latest input was published.

Features are then materialized onto a calendar grid with a backward as-of
join on availability.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Periods per month by native frequency.
_PPM = {"d": 21, "w": 13 / 3, "m": 1, "q": 1 / 3}


def periods_for_months(frequency: str, months: float) -> int:
    return max(1, int(round(_PPM[frequency] * months)))


def _rolling(v: pd.Series, n: int, fn: str, min_frac: float = 0.6) -> pd.Series:
    minp = max(2, int(n * min_frac))
    r = v.rolling(n, min_periods=minp)
    return getattr(r, fn)()


def _since_last_true(mask: pd.Series, cap: int) -> pd.Series:
    idx = np.arange(len(mask), dtype=float)
    last_true = pd.Series(np.where(mask.to_numpy(), idx, np.nan)).ffill().to_numpy()
    dist = idx - last_true
    dist[np.isnan(dist)] = cap
    return pd.Series(np.minimum(dist, cap), index=mask.index)


def transform_values(v: pd.Series, code: str, frequency: str) -> pd.Series:
    """Apply one named transform to a value series (period-ordered)."""
    f = frequency

    if code == "level":
        return v

    m = re.fullmatch(r"chg(\d+)m", code)
    if m:
        return v - v.shift(periods_for_months(f, int(m.group(1))))

    m = re.fullmatch(r"(?:pct|ret)(\d+)m", code)
    if m:
        return v / v.shift(periods_for_months(f, int(m.group(1)))) - 1.0

    if code == "yoy":
        return v / v.shift(periods_for_months(f, 12)) - 1.0

    m = re.fullmatch(r"ann(\d+)m", code)
    if m:
        p = int(m.group(1))
        n = periods_for_months(f, p)
        ratio = v / v.shift(n)
        return np.sign(ratio) * np.abs(ratio) ** (12.0 / p) - 1.0

    if code == "accel3m":
        n = periods_for_months(f, 3)
        ann3 = (v / v.shift(n)) ** (12.0 / 3) - 1.0
        return ann3 - ann3.shift(n)

    m = re.fullmatch(r"z(\d+)y", code)
    if m:
        n = periods_for_months(f, 12 * int(m.group(1)))
        mu = _rolling(v, n, "mean")
        sd = _rolling(v, n, "std")
        return (v - mu) / sd.replace(0.0, np.nan)

    m = re.fullmatch(r"pctile(\d+)y", code)
    if m:
        n = periods_for_months(f, 12 * int(m.group(1)))
        minp = max(2, int(n * 0.6))
        return v.rolling(n, min_periods=minp).rank(pct=True)

    m = re.fullmatch(r"dmax(\d+)y", code)
    if m:
        n = periods_for_months(f, 12 * int(m.group(1)))
        return v / _rolling(v, n, "max") - 1.0

    m = re.fullmatch(r"avg(\d+)m", code)
    if m:
        return _rolling(v, periods_for_months(f, int(m.group(1))), "mean")

    if code == "vol3m":  # realized vol of daily log returns, annualized
        n = periods_for_months(f, 3)
        lr = np.log(v / v.shift(1))
        return _rolling(lr, n, "std") * np.sqrt(252 if f == "d" else 52)

    if code == "dd1y":  # drawdown from trailing 1-year peak
        n = periods_for_months(f, 12)
        return v / _rolling(v, n, "max") - 1.0

    if code == "dma200":  # distance to 200-day moving average
        n = 200 if f == "d" else periods_for_months(f, 9)
        return v / _rolling(v, n, "mean") - 1.0

    if code == "inv_share1y":  # share of trailing year spent inverted
        n = periods_for_months(f, 12)
        return _rolling((v < 0).astype(float), n, "mean")

    if code == "resteep":  # steepening off the trailing 6-month low, if inverted
        n = periods_for_months(f, 6)
        lo = _rolling(v, n, "min", min_frac=0.02)  # a trailing low needs little warm-up
        return pd.Series(np.where(lo < 0, v - lo, 0.0), index=v.index)

    if code == "since_inv":  # years since the curve was last inverted (capped at 3)
        cap = periods_for_months(f, 36)
        return _since_last_true(v < 0, cap) / periods_for_months(f, 12)

    if code == "infl_accel":  # 3m annualized inflation minus yoy inflation
        n3 = periods_for_months(f, 3)
        n12 = periods_for_months(f, 12)
        ann3 = (v / v.shift(n3)) ** 4.0 - 1.0
        yoy = v / v.shift(n12) - 1.0
        return ann3 - yoy

    raise ValueError(f"Unknown transform code: {code}")


def transform_frame(df: pd.DataFrame, code: str, frequency: str) -> pd.DataFrame:
    """Apply a transform to a period/value/avail frame, propagating availability."""
    df = df.sort_values("period").reset_index(drop=True)
    out = pd.DataFrame({
        "period": df["period"],
        "value": transform_values(df["value"].astype(float), code, frequency).to_numpy(),
        # A trailing-window statistic is knowable once its most recent input is
        # published; cummax guards against any non-monotone release dates.
        "avail": df["avail"].cummax(),
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


def as_of_series(df: pd.DataFrame, grid: pd.DatetimeIndex, value_col: str = "value") -> pd.Series:
    """Backward as-of join: for each grid date, the last value whose
    availability date is <= that grid date."""
    if df.empty:
        return pd.Series(np.nan, index=grid)
    d = df.sort_values(["avail", "period"]).reset_index(drop=True)
    merged = pd.merge_asof(
        pd.DataFrame({"t": grid}),
        d.rename(columns={"avail": "t", value_col: "_v"})[["t", "_v"]],
        on="t",
        direction="backward",
        allow_exact_matches=True,
    )
    return pd.Series(merged["_v"].to_numpy(), index=grid)


def build_feature_panel(
    feature_frames: dict[str, pd.DataFrame], grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """Materialize all feature frames onto a common calendar grid."""
    cols = {name: as_of_series(df, grid) for name, df in feature_frames.items()}
    return pd.DataFrame(cols, index=grid)
