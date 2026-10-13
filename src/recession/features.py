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

    m = re.fullmatch(r"dpeak(\d+)y", code)
    if m:  # level drop from the trailing peak (for rates, where ratios misbehave near zero)
        n = periods_for_months(f, 12 * int(m.group(1)))
        return v - _rolling(v, n, "max")

    m = re.fullmatch(r"lag(\d+)m", code)
    if m:  # the value N months ago (distributed-lag predictors, e.g. the curve 6m back)
        return v.shift(periods_for_months(f, int(m.group(1))))

    m = re.fullmatch(r"min(\d+)y", code)
    if m:  # trailing N-year minimum
        return _rolling(v, periods_for_months(f, 12 * int(m.group(1))), "min")

    m = re.fullmatch(r"rise(\d+)y", code)
    if m:  # rise from the trailing N-year low (cumulative tightening)
        return v - _rolling(v, periods_for_months(f, 12 * int(m.group(1))), "min")

    m = re.fullmatch(r"invdepth(\d+)y", code)
    if m:  # average inversion depth over N years: mean of max(-v, 0)
        return _rolling(np.maximum(-v, 0.0), periods_for_months(f, 12 * int(m.group(1))), "mean")

    m = re.fullmatch(r"nopi(\d+)y", code)
    if m:  # Hamilton net oil price increase: log excess over the prior N-year peak
        n = periods_for_months(f, 12 * int(m.group(1)))
        prev_peak = _rolling(v.shift(1), n, "max")
        return np.maximum(np.log(v) - np.log(prev_peak), 0.0)

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

    m = re.fullmatch(r"hpgap(?:(\d+)k)?", code)
    if m:  # one-sided HP gap: hpgap (standard lambda) or hpgap400k (BIS credit gap)
        lam = int(m.group(1)) * 1000 if m.group(1) else _HP_LAMBDA[f]
        return one_sided_hp_gap(v, lam, min_periods=periods_for_months(f, 120))

    raise ValueError(f"Unknown transform code: {code}")


_HP_LAMBDA = {"q": 1600, "m": 129_600, "w": 1600 * 13 ** 4, "d": 1600 * 63 ** 4}


def one_sided_hp_gap(v: pd.Series, lam: float, min_periods: int) -> pd.Series:
    """Deviation from a Hodrick-Prescott trend fitted on the data up to each
    date only (the 'real-time' gap of the credit-gap literature). The usual
    two-sided filter would let the trend at t depend on t+1..T."""
    from statsmodels.tsa.filters.hp_filter import hpfilter

    x = v.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        seg = x[: t + 1]
        if np.isnan(seg[-1]) or np.sum(~np.isnan(seg)) < min_periods:
            continue
        s = pd.Series(seg).interpolate(limit_direction="both")  # gaps inside the window only
        cycle, _trend = hpfilter(s.to_numpy(), lamb=lam)
        out[t] = cycle[-1]
    return pd.Series(out, index=v.index)


_PERIOD_FREQ = {"m": "MS", "q": "QS"}
_WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def regularize_frame(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Reindex a period/value/avail frame to its complete calendar grid.

    Transforms shift by rows, so a row must be a calendar period: a missing
    month (a shutdown, a quarterly stretch of a monthly series, a sparse
    series) otherwise silently turns a 3-month change into a 4-month one.
    Missing periods become NaN rows, which the transforms cannot see through
    and which are dropped again afterwards. Business-daily series are left
    as they are (their row logic is trading days by design)."""
    if df.empty or frequency == "d":
        return df
    df = df.sort_values("period")
    if frequency in ("m", "q"):
        key = df["period"].dt.to_period("M" if frequency == "m" else "Q").dt.to_timestamp()
        df = df.assign(period=key).drop_duplicates("period", keep="last")
        grid = pd.date_range(key.min(), key.max(), freq=_PERIOD_FREQ[frequency])
    else:  # weekly: anchor on the series' own weekday
        dow = int(df["period"].dt.dayofweek.mode().iloc[0])
        on_grid = df["period"].dt.dayofweek == dow
        if on_grid.mean() < 0.98:  # irregular stamps: not a weekly grid
            return df.reset_index(drop=True)
        df = df[on_grid].drop_duplicates("period", keep="last")
        grid = pd.date_range(df["period"].min(), df["period"].max(), freq=f"W-{_WEEKDAYS[dow]}")
    out = df.set_index("period").reindex(grid)
    out.index.name = "period"
    return out.reset_index()


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
    return out.dropna(subset=["value", "avail"]).reset_index(drop=True)


def as_of_series(
    df: pd.DataFrame, grid: pd.DatetimeIndex, value_col: str = "value",
    max_age_days: int | None = None,
) -> pd.Series:
    """Backward as-of join: for each grid date, the last value whose
    availability date is <= that grid date. With `max_age_days`, a value
    whose period started more than that many days before the grid date is
    treated as missing (a dead or paused series degrades to NaN instead of
    being carried forward forever)."""
    if df.empty:
        return pd.Series(np.nan, index=grid)
    d = df.sort_values(["avail", "period"]).reset_index(drop=True)
    merged = pd.merge_asof(
        pd.DataFrame({"t": grid}),
        d.rename(columns={"avail": "t", value_col: "_v", "period": "_p"})[["t", "_v", "_p"]],
        on="t",
        direction="backward",
        allow_exact_matches=True,
    )
    v = merged["_v"].to_numpy(dtype=float)
    if max_age_days is not None:
        age = (merged["t"] - merged["_p"]).dt.days.to_numpy()
        v = np.where(np.isnan(age) | (age > max_age_days), np.nan, v)
    return pd.Series(v, index=grid)


def build_feature_panel(
    feature_frames: dict[str, pd.DataFrame], grid: pd.DatetimeIndex,
    max_age_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Materialize all feature frames onto a common calendar grid."""
    max_age_days = max_age_days or {}
    cols = {name: as_of_series(df, grid, max_age_days=max_age_days.get(name))
            for name, df in feature_frames.items()}
    return pd.DataFrame(cols, index=grid)
