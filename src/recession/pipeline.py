"""Data pipeline: fetch, derive, and assemble point-in-time feature panels.

Two panels are produced from the same code path:

* ``train`` panel — uses ALFRED first-release values where available, so each
  historical row only contains numbers a forecaster could actually have seen
  on that date (both timing and, where vintages exist, magnitude).
* ``now`` panel — uses latest-vintage values, which is the point-in-time
  correct choice for *today's* prediction (today's forecaster knows current
  revised data).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import FREQ_DAYS, Indicator, IndicatorConfig, load_config
from .features import build_feature_panel, periods_for_months, regularize_frame, transform_frame
from .fred_client import SeriesResult, fetch_series, fetch_url_series, period_end
from .paths import get_logger

log = get_logger(__name__)

GRID_START = "1962-01-01"
SPLICE_OVERLAP_OBS = 4  # observations used to level a splice at the switch
# A feature whose latest period started more than this long ago is missing,
# not "unchanged": k periods of the native frequency plus the publication lag.
STALE_PERIODS = 5
STALE_GRACE_DAYS = 14


def max_feature_age_days(ind: Indicator) -> int:
    return STALE_PERIODS * FREQ_DAYS[ind.frequency] + max(ind.publication_lag_days, 0) + STALE_GRACE_DAYS


# ------------------------------------------------------------------ fetching

def fetch_all(
    cfg: IndicatorConfig | None = None,
    ttl_hours: float = 12.0,
    offline: bool = False,
) -> dict[str, dict[str, SeriesResult]]:
    """Fetch every downloaded series (FRED and URL indicators + label)."""
    cfg = cfg or load_config()
    fetched: dict[str, dict[str, SeriesResult]] = {}
    for ind in cfg.indicators:
        if not ind.fetched or not ind.enabled:
            continue
        if ind.source == "url":
            fetched[ind.id] = fetch_url_series(
                ind.id, ind.url, ind.frequency, ind.publication_lag_days,
                date_col=ind.url_date_col, value_col=ind.url_value_col, sheet=ind.url_sheet,
                ttl_hours=ttl_hours, offline=offline,
            )
            continue
        fetched[ind.id] = fetch_series(
            ind.id, ind.frequency, ind.publication_lag_days,
            want_vintage=ind.vintage, ttl_hours=ttl_hours, offline=offline,
        )
    # Ground-truth label chronology (monthly NBER indicator, latest values).
    fetched[cfg.label_series] = fetch_series(
        cfg.label_series, "m", 0, want_vintage=False, ttl_hours=ttl_hours, offline=offline,
    )
    return fetched


def _series_ok(ind: Indicator, res: dict[str, SeriesResult]) -> bool:
    latest = res.get("latest")
    if latest is None or not latest.ok:
        return False
    if ind.min_history_start is not None:
        first = latest.df["period"].min()
        if pd.isna(first) or first > pd.Timestamp(ind.min_history_start):
            log.warning(
                "%s disabled: history starts %s, later than required %s "
                "(likely license-limited without an API key); proxies carry the signal.",
                ind.id, first, ind.min_history_start,
            )
            return False
    return True


# ------------------------------------------------------------ derived series

def _align_inputs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Align parent series onto the first parent's period axis with backward
    as-of joins; combined availability is the max across inputs used."""
    base = frames[0].sort_values("period").reset_index(drop=True)
    out = base.rename(columns={"value": "v0", "avail": "a0"})
    for i, f in enumerate(frames[1:], start=1):
        f = f.sort_values("period").reset_index(drop=True)
        out = pd.merge_asof(
            out,
            f.rename(columns={"value": f"v{i}", "avail": f"a{i}"}),
            on="period", direction="backward",
        )
    avail = out[[c for c in out.columns if c.startswith("a")]].max(axis=1)
    vals = out[[c for c in out.columns if c.startswith("v")]]
    res = pd.DataFrame({"period": out["period"], "avail": avail})
    for c in vals.columns:
        res[c] = vals[c]
    return res.dropna(subset=["v0"]).reset_index(drop=True)


def _align_union(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join parents on period (same frequency); for a splice the
    successor's periods must survive after the predecessor ends."""
    out = None
    for i, f in enumerate(frames):
        g = (f.sort_values("period")
             .rename(columns={"value": f"v{i}", "avail": f"a{i}"})[["period", f"v{i}", f"a{i}"]])
        out = g if out is None else pd.merge(out, g, on="period", how="outer")
    out = out.sort_values("period").reset_index(drop=True)
    res = pd.DataFrame({"period": out["period"],
                        "avail": out[[c for c in out.columns if c.startswith("a")]].max(axis=1)})
    for c in [c for c in out.columns if c.startswith("v")]:
        res[c] = out[c]
    return res


def derive_series(ind: Indicator, parents: list[pd.DataFrame]) -> pd.DataFrame:
    a = _align_union(parents) if ind.derive == "splice" else _align_inputs(parents)
    if ind.derive == "spread":
        value = a["v0"] - a["v1"]
    elif ind.derive == "curvature":  # mid yield minus average of wings
        value = a["v1"] - (a["v0"] + a["v2"]) / 2.0
    elif ind.derive == "real_rate":  # policy rate minus trailing core inflation
        n = periods_for_months(ind.frequency, 12)
        infl = (a["v1"] / a["v1"].shift(n) - 1.0) * 100.0
        value = a["v0"] - infl
    elif ind.derive == "deflate":  # nominal series in real terms
        value = a["v0"] / a["v1"] * 100.0
    elif ind.derive == "avg4w":
        value = a["v0"].rolling(4, min_periods=2).mean()
    elif ind.derive == "sahm":  # 3m avg unemployment minus its 12m low
        n3, n12 = periods_for_months(ind.frequency, 3), periods_for_months(ind.frequency, 12)
        u3 = a["v0"].rolling(n3, min_periods=n3).mean()
        value = u3 - u3.rolling(n12, min_periods=n12 // 2).min()
    elif ind.derive == "ratio":  # v0 / v1 (optionally scaled)
        value = a["v0"] / a["v1"].replace(0.0, np.nan) * float(ind.derive_params.get("scale", 1.0))
    elif ind.derive == "flow_rate":  # 100 * flow(t) / stock(t-1), e.g. E->U flows over employment
        value = 100.0 * a["v0"] / a["v1"].shift(1).replace(0.0, np.nan)
    elif ind.derive == "growth_gap":  # yoy growth of v0 minus yoy growth of v1, in points
        n = periods_for_months(ind.frequency, 12)
        value = 100.0 * ((a["v0"] / a["v0"].shift(n)) - (a["v1"] / a["v1"].shift(n)))
    elif ind.derive == "wsum":  # fixed-weight sum of the inputs (e.g. ISM weights on survey components)
        weights = [float(w) for w in ind.derive_params["weights"]]
        value = sum(w * a[f"v{i}"] for i, w in enumerate(weights))
    elif ind.derive == "ntfs":
        # Near-term forward spread proxy (Engstrom-Sharpe): the 3-month rate
        # implied ~18 months ahead minus the spot bill, from the 2y/1y/3m
        # points: 2.25*DGS2 - 1.25*DGS1 - DTB3 (inputs in that order)
        value = 2.25 * a["v0"] - 1.25 * a["v1"] - a["v2"]
    elif ind.derive == "taylor_gap":
        # Policy rate minus a Taylor (1993) rule rate, inputs [FEDFUNDS, core CPI,
        # UNRATE]. The unemployment gap uses the trailing 10-year mean rate as
        # u*: a real-time stand-in for the CBO natural rate, whose history is
        # re-estimated with hindsight. Positive = policy tighter than the rule.
        n12 = periods_for_months(ind.frequency, 12)
        n120 = periods_for_months(ind.frequency, 120)
        pi = 100.0 * (a["v1"] / a["v1"].shift(n12) - 1.0)
        ugap = a["v2"] - a["v2"].rolling(n120, min_periods=n120 // 2).mean()
        rule = 2.0 + pi + 0.5 * (pi - 2.0) - 1.0 * ugap
        value = a["v0"] - rule
    elif ind.derive == "splice":  # v0 before switch_date, v1 from it; optional level offset
        switch = pd.Timestamp(ind.derive_params["switch_date"])
        v0, v1 = a["v0"], a["v1"]
        offset = ind.derive_params.get("offset", 0.0)
        if offset == "auto":
            # level difference on the first overlapping observations only:
            # the rows before the overlap receive a shift that was knowable
            # at the start of the overlap, not one averaged over its future
            both = (v0.notna() & v1.notna()).to_numpy()
            k = min(int(both.sum()), SPLICE_OVERLAP_OBS)
            offset = float((v1 - v0)[both].iloc[:k].mean()) if k else 0.0
        value = pd.Series(np.where(a["period"] < switch, v0 + float(offset), v1), index=a.index)
    else:
        raise ValueError(f"Unknown derive recipe {ind.derive!r} for {ind.id}")
    out = pd.DataFrame({"period": a["period"], "value": value, "avail": a["avail"].cummax()})
    return out.dropna(subset=["value"]).reset_index(drop=True)


CHAIN_LINK_THRESHOLD = 0.10  # log change in first/latest ratio that marks a rebasing
CHAIN_LINK_WINDOW = 6        # periods of ratio on each side used to confirm it


def chain_link_first_release(first: pd.DataFrame, latest: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Put first-release levels on one base across index/unit changes.

    ALFRED initial releases of an index series sit on whichever base year the
    agency used at the time (industrial production: 1987=100, then 1992=100,
    ...), so growth rates computed across a rebasing are nonsense (-66%
    annualised in 1990). A rebasing shows up as a persistent shift in the
    ratio first/latest, which is otherwise close to constant; it is detected
    on medians of the ratio before and after each date (so one revised
    observation is not a break) and every earlier value is scaled onto the
    later base. The latest vintage is used only to locate and size the level
    shift, never to supply values."""
    if first.empty or latest.empty or (first["value"] <= 0).any():
        return first, []
    m = first[["period", "value"]].merge(
        latest[["period", "value"]].rename(columns={"value": "latest"}), on="period", how="left"
    )
    r = np.log(m["value"] / m["latest"].where(m["latest"] > 0))
    if r.notna().sum() < 4 * CHAIN_LINK_WINDOW:
        return first, []
    before = r.shift(1).rolling(CHAIN_LINK_WINDOW, min_periods=2).median()
    after = r[::-1].rolling(CHAIN_LINK_WINDOW, min_periods=2).median()[::-1]
    jump = (after - before)
    # the row itself must already sit on the new base (otherwise the last
    # pre-break row, whose forward window is mostly post-break, ties)
    own = (r - before).abs() > CHAIN_LINK_THRESHOLD
    cand = np.flatnonzero(((jump.abs() > CHAIN_LINK_THRESHOLD) & own).to_numpy())
    # the shift is detected over a stretch of rows around the break: group
    # neighbouring candidates and keep the row where the shift is largest
    breaks: list[int] = []
    cluster: list[int] = []
    for i in cand:
        if cluster and i - cluster[-1] > CHAIN_LINK_WINDOW:
            breaks.append(int(max(cluster, key=lambda j: abs(jump.iloc[j]))))
            cluster = []
        cluster.append(int(i))
    if cluster:
        breaks.append(int(max(cluster, key=lambda j: abs(jump.iloc[j]))))
    if not breaks:
        return first, []
    out = first.copy()
    vals = out["value"].to_numpy(dtype=float).copy()
    for i in breaks:  # chronological: earlier rows accumulate every later factor
        factor = float(np.exp(jump.iloc[i]))
        vals[:i] = vals[:i] * factor
    out["value"] = vals
    return out, [str(m["period"].iloc[i].date()) for i in breaks]


def build_series_frames(
    cfg: IndicatorConfig,
    fetched: dict[str, dict[str, SeriesResult]],
    mode: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Resolve every enabled indicator to a period/value/avail frame.

    mode='train' prefers ALFRED first-release frames; mode='now' uses latest.
    Returns (frames, unavailable_ids).
    """
    assert mode in ("train", "now")
    frames: dict[str, pd.DataFrame] = {}
    unavailable: list[str] = []

    for ind in cfg.indicators:
        if not ind.fetched or not ind.enabled:
            continue
        res = fetched.get(ind.id, {})
        if not _series_ok(ind, res):
            unavailable.append(ind.id)
            continue
        if mode == "train" and ind.vintage and "first_release" in res and res["first_release"].ok:
            # ALFRED only carries vintages from some start date per series.
            # Splice: latest-vintage values (lag-based availability, residual
            # revision bias documented) before that date, true first releases
            # with actual publication dates after it.
            fr = res["first_release"].df
            if ind.chain_link:
                fr, breaks = chain_link_first_release(fr, res["latest"].df)
                if breaks:
                    log.info("%s: first-release levels chain-linked at %s", ind.id, ", ".join(breaks))
            older = res["latest"].df[res["latest"].df["period"] < fr["period"].min()]
            frame = (
                pd.concat([older, fr], ignore_index=True)
                .sort_values("period")
                .reset_index(drop=True)
            )
        else:
            frame = res["latest"].df.copy()
        frame = frame[["period", "value", "avail"]]
        if mode == "train" and ind.real_time_from:
            # A model-based index is re-estimated over its whole history: the
            # values dated before its first publication were never observable
            # in real time, so training never sees them.
            frame = frame[frame["period"] >= pd.Timestamp(ind.real_time_from)]
        frames[ind.id] = regularize_frame(frame, ind.frequency)

    for ind in cfg.indicators:
        if ind.source != "derived" or not ind.enabled:
            continue
        missing = [p for p in ind.inputs if p not in frames]
        if missing:
            log.warning("%s unavailable (missing inputs %s)", ind.id, missing)
            unavailable.append(ind.id)
            continue
        frames[ind.id] = derive_series(ind, [frames[p] for p in ind.inputs])

    return frames, unavailable


# --------------------------------------------------------------- assembling

def business_day_grid(end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().normalize()
    return pd.bdate_range(GRID_START, end)


def build_feature_frames(
    cfg: IndicatorConfig, frames: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Apply each configured transform, producing named feature frames."""
    out: dict[str, pd.DataFrame] = {}
    for ind in cfg.indicators:
        if not ind.enabled or ind.id not in frames:
            continue
        for t in ind.transforms:
            name = f"{ind.id}__{t.code}"
            try:
                out[name] = transform_frame(frames[ind.id], t.code, ind.frequency)
            except Exception as exc:
                log.error("Transform %s failed: %s", name, exc)
    return out


def build_panel(
    cfg: IndicatorConfig,
    fetched: dict[str, dict[str, SeriesResult]],
    mode: str,
    grid: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    frames, _ = build_series_frames(cfg, fetched, mode)
    feats = build_feature_frames(cfg, frames)
    grid = grid if grid is not None else business_day_grid()
    specs = cfg.feature_specs()
    ages = {name: max_feature_age_days(specs[name][0]) for name in feats if name in specs}
    panel = build_feature_panel(feats, grid, max_age_days=ages)
    if any(i.source == "composite" and i.enabled for i in cfg.indicators):
        from .composites import composite_features  # local import: composites use features
        comp = composite_features(panel, cfg)
        panel = pd.concat([panel, comp.drop(columns=[c for c in comp.columns if c.endswith("__coverage")])], axis=1)
    panel.index.name = "date"
    return panel


def weekly_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Friday sub-sample of a daily panel (modeling rows)."""
    return panel[panel.index.dayofweek == 4]


# -------------------------------------------------------------- data health

def data_health(
    cfg: IndicatorConfig,
    fetched: dict[str, dict[str, SeriesResult]],
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-source freshness report, including the label series."""
    now = now or pd.Timestamp.now()
    rows = []
    for ind in cfg.indicators:
        if not ind.fetched or not ind.enabled:
            continue
        res = fetched.get(ind.id, {})
        latest = res.get("latest")
        row = {
            "id": ind.id, "name": ind.name, "category": ind.category,
            "tier": ind.tier, "frequency": ind.frequency,
            "vintage_mode": "first_release" in res,
            "source": None, "last_period": None, "available_since": None,
            "fetched_at": None, "days_stale": None, "status": "error",
            "error": None,
        }
        if latest is not None and latest.ok:
            last_p = latest.df["period"].max()
            pe = period_end(pd.Series([last_p]), ind.frequency).iloc[0]
            expected_gap = FREQ_DAYS[ind.frequency] + max(ind.publication_lag_days, 0)
            days_stale = (now.normalize() - pe).days
            row.update({
                "source": latest.meta.get("source_detail"),
                "last_period": last_p,
                "available_since": latest.df["avail"].max(),
                "fetched_at": latest.meta.get("fetched_at"),
                "days_stale": days_stale,
                "status": (
                    "cache_fallback" if latest.fetch_error else
                    "archived" if ind.discontinued else
                    "stale" if days_stale > 2 * expected_gap + 7 else "fresh"
                ),
                "error": latest.fetch_error,
            })
        elif latest is not None:
            row["error"] = latest.fetch_error or "no data"
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_version(fetched: dict[str, dict[str, SeriesResult]]) -> str:
    sig = {
        sid: str(res["latest"].df["period"].max()) if res.get("latest") and res["latest"].ok else "NA"
        for sid, res in sorted(fetched.items())
    }
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:12]


def refresh_summary(fetched: dict[str, dict[str, SeriesResult]]) -> dict:
    n_err = sum(1 for r in fetched.values() if r.get("latest") and r["latest"].fetch_error)
    n_vintage = sum(1 for r in fetched.values() if "first_release" in r)
    return {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "n_series": len(fetched),
        "n_fetch_errors": n_err,
        "n_vintage_series": n_vintage,
        "dataset_version": dataset_version(fetched),
    }
