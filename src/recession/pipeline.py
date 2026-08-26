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

import pandas as pd

from .config import FREQ_DAYS, Indicator, IndicatorConfig, load_config
from .features import build_feature_panel, periods_for_months, transform_frame
from .fred_client import SeriesResult, fetch_series, period_end
from .paths import get_logger

log = get_logger(__name__)

GRID_START = "1962-01-01"


# ------------------------------------------------------------------ fetching

def fetch_all(
    cfg: IndicatorConfig | None = None,
    ttl_hours: float = 12.0,
    offline: bool = False,
) -> dict[str, dict[str, SeriesResult]]:
    """Fetch every FRED-sourced series (indicators + label)."""
    cfg = cfg or load_config()
    fetched: dict[str, dict[str, SeriesResult]] = {}
    for ind in cfg.indicators:
        if ind.source != "fred" or not ind.enabled:
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


def derive_series(ind: Indicator, parents: list[pd.DataFrame]) -> pd.DataFrame:
    a = _align_inputs(parents)
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
        u3 = a["v0"].rolling(3, min_periods=3).mean()
        value = u3 - u3.rolling(12, min_periods=6).min()
    else:
        raise ValueError(f"Unknown derive recipe {ind.derive!r} for {ind.id}")
    out = pd.DataFrame({"period": a["period"], "value": value, "avail": a["avail"].cummax()})
    return out.dropna(subset=["value"]).reset_index(drop=True)


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
        if ind.source != "fred" or not ind.enabled:
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
            older = res["latest"].df[res["latest"].df["period"] < fr["period"].min()]
            frames[ind.id] = (
                pd.concat([older, fr], ignore_index=True)
                .sort_values("period")
                .reset_index(drop=True)
            )
        else:
            frames[ind.id] = res["latest"].df.copy()

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
    panel = build_feature_panel(feats, grid)
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
        if ind.source != "fred" or not ind.enabled:
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
            expected_gap = FREQ_DAYS[ind.frequency] + ind.publication_lag_days
            days_stale = (now.normalize() - pe).days
            row.update({
                "source": latest.meta.get("source_detail"),
                "last_period": last_p,
                "available_since": latest.df["avail"].max(),
                "fetched_at": latest.meta.get("fetched_at"),
                "days_stale": days_stale,
                "status": (
                    "cache_fallback" if latest.fetch_error else
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
