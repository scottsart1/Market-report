"""Category stress scores (0-100), diagnostic only.

Each feature is mapped to its *expanding historical percentile* (trailing
data only), oriented so that higher = more recession-typical using the
configured risk direction, then averaged within its category. These scores
visualize where stress sits; they are NOT inputs to, or substitutes for,
the probability model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORY_LABELS, IndicatorConfig

MIN_HISTORY_WEEKS = 260  # require ~5 years before a percentile is meaningful


def oriented_percentiles(panel: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    specs = cfg.feature_specs()
    out = {}
    for col in panel.columns:
        if col not in specs:
            continue
        _, t = specs[col]
        pct = panel[col].expanding(min_periods=MIN_HISTORY_WEEKS).rank(pct=True)
        out[col] = pct if t.risk > 0 else 1.0 - pct
    return pd.DataFrame(out, index=panel.index)


def category_scores(panel: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """Weekly 0-100 stress score per category."""
    oriented = oriented_percentiles(panel, cfg)
    specs = cfg.feature_specs()
    by_cat: dict[str, list[str]] = {}
    for col in oriented.columns:
        ind, _ = specs[col]
        by_cat.setdefault(ind.category, []).append(col)
    scores = {
        cat: oriented[cols].mean(axis=1, skipna=True) * 100.0
        for cat, cols in by_cat.items()
    }
    return pd.DataFrame(scores, index=panel.index)


def category_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Current score, week/month ago, percentile and trend per category."""
    rows = []
    if scores.empty:
        return pd.DataFrame()
    last = scores.index.max()
    for cat in scores.columns:
        s = scores[cat].dropna()
        if s.empty:
            continue
        cur = float(s.iloc[-1])
        wk = s[s.index <= last - pd.Timedelta(days=7)]
        mo = s[s.index <= last - pd.Timedelta(days=30)]
        prev_w = float(wk.iloc[-1]) if len(wk) else np.nan
        prev_m = float(mo.iloc[-1]) if len(mo) else np.nan
        hist_pct = float((s <= cur).mean() * 100)
        trend = "rising" if cur - prev_m > 2 else ("falling" if cur - prev_m < -2 else "flat")
        rows.append({
            "category": cat,
            "label": CATEGORY_LABELS.get(cat, cat),
            "score": round(cur, 1),
            "prev_week": round(prev_w, 1) if not np.isnan(prev_w) else None,
            "prev_month": round(prev_m, 1) if not np.isnan(prev_m) else None,
            "hist_percentile": round(hist_pct, 1),
            "trend": trend,
        })
    return pd.DataFrame(rows)
