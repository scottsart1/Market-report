"""Breadth composites: how many things look recessionary at once.

With ~8 recessions, a fixed-membership count of pre-specified conditions is
a far more robust predictor than any one of its members, and it carries no
fitted parameters. Composites are computed on the daily as-of feature panel,
so every input value at date t was knowable at t, and every statistic is a
trailing window: the result is as leakage-safe as the panel itself.

Kinds (config `composites:` entries, see docs/v2_plan.md):

* breadth_pctile — mean over members of the risk-oriented trailing 10-year
  percentile (1.0 = the most recessionary decile of the member's own history)
* rule_share     — share of pre-specified rules (feature op value) that hold
* inv_share      — share of members below zero (e.g. curve pairs inverted)
* zmean          — mean trailing 10-year z-score of the members (same sign)

NaN members are skipped so a composite exists as soon as any member does;
the share of members present is exposed as `<ID>__coverage` for diagnostics.
"""
from __future__ import annotations

import operator

import numpy as np
import pandas as pd

from .config import Indicator, IndicatorConfig
from .features import transform_values

WINDOW_DAYS = 2520        # trailing 10 business-years
MIN_HISTORY_DAYS = 1260   # 5 years before a percentile or z-score is used
_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def _oriented_percentile(s: pd.Series, risk: int) -> pd.Series:
    pct = s.rolling(WINDOW_DAYS, min_periods=MIN_HISTORY_DAYS).rank(pct=True)
    return pct if risk > 0 else 1.0 - pct


def _zscore(s: pd.Series) -> pd.Series:
    r = s.rolling(WINDOW_DAYS, min_periods=MIN_HISTORY_DAYS)
    return (s - r.mean()) / r.std().replace(0.0, np.nan)


def composite_series(ind: Indicator, panel: pd.DataFrame, specs: dict) -> pd.Series:
    """The composite's base series on the panel's grid."""
    kind = ind.composite_kind
    if kind == "rule_share":
        cols = []
        for rule in ind.rules:
            feat = rule["feature"]
            if feat not in panel.columns:
                continue
            v = panel[feat]
            hit = _OPS[rule["op"]](v, float(rule["value"])).astype(float).where(v.notna())
            cols.append(hit)
        return pd.concat(cols, axis=1).mean(axis=1, skipna=True) if cols else pd.Series(np.nan, index=panel.index)
    members = [m for m in ind.members if m in panel.columns]
    if not members:
        return pd.Series(np.nan, index=panel.index)
    if kind == "breadth_pctile":
        parts = [_oriented_percentile(panel[m], specs[m][1].risk if m in specs else 1) for m in members]
    elif kind == "inv_share":
        parts = [(panel[m] < 0).astype(float).where(panel[m].notna()) for m in members]
    elif kind == "zmean":
        parts = [_zscore(panel[m]) for m in members]
    else:
        raise ValueError(f"Unknown composite kind {kind!r} for {ind.id}")
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def composite_features(panel: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """All configured composites' transforms, as panel columns."""
    specs = cfg.feature_specs()
    out: dict[str, pd.Series] = {}
    for ind in cfg.indicators:
        if ind.source != "composite" or not ind.enabled:
            continue
        base = composite_series(ind, panel, specs)
        present = [m for m in (ind.members or [r["feature"] for r in ind.rules]) if m in panel.columns]
        total = max(len(ind.members) or len(ind.rules), 1)
        out[f"{ind.id}__coverage"] = panel[present].notna().sum(axis=1) / total if present else pd.Series(0.0, index=panel.index)
        for t in ind.transforms:
            out[f"{ind.id}__{t.code}"] = transform_values(base, t.code, "d")
    return pd.DataFrame(out, index=panel.index)
