"""Model-grounded explanations.

Attribution method: **median substitution** (a local ablation measure that
works identically for the linear, boosted and ensemble models). For feature
j, its contribution is

    delta_j = p(x) - p(x with feature j replaced by its training median)

i.e. how much today's probability changes because feature j is where it is
rather than at its historical neutral level. The same machinery run against
"last week's value" instead of the median decomposes the weekly probability
change. Phrasing is a deterministic template over config metadata — nothing
is invented outside the model and its inputs.

For the elastic-net model, standardized coefficients are also exposed for
the diagnostics page.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import CATEGORY_LABELS, IndicatorConfig


def substitution_attribution(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    x_row: pd.Series,
    baseline: pd.Series,
    feature_cols: list[str],
) -> pd.Series:
    """delta_j = p(x) - p(x_j -> baseline_j), for every feature at once."""
    x = x_row.reindex(feature_cols)
    base = baseline.reindex(feature_cols)
    rows = [x.to_numpy(dtype=float)]
    changed: list[str] = []
    for c in feature_cols:
        xv, bv = x[c], base[c]
        if pd.isna(bv) or (pd.notna(xv) and np.isclose(float(xv), float(bv), equal_nan=True)):
            continue
        r = x.copy()
        r[c] = bv
        rows.append(r.to_numpy(dtype=float))
        changed.append(c)
    X = pd.DataFrame(rows, columns=feature_cols)
    p = predict_fn(X)
    deltas = pd.Series(0.0, index=feature_cols)
    deltas[changed] = p[0] - p[1:]
    return deltas


def describe_factor(
    feature: str,
    delta_p: float,
    x_now: float | None,
    cfg: IndicatorConfig,
) -> dict:
    specs = cfg.feature_specs()
    if feature in specs:
        ind, t = specs[feature]
        desc, category = t.desc, ind.category
    else:
        desc, category = feature, "other"
    direction = "upward" if delta_p > 0 else "downward"
    return {
        "feature": feature,
        "label": desc,
        "category": CATEGORY_LABELS.get(category, category),
        "delta_pp": round(100 * delta_p, 2),
        "value": None if x_now is None or pd.isna(x_now) else float(x_now),
        "text": f"{desc} — {direction} pressure on 90-day risk ({abs(delta_p) * 100:.1f} pp of model score)",
    }


def top_factors(
    deltas: pd.Series,
    x_row: pd.Series,
    cfg: IndicatorConfig,
    n: int = 5,
    min_pp: float = 0.05,
) -> dict[str, list[dict]]:
    d = deltas[deltas.abs() * 100 >= min_pp]
    up = d[d > 0].sort_values(ascending=False).head(n)
    down = d[d < 0].sort_values().head(n)
    return {
        "increasing": [describe_factor(f, v, x_row.get(f), cfg) for f, v in up.items()],
        "decreasing": [describe_factor(f, v, x_row.get(f), cfg) for f, v in down.items()],
    }


def elastic_net_coefficients(model, cfg: IndicatorConfig) -> pd.DataFrame:
    """Standardized coefficients of the fitted elastic-net model."""
    coefs = model.coefficients()
    specs = cfg.feature_specs()
    rows = []
    for feat, beta in coefs.items():
        if feat in specs:
            ind, t = specs[feat]
            rows.append({
                "feature": feat, "label": t.desc,
                "category": CATEGORY_LABELS.get(ind.category, ind.category),
                "coef": float(beta),
            })
    return (
        pd.DataFrame(rows)
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
