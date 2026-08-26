"""Uncertainty bands via moving-block bootstrap.

With only ~8 recessions in the usable sample, point probabilities carry real
parameter uncertainty. We resample the weekly training panel in contiguous
2-year blocks (preserving serial dependence and whole recession episodes),
refit the elastic-net model on each resample with fixed hyperparameters, and
keep the fitted pipelines. At prediction time the spread of their calibrated
probabilities at the current feature vector gives a 10th-90th percentile
band, mapped through the hazard term structure for every horizon.

The band reflects estimation uncertainty of the interpretable model; it is
an honest lower bound on total uncertainty (model-choice uncertainty comes
on top), which the methodology page states explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ElasticNetLogit, training_weights
from .paths import get_logger

log = get_logger(__name__)

BLOCK_WEEKS = 104
N_BOOT = 120
QUANTILES = (0.10, 0.90)


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
    return np.sort(idx)


def fit_bootstrap_models(
    X: pd.DataFrame,
    y: np.ndarray,
    next_onset: pd.Series,
    feature_cols: list[str],
    fixed_C: float,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    fitted = []
    n = len(X)
    for b in range(n_boot):
        idx = block_bootstrap_indices(n, BLOCK_WEEKS, rng)
        yb = y[idx]
        if yb.sum() < 5 or yb.sum() == len(yb):
            continue
        Xb = X.iloc[idx]
        wb = training_weights(yb, next_onset.iloc[idx])
        m = ElasticNetLogit(columns=feature_cols, Cs=(fixed_C,))
        try:
            m.fit(Xb, yb, sample_weight=wb)
            fitted.append(m)
        except Exception as exc:  # pragma: no cover
            log.warning("Bootstrap fit %d failed: %s", b, exc)
    log.info("Bootstrap: %d/%d fits kept", len(fitted), n_boot)
    return fitted


def bootstrap_band(
    boot_models: list,
    x_row: pd.DataFrame,
    calibrate_fn,
    term_structure_fn,
) -> dict[int, tuple[float, float]]:
    """10-90% band per horizon at the given feature row."""
    if not boot_models:
        return {}
    ps = np.array([m.predict_proba(x_row)[0, 1] for m in boot_models])
    ps_cal = calibrate_fn(ps)
    lo, hi = np.quantile(ps_cal, QUANTILES)
    band = {}
    for p_edge, key in ((lo, "lo"), (hi, "hi")):
        ts = term_structure_fn(np.array([p_edge]))
        for h, v in ts.items():
            band.setdefault(h, {})[key] = float(v[0])
    return {h: (d["lo"], d["hi"]) for h, d in band.items()}
