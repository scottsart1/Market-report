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


def anchored_band(ps_cal: np.ndarray, point: float) -> tuple[float, float]:
    """10-90% range of bootstrap probabilities, rescaled so its median lands on
    the production point estimate.

    The bootstrap refits a single member model, so only its relative spread is
    borrowed. Scaling rather than shifting keeps both edges nonnegative and
    the point estimate inside the band at any probability level.
    """
    lo, med, hi = np.quantile(ps_cal, (QUANTILES[0], 0.5, QUANTILES[1]))
    k = point / med
    return float(np.clip(lo * k, 0.0, 1.0)), float(np.clip(hi * k, 0.0, 1.0))


def bootstrap_band(
    boot_models: list,
    x_row: pd.DataFrame,
    calibrate_fn,
    term_structure_fn,
    point: float,
) -> dict[int, tuple[float, float]]:
    """10-90% band per horizon: anchored in base-probability space at `point`,
    then mapped through the (monotone) term structure to every horizon."""
    if not boot_models:
        return {}
    ps_cal = calibrate_fn(np.array([m.predict_proba(x_row)[0, 1] for m in boot_models]))
    lo, hi = anchored_band(ps_cal, point)
    ts_lo, ts_hi = term_structure_fn(np.array([lo])), term_structure_fn(np.array([hi]))
    return {h: (float(ts_lo[h][0]), float(ts_hi[h][0])) for h in ts_lo}
