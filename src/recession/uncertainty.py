"""Uncertainty bands via moving-block bootstrap.

With only ~8 recessions in the usable sample, point probabilities carry real
parameter uncertainty. We resample the weekly training panel in contiguous
3-year blocks (preserving serial dependence and whole recession episodes,
including the slow build-up before an onset), refit every served ensemble
member on each resample with the production hyperparameters, and keep the
fitted models. At prediction time each replicate is scored exactly like
production (members calibrated with the production calibrators, then
averaged); the spread of those probabilities at the current feature vector
gives a 10th-90th percentile band, mapped through the hazard term structure
for every horizon.

The band reflects estimation uncertainty of the served ensemble given its
architecture and calibrators; it is an honest lower bound on total
uncertainty (architecture and calibrator-choice uncertainty come on top),
which the methodology page states explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone

from .models import ElasticNetLogit, training_weights
from .paths import get_logger

log = get_logger(__name__)

BLOCK_WEEKS = 156   # 3-year blocks
N_BOOT = 60     # every replicate now refits two boosted models (~0.2 MB each): 60 keeps the bundle near 25 MB
QUANTILES = (0.10, 0.90)


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
    return np.sort(idx)


def fit_bootstrap_ensembles(
    X: pd.DataFrame,
    y: np.ndarray,
    next_onset: pd.Series,
    protos: dict,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> list[dict]:
    """Refit every served member (`protos`: name -> fitted production model,
    cloned with its hyperparameters) on `n_boot` block resamples. Returns one
    {member: fitted model} dict per kept replicate."""
    rng = np.random.default_rng(seed)
    fitted: list[dict] = []
    n = len(X)
    for b in range(n_boot):
        idx = block_bootstrap_indices(n, BLOCK_WEEKS, rng)
        yb = y[idx]
        if yb.sum() < 5 or yb.sum() == len(yb):
            continue
        Xb = X.iloc[idx]
        wb = training_weights(yb, next_onset.iloc[idx])
        rep = {}
        try:
            for name, proto in protos.items():
                rep[name] = clone(proto).fit(Xb, yb, sample_weight=wb)
            fitted.append(rep)
        except Exception as exc:  # pragma: no cover
            log.warning("Bootstrap fit %d failed: %s", b, exc)
    log.info("Bootstrap: %d/%d replicates kept (%s)", len(fitted), n_boot, ", ".join(protos))
    return fitted


def fit_bootstrap_models(
    X: pd.DataFrame,
    y: np.ndarray,
    next_onset: pd.Series,
    feature_cols: list[str],
    fixed_C: float,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> list:
    """Elastic-net-only bootstrap (pre-2.0 bundles)."""
    reps = fit_bootstrap_ensembles(X, y, next_onset,
                                   {"B_elastic_net": ElasticNetLogit(columns=feature_cols, Cs=(fixed_C,))},
                                   n_boot=n_boot, seed=seed)
    return [r["B_elastic_net"] for r in reps]


def bootstrap_scores(replicates: list, x_row: pd.DataFrame, calibrators: dict) -> np.ndarray:
    """Calibrated probability of each bootstrap replicate at `x_row`, scored
    like production: every member through its own calibrator, then averaged.
    Accepts the pre-2.0 list of elastic-net models (single member)."""
    out = []
    for rep in replicates:
        members = rep if isinstance(rep, dict) else {"B_elastic_net": rep}
        out.append(np.mean([calibrators[m].transform(np.array([model.predict_proba(x_row)[0, 1]]))[0]
                            for m, model in members.items()]))
    return np.asarray(out, dtype=float)


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
    replicates: list,
    x_row: pd.DataFrame,
    calibrators: dict,
    term_structure_fn,
    point: float,
) -> dict[int, tuple[float, float]]:
    """10-90% band per horizon: anchored in base-probability space at `point`,
    then mapped through the (monotone) term structure to every horizon."""
    if not replicates:
        return {}
    ps_cal = bootstrap_scores(replicates, x_row, calibrators)
    lo, hi = anchored_band(ps_cal, point)
    ts_lo, ts_hi = term_structure_fn(np.array([lo])), term_structure_fn(np.array([hi]))
    return {h: (float(ts_lo[h][0]), float(ts_hi[h][0])) for h in ts_lo}
