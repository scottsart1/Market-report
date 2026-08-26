"""Probability calibration and the multi-horizon term structure.

Design (Approach 2 from the spec, hazard-based):

1. A single base classifier predicts the probability of a recession onset
   within the **90-day** window — the horizon with the most historical
   events. Separate 15-day classifiers would rest on a handful of positive
   windows and calibrate terribly; this is the statistically appropriate
   pooling given monthly NBER ground truth.
2. The base probability is calibrated (Platt vs isotonic chosen on
   out-of-sample data inside the walk-forward backtest).
3. Shorter horizons come from a discrete-time hazard identity. With a
   constant hazard over the window, P(h) = 1 - (1 - P90)^(h/90). We
   generalize the exponent to h-specific parameters theta_h fitted by
   maximum likelihood on out-of-sample predictions against the actual
   y15..y60 labels (a power/proportional-hazards calibration), then enforce
   theta_15 <= ... <= theta_90 by pool-adjacent-violators. Because
   x -> 1-(1-p)^theta is increasing in theta for p in (0,1), monotone thetas
   guarantee P15 <= P30 <= P45 <= P60 <= P90 for every prediction — a
   structural property, not an after-the-fact sort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .labels import HORIZONS

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return _clip(p)


class PlattCalibrator:
    """Sigmoid (Platt) calibration: logistic regression on the logit score."""

    name = "platt"

    def fit(self, p, y):
        z = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, dtype=int))
        return self

    def transform(self, p):
        z = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
        return _clip(self.lr_.predict_proba(z)[:, 1])


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(
            y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True
        ).fit(_clip(p), np.asarray(y, dtype=float))
        return self

    def transform(self, p):
        return _clip(self.iso_.predict(_clip(p)))


CALIBRATORS = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def log_loss_w(y, p, w=None) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


class HazardTermStructure:
    """Maps calibrated P(onset within 90d) to shorter horizons."""

    def __init__(self, horizons: tuple[int, ...] = HORIZONS, base_h: int = 90):
        self.horizons = horizons
        self.base_h = base_h
        self.theta_ = {h: h / base_h for h in horizons}

    def fit(self, p_base: np.ndarray, labels: pd.DataFrame) -> "HazardTermStructure":
        """MLE of theta_h per horizon on out-of-sample predictions, then PAV
        across horizons. labels must contain y{h} columns aligned to p_base."""
        p = _clip(p_base)
        thetas, hs = [], []
        for h in self.horizons:
            y = labels[f"y{h}"].to_numpy()
            default = h / self.base_h
            grid = np.unique(np.concatenate([
                np.linspace(0.2, 2.5, 40) * default, [default, 1.0]
            ]))
            grid = grid[(grid > 0.01) & (grid <= 3.0)]
            losses = [log_loss_w(y, 1 - (1 - p) ** th) for th in grid]
            thetas.append(float(grid[int(np.argmin(losses))]))
            hs.append(h)
        # pool-adjacent-violators: thetas must be nondecreasing in horizon
        iso = IsotonicRegression(increasing=True).fit(hs, thetas)
        fitted = iso.predict(hs)
        self.theta_ = {h: float(t) for h, t in zip(hs, fitted)}
        return self

    def term_structure(self, p_base) -> dict[int, np.ndarray]:
        p = _clip(np.atleast_1d(np.asarray(p_base, dtype=float)))
        return {h: 1 - (1 - p) ** self.theta_[h] for h in self.horizons}
