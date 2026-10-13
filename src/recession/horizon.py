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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from .labels import HORIZONS

EPS = 1e-6
# A calibration window can hold as little as one recession. A Platt fit that
# flattens the score's logit dispersion by more than 4x (or inverts it) is
# asserting the model has no ranking skill, which that window cannot
# establish; such a fit is rejected in favor of the uncalibrated score.
MIN_PLATT_SLOPE = 0.25
# Prior weight pulling the Platt slope toward 1 (in log-likelihood units):
# with one or two onsets in the window the slope is weakly identified, and
# an unshrunk fit can quadruple or flatten the score's dispersion.
PLATT_SLOPE_PRIOR = 5.0
# A calibrator may only be applied where it was fitted: if more than this
# share of the scores it is asked to map lie outside its fitted score range,
# the fit says nothing about them and the raw score is served instead.
MAX_OUT_OF_RANGE = 0.10
CAL_FLOOR = 1e-4  # smallest calibrated probability a block may assert


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return _clip(p)

    def is_informative(self) -> bool:
        return True

    def covers(self, p) -> bool:
        return True


class PlattCalibrator:
    """Sigmoid (Platt) calibration on the logit score, slope shrunk toward 1.

    Minimises the log loss plus PLATT_SLOPE_PRIOR * (slope - 1)^2, so the
    intercept (the prior shift) is fitted freely while the slope moves away
    from 1 only when the window carries evidence for it."""

    name = "platt"

    def fit(self, p, y):
        z, yy = _logit(p), np.asarray(y, dtype=float)

        def nll(ab):
            q = _clip(expit(ab[0] * z + ab[1]))
            return -np.sum(yy * np.log(q) + (1 - yy) * np.log(1 - q)) + PLATT_SLOPE_PRIOR * (ab[0] - 1.0) ** 2

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
        self.slope_, self.intercept_ = float(res.x[0]), float(res.x[1])
        self.fit_range_ = (float(z.min()), float(z.max()))
        return self

    def transform(self, p):
        return _clip(expit(self.slope_ * _logit(p) + self.intercept_))

    def is_informative(self) -> bool:
        return self.slope_ >= MIN_PLATT_SLOPE

    def covers(self, p) -> bool:
        # a sigmoid extrapolates sensibly, so only a gross mismatch is refused
        z = _logit(p)
        lo, hi = self.fit_range_
        span = max(hi - lo, 1.0)
        return float(np.mean((z < lo - span) | (z > hi + span))) <= MAX_OUT_OF_RANGE


class SmoothIsotonicCalibrator:
    """Isotonic calibration interpolated between block centers.

    Plain isotonic regression is a step function: with few positive events
    its blocks are wide, so a daily monitor's headline sits frozen on one
    step while the model score moves, then jumps at block edges. This keeps
    the pool-adjacent-violators fit but anchors each block's value at the
    mean score of the observations in that block and interpolates linearly
    between anchors, giving a continuous, nondecreasing map.
    """

    name = "isotonic_smooth"

    def fit(self, p, y):
        x = _clip(p)
        iso = IsotonicRegression(
            y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True
        ).fit(x, np.asarray(y, dtype=float))
        blocks = pd.DataFrame({"x": x, "v": iso.predict(x)}).groupby("v")["x"].mean()
        self.knots_x_ = blocks.to_numpy()
        # a block with no positives is "very unlikely", not impossible
        self.knots_y_ = np.clip(blocks.index.to_numpy(), CAL_FLOOR, 1 - CAL_FLOOR)
        order = np.argsort(self.knots_x_)
        self.knots_x_, self.knots_y_ = self.knots_x_[order], self.knots_y_[order]
        return self

    def transform(self, p):
        """Interpolate between knots; outside them extrapolate linearly toward
        (0, 0) and (1, 1) instead of clamping, so scores beyond the fitted
        range keep their ranking (the clamp once mapped a whole test fold to
        a constant)."""
        x = _clip(p)
        kx, ky = self.knots_x_, self.knots_y_
        out = np.interp(x, kx, ky)
        out = np.where(x < kx[0], ky[0] * x / kx[0], out)
        out = np.where(x > kx[-1], ky[-1] + (1 - ky[-1]) * (x - kx[-1]) / (1 - kx[-1]), out)
        return _clip(out)

    def is_informative(self) -> bool:
        # at least two distinct levels, and a range that is not a hairline
        return len(self.knots_y_) >= 2 and (self.knots_y_[-1] - self.knots_y_[0]) >= 0.02

    def covers(self, p) -> bool:
        x = _clip(p)
        return float(np.mean((x < self.knots_x_[0]) | (x > self.knots_x_[-1]))) <= MAX_OUT_OF_RANGE


CALIBRATORS = {
    "none": IdentityCalibrator,
    "platt": PlattCalibrator,
    "isotonic_smooth": SmoothIsotonicCalibrator,
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
        """MLE of theta_h for the sub-horizons on out-of-sample predictions,
        then PAV across horizons. The base horizon's exponent is 1 by
        construction: the served P90 IS the calibrated probability. (A fitted
        theta_90 silently re-calibrated the headline number, and did so on the
        rows it was then scored on.) labels must contain y{h} columns aligned
        to p_base."""
        p = _clip(p_base)
        thetas, hs = [], []
        for h in self.horizons:
            if h == self.base_h:
                thetas.append(1.0)
                hs.append(h)
                continue
            y = labels[f"y{h}"].to_numpy()
            default = h / self.base_h
            grid = np.unique(np.concatenate([np.linspace(0.2, 2.5, 40) * default, [default]]))
            grid = grid[(grid > 0.01) & (grid <= 1.0)]  # a sub-horizon cannot exceed the base
            losses = [log_loss_w(y, 1 - (1 - p) ** th) for th in grid]
            thetas.append(float(grid[int(np.argmin(losses))]))
            hs.append(h)
        # pool-adjacent-violators: thetas must be nondecreasing in horizon
        iso = IsotonicRegression(increasing=True).fit(hs, thetas)
        fitted = np.minimum(iso.predict(hs), 1.0)
        self.theta_ = {h: float(t) for h, t in zip(hs, fitted)}
        self.theta_[self.base_h] = 1.0
        return self

    def term_structure(self, p_base) -> dict[int, np.ndarray]:
        p = _clip(np.atleast_1d(np.asarray(p_base, dtype=float)))
        return {h: 1 - (1 - p) ** self.theta_[h] for h in self.horizons}
