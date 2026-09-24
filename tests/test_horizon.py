"""Calibration and hazard term structure invariants."""
import numpy as np
import pandas as pd

from recession.horizon import (
    HazardTermStructure, PlattCalibrator, SmoothIsotonicCalibrator,
)
from recession.labels import HORIZONS


def _fake_oos(n=800, seed=1):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1.2, 8, n), 1e-4, 1 - 1e-4)
    lab = {}
    y90 = (rng.random(n) < p).astype(int)
    lab["y90"] = y90
    for h in (15, 30, 45, 60):
        keep = rng.random(n) < h / 90
        lab[f"y{h}"] = (y90 & (keep)).astype(int)
    return p, pd.DataFrame(lab)


def test_term_structure_monotone_in_horizon():
    p, lab = _fake_oos()
    ts = HazardTermStructure().fit(p, lab)
    thetas = [ts.theta_[h] for h in HORIZONS]
    assert all(t2 >= t1 for t1, t2 in zip(thetas, thetas[1:]))
    out = ts.term_structure(np.array([0.001, 0.05, 0.2, 0.6, 0.95]))
    stacked = np.vstack([out[h] for h in HORIZONS])
    assert (np.diff(stacked, axis=0) >= -1e-12).all(), "P(h) must be nondecreasing in h"
    for h in HORIZONS:
        assert ((out[h] >= 0) & (out[h] <= 1)).all()


def test_default_theta_matches_constant_hazard():
    ts = HazardTermStructure()
    p = np.array([0.3])
    out = ts.term_structure(p)
    assert np.isclose(out[90][0], 0.3)
    assert np.isclose(out[45][0], 1 - (1 - 0.3) ** 0.5)


def test_calibrators_bounded_and_monotone():
    p, lab = _fake_oos()
    y = lab["y90"].to_numpy()
    for cal in (PlattCalibrator().fit(p, y), SmoothIsotonicCalibrator().fit(p, y)):
        grid = np.linspace(0.001, 0.999, 200)
        out = cal.transform(grid)
        assert ((out > 0) & (out < 1)).all()
        assert (np.diff(out) >= -1e-9).all(), "calibration must preserve ranking"


def test_smooth_isotonic_has_no_plateaus_or_cliffs():
    """A daily monitor must respond to small score moves and must not jump
    at arbitrary block edges (the failure mode of step isotonic)."""
    p, lab = _fake_oos(n=3000, seed=7)
    cal = SmoothIsotonicCalibrator().fit(p, lab["y90"].to_numpy())
    lo, hi = cal.knots_x_[0], cal.knots_x_[-1]
    grid = np.linspace(lo, hi, 400)
    out = cal.transform(grid)
    assert (np.diff(out) >= -1e-12).all()
    # between the outer knots, distinct scores map to distinct probabilities
    assert np.unique(out.round(10)).size > 0.9 * len(grid)
    # continuity: no single small step in score moves the output by much
    assert np.abs(np.diff(out)).max() < 0.05


def test_uninformative_platt_is_rejected():
    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(1.2, 8, 4000), 1e-4, 1 - 1e-4)
    informative = PlattCalibrator().fit(p, (rng.random(p.size) < p).astype(int))
    unrelated = PlattCalibrator().fit(p, (rng.random(p.size) < 0.1).astype(int))
    assert informative.is_informative()
    assert not unrelated.is_informative()
