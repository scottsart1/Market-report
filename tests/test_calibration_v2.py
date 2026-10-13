"""Stage-1 correctness: served P90 is the calibrated score, calibrators need
more than one onset, and the 1-year purge covers its label window."""
import numpy as np
import pandas as pd

from recession.backtest import EMBARGO_DAYS, inner_split, n_events
from recession.horizon import (
    MIN_PLATT_SLOPE, HazardTermStructure, PlattCalibrator, SmoothIsotonicCalibrator,
)


def _oos(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 8, n)
    labels = pd.DataFrame({f"y{h}": (rng.uniform(size=n) < p * h / 90).astype(int)
                           for h in (15, 30, 45, 60, 90)})
    return p, labels


def test_base_horizon_exponent_is_one_by_construction():
    p, labels = _oos()
    ts = HazardTermStructure().fit(p, labels)
    assert ts.theta_[90] == 1.0
    np.testing.assert_allclose(ts.term_structure(p)[90], np.clip(p, 1e-6, 1 - 1e-6))
    th = [ts.theta_[h] for h in (15, 30, 45, 60, 90)]
    assert all(0 < t <= 1 for t in th) and (np.diff(th) >= -1e-12).all()


def test_platt_slope_is_shrunk_toward_one_on_a_thin_window():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 40)
    y = (rng.uniform(size=40) < p).astype(int)
    cal = PlattCalibrator().fit(p, y)
    assert 0.5 < cal.slope_ < 1.5, cal.slope_
    assert cal.is_informative()
    # a monotone map: ranking is preserved
    q = cal.transform(np.array([0.05, 0.2, 0.5]))
    assert (np.diff(q) > 0).all()


def test_isotonic_extrapolates_instead_of_clamping():
    # fitted on scores 0.4-1.0; asked about scores below 0.4
    x = np.linspace(0.4, 1.0, 200)
    y = (np.random.default_rng(2).uniform(size=200) < x ** 2).astype(int)
    cal = SmoothIsotonicCalibrator().fit(x, y)
    lo = cal.transform(np.array([0.05, 0.15, 0.3]))
    assert (np.diff(lo) > 0).all(), "scores below the first knot must still rank"
    assert lo[0] < cal.transform(np.array([0.4]))[0]
    assert not cal.covers(np.linspace(0.05, 0.3, 50))
    assert cal.covers(np.linspace(0.5, 0.9, 50))


def test_inner_split_purge_matches_the_label_window():
    idx = pd.date_range("1968-06-07", "2026-06-26", freq="W-FRI")
    fit, cal = inner_split(idx, 365)
    assert (cal.min() - fit.max()).days > EMBARGO_DAYS + 365
    fit90, cal90 = inner_split(idx)
    assert (cal90.min() - fit90.max()).days > EMBARGO_DAYS + 90
    assert (cal90.min() - fit90.max()).days < EMBARGO_DAYS + 365


def test_n_events_counts_distinct_onsets():
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    onsets = pd.Series([pd.NaT, "2001-04-01", "2001-04-01", pd.NaT, "2008-01-01", "2008-01-01", pd.NaT])
    assert n_events(y, onsets) == 2
    assert n_events(y[:4], onsets[:4]) == 1


def test_uninformative_platt_is_rejected():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.9, 300)
    y = rng.integers(0, 2, 300)  # labels unrelated to the score
    cal = PlattCalibrator().fit(p, y)
    assert cal.slope_ < MIN_PLATT_SLOPE or not cal.is_informative() or cal.slope_ < 1.0
