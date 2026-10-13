"""Uncertainty bands must be well-formed at any probability level."""
import numpy as np

from recession.horizon import HazardTermStructure
from recession.uncertainty import anchored_band


def test_band_contains_point_and_is_ordered_across_horizons():
    rng = np.random.default_rng(0)
    ts = HazardTermStructure()
    ts.theta_ = {15: 0.07, 30: 0.16, 45: 0.25, 60: 0.33, 90: 0.55}
    for point in (1e-4, 0.002, 0.03, 0.4, 0.97):
        ps = np.clip(rng.lognormal(np.log(0.01), 1.2, 120), 1e-6, 1)
        lo, hi = anchored_band(ps, point)
        assert 0.0 <= lo <= point <= hi <= 1.0
        mid = ts.term_structure(np.array([point]))
        low, high = ts.term_structure(np.array([lo])), ts.term_structure(np.array([hi]))
        los = [low[h][0] for h in sorted(low)]
        assert all(low[h][0] <= mid[h][0] <= high[h][0] for h in low)
        assert all(a <= b for a, b in zip(los, los[1:]))


def test_bootstrap_scores_average_calibrated_members():
    import pandas as pd

    class _M:
        def __init__(self, p): self.p = p
        def predict_proba(self, X): return np.array([[1 - self.p, self.p]])

    class _C:
        def __init__(self, k): self.k = k
        def transform(self, p): return np.asarray(p) * self.k

    from recession.uncertainty import bootstrap_scores
    x = pd.DataFrame([[0.0]], columns=["f"])
    reps = [{"B_elastic_net": _M(0.2), "C_grad_boost": _M(0.4)}, {"B_elastic_net": _M(0.1), "C_grad_boost": _M(0.1)}]
    cals = {"B_elastic_net": _C(1.0), "C_grad_boost": _C(0.5)}
    np.testing.assert_allclose(bootstrap_scores(reps, x, cals), [(0.2 + 0.2) / 2, (0.1 + 0.05) / 2])
    # pre-2.0 bundles stored a flat list of elastic-net models
    np.testing.assert_allclose(bootstrap_scores([_M(0.3)], x, cals), [0.3])
