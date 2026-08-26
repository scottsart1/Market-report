"""Walk-forward validation: temporal ordering, purge gap, sane outputs."""
import numpy as np
import pandas as pd

from recession.backtest import EMBARGO_DAYS, HORIZON_DAYS, make_folds, run_backtest
from recession.labels import build_recession_calendar, make_labels
from recession.models import training_weights


def test_fold_purge_gap():
    idx = pd.date_range("1968-06-06", "2026-06-01", freq="W-FRI")
    folds = make_folds(idx)
    assert len(folds) >= 6
    for f in folds:
        gap = (f["test_start"] - f["train_end"]).days
        assert gap >= EMBARGO_DAYS + HORIZON_DAYS, "training must end before test minus purge"


def _synthetic_dataset(synthetic_usrec):
    """Small feature panel with one informative feature."""
    cal = build_recession_calendar(synthetic_usrec)
    grid = pd.date_range("1986-01-02", "2012-06-29", freq="W-FRI")
    lab = make_labels(grid, cal)
    rng = np.random.default_rng(3)
    signal = lab["y90"].rolling(8, min_periods=1).max().shift(1).fillna(0)
    X = pd.DataFrame({
        "SPREAD_10Y3M__level": -2 * signal + rng.normal(0, 0.7, len(grid)),
        "SPREAD_10Y3M__chg12m": rng.normal(0, 1, len(grid)),
        "noise1": rng.normal(0, 1, len(grid)),
        "noise2": rng.normal(0, 1, len(grid)),
    }, index=grid)
    return X, lab


def test_backtest_outputs(synthetic_usrec, monkeypatch):
    import recession.backtest as bt
    monkeypatch.setattr(bt, "TRAIN_START", pd.Timestamp("1986-01-01"))
    monkeypatch.setattr(bt, "FOLDS", [
        ("T1", "1994-01-01", "2001-12-31"),
        ("T2", "2002-01-01", "2007-12-31"),
        ("T3", "2008-01-01", "2011-12-31"),
    ])
    X, lab = _synthetic_dataset(synthetic_usrec)
    res = bt.run_backtest(X, lab, list(X.columns))

    # probabilities bounded
    pcols = [c for c in res.oos.columns if c.startswith(("p_raw_", "p_cal_", "p1", "p3", "p4", "p6", "p9"))]
    for c in pcols:
        v = res.oos[c].dropna()
        assert ((v >= 0) & (v <= 1)).all(), c

    # horizon monotonicity of the final term structure
    P = res.oos[[f"p{h}" for h in (15, 30, 45, 60, 90)]].dropna().to_numpy()
    assert (np.diff(P, axis=1) >= -1e-9).all()

    # every OOS prediction date lies strictly after its fold's train_end
    for f in res.fold_info:
        rows = res.oos[res.oos["fold"] == f["k"]]
        if len(rows):
            assert rows.index.min() > f["train_end"]

    assert res.selection["production_model"] in res.metrics


def test_event_weights_balance():
    y = np.array([0] * 90 + [1] * 6 + [0] * 90 + [1] * 3)
    onsets = pd.Series(
        [pd.NaT] * 90 + [pd.Timestamp("2000-01-01")] * 6
        + [pd.NaT] * 90 + [pd.Timestamp("2005-01-01")] * 3
    )
    w = training_weights(y, onsets)
    w_e1 = w[90:96].sum()
    w_e2 = w[186:189].sum()
    assert np.isclose(w_e1, w_e2, rtol=1e-6), "each recession event carries equal total weight"
    assert np.isclose(w.mean(), 1.0, rtol=1e-6)
