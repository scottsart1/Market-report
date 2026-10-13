"""Stage-2 experiment switches: natural-prior weights, ensemble membership,
prior-OOS calibration, relative event thresholds."""
import numpy as np
import pandas as pd

import recession.backtest as bt
from recession.backtest import ensemble_members, event_analysis, event_thresholds
from recession.models import training_weights
from tests.test_backtest import _synthetic_dataset


def test_natural_weights_keep_positives_at_their_share():
    y = np.array([0] * 94 + [1] * 3 + [0] * 100 + [1] * 3)
    onsets = pd.Series([pd.NaT] * 94 + [pd.Timestamp("2001-04-01")] * 3
                       + [pd.NaT] * 100 + [pd.Timestamp("2008-01-01")] * 3)
    w_bal = training_weights(y, onsets, class_balance=True)
    w_nat = training_weights(y, onsets, class_balance=False)
    pos = y == 1
    # balanced: positives carry half the total weight; natural: their own share
    assert np.isclose(w_bal[pos].sum() / w_bal.sum(), 0.5)
    assert np.isclose(w_nat[pos].sum() / w_nat.sum(), pos.mean())
    # either way each event carries the same total weight and the mean is 1
    for w in (w_bal, w_nat):
        assert np.isclose(w[94:97].sum(), w[197:200].sum()) and np.isclose(w.mean(), 1.0)


def test_ensemble_switch_parsing(monkeypatch):
    monkeypatch.delenv("RECESSION_VARIANTS", raising=False)
    assert ensemble_members() == ["B_elastic_net", "C_grad_boost"]
    monkeypatch.setenv("RECESSION_VARIANTS", "ensemble=A,B,C")
    assert ensemble_members() == ["A_yield_curve", "B_elastic_net", "C_grad_boost"]


def test_relative_thresholds_and_early_warnings():
    idx = pd.date_range("2000-01-07", periods=400, freq="W-FRI")
    onset = pd.Timestamp("2004-01-01")
    y90 = ((idx < onset) & (idx >= onset - pd.Timedelta(days=90))).astype(int)
    oos = pd.DataFrame({"y90": y90, "eligible": True, "next_onset": pd.NaT}, index=idx)
    oos.loc[y90 == 1, "next_onset"] = onset
    base = y90.mean()
    det, alarm = event_thresholds(base)
    assert np.isclose(det, 3 * base) and np.isclose(alarm, 6 * base)
    # a probability that sits just above the relative detection level 200 days
    # ahead, and an alarm episode 300 days ahead (early warning, not false)
    p = pd.Series(0.0, index=idx)
    assert alarm + 0.001 < 0.20  # relative levels sit far below the absolute ones here
    p[(idx >= onset - pd.Timedelta(days=330)) & (idx < onset - pd.Timedelta(days=290))] = alarm + 0.001
    p[(idx >= onset - pd.Timedelta(days=200)) & (idx < onset)] = det + 0.005
    events, false_alarms = event_analysis(oos.assign(p=p), "p")
    assert events[0]["detected"] and not events[0]["detected_abs"]
    assert false_alarms == []  # the 300-day-ahead episode is an early warning
    # the same episode with no onset afterwards is a false alarm
    oos2 = oos.copy(); oos2["y90"] = 0; oos2["next_onset"] = pd.NaT
    q = pd.Series(0.0, index=idx); q.iloc[50:60] = 0.9
    _, fa2 = event_analysis(oos2.assign(p=q), "p")
    assert len(fa2) == 1 and fa2[0]["kind"] == "false"


def test_prior_oos_calibration_path_runs(synthetic_usrec, monkeypatch):
    monkeypatch.setattr(bt, "TRAIN_START", pd.Timestamp("1986-01-01"))
    monkeypatch.setattr(bt, "FOLDS", [
        ("T1", "1994-01-01", "2001-12-31"),
        ("T2", "2002-01-01", "2007-12-31"),
        ("T3", "2008-01-01", "2011-12-31"),
    ])
    monkeypatch.setenv("RECESSION_VARIANTS", "class_balance=0;calibration=prior_oos;ensemble=A,B,C;hgb_monotone=1")
    X, lab = _synthetic_dataset(synthetic_usrec)
    res = bt.run_backtest(X, lab, list(X.columns))
    assert all(v == "prior_oos" for k, v in res.calibrator_choice.items() if k != "D_ensemble")
    d = res.oos["p_cal_D_ensemble"].dropna()
    assert ((d >= 0) & (d <= 1)).all() and len(d) > 100
    assert res.theta[90] == 1.0
    assert res.selection["theta_walk_forward"]
    long = bt.run_long_backtest(X, lab, list(X.columns))
    assert set(long["calibrator_choice"].values()) == {"prior_oos"}
    assert "p_raw_B_elastic_net" in long["oos"].columns
