"""Model zoo behaviour that the backtest relies on."""
import numpy as np
import pandas as pd

from recession.models import HGBModel, build_models


def _data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "claims": rng.normal(0, 1, n),
        "spread": rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
    })
    logit = 1.5 * X["claims"] - 1.5 * X["spread"] + 0.3 * X["noise"] - 2.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_monotone_constraint_is_respected():
    X, y = _data()
    m = HGBModel(columns=list(X.columns), monotone={"claims": +1, "spread": -1}).fit(X, y)
    grid = pd.DataFrame({"claims": np.linspace(-3, 3, 61), "spread": 0.0, "noise": 0.0})
    p = m.predict_proba(grid)[:, 1]
    assert (np.diff(p) >= -1e-12).all(), "risk must not fall as claims rise"
    grid = pd.DataFrame({"claims": 0.0, "spread": np.linspace(-3, 3, 61), "noise": 0.0})
    p = m.predict_proba(grid)[:, 1]
    assert (np.diff(p) <= 1e-12).all(), "risk must not rise as the curve steepens"


def test_monotone_constraints_survive_dropped_empty_columns():
    X, y = _data()
    X["never_seen"] = np.nan  # a series that starts after this training window
    cols = ["never_seen", "claims", "spread", "noise"]
    m = HGBModel(columns=cols, monotone={"claims": +1, "spread": -1}).fit(X, y)
    assert list(m.keep_) == [False, True, True, True]
    assert list(m.clf_.monotonic_cst) == [1, -1, 0]


def test_build_models_monotone_switch(monkeypatch):
    cols = ["SPREAD_10Y3M__level", "ICSA_4WK__pct3m", "not_in_config"]
    monkeypatch.delenv("RECESSION_VARIANTS", raising=False)
    assert build_models(cols)["C_grad_boost"].monotone is None
    monkeypatch.setenv("RECESSION_VARIANTS", "hgb_monotone=1")
    mono = build_models(cols)["C_grad_boost"].monotone
    assert mono == {"SPREAD_10Y3M__level": -1, "ICSA_4WK__pct3m": 1}


def test_feature_routing_by_core_flag_and_horizon(tmp_path, monkeypatch):
    """Core features feed every member; wide-only ones reach the boosted
    model; horizon tags drop a column from the other target's models."""
    from recession import models as M
    from recession.config import load_config
    y = tmp_path / "c.yaml"
    base = ("    category: rates\n    source: fred\n    frequency: d\n    publication_lag_days: 1\n"
            "    vintage: false\n    tier: 1\n    license: public\n")
    y.write_text(
        "indicators:\n"
        "  - id: CORE\n    name: c\n" + base + "    transforms: [{t: level, risk: 1, desc: core one}]\n"
        "  - id: WIDE\n    name: w\n    core: false\n" + base +
        "    transforms: [{t: level, risk: 1, desc: wide one}, {t: chg3m, risk: 1, desc: short only, horizons: [90]}]\n"
    )
    cfg = load_config(y)
    monkeypatch.setattr(M, "load_config", lambda: cfg)
    cols = ["CORE__level", "WIDE__level", "WIDE__chg3m", "UNKNOWN__x"]
    r90 = M.route_features(cols, 90)
    assert r90["core"] == ["CORE__level", "UNKNOWN__x"]
    assert r90["wide"] == cols
    r365 = M.route_features(cols, 365)
    assert "WIDE__chg3m" not in r365["wide"] and r365["core"] == ["CORE__level", "UNKNOWN__x"]
    ms = M.build_models(cols)
    assert ms["B_elastic_net"].columns == r90["core"] and ms["C_grad_boost"].columns == cols
    monkeypatch.setenv("RECESSION_VARIANTS", "routing=False")
    assert M.route_features(cols, 90)["core"] == cols


def test_modeled_flag_keeps_series_in_panel_but_out_of_models(tmp_path):
    from recession.config import load_config
    y = tmp_path / "c.yaml"
    base = ("    category: rates\n    source: fred\n    frequency: d\n    publication_lag_days: 1\n"
            "    vintage: false\n    tier: 1\n    license: public\n")
    y.write_text(
        "indicators:\n"
        "  - id: KEEP\n    name: k\n" + base + "    transforms: [{t: level, risk: 1, desc: kept one}]\n"
        "  - id: PANEL\n    name: p\n    modeled: false\n" + base + "    transforms: [{t: level, risk: 1, desc: panel only}]\n"
    )
    cfg = load_config(y)
    assert "PANEL__level" in cfg.feature_specs()                      # still built into the panel
    assert cfg.modeled_features(["KEEP__level", "PANEL__level", "X"]) == ["KEEP__level", "X"]
