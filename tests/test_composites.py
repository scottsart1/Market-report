"""Composites, new transform codes, the NTFS recipe and chain-linking."""
import numpy as np
import pandas as pd
import pytest

from recession.composites import composite_series
from recession.config import Indicator, IndicatorConfig
from recession.features import transform_values
from recession.pipeline import chain_link_first_release, derive_series


def _ind(**kw):
    base = dict(id="X", name="x", category="composite", source="composite", frequency="d",
                publication_lag_days=0, vintage=False, tier=1, license="public")
    base.update(kw)
    return Indicator(**base)


def _panel(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({
        "A__level": rng.normal(0, 1, n),
        "B__level": rng.normal(0, 1, n),
        "S1__level": np.linspace(1, -1, n),     # a spread that inverts half way
        "S2__level": np.full(n, 0.5),
    }, index=idx)


def test_rule_share_and_inv_share():
    p = _panel()
    rs = composite_series(_ind(composite_kind="rule_share",
                               rules=({"feature": "S1__level", "op": "<", "value": 0},
                                      {"feature": "S2__level", "op": "<", "value": 0},
                                      {"feature": "missing", "op": ">", "value": 0})), p, {})
    assert rs.iloc[0] == 0.0 and rs.iloc[-1] == 0.5          # S1 inverted, S2 not; missing skipped
    inv = composite_series(_ind(composite_kind="inv_share", members=("S1__level", "S2__level")), p, {})
    assert inv.iloc[0] == 0.0 and inv.iloc[-1] == 0.5


def test_breadth_pctile_is_oriented_and_trailing():
    p = _panel()
    spec_a = (None, type("T", (), {"risk": 1})())
    spec_b = (None, type("T", (), {"risk": -1})())
    comp = composite_series(_ind(composite_kind="breadth_pctile", members=("A__level", "B__level")), p,
                            {"A__level": spec_a, "B__level": spec_b})
    assert comp.iloc[:1259].isna().all() and comp.iloc[1300:].notna().all()
    # tamper with the future: the past must not change
    p2 = p.copy(); p2.iloc[3000:, 0] += 50
    comp2 = composite_series(_ind(composite_kind="breadth_pctile", members=("A__level", "B__level")), p2,
                             {"A__level": spec_a, "B__level": spec_b})
    pd.testing.assert_series_equal(comp.iloc[:3000], comp2.iloc[:3000])


def test_new_transform_codes():
    v = pd.Series(np.r_[np.linspace(2, -1, 24), np.linspace(-1, 1, 24)], dtype=float)
    assert transform_values(v, "lag6m", "m").iloc[10] == v.iloc[4]
    rise = transform_values(v, "rise1y", "m")
    assert rise.iloc[-1] == pytest.approx(v.iloc[-1] - v.iloc[-12:].min())
    depth = transform_values(v, "invdepth1y", "m")
    assert depth.iloc[30] > 0 and depth.iloc[11] == pytest.approx(0.0)
    oil = pd.Series(np.r_[np.full(40, 50.0), [60.0, 55.0]])
    nopi = transform_values(oil, "nopi3y", "m")
    assert nopi.iloc[40] == pytest.approx(np.log(60 / 50)) and nopi.iloc[41] == 0.0


def test_ntfs_recipe():
    periods = pd.date_range("2020-01-01", periods=3, freq="D")
    mk = lambda vals: pd.DataFrame({"period": periods, "value": vals, "avail": periods + pd.Timedelta(days=1)})
    ind = Indicator(id="N", name="n", category="rates", source="derived", frequency="d",
                    publication_lag_days=1, vintage=False, tier=1, license="public",
                    derive="ntfs", inputs=("DGS2", "DGS1", "DTB3"))
    s = derive_series(ind, [mk([4.0, 4.0, 4.0]), mk([4.0, 4.5, 5.0]), mk([4.0, 4.5, 5.5])])
    np.testing.assert_allclose(s["value"], [2.25 * 4 - 1.25 * 4 - 4, 2.25 * 4 - 1.25 * 4.5 - 4.5, 2.25 * 4 - 1.25 * 5 - 5.5])


def test_chain_link_restores_one_base_across_a_rebasing():
    periods = pd.date_range("1985-01-01", periods=60, freq="MS")
    latest = pd.DataFrame({"period": periods, "value": 100 * 1.003 ** np.arange(60)})
    first = latest.copy()
    first.loc[30:, "value"] = first.loc[30:, "value"] * 0.75       # rebased to a new base year from row 30
    first["value"] = first["value"] * (1 + np.random.default_rng(0).normal(0, 0.004, 60))  # ordinary revisions
    first["avail"] = periods + pd.Timedelta(days=20)
    fixed, breaks = chain_link_first_release(first, latest)
    assert breaks == ["1987-07-01"]
    g = fixed["value"].pct_change()
    assert abs(g.iloc[30]) < 0.02                                  # no -25% growth at the break
    untouched, nb = chain_link_first_release(latest.assign(avail=first["avail"]), latest)
    assert nb == [] and untouched["value"].equals(latest["value"])


def test_config_parses_composites(tmp_path):
    from recession.config import load_config
    y = tmp_path / "c.yaml"
    y.write_text(
        "indicators:\n  - id: DGS10\n    name: t\n    category: rates\n    source: fred\n    frequency: d\n"
        "    publication_lag_days: 1\n    vintage: false\n    tier: 1\n    license: public\n"
        "    transforms: [{t: level, risk: -1, desc: d}]\n"
        "composites:\n  - id: BR\n    kind: breadth_pctile\n    members: [DGS10__level]\n"
        "    transforms: [{t: level, risk: 1, desc: breadth, horizons: [90]}]\n"
    )
    cfg = load_config(y)
    br = cfg.by_id()["BR"]
    assert br.source == "composite" and br.composite_kind == "breadth_pctile" and br.frequency == "d"
    assert "BR__level" in cfg.feature_specs() and cfg.features_for(365) == ["DGS10__level"]
    assert isinstance(cfg, IndicatorConfig)
