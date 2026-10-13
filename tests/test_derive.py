"""Derived-series recipes and the one-sided HP gap: values and leakage safety."""
import numpy as np
import pandas as pd
import pytest

from recession.config import Indicator
from recession.features import transform_frame, transform_values
from recession.pipeline import derive_series


def _frame(values, start="2000-01-01", freq="MS", lag=7):
    periods = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"period": periods, "value": np.asarray(values, dtype=float),
                         "avail": periods + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)})


def _ind(derive, inputs, params=None, freq="m"):
    return Indicator(id="X", name="x", category="labor", source="derived", frequency=freq,
                     publication_lag_days=7, vintage=False, tier=1, license="public",
                     derive=derive, inputs=tuple(inputs), derive_params=params or {})


def test_ratio_and_flow_rate():
    a, b = _frame([2, 4, 6, 8]), _frame([1, 2, 3, 4])
    r = derive_series(_ind("ratio", ["A", "B"]), [a, b])
    assert list(r["value"]) == [2, 2, 2, 2]
    fr = derive_series(_ind("flow_rate", ["A", "B"]), [a, b])  # 100 * a_t / b_{t-1}
    assert np.allclose(fr["value"], [400, 300, 800 / 3])


def test_growth_gap_is_yoy_difference_in_points():
    a = _frame(np.r_[np.ones(12), np.full(6, 1.10)])     # +10% yoy at the end
    b = _frame(np.r_[np.ones(12), np.full(6, 1.04)])     # +4% yoy
    g = derive_series(_ind("growth_gap", ["A", "B"]), [a, b])
    assert np.isclose(g["value"].iloc[-1], 6.0)


def test_splice_switches_series_and_levels_the_overlap():
    old = _frame([10, 11, 12, 13, np.nan, np.nan], start="2000-01-01")
    new = _frame([np.nan, np.nan, 14, 15, 16, 17], start="2000-01-01")
    s = derive_series(_ind("splice", ["OLD", "NEW"], {"switch_date": "2000-05-01", "offset": "auto"}), [old, new])
    # overlap (Mar, Apr): new - old = 2 -> the old segment is shifted up by 2
    assert list(s["value"]) == [12, 13, 14, 15, 16, 17]
    s0 = derive_series(_ind("splice", ["OLD", "NEW"], {"switch_date": "2000-05-01"}), [old, new])
    assert list(s0["value"]) == [10, 11, 12, 13, 16, 17]


def test_taylor_gap_signs():
    n = 150
    ffr = _frame(np.full(n, 5.0))
    cpi = _frame(100 * (1.02 ** (np.arange(n) / 12)))        # 2% core inflation
    un = _frame(np.r_[np.full(n - 1, 5.0), 4.0])             # last month: tight labor market
    g = derive_series(_ind("taylor_gap", ["FEDFUNDS", "CPILFESL", "UNRATE"]), [ffr, cpi, un])
    v = g.set_index("period")["value"]
    # steady state: rule = 2 + 2 + 0 - 0 = 4, policy at 5 -> gap +1
    assert np.isclose(v.iloc[-2], 1.0, atol=0.02)
    # lower unemployment raises the rule rate, so the gap shrinks
    assert v.iloc[-1] < v.iloc[-2]


def test_dpeak_is_level_drop_from_trailing_peak():
    v = pd.Series([1.0, 3.0, 2.0, 2.5, 1.0] + [1.0] * 20)
    d = transform_values(v, "dpeak1y", "m")
    assert d.iloc[4] == pytest.approx(-2.0) and d.iloc[1] == pytest.approx(0.0)


def test_sahm_works_on_weekly_frequency():
    weeks = pd.date_range("2000-01-07", periods=80, freq="W-FRI")
    v = np.r_[np.full(60, 3.0), np.linspace(3.0, 4.0, 20)]
    f = pd.DataFrame({"period": weeks, "value": v, "avail": weeks + pd.Timedelta(days=12)})
    s = derive_series(_ind("sahm", ["IURSA"], freq="w"), [f]).set_index("period")["value"]
    assert s.iloc[-1] > 0.5 and abs(s.loc[weeks[59]]) < 1e-9
    assert s.index[0] == weeks[37]  # 13-week mean, then a 26-week minimum warm-up


def test_one_sided_hp_gap_uses_past_only():
    rng = np.random.default_rng(1)
    monthly_series = _frame(100 + np.cumsum(rng.normal(0.3, 1.0, 300)), start="1990-01-01")
    base = transform_frame(monthly_series, "hpgap", "m")
    tampered = monthly_series.copy()
    cut = 200
    tampered.loc[cut:, "value"] = tampered.loc[cut:, "value"] * 3 + 50
    tam = transform_frame(tampered, "hpgap", "m")
    cutoff = monthly_series["period"].iloc[cut]
    b = base[base["period"] < cutoff].set_index("period")["value"]
    t = tam[tam["period"] < cutoff].set_index("period")["value"]
    common = b.index.intersection(t.index)
    assert len(common) > 20
    np.testing.assert_allclose(b.loc[common], t.loc[common], rtol=1e-10)


def test_hp_gap_lambda_parsing_and_warmup():
    q = pd.Series(np.linspace(100, 200, 60) + np.sin(np.arange(60)) * 5)
    g = transform_values(q, "hpgap400k", "q")
    assert g.isna().sum() >= 39  # 10-year warm-up on quarterly data
    assert g.notna().sum() > 15
    with pytest.raises(ValueError):
        transform_values(q, "hpgapx", "q")
