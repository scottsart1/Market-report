"""Transformations must use trailing data only and propagate availability."""
import numpy as np
import pandas as pd
import pytest

from recession.features import (
    as_of_series, periods_for_months, transform_frame, transform_values,
)

CODES = ["level", "chg3m", "pct6m", "yoy", "ann3m", "accel3m", "z5y",
         "pctile3y", "dmax2y", "avg3m", "infl_accel"]


@pytest.mark.parametrize("code", CODES)
def test_transforms_use_past_data_only(monthly_series, code):
    """Changing FUTURE observations must not change the feature at time t."""
    base = transform_frame(monthly_series, code, "m")
    tampered = monthly_series.copy()
    cut = 100
    tampered.loc[cut:, "value"] = tampered.loc[cut:, "value"] * 5 + 123.0
    tam = transform_frame(tampered, code, "m")
    cutoff_period = monthly_series["period"].iloc[cut]
    b = base[base["period"] < cutoff_period].set_index("period")["value"]
    t = tam[tam["period"] < cutoff_period].set_index("period")["value"]
    common = b.index.intersection(t.index)
    assert len(common) > 20
    np.testing.assert_allclose(b.loc[common], t.loc[common], rtol=1e-12)


def test_chg_and_yoy_values(monthly_series):
    v = monthly_series["value"]
    chg = transform_values(v, "chg3m", "m")
    assert np.isclose(chg.iloc[10], v.iloc[10] - v.iloc[7])
    yoy = transform_values(v, "yoy", "m")
    assert np.isclose(yoy.iloc[20], v.iloc[20] / v.iloc[8] - 1)


def test_since_inv_and_resteep():
    v = pd.Series([1.0, 0.5, -0.2, -0.5, -0.1, 0.3, 0.8, 1.0] + [1.0] * 30)
    si = transform_values(v, "since_inv", "d")
    assert si.iloc[4] == 0.0  # currently inverted
    assert si.iloc[6] > si.iloc[5] > 0  # counting up after inversion ends
    rs = transform_values(v, "resteep", "d")
    assert rs.iloc[5] > 0  # steepened off an inverted trough
    assert rs.iloc[0] == 0.0


def test_availability_is_cummax(monthly_series):
    shuffled = monthly_series.copy()
    # inject one out-of-order availability (late revision released early)
    shuffled.loc[50, "avail"] = shuffled.loc[10, "avail"]
    out = transform_frame(shuffled, "chg3m", "m")
    assert out["avail"].is_monotonic_increasing


def test_as_of_join_never_early(monthly_series):
    feat = transform_frame(monthly_series, "level", "m")
    grid = pd.date_range("2000-01-01", "2011-06-30", freq="B")
    s = as_of_series(feat, grid)
    for _, row in feat.iloc[[5, 60, 100]].iterrows():
        before = s[grid < row["avail"]]
        # value must NOT appear before its availability date
        assert not np.any(np.isclose(before.to_numpy(), row["value"], rtol=0, atol=1e-12))
        at = s[grid >= row["avail"]]
        assert np.isclose(at.iloc[0], row["value"])


def test_periods_for_months():
    assert periods_for_months("m", 12) == 12
    assert periods_for_months("w", 12) == 52
    assert periods_for_months("d", 3) == 63
