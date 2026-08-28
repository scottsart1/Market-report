"""Cross-cutting sanity checks: no label leakage into predictors, stale-data
detection, availability shifting on the raw client, config hygiene."""
import numpy as np
import pandas as pd
import pytest

from recession.config import load_config
from recession.fred_client import (
    SeriesResult, _availability_from_lag, _availability_from_release, _validate_obs,
)
from recession.pipeline import data_health


def test_label_never_a_predictor():
    cfg = load_config()
    assert cfg.label_series == "USREC"
    for ind in cfg.indicators:
        assert ind.id != cfg.label_series
        assert cfg.label_series not in ind.inputs
    for feat in cfg.feature_specs():
        assert "USREC" not in feat


def test_config_features_have_metadata():
    cfg = load_config()
    specs = cfg.feature_specs()
    assert len(specs) > 50
    for name, (ind, t) in specs.items():
        assert t.risk in (-1, 1), name
        assert len(t.desc) > 5, name
        assert ind.frequency in ("d", "w", "m", "q")
        assert ind.publication_lag_days >= 0


def test_monthly_availability_lag():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "value": [1.0, 2.0],
    })
    avail = _availability_from_lag(df, "m", 7)
    # January value describes the month ending Jan 31 -> known Feb 7
    assert avail.iloc[0] == pd.Timestamp("2020-02-07")


def test_vintage_backfill_guard():
    """Observations older than ALFRED's first vintage must fall back to the
    configured lag instead of pretending they appeared decades later."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["1980-01-05", "2020-01-04"]),
        "value": [200.0, 210.0],
        "release_date": pd.to_datetime(["2009-05-28", "2020-01-09"]),
    })
    avail = _availability_from_release(df, "w", 5)
    assert avail.iloc[0] == pd.Timestamp("1980-01-10")  # period end + 5d, NOT 2009
    assert avail.iloc[1] == pd.Timestamp("2020-01-09")  # true first release kept


def test_impossible_dates_dropped():
    df = pd.DataFrame({
        "period": pd.to_datetime(["1800-01-01", "2020-01-01", "2300-01-01"]),
        "value": [1.0, 2.0, 3.0],
    })
    out = _validate_obs(df, "X")
    assert list(out["value"]) == [2.0]


def _mk_result(series_id, last_period, freq="w", err=None):
    df = pd.DataFrame({
        "period": pd.date_range(end=last_period, periods=30,
                                freq={"w": "W-SAT", "m": "MS", "d": "B"}[freq]),
        "value": np.arange(30, dtype=float),
    })
    df["avail"] = df["period"] + pd.Timedelta(days=5)
    r = SeriesResult(series_id, "latest", df, meta={"source_detail": "test"})
    r.fetch_error = err
    return {"latest": r}


def test_stale_data_flagged():
    cfg = load_config()
    now = pd.Timestamp("2026-08-25")
    fetched = {}
    for ind in cfg.indicators:
        if ind.source != "fred" or not ind.enabled:
            continue
        if ind.id == "ICSA":  # weekly series 100 days old -> stale
            fetched[ind.id] = _mk_result(ind.id, now - pd.Timedelta(days=100))
        elif ind.id == "NFCI":  # API failed, cache fallback
            fetched[ind.id] = _mk_result(ind.id, now - pd.Timedelta(days=6), err="boom")
        else:
            fetched[ind.id] = _mk_result(ind.id, now - pd.Timedelta(days=3))
    h = data_health(cfg, fetched, now)
    assert h.loc[h["id"] == "ICSA", "status"].iloc[0] == "stale"
    assert h.loc[h["id"] == "NFCI", "status"].iloc[0] == "cache_fallback"
    assert (h.loc[~h["id"].isin(["ICSA", "NFCI"]), "status"] == "fresh").all()


def test_risk_bands_cover_unit_interval():
    from recession.predict import classify
    assert classify(0.001) == "Very Low"
    assert classify(0.07) == "Low"
    assert classify(0.15) == "Moderate"
    assert classify(0.25) == "Elevated"
    assert classify(0.4) == "High"
    assert classify(0.9) == "Extreme"
