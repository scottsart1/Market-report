"""URL-file indicators: parsed, lag-dated, cached, flagged as revised history."""
import pandas as pd
import pytest

import recession.fred_client as fc


class _Resp:
    def __init__(self, text):
        self.text = text
        self.content = text.encode()


def test_url_series_parses_and_dates_availability(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "RAW_DIR", tmp_path)
    csv = "date,gz_spread,ebp\n2026-07-01,0.84,-0.31\n2026-08-01,0.88,-0.23\n2026-09-01,0.93,-0.39\n"
    calls = []
    monkeypatch.setattr(fc, "_request", lambda url, params, timeout=75, delays=None: calls.append(url) or _Resp(csv))

    out = fc.fetch_url_series("EBP", "https://example.test/ebp.csv", "m", 10,
                              date_col="date", value_col="ebp", ttl_hours=12)
    df = out["latest"].df
    assert list(df["value"]) == [-0.31, -0.23, -0.39]
    # monthly value for September is known 10 days after the month ends
    assert df["avail"].iloc[-1] == pd.Timestamp("2026-10-10")
    assert out["latest"].meta["vintage_fallback"] is True
    assert out["latest"].meta["source_detail"] == "url_file"

    # second call within the TTL comes from the cache, no download
    again = fc.fetch_url_series("EBP", "https://example.test/ebp.csv", "m", 10,
                                date_col="date", value_col="ebp", ttl_hours=12)
    assert again["latest"].from_cache and len(calls) == 1


def test_url_series_missing_column_is_an_error_not_zeros(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fc, "_request", lambda *a, **k: _Resp("date,x\n2026-01-01,1\n"))
    out = fc.fetch_url_series("BAD", "https://example.test/x.csv", "m", 5, value_col="ebp")
    assert not out["latest"].ok and "ebp" in out["latest"].fetch_error


def test_config_rejects_url_indicator_without_value_col(tmp_path):
    from recession.config import load_config
    cfg = tmp_path / "ind.yaml"
    cfg.write_text(
        "indicators:\n  - id: X\n    name: x\n    category: credit\n    source: url\n"
        "    url: https://example.test/x.csv\n    frequency: m\n    publication_lag_days: 5\n"
        "    vintage: false\n    tier: 2\n    license: public\n    transforms: []\n"
    )
    with pytest.raises(ValueError, match="url_value_col"):
        load_config(cfg)
