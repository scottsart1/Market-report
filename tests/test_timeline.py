"""History chart timeline: backtest up to the training cutoff, live model after."""
import pandas as pd

from recession.labels import build_recession_calendar
from recession.timeline import stitch_timeline


def _inputs():
    bt_idx = pd.date_range("2026-01-02", "2026-09-18", freq="W-FRI")
    oos = pd.DataFrame({"p90": 0.01, "p365": 0.06, "y90": 0, "in_recession": False}, index=bt_idx)
    live_idx = pd.bdate_range("2026-03-02", "2026-10-12")
    curve = pd.DataFrame({"p90": 0.002, "p365": 0.03}, index=live_idx)
    return oos, curve


def test_switches_at_cutoff_without_overlap():
    oos, curve = _inputs()
    tl = stitch_timeline(oos, curve, "2026-05-29")
    assert tl.index.is_monotonic_increasing and tl.index.is_unique
    assert list(tl.columns) == ["p90", "p365", "in_recession", "source"]
    assert (tl.loc[:"2026-05-29", "source"] == "backtest").all()
    assert (tl.loc["2026-05-30":, "source"] == "live").all()
    # the chart reaches the latest live date, not the end of the backtest file
    assert tl.index.max() == curve.index.max()
    assert (tl.loc[tl["source"] == "live", "p90"] == 0.002).all()
    assert (tl.loc[tl["source"] == "backtest", "p90"] == 0.01).all()


def test_live_rows_take_recession_shading_from_calendar():
    oos, curve = _inputs()
    usrec = pd.DataFrame({
        "period": pd.date_range("2025-01-01", "2026-09-01", freq="MS"),
        "value": 0,
    })
    usrec.loc[usrec["period"] == "2026-09-01", "value"] = 1
    tl = stitch_timeline(oos, curve, "2026-05-29", build_recession_calendar(usrec))
    live = tl[tl["source"] == "live"]
    assert live.loc["2026-09-01":"2026-09-30", "in_recession"].all()
    assert not live.loc[:"2026-08-31", "in_recession"].any()
    assert not live.loc["2026-10-01":, "in_recession"].any()  # beyond the last label


def test_no_live_rows_before_cutoff_is_reached():
    oos, curve = _inputs()
    tl = stitch_timeline(oos, curve, "2026-12-31")
    assert (tl["source"] == "backtest").all()
    assert tl.index.max() == oos.index.max()
