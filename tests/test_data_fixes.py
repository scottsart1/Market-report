"""Stage-1 data fixes: calendar-regular transforms, structural backfill
detection, staleness tolerance, point-in-time splice offsets, label maturity."""
import numpy as np
import pandas as pd

from recession.config import Indicator
from recession.features import as_of_series, regularize_frame, transform_frame
from recession.fred_client import _availability_from_release
from recession.labels import LABEL_MATURITY_DAYS, build_recession_calendar, make_labels
from recession.pipeline import derive_series


def _monthly(values, start="2024-01-01", lag=7):
    periods = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"period": periods, "value": np.asarray(values, dtype=float),
                         "avail": periods + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)})


def test_missing_month_does_not_stretch_the_horizon():
    full = _monthly(np.arange(1, 25, dtype=float))          # 1..24, +1 per month
    gap = full.drop(index=[20]).reset_index(drop=True)      # Sep-2025 never published
    chg = transform_frame(regularize_frame(gap, "m"), "chg3m", "m").set_index("period")["value"]
    # Nov-2025 = 23, three months earlier (Aug) = 20 -> +3, even though a row is missing
    assert chg.loc["2025-11-01"] == 3.0 and chg.loc["2025-10-01"] == 3.0
    # Dec-2025 would need Sep (missing): no value, rather than a stretched window
    assert pd.Timestamp("2025-12-01") not in chg.index and pd.Timestamp("2025-09-01") not in chg.index
    # without regularization the same change would silently span four months
    raw = transform_frame(gap, "chg3m", "m").set_index("period")["value"]
    assert raw.loc["2025-11-01"] == 4.0


def test_regularize_weekly_keeps_its_weekday_grid():
    weeks = pd.date_range("2025-01-04", periods=10, freq="W-SAT")
    f = pd.DataFrame({"period": weeks, "value": 1.0, "avail": weeks + pd.Timedelta(days=5)})
    g = regularize_frame(f.drop(index=[4]), "w")
    assert len(g) == 10 and g["value"].isna().sum() == 1
    assert (g["period"].dt.dayofweek == 5).all()


def test_late_release_keeps_its_true_date_but_backfill_falls_back():
    periods = pd.date_range("2010-01-01", periods=6, freq="MS")
    first_vintage = pd.Timestamp("2010-04-20")
    rel = [first_vintage, first_vintage, first_vintage,          # Jan-Mar: backfilled on the first vintage date
           first_vintage,                                         # Apr: its genuine first release on that date
           pd.Timestamp("2010-06-20"),                            # May: normal
           pd.Timestamp("2010-09-18")]                            # Jun: 80 days late (shutdown) -> stays late
    df = pd.DataFrame({"period": periods, "value": 1.0, "release_date": rel})
    avail = _availability_from_release(df, "m", lag_days=18)
    assert avail.iloc[0] == pd.Timestamp("2010-02-18")          # period_end + lag
    assert avail.iloc[3] == first_vintage                         # real
    assert avail.iloc[5] == pd.Timestamp("2010-09-18")           # late, not relabelled early


def test_dead_series_degrades_to_nan():
    f = _monthly([1, 2, 3], start="2020-01-01")
    grid = pd.bdate_range("2020-01-01", "2021-06-30")
    carried = as_of_series(f, grid)
    tol = as_of_series(f, grid, max_age_days=170)
    assert carried.loc["2021-06-30"] == 3.0
    assert np.isnan(tol.loc["2021-06-30"]) and tol.loc["2020-06-15"] == 3.0


def test_splice_offset_uses_the_first_overlap_observations():
    old = _monthly([10, 10, 10, 10, 10, 10, 10, np.nan], start="2000-01-01")
    new = _monthly([np.nan, np.nan, 12, 12, 12, 12, 20, 20], start="2000-01-01")  # overlap jumps later
    ind = Indicator(id="S", name="s", category="credit", source="derived", frequency="m",
                    publication_lag_days=7, vintage=False, tier=1, license="public",
                    derive="splice", inputs=("OLD", "NEW"),
                    derive_params={"switch_date": "2000-07-01", "offset": "auto"})
    s = derive_series(ind, [old, new]).set_index("period")["value"]
    # offset = mean over the first four overlap rows (Mar-Jun: +2), not the whole overlap (+3.6)
    assert s.loc["2000-01-01"] == 12.0 and s.loc["2000-07-01"] == 20.0


def test_label_maturity_excludes_the_unconfirmed_tail(synthetic_usrec):
    cal = build_recession_calendar(synthetic_usrec)
    grid = pd.date_range("2005-01-07", cal.last_label_date, freq="W-FRI")
    lab = make_labels(grid, cal)
    last_eligible = lab.index[lab["eligible"]].max()
    assert (cal.last_label_date - last_eligible).days >= 90 + LABEL_MATURITY_DAYS
    assert (cal.last_label_date - last_eligible).days < 90 + LABEL_MATURITY_DAYS + 14
    lab0 = make_labels(grid, cal, maturity_days=0)
    assert lab0["eligible"].sum() > lab["eligible"].sum()
