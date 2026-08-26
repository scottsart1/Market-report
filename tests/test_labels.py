"""Label construction: onset convention, windows, eligibility."""
import numpy as np
import pandas as pd

from recession.labels import build_recession_calendar, make_labels


def test_onset_convention(synthetic_usrec):
    cal = build_recession_calendar(synthetic_usrec)
    assert cal.onsets == [pd.Timestamp("1990-01-01"), pd.Timestamp("2008-06-01")]
    assert cal.ends[0] == pd.Timestamp("1990-03-31")


def test_label_windows(synthetic_usrec):
    cal = build_recession_calendar(synthetic_usrec)
    grid = pd.date_range("1988-01-01", "2011-12-31", freq="W-FRI")
    lab = make_labels(grid, cal)

    t_10d_before = pd.Timestamp("2008-05-23")   # Friday, 9 days before onset
    t_80d_before = pd.Timestamp("2008-03-14")
    row_near = lab.loc[t_10d_before]
    row_far = lab.loc[t_80d_before]
    assert row_near["y15"] == 1 and row_near["y90"] == 1
    assert row_far["y15"] == 0 and row_far["y30"] == 0 and row_far["y90"] == 1

    # cumulative windows are monotone in h by construction
    ys = lab[[f"y{h}" for h in (15, 30, 45, 60, 90)]].to_numpy()
    assert (np.diff(ys, axis=1) >= 0).all()


def test_eligibility(synthetic_usrec):
    cal = build_recession_calendar(synthetic_usrec)
    grid = pd.date_range("1988-01-01", "2012-12-31", freq="W-FRI")
    lab = make_labels(grid, cal)
    # rows inside a recession are ineligible
    inside = lab.loc["2008-08-01":"2009-05-01"]
    assert not inside["eligible"].any()
    assert inside["in_recession"].all()
    # rows whose 90d window extends past the labeled period are ineligible
    tail = lab[lab.index > cal.last_label_date - pd.Timedelta(days=90)]
    assert not tail["eligible"].any()
    # normal expansion rows are eligible
    assert lab.loc["1995-06-02", "eligible"]


def test_next_onset_assignment(synthetic_usrec):
    cal = build_recession_calendar(synthetic_usrec)
    grid = pd.date_range("1988-01-01", "2010-12-31", freq="W-FRI")
    lab = make_labels(grid, cal)
    pos = lab[lab["y90"] == 1]
    assert set(pd.DatetimeIndex(pos["next_onset"]).unique()) <= set(cal.onsets)
    # every positive row's onset lies within (t, t+90d]
    for t, row in pos.iterrows():
        o = pd.Timestamp(row["next_onset"])
        assert t < o <= t + pd.Timedelta(days=90)
