"""Recession ground truth from the NBER chronology (FRED series USREC).

Day-count convention (documented in the dashboard methodology):

* NBER dates business cycles at **monthly** precision. FRED's USREC equals 1
  for every month from the peak-following month through the trough month.
* We define the **onset day** of a recession as the *first calendar day of
  the first USREC==1 month* after an expansion month.
* "Recession within h days" at date t means: an onset day falls in the
  window (t, t + h days]. Because the underlying truth is monthly, 15/30/45-
  day distinctions inherit monthly granularity smoothed by the model's
  hazard term structure — the labels do not pretend daily precision exists.
* Dates already inside a recession are excluded from training and
  evaluation: the model estimates the probability of *entering* a recession,
  conditional on not currently being in one.

The label series is used ONLY as the target. Config validation enforces that
it can never appear among predictors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = (15, 30, 45, 60, 90)


@dataclass
class RecessionCalendar:
    onsets: list[pd.Timestamp]          # first day of each recession
    ends: list[pd.Timestamp]            # last day of each recession (trough month end)
    monthly: pd.Series                  # USREC by month start
    last_label_date: pd.Timestamp       # end of last month with a published label

    def in_recession(self, grid: pd.DatetimeIndex) -> pd.Series:
        m = self.monthly.reindex(
            pd.period_range(self.monthly.index.min(), grid.max(), freq="M").to_timestamp()
        )
        keys = pd.DatetimeIndex(grid).to_period("M").to_timestamp()
        vals = m.reindex(keys).to_numpy()
        return pd.Series(np.nan_to_num(vals, nan=0.0).astype(bool), index=grid)


def build_recession_calendar(usrec: pd.DataFrame) -> RecessionCalendar:
    """usrec: frame with columns period (month start) and value (0/1)."""
    s = (
        usrec.dropna(subset=["value"])
        .assign(period=lambda d: pd.to_datetime(d["period"]))
        .set_index("period")["value"]
        .astype(int)
        .sort_index()
    )
    flips_up = s[(s == 1) & (s.shift(1, fill_value=0) == 0)].index
    flips_down = s[(s == 0) & (s.shift(1, fill_value=0) == 1)].index
    onsets = [pd.Timestamp(d) for d in flips_up]
    ends = [pd.Timestamp(d) - pd.Timedelta(days=1) for d in flips_down]
    if len(ends) < len(onsets):  # currently in recession
        ends.append(s.index.max() + pd.offsets.MonthEnd(0))
    last_label = s.index.max() + pd.offsets.MonthEnd(0)
    return RecessionCalendar(onsets=onsets, ends=ends, monthly=s, last_label_date=last_label)


def make_labels(
    grid: pd.DatetimeIndex,
    cal: RecessionCalendar,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Per-date targets y{h} plus eligibility masks.

    y{h}[t] = 1 iff a recession onset occurs in (t, t + h days].
    eligible[t] = not currently in recession AND the h=max window is fully
    inside the labeled period (immature labels are excluded from training).
    """
    onsets = pd.DatetimeIndex(cal.onsets)
    out = pd.DataFrame(index=grid)
    t = grid.to_numpy()
    for h in horizons:
        hi = grid + pd.Timedelta(days=h)
        # count of onsets <= x for each bound
        y = (
            np.searchsorted(onsets.to_numpy(), hi.to_numpy(), side="right")
            - np.searchsorted(onsets.to_numpy(), t, side="right")
        ) > 0
        out[f"y{h}"] = y.astype(int)
    in_rec = cal.in_recession(grid)
    out["in_recession"] = in_rec.to_numpy()
    hmax = max(horizons)
    mature = (grid + pd.Timedelta(days=hmax)) <= cal.last_label_date
    out["eligible"] = (~in_rec.to_numpy()) & mature
    # onset id for event-aware weighting: which onset (if any) a positive
    # y{hmax} row is anticipating (the next onset after t)
    nxt = np.searchsorted(onsets.to_numpy(), t, side="right")
    onset_for_row = np.where(nxt < len(onsets), onsets.to_numpy()[np.minimum(nxt, len(onsets) - 1)], np.datetime64("NaT"))
    out["next_onset"] = onset_for_row
    return out
