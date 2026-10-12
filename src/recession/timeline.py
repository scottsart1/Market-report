"""Probability timeline behind the historical chart.

The walk-forward backtest predictions are frozen when the model is trained,
so on their own the chart stops at the last retrain. The timeline continues
with the production model's own predictions after its training cutoff, so
every point still comes from a model that had not seen that date's outcome:
backtest models up to the cutoff, the production model after it.
"""
from __future__ import annotations

import pandas as pd


def stitch_timeline(
    oos: pd.DataFrame,
    curve: pd.DataFrame,
    train_end,
    calendar=None,
) -> pd.DataFrame:
    """Backtest rows through `train_end`, production-model rows after it.

    Returns the probability columns shared by both inputs plus `in_recession`
    and `source` ("backtest" or "live"), sorted by date with no overlap.
    `calendar` (a RecessionCalendar) shades live dates; without it they are
    treated as non-recession, which is all NBER can have dated that recently.
    """
    cutoff = pd.Timestamp(train_end)
    cols = [c for c in curve.columns if c in oos.columns]
    backtest = oos.loc[oos.index <= cutoff, cols + ["in_recession"]].assign(source="backtest")
    live = curve.loc[curve.index > cutoff, cols].copy()
    live["in_recession"] = (
        calendar.in_recession(live.index).to_numpy()
        if calendar is not None and len(live) else False
    )
    live["source"] = "live"
    out = pd.concat([f for f in (backtest, live) if len(f)] or [backtest]).sort_index()
    out["in_recession"] = out["in_recession"].astype(bool)
    return out
