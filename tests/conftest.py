import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def monthly_series():
    """Simple monthly series frame with availability = period end + 10 days."""
    periods = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    rng = np.random.default_rng(0)
    values = 100 + np.cumsum(rng.normal(0, 1, len(periods)))
    avail = periods + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10)
    return pd.DataFrame({"period": periods, "value": values, "avail": avail})


@pytest.fixture
def synthetic_usrec():
    """USREC-like frame: recession Jan1990-Mar1990 and Jun2008-Jun2009."""
    periods = pd.date_range("1985-01-01", "2012-12-01", freq="MS")
    v = pd.Series(0, index=periods)
    v.loc["1990-01-01":"1990-03-01"] = 1
    v.loc["2008-06-01":"2009-06-01"] = 1
    return pd.DataFrame({"period": periods, "value": v.to_numpy()})
