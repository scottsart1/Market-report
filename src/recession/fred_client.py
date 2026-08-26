"""FRED / ALFRED data client.

Two access modes:

* **Keyed** (``FRED_API_KEY`` set): uses the official FRED API. For
  revision-prone series it additionally downloads ALFRED *first-release*
  observations (``output_type=4``), i.e. the value as it was originally
  published together with the actual publication date (``realtime_start``).
  This is what makes historical features point-in-time correct.
* **Keyless fallback**: uses the public ``fredgraph.csv`` endpoint, which
  serves latest-vintage values only. Availability dates are then
  approximated as ``period_end + publication_lag_days`` from the indicator
  config. Timing look-ahead is removed, but revision look-ahead remains and
  is flagged in Data Health.

All downloads are cached as Parquet + JSON metadata under ``data/raw`` and
only refreshed when older than a TTL. On network failure the most recent
cache is used and flagged as stale — the pipeline never silently substitutes
zeros.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

from .paths import RAW_DIR, fred_api_key, get_logger

log = get_logger(__name__)

FRED_API = "https://api.stlouisfed.org/fred"
FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFRED_EPOCH = "1776-07-04"
ALFRED_HORIZON = "9999-12-31"

RETRY_DELAYS = [2, 4, 8, 16]  # seconds, exponential backoff
MIN_REQUEST_INTERVAL = 0.25  # stay well under FRED's 120 req/min limit

_last_request_ts = 0.0


@dataclass
class SeriesResult:
    """One fetched series in one mode ('latest' or 'first_release')."""

    series_id: str
    mode: str
    df: pd.DataFrame  # columns: period (Timestamp), value (float), avail (Timestamp)
    meta: dict = field(default_factory=dict)
    from_cache: bool = False
    fetch_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.df is not None and len(self.df) > 0


def _throttle() -> None:
    global _last_request_ts
    wait = MIN_REQUEST_INTERVAL - (time.time() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def _request(url: str, params: dict, timeout: int = 75, delays: list[int] | None = None) -> requests.Response:
    """GET with retries, backoff and 429 handling."""
    if delays is None:
        delays = RETRY_DELAYS
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            _throttle()
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                log.warning("Rate limited on %s, sleeping 20s", url)
                time.sleep(20)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # includes HTTPError
            last_exc = exc
            log.warning("Request failed (attempt %d): %s", attempt + 1, exc)
    raise ConnectionError(f"FRED request failed after retries: {last_exc}")


def period_end(period: pd.Series, frequency: str) -> pd.Series:
    """FRED dates observations at period *start* for monthly series and at the
    week-ending date for weekly series. Return the calendar end of the period
    the value describes."""
    if frequency == "m":
        return period + pd.offsets.MonthEnd(0)
    return pd.Series(period.values, index=period.index)  # d, w: date is the period end


# --------------------------------------------------------------------- cache

def _cache_paths(series_id: str, mode: str):
    base = RAW_DIR / f"{series_id}__{mode}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


def _read_cache(series_id: str, mode: str) -> SeriesResult | None:
    pq, mj = _cache_paths(series_id, mode)
    if not pq.exists() or not mj.exists():
        return None
    try:
        df = pd.read_parquet(pq)
        meta = json.loads(mj.read_text())
        return SeriesResult(series_id, mode, df, meta, from_cache=True)
    except Exception as exc:
        log.warning("Corrupt cache for %s/%s: %s", series_id, mode, exc)
        return None


def _write_cache(res: SeriesResult) -> None:
    pq, mj = _cache_paths(res.series_id, res.mode)
    res.df.to_parquet(pq, index=False)
    mj.write_text(json.dumps(res.meta, default=str, indent=1))


def _cache_age_hours(series_id: str, mode: str) -> float | None:
    cached = _read_cache(series_id, mode)
    if cached is None:
        return None
    ts = cached.meta.get("fetched_at")
    if not ts:
        return None
    fetched = datetime.fromisoformat(ts)
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600


# ------------------------------------------------------------------ fetchers

def _validate_obs(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Basic data validation: parseable dates, numeric values, no impossible
    dates, sorted, deduplicated."""
    df = df.dropna(subset=["period", "value"]).copy()
    now_plus = pd.Timestamp.now() + pd.Timedelta(days=7)
    bad = (df["period"] < pd.Timestamp("1850-01-01")) | (df["period"] > now_plus)
    if bad.any():
        log.warning("%s: dropping %d observations with impossible dates", series_id, int(bad.sum()))
        df = df[~bad]
    df = df.sort_values("period").drop_duplicates(subset="period", keep="last")
    return df.reset_index(drop=True)


def _fetch_api_latest(series_id: str, api_key: str) -> pd.DataFrame:
    resp = _request(
        f"{FRED_API}/series/observations",
        {"series_id": series_id, "api_key": api_key, "file_type": "json", "limit": 100000},
    )
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["period", "value"])
    df["period"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return _validate_obs(df[["period", "value"]], series_id)


def _empty_release_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "period": pd.Series(dtype="datetime64[ns]"),
        "value": pd.Series(dtype="float64"),
        "release_date": pd.Series(dtype="datetime64[ns]"),
    })


def _first_release_query(
    series_id: str, api_key: str,
    obs_start: str | None, obs_end: str | None,
    delays: list[int] | None = None,
) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 4,  # initial release only
        "realtime_start": ALFRED_EPOCH,
        "realtime_end": ALFRED_HORIZON,
        "limit": 100000,
    }
    if obs_start:
        params["observation_start"] = obs_start
    if obs_end:
        params["observation_end"] = obs_end
    resp = _request(f"{FRED_API}/series/observations", params, delays=delays)
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return _empty_release_frame()
    df["period"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["realtime_start"], errors="coerce")
    return _validate_obs(df[["period", "value", "release_date"]], series_id)


def _fetch_api_first_release(series_id: str, api_key: str) -> pd.DataFrame:
    """ALFRED first-release values with their actual publication date.

    The initial-release extraction is expensive on FRED's side; for
    vintage-heavy series a full-history query can 504. Try one full query,
    then fall back to chunked observation windows.
    """
    try:
        return _first_release_query(series_id, api_key, None, None, delays=[2])
    except ConnectionError:
        log.warning("%s: full first-release query failed; retrying in chunks", series_id)
    edges = ["1962-01-01", "1985-01-01", "2000-01-01", "2010-01-01", "2018-01-01", None]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        b_end = (pd.Timestamp(b) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if b else None
        parts.append(_first_release_query(series_id, api_key, a, b_end))
    parts = [p for p in parts if not p.empty] or [_empty_release_frame()]
    out = pd.concat(parts, ignore_index=True)
    return _validate_obs(out, series_id)


def _fetch_csv_latest(series_id: str) -> pd.DataFrame:
    """Keyless fallback via the public fredgraph CSV endpoint."""
    resp = _request(FREDGRAPH_CSV, {"id": series_id})
    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected CSV for {series_id}")
    df.columns = ["period", "value"] + list(df.columns[2:])
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return _validate_obs(df[["period", "value"]], series_id)


def _fetch_api_meta(series_id: str, api_key: str) -> dict:
    resp = _request(
        f"{FRED_API}/series",
        {"series_id": series_id, "api_key": api_key, "file_type": "json"},
    )
    ser = resp.json().get("seriess", [])
    return ser[0] if ser else {}


# ------------------------------------------------------------------ assembly

def _availability_from_lag(df: pd.DataFrame, frequency: str, lag_days: int) -> pd.Series:
    return period_end(df["period"], frequency) + pd.Timedelta(days=lag_days)


def _availability_from_release(
    df: pd.DataFrame, frequency: str, lag_days: int
) -> pd.Series:
    """Availability = actual first-release date, except for 'backfill' rows.

    ALFRED only tracks vintages from some start date per series; observations
    older than the first vintage all carry that first vintage date as their
    realtime_start. Treat a release date far beyond the configured lag as
    backfill and fall back to period_end + lag for the timing (values remain
    earliest-vintage; residual revision bias documented).
    """
    pe = period_end(df["period"], frequency)
    lag_based = pe + pd.Timedelta(days=lag_days)
    gap = (df["release_date"] - pe).dt.days
    backfill_threshold = max(3 * lag_days, lag_days + 45)
    is_backfill = gap > backfill_threshold
    avail = df["release_date"].where(~is_backfill, lag_based)
    # A release date can never precede the data it describes by more than the
    # period itself; guard against metadata glitches.
    return avail.clip(lower=pe - pd.Timedelta(days=90))


def fetch_series(
    series_id: str,
    frequency: str,
    lag_days: int,
    want_vintage: bool,
    ttl_hours: float = 12.0,
    offline: bool = False,
) -> dict[str, SeriesResult]:
    """Fetch one FRED series in the modes needed.

    Returns dict with keys among {'latest', 'first_release'}; 'latest' is
    always present. Each SeriesResult.df has columns period/value/avail.
    """
    api_key = fred_api_key()
    out: dict[str, SeriesResult] = {}

    modes = ["latest"]
    # RECESSION_SKIP_VINTAGE=1 skips first-release downloads — prediction-only
    # refreshes use latest-vintage data anyway; training needs the vintages.
    skip_vintage = os.environ.get("RECESSION_SKIP_VINTAGE", "").strip() not in ("", "0")
    if want_vintage and api_key and not skip_vintage:
        modes.append("first_release")

    for mode in modes:
        cached = _read_cache(series_id, mode)
        age = _cache_age_hours(series_id, mode)
        if cached is not None and (offline or (age is not None and age < ttl_hours)):
            out[mode] = cached
            continue

        try:
            meta: dict = {}
            if mode == "first_release":
                df = _fetch_api_first_release(series_id, api_key)  # type: ignore[arg-type]
                df["avail"] = _availability_from_release(df, frequency, lag_days)
                df = df[["period", "value", "avail"]]
                source_detail = "alfred_first_release"
            else:
                if api_key:
                    df = _fetch_api_latest(series_id, api_key)
                    meta = _fetch_api_meta(series_id, api_key)
                    source_detail = "fred_api"
                else:
                    df = _fetch_csv_latest(series_id)
                    source_detail = "fredgraph_csv"
                df["avail"] = _availability_from_lag(df, frequency, lag_days)
                df = df[["period", "value", "avail"]]

            res = SeriesResult(
                series_id,
                mode,
                df,
                meta={
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source_detail": source_detail,
                    "n_obs": len(df),
                    "first_period": str(df["period"].min()) if len(df) else None,
                    "last_period": str(df["period"].max()) if len(df) else None,
                    "api_title": meta.get("title"),
                    "api_obs_start": meta.get("observation_start"),
                    "api_obs_end": meta.get("observation_end"),
                    "api_frequency": meta.get("frequency_short"),
                    "api_last_updated": meta.get("last_updated"),
                },
            )
            _write_cache(res)
            out[mode] = res
        except Exception as exc:
            log.error("Fetch failed for %s/%s: %s", series_id, mode, exc)
            if cached is not None:
                cached.fetch_error = str(exc)
                out[mode] = cached
            else:
                out[mode] = SeriesResult(
                    series_id, mode,
                    pd.DataFrame(columns=["period", "value", "avail"]),
                    meta={}, fetch_error=str(exc),
                )

    # If vintage was requested but no key is available, note the fallback.
    if want_vintage and "first_release" not in out and "latest" in out:
        out["latest"].meta.setdefault("vintage_fallback", True)
    return out
