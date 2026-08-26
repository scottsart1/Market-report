"""Local prediction history (SQLite). One row per successful refresh."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .paths import HISTORY_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    data_date TEXT NOT NULL,
    p15 REAL, p30 REAL, p45 REAL, p60 REAL, p90 REAL,
    p90_lo REAL, p90_hi REAL,
    model_version TEXT,
    model_name TEXT,
    calibration TEXT,
    train_end TEXT,
    dataset_version TEXT,
    n_indicators_ok INTEGER,
    n_indicators_missing INTEGER,
    features_json TEXT
);
"""


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(HISTORY_DB)
    con.execute(SCHEMA)
    try:  # schema migration: 1-year horizon added in v1.1
        con.execute("ALTER TABLE predictions ADD COLUMN p365 REAL")
    except sqlite3.OperationalError:
        pass
    return con


def save_prediction(pred: dict) -> None:
    con = _conn()
    with con:
        con.execute(
            """INSERT INTO predictions
               (ts_utc, data_date, p15, p30, p45, p60, p90, p365, p90_lo, p90_hi,
                model_version, model_name, calibration, train_end,
                dataset_version, n_indicators_ok, n_indicators_missing, features_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                str(pred["data_date"]),
                *(float(pred["probabilities"][h]) for h in (15, 30, 45, 60, 90)),
                float(pred["probabilities"][365]) if 365 in pred["probabilities"] else None,
                float(pred.get("band", {}).get(90, (None, None))[0] or 0) or None,
                float(pred.get("band", {}).get(90, (None, None))[1] or 0) or None,
                pred.get("model_version"),
                pred.get("model_name"),
                pred.get("calibration"),
                str(pred.get("train_end")),
                pred.get("dataset_version"),
                int(pred.get("n_indicators_ok", 0)),
                int(pred.get("n_indicators_missing", 0)),
                json.dumps(pred.get("key_features", {}), default=str),
            ),
        )
    con.close()


def load_history() -> pd.DataFrame:
    con = _conn()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY ts_utc", con)
    con.close()
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], format="ISO8601", utc=True)
        df["data_date"] = pd.to_datetime(df["data_date"])
    return df
