"""Project paths, environment loading and logging setup."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.environ.get("RECESSION_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
CONFIG_PATH = PROJECT_ROOT / "config" / "indicators.yaml"
HISTORY_DB = DATA_DIR / "history.sqlite"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def fred_api_key() -> str | None:
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key or None


_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        level = os.environ.get("RECESSION_LOG_LEVEL", "INFO").upper()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(level=level, format=fmt)
        try:
            fh = logging.FileHandler(LOG_DIR / "recession.log")
            fh.setFormatter(logging.Formatter(fmt))
            logging.getLogger().addHandler(fh)
        except OSError:
            pass
        _LOG_CONFIGURED = True
    return logging.getLogger(name)
