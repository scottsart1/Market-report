"""Experiment switches for A/B tests inside the walk-forward backtest.

The experiment harness (scripts/experiment.py) runs the same backtest code
under different settings by setting RECESSION_VARIANTS="key=value;key2=value2".
Modeling code asks ``variant("name", default)``; with the variable unset every
call returns its default, so the committed defaults ARE the production
configuration and a variant is only ever a deliberate deviation from it.
"""
from __future__ import annotations

import os

ENV_VAR = "RECESSION_VARIANTS"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def active_variants() -> dict[str, str]:
    """Parsed contents of RECESSION_VARIANTS (empty when unset)."""
    raw = os.environ.get(ENV_VAR, "").strip()
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def variant(key: str, default):
    """Value of switch `key`, coerced to the type of `default`."""
    raw = active_variants().get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        low = raw.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"Variant {key}={raw!r} is not a boolean")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw
