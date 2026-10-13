"""Load and validate the indicator configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import CONFIG_PATH

CATEGORIES = [
    "rates", "policy", "credit", "markets", "labor",
    "real_activity", "housing", "consumer", "inflation", "alt_hf",
]

CATEGORY_LABELS = {
    "rates": "Yield Curve / Rates",
    "policy": "Monetary Policy",
    "credit": "Credit & Financial Stress",
    "markets": "Financial Markets",
    "labor": "Labor Market",
    "real_activity": "Real Activity",
    "housing": "Housing",
    "consumer": "Consumer",
    "inflation": "Inflation",
    "alt_hf": "Alternative / High Frequency",
}

FREQ_DAYS = {"d": 1, "w": 7, "m": 31, "q": 92}


@dataclass(frozen=True)
class TransformSpec:
    code: str
    risk: int  # +1: higher value -> higher recession risk; -1: opposite
    desc: str


FETCHED_SOURCES = ("fred", "url")  # sources that are downloaded (vs derived)


@dataclass(frozen=True)
class Indicator:
    id: str
    name: str
    category: str
    source: str  # "fred" | "url" | "derived"
    frequency: str  # "d" | "w" | "m" | "q"
    publication_lag_days: int
    vintage: bool
    tier: int
    license: str
    transforms: tuple[TransformSpec, ...] = ()
    derive: str | None = None
    inputs: tuple[str, ...] = ()
    derive_params: dict = field(default_factory=dict)  # e.g. splice switch_date
    enabled: bool = True
    min_history_start: str | None = None
    # source == "url": a public CSV/XLSX file (e.g. a Fed research dataset)
    url: str | None = None
    url_date_col: str = "date"
    url_value_col: str | None = None
    url_sheet: str | None = None

    @property
    def fetched(self) -> bool:
        return self.source in FETCHED_SOURCES

    def feature_names(self) -> list[str]:
        return [f"{self.id}__{t.code}" for t in self.transforms]


@dataclass(frozen=True)
class IndicatorConfig:
    indicators: tuple[Indicator, ...]
    label_series: str = "USREC"

    def by_id(self) -> dict[str, Indicator]:
        return {i.id: i for i in self.indicators}

    def fred_ids(self) -> list[str]:
        return [i.id for i in self.indicators if i.source == "fred" and i.enabled]

    def feature_specs(self) -> dict[str, tuple[Indicator, TransformSpec]]:
        """feature name -> (indicator, transform spec)."""
        out: dict[str, tuple[Indicator, TransformSpec]] = {}
        for ind in self.indicators:
            if not ind.enabled:
                continue
            for t in ind.transforms:
                out[f"{ind.id}__{t.code}"] = (ind, t)
        return out


def _parse_indicator(d: dict) -> Indicator:
    transforms = tuple(
        TransformSpec(code=t["t"], risk=int(t.get("risk", 1)), desc=t.get("desc", t["t"]))
        for t in d.get("transforms", []) or []
    )
    return Indicator(
        id=d["id"],
        name=d["name"],
        category=d["category"],
        source=d["source"],
        frequency=d["frequency"],
        publication_lag_days=int(d.get("publication_lag_days", 1)),
        vintage=bool(d.get("vintage", False)),
        tier=int(d.get("tier", 2)),
        license=d.get("license", "public"),
        transforms=transforms,
        derive=d.get("derive"),
        inputs=tuple(d.get("inputs", []) or []),
        derive_params=dict(d.get("derive_params", {}) or {}),
        enabled=bool(d.get("enabled", True)),
        min_history_start=d.get("min_history_start"),
        url=d.get("url"),
        url_date_col=d.get("url_date_col", "date"),
        url_value_col=d.get("url_value_col"),
        url_sheet=d.get("url_sheet"),
    )


@lru_cache(maxsize=4)
def load_config(path: str | Path = CONFIG_PATH) -> IndicatorConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    inds = tuple(_parse_indicator(d) for d in raw["indicators"])
    label_id = raw.get("labels", [{"id": "USREC"}])[0]["id"]

    ids = [i.id for i in inds]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate indicator ids in config")
    for i in inds:
        if i.category not in CATEGORIES:
            raise ValueError(f"Unknown category {i.category!r} for {i.id}")
        if i.source not in (*FETCHED_SOURCES, "derived"):
            raise ValueError(f"Unknown source {i.source!r} for {i.id}")
        if i.source == "derived" and not i.inputs:
            raise ValueError(f"Derived indicator {i.id} needs inputs")
        if i.source == "url" and not (i.url and i.url_value_col):
            raise ValueError(f"URL indicator {i.id} needs url and url_value_col")
        if i.source == "url" and i.vintage:
            raise ValueError(f"URL indicator {i.id} cannot be vintage: no ALFRED history")
        # The recession label must never appear as a predictor input.
        if label_id in (i.id, *i.inputs):
            raise ValueError(f"Label series {label_id} may not be used as an indicator")
    return IndicatorConfig(indicators=inds, label_series=label_id)
