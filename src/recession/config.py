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
    "survey", "fiscal", "trade", "global", "composite",
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
    "survey": "Business Surveys",
    "fiscal": "Fiscal",
    "trade": "Trade & Dollar",
    "global": "Global",
    "composite": "Breadth Composites",
}

FREQ_DAYS = {"d": 1, "w": 7, "m": 31, "q": 92}
ALL_HORIZON_TAGS = (90, 365)


@dataclass(frozen=True)
class TransformSpec:
    code: str
    risk: int  # +1: higher value -> higher recession risk; -1: opposite
    desc: str
    horizons: tuple[int, ...] = ALL_HORIZON_TAGS  # which target models use it


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
    # first real publication of a re-estimated index: earlier values are
    # backcasts and are masked in training
    real_time_from: str | None = None
    # feature set routing: core features feed every member; the rest only
    # the constrained boosted model (see docs/v2_plan.md)
    core: bool = True
    replaces: str | None = None
    # a series that ended (kept as the early leg of a splice): never stale
    discontinued: bool = False
    # put ALFRED first-release levels on one base across rebasings (train mode)
    chain_link: bool = True
    # source == "composite": breadth measures computed on the as-of panel
    composite_kind: str | None = None   # breadth_pctile | rule_share | inv_share | zmean
    members: tuple[str, ...] = ()       # feature names (ID__code)
    rules: tuple[dict, ...] = ()        # rule_share: {feature, op, value}

    @property
    def fetched(self) -> bool:
        return self.source in FETCHED_SOURCES

    def feature_names(self) -> list[str]:
        return [f"{self.id}__{t.code}" for t in self.transforms]


@dataclass(frozen=True)
class IndicatorConfig:
    indicators: tuple[Indicator, ...]
    label_series: str = "USREC"
    composites: tuple[dict, ...] = ()  # breadth composites (see composites.py)

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

    def features_for(self, horizon: int, core_only: bool = False) -> list[str]:
        """Feature names tagged for a target horizon (90 or 365), optionally
        only those routed to every member."""
        return [name for name, (ind, t) in self.feature_specs().items()
                if horizon in t.horizons and (ind.core or not core_only)]


def _parse_transform(t: dict) -> TransformSpec:
    hz = t.get("horizons")
    return TransformSpec(
        code=t["t"], risk=int(t.get("risk", 1)), desc=t.get("desc", t["t"]),
        horizons=tuple(int(h) for h in hz) if hz else ALL_HORIZON_TAGS,
    )


def _parse_indicator(d: dict) -> Indicator:
    transforms = tuple(_parse_transform(t) for t in d.get("transforms", []) or [])
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
        real_time_from=d.get("real_time_from"),
        core=bool(d.get("core", True)),
        replaces=d.get("replaces"),
        discontinued=bool(d.get("discontinued", False)),
        chain_link=bool(d.get("chain_link", True)),
        composite_kind=d.get("kind") if d.get("source") == "composite" else None,
        members=tuple(d.get("members", []) or []),
        rules=tuple(dict(r) for r in (d.get("rules", []) or [])),
    )


def _parse_composite(d: dict) -> Indicator:
    """A composite entry (top-level `composites:` list) is an indicator with
    source 'composite', computed on the daily as-of panel."""
    d = dict(d)
    d.setdefault("source", "composite")
    d.setdefault("frequency", "d")
    d.setdefault("publication_lag_days", 0)
    d.setdefault("vintage", False)
    d.setdefault("tier", 1)
    d.setdefault("license", "public")
    d.setdefault("category", "composite")
    d.setdefault("name", d["id"])
    return _parse_indicator(d)


@lru_cache(maxsize=4)
def load_config(path: str | Path = CONFIG_PATH) -> IndicatorConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    inds = tuple(_parse_indicator(d) for d in raw["indicators"])
    composites = tuple(dict(c) for c in raw.get("composites", []) or [])
    inds = inds + tuple(_parse_composite(c) for c in composites)
    label_id = raw.get("labels", [{"id": "USREC"}])[0]["id"]

    ids = [i.id for i in inds]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate indicator ids in config")
    for i in inds:
        if i.category not in CATEGORIES:
            raise ValueError(f"Unknown category {i.category!r} for {i.id}")
        if i.source not in (*FETCHED_SOURCES, "derived", "composite"):
            raise ValueError(f"Unknown source {i.source!r} for {i.id}")
        if i.source == "composite":
            if i.composite_kind not in ("breadth_pctile", "rule_share", "inv_share", "zmean"):
                raise ValueError(f"Composite {i.id}: unknown kind {i.composite_kind!r}")
            if i.composite_kind == "rule_share" and not i.rules:
                raise ValueError(f"Composite {i.id} needs rules")
            if i.composite_kind != "rule_share" and not i.members:
                raise ValueError(f"Composite {i.id} needs members")
        if i.source == "derived" and not i.inputs:
            raise ValueError(f"Derived indicator {i.id} needs inputs")
        if i.source == "url" and not (i.url and i.url_value_col):
            raise ValueError(f"URL indicator {i.id} needs url and url_value_col")
        if i.source == "url" and i.vintage:
            raise ValueError(f"URL indicator {i.id} cannot be vintage: no ALFRED history")
        # The recession label must never appear as a predictor input.
        if label_id in (i.id, *i.inputs):
            raise ValueError(f"Label series {label_id} may not be used as an indicator")
        for t in i.transforms:
            if not set(t.horizons) <= set(ALL_HORIZON_TAGS):
                raise ValueError(f"{i.id}: horizons must be among {ALL_HORIZON_TAGS}")
    return IndicatorConfig(indicators=inds, label_series=label_id, composites=composites)
