"""Production prediction: refresh data, run the trained bundle, explain,
persist. Run:  python -m recession.predict [--offline] [--no-save]
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .categories import category_scores, category_summary, oriented_percentiles
from .config import CATEGORY_LABELS, load_config
from .explain import substitution_attribution, top_factors
from .fred_client import period_end
from .history import save_prediction
from .horizon import HazardTermStructure
from .labels import HORIZONS, build_recession_calendar
from .paths import get_logger
from .pipeline import (
    build_panel, build_series_frames, data_health, fetch_all,
    refresh_summary, weekly_grid,
)
from .train import BUNDLE_PATH
from .uncertainty import bootstrap_band

log = get_logger(__name__)

RISK_BANDS = [
    (0.05, "Very Low"), (0.10, "Low"), (0.20, "Moderate"),
    (0.35, "Elevated"), (0.55, "High"), (1.01, "Extreme"),
]


def classify(p: float) -> str:
    for cut, name in RISK_BANDS:
        if p < cut:
            return name
    return "Extreme"


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {BUNDLE_PATH}. Run `python -m recession.train` first."
        )
    return joblib.load(BUNDLE_PATH)


def bundle_p90_raw_cal(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw and calibrated 90-day probability from the production model."""
    prod = bundle["production_model"]
    if prod == "D_ensemble":
        ps = []
        for m in bundle["ensemble_members"]:
            raw = bundle["models"][m].predict_proba(X)[:, 1]
            ps.append(bundle["calibrators"][m].transform(raw))
        cal = np.mean(ps, axis=0)
        return cal, cal  # ensemble is defined in calibrated space
    raw = bundle["models"][prod].predict_proba(X)[:, 1]
    return raw, bundle["calibrators"][prod].transform(raw)


def bundle_predictor(bundle: dict):
    def fn(X: pd.DataFrame) -> np.ndarray:
        return bundle_p90_raw_cal(bundle, X)[1]
    return fn


def bundle_long_cal(bundle: dict, X: pd.DataFrame) -> np.ndarray | None:
    """Calibrated 1-year onset probability (mean of calibrated members),
    before the P90 monotonicity floor. None for pre-1.1 bundles."""
    if "models_long" not in bundle:
        return None
    ps = []
    for m, model in bundle["models_long"].items():
        raw = model.predict_proba(X)[:, 1]
        ps.append(bundle["calibrators_long"][m].transform(raw))
    return np.mean(ps, axis=0)


def bundle_score_predictor(bundle: dict):
    """Smooth pre-calibration ensemble probability, used for attribution.

    Isotonic calibrators are step functions: near a plateau every small
    feature substitution maps to a zero delta, which would make explanations
    silently empty. The raw member-average is smooth, and calibration is
    monotone, so signs and rankings of contributions carry through to the
    final probability. Displayed as 'model-score' contributions.
    """
    prod = bundle["production_model"]
    members = bundle["ensemble_members"] if prod == "D_ensemble" else [prod]

    def fn(X: pd.DataFrame) -> np.ndarray:
        return np.mean([bundle["models"][m].predict_proba(X)[:, 1] for m in members], axis=0)
    return fn


def _term_structure(bundle: dict):
    ts = HazardTermStructure()
    ts.theta_ = dict(bundle["theta"])
    return ts


def indicator_snapshot(cfg, frames_now: dict, panel_weekly: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Section-4 table: latest raw value, change, z-score, percentile, signal, age."""
    oriented = oriented_percentiles(panel_weekly, cfg)
    last_oriented = oriented.iloc[-1] if len(oriented) else pd.Series(dtype=float)
    rows = []
    for ind in cfg.indicators:
        if not ind.enabled or not ind.transforms or ind.id not in frames_now:
            continue
        df = frames_now[ind.id].dropna(subset=["value"])
        if df.empty:
            continue
        v = df["value"]
        latest, prev = float(v.iloc[-1]), float(v.iloc[-2]) if len(v) > 1 else np.nan
        tail = v.tail(520 if ind.frequency == "w" else (2520 if ind.frequency == "d" else 120))
        z = (latest - tail.mean()) / tail.std() if tail.std() > 0 else np.nan
        pctile = float((tail <= latest).mean() * 100)
        feat_names = [f"{ind.id}__{t.code}" for t in ind.transforms]
        sig_vals = [last_oriented.get(f) for f in feat_names if f in last_oriented.index]
        sig_vals = [s for s in sig_vals if pd.notna(s)]
        stress = float(np.mean(sig_vals) * 100) if sig_vals else np.nan
        signal = ("risk ↑" if stress >= 70 else "risk ↓" if stress <= 30 else "neutral") if not np.isnan(stress) else "n/a"
        pe = period_end(pd.Series([df["period"].iloc[-1]]), ind.frequency).iloc[0]
        rows.append({
            "Indicator": ind.name,
            "Category": CATEGORY_LABELS.get(ind.category, ind.category),
            "Latest": latest,
            "Change": latest - prev if not np.isnan(prev) else np.nan,
            "Z-score (10y)": round(float(z), 2) if pd.notna(z) else None,
            "Percentile (10y)": round(pctile, 1),
            "Stress (0-100)": round(stress, 1) if not np.isnan(stress) else None,
            "Signal": signal,
            "Data age (days)": int((now.normalize() - pe).days),
            "Tier": ind.tier,
        })
    return pd.DataFrame(rows)


def run_prediction(ttl_hours: float = 6.0, offline: bool = False, save: bool = True) -> dict:
    bundle = load_bundle()
    cfg = load_config()
    fetched = fetch_all(cfg, ttl_hours=ttl_hours, offline=offline)
    summary = refresh_summary(fetched)

    frames_now, unavailable = build_series_frames(cfg, fetched, mode="now")
    panel_now = build_panel(cfg, fetched, mode="now")
    Xw = weekly_grid(panel_now)
    now = pd.Timestamp.today().normalize()

    feature_cols = bundle["feature_cols"]
    X_all = panel_now.reindex(columns=feature_cols)

    # Retrospective daily curve (current model, latest data) for deltas/chart.
    tail = X_all.tail(1100)  # ~4 years of business days
    _, p90_daily = bundle_p90_raw_cal(bundle, tail)
    ts = _term_structure(bundle)
    curve = pd.DataFrame(
        {f"p{h}": v for h, v in ts.term_structure(p90_daily).items()}, index=tail.index
    )
    display_horizons = list(HORIZONS)
    p_long_daily = bundle_long_cal(bundle, tail)
    if p_long_daily is not None:
        lh = int(bundle.get("long_horizon", 365))
        # monotonicity floor: the 1-year probability can never sit below P90
        curve[f"p{lh}"] = np.maximum(p_long_daily, curve["p90"].to_numpy())
        display_horizons.append(lh)

    x_today = X_all.iloc[[-1]]
    data_date = X_all.index[-1]
    p90_today = float(p90_daily[-1])
    probs = {h: float(curve[f"p{h}"].iloc[-1]) for h in display_horizons}

    def deltas_vs(days: int) -> dict[int, float]:
        cutoff = data_date - pd.Timedelta(days=days)
        past = curve[curve.index <= cutoff]
        if past.empty:
            return {h: np.nan for h in display_horizons}
        return {h: probs[h] - float(past[f"p{h}"].iloc[-1]) for h in display_horizons}

    changes = {"1d": deltas_vs(1), "7d": deltas_vs(7), "30d": deltas_vs(30)}

    # ------------------------------------------------------------ explanation
    score_fn = bundle_score_predictor(bundle)
    medians = bundle["train_medians"].reindex(feature_cols)
    attr_level = substitution_attribution(score_fn, x_today.iloc[0], medians, feature_cols)
    factors = top_factors(attr_level, x_today.iloc[0], cfg, n=8)

    week_ago_rows = X_all[X_all.index <= data_date - pd.Timedelta(days=7)]
    factors_change = {}
    if len(week_ago_rows):
        attr_change = substitution_attribution(
            score_fn, x_today.iloc[0], week_ago_rows.iloc[-1], feature_cols
        )
        factors_change = top_factors(attr_change, x_today.iloc[0], cfg, n=5, min_pp=0.02)

    # ------------------------------------------------------------ uncertainty
    prod = bundle["production_model"]
    cal_b = bundle["calibrators"]["B_elastic_net"]
    band = bootstrap_band(
        bundle.get("boot_models", []), x_today,
        calibrate_fn=lambda p: cal_b.transform(p),
        term_structure_fn=ts.term_structure,
    )
    # Center the bootstrap band on the production probability (band width from
    # the interpretable member; documented approximation when prod != B).
    if band and prod != "B_elastic_net":
        shift = probs[90] - float(np.mean(band[90]))
        band = {h: (max(0.0, lo + shift), min(1.0, hi + shift)) for h, (lo, hi) in band.items()}

    # 1-year band from the dedicated long-horizon bootstrap
    if p_long_daily is not None and bundle.get("boot_models_long"):
        cal_bl = bundle["calibrators_long"]["B_elastic_net"]
        ps = cal_bl.transform(np.array(
            [m.predict_proba(x_today)[0, 1] for m in bundle["boot_models_long"]]
        ))
        lo, hi = (float(q) for q in np.quantile(ps, (0.10, 0.90)))
        shift = probs[lh] - (lo + hi) / 2
        band[lh] = (max(0.0, lo + shift), min(1.0, hi + shift))

    # ------------------------------------------------------- categories/health
    scores = category_scores(Xw.reindex(columns=feature_cols), cfg)
    cat_summary = category_summary(scores)
    health = data_health(cfg, fetched, now)
    snapshot = indicator_snapshot(cfg, frames_now, Xw.reindex(columns=feature_cols), now)

    n_ok = int((health["status"] == "fresh").sum())
    n_bad = int(len(health) - n_ok)
    tier1 = health[health["tier"] == 1]
    tier1_fresh = float((tier1["status"] == "fresh").mean()) if len(tier1) else 0.0
    band_width = (band[90][1] - band[90][0]) if 90 in band else np.nan
    confidence = (
        "High" if tier1_fresh >= 0.9 and summary["n_fetch_errors"] == 0 and (np.isnan(band_width) or band_width < 0.25)
        else "Low" if tier1_fresh < 0.6 or summary["n_fetch_errors"] > 3
        else "Medium"
    )

    usrec = fetched[cfg.label_series]["latest"]
    cal_obj = build_recession_calendar(usrec.df) if usrec.ok else None
    currently_in_recession = (
        bool(cal_obj.in_recession(pd.DatetimeIndex([min(data_date, cal_obj.last_label_date)])).iloc[0])
        if cal_obj else False
    )

    key_features = {
        f: (None if pd.isna(x_today.iloc[0][f]) else round(float(x_today.iloc[0][f]), 5))
        for f in attr_level.abs().sort_values(ascending=False).head(15).index
    }

    pred = {
        "data_date": str(data_date.date()),
        "generated_at": summary["refreshed_at"],
        "probabilities": probs,
        "p90_raw_base": p90_today,
        "changes": changes,
        "classification": {h: classify(probs[h]) for h in display_horizons},
        "overall_classification": classify(probs[90]),
        "horizons": display_horizons,
        "band": band,
        "confidence": confidence,
        "confidence_detail": {
            "tier1_fresh_share": round(tier1_fresh, 3),
            "fetch_errors": summary["n_fetch_errors"],
            "band_width_90d": None if np.isnan(band_width) else round(float(band_width), 4),
        },
        "factors": factors,
        "factors_change_7d": factors_change,
        "model_version": bundle["model_version"],
        "model_name": prod,
        "calibration": bundle["calibrator_choice"].get(prod, "members"),
        "train_end": bundle["train_end"],
        "dataset_version": summary["dataset_version"],
        "n_indicators_ok": n_ok,
        "n_indicators_missing": n_bad + len(unavailable),
        "currently_in_recession_labeled": currently_in_recession,
        "key_features": key_features,
        "theta": bundle["theta"],
    }

    if save:
        try:
            save_prediction(pred)
        except Exception as exc:
            log.error("Could not save prediction history: %s", exc)

    return {
        "prediction": pred,
        "curve": curve,
        "category_scores": scores,
        "category_summary": cat_summary,
        "health": health,
        "snapshot": snapshot,
        "refresh": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--ttl-hours", type=float, default=6.0)
    args = ap.parse_args()
    out = run_prediction(ttl_hours=args.ttl_hours, offline=args.offline, save=not args.no_save)
    p = out["prediction"]
    print(json.dumps({
        "data_date": p["data_date"],
        "probabilities_pct": {f"{h}d": round(100 * v, 2) for h, v in p["probabilities"].items()},
        "band_90d_pct": [round(100 * b, 1) for b in p["band"].get(90, ())],
        "classification": p["overall_classification"],
        "confidence": p["confidence"],
        "model": f"{p['model_name']} v{p['model_version']} (cal={p['calibration']})",
        "top_up": [f["text"] for f in p["factors"]["increasing"][:5]],
        "top_down": [f["text"] for f in p["factors"]["decreasing"][:5]],
    }, indent=2))


if __name__ == "__main__":
    main()
