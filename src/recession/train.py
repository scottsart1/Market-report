"""End-to-end training: fetch -> point-in-time features -> backtest ->
production fit -> artifacts.

Run:  python -m recession.train [--fast] [--offline] [--ttl-hours H]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import MODEL_VERSION
from .backtest import (
    LONG_H, TRAIN_START, fit_fold_calibrators, inner_split, run_backtest,
    run_long_backtest,
)
from .config import load_config
from .horizon import HazardTermStructure
from .labels import HORIZONS, build_recession_calendar, make_labels
from .models import build_models
from .paths import MODELS_DIR, get_logger
from .pipeline import build_panel, dataset_version, fetch_all, weekly_grid
from .uncertainty import fit_bootstrap_models

log = get_logger(__name__)

BUNDLE_PATH = MODELS_DIR / "production_bundle.joblib"
RESULTS_PATH = MODELS_DIR / "backtest_results.json"
OOS_PATH = MODELS_DIR / "oos_predictions.parquet"
CARD_PATH = MODELS_DIR / "model_card.md"


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_training_data(ttl_hours: float = 12.0, offline: bool = False):
    """Fetch everything and assemble the weekly training matrix + labels."""
    cfg = load_config()
    fetched = fetch_all(cfg, ttl_hours=ttl_hours, offline=offline)

    usrec = fetched[cfg.label_series]["latest"]
    if not usrec.ok:
        raise RuntimeError("Cannot proceed: NBER label series unavailable and not cached")
    cal = build_recession_calendar(usrec.df)

    panel_train = build_panel(cfg, fetched, mode="train")
    Xw = weekly_grid(panel_train)
    labels = make_labels(Xw.index, cal, HORIZONS)
    return cfg, fetched, cal, panel_train, Xw, labels


def train(fast: bool = False, ttl_hours: float = 12.0, offline: bool = False) -> dict:
    cfg, fetched, cal, panel_train, Xw, labels = build_training_data(ttl_hours, offline)
    feature_cols = [c for c in Xw.columns]
    log.info("Weekly training panel: %d rows x %d features", *Xw.shape)

    # ------------------------------------------------ walk-forward backtest
    bt = run_backtest(Xw, labels, feature_cols)
    production = bt.selection["production_model"]
    log.info("Production model selected: %s", production)

    # dedicated 1-year onset classifier, same validation discipline
    long_bt = run_long_backtest(Xw, labels, feature_cols)
    log.info("1-year model backtest done (ensemble Brier %.4f)",
             long_bt["metrics"]["ensemble_1y"]["brier"])

    # -------------------------------------- production fit ("fold 8")
    # The production predictor uses EXACTLY the construction the walk-forward
    # validated seven times: model fitted on the first 75% of eligible
    # history, calibrator fitted on the purged most-recent tail, hazard
    # exponents from pooled out-of-sample calibrated predictions. A
    # full-sample refit was rejected: in-sample the boosted model memorizes
    # the most recent no-recession regime and its score scale detaches from
    # everything the backtest validated.
    eligible = labels["eligible"].astype(bool)
    tr = (Xw.index >= TRAIN_START) & eligible.to_numpy()
    X_tr = Xw[tr]
    train_end = X_tr.index.max()
    train_idx = Xw.index[tr]
    fit_idx, cal_idx = inner_split(train_idx)
    y90_ser = labels["y90"]

    models = build_models(feature_cols)
    fitted, calibrators = {}, {}
    for name in ("A_yield_curve", "B_elastic_net", "C_grad_boost", "bl_sahm", "bl_nfci", "bl_constant"):
        inner, cals = fit_fold_calibrators(models[name], Xw, y90_ser, labels, fit_idx, cal_idx)
        fitted[name] = inner
        method = bt.calibrator_choice.get(name, "none")
        calibrators[name] = cals.get(method, cals["none"])

    theta = HazardTermStructure()
    theta.theta_ = dict(bt.theta)

    # production 1-year models: identical fold-8 construction on the
    # long-eligible rows (which end a year before the labeled edge)
    tr_long = (Xw.index >= TRAIN_START) & labels["eligible_long"].to_numpy()
    long_idx = Xw.index[tr_long]
    fit_idx_l, cal_idx_l = inner_split(long_idx)
    y365_ser = labels[f"y{LONG_H}"]
    models_long, calibrators_long = {}, {}
    for name in ("B_elastic_net", "C_grad_boost"):
        inner, cals = fit_fold_calibrators(
            build_models(feature_cols)[name], Xw, y365_ser, labels, fit_idx_l, cal_idx_l
        )
        models_long[name] = inner
        method = long_bt["calibrator_choice"].get(name, "none")
        calibrators_long[name] = cals.get(method, cals["none"])

    # ------------------------------------------------ bootstrap uncertainty
    boot_models, boot_models_long = [], []
    if not fast:
        boot_models = fit_bootstrap_models(
            Xw.loc[fit_idx], y90_ser.loc[fit_idx].to_numpy(),
            labels.loc[fit_idx, "next_onset"], feature_cols,
            fixed_C=fitted["B_elastic_net"].C_,
        )
        boot_models_long = fit_bootstrap_models(
            Xw.loc[fit_idx_l], y365_ser.loc[fit_idx_l].to_numpy(),
            labels.loc[fit_idx_l, "next_onset"], feature_cols,
            fixed_C=models_long["B_elastic_net"].C_,
        )

    train_medians = X_tr.median(numeric_only=True)

    bundle = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_model": production,
        "ensemble_members": ["B_elastic_net", "C_grad_boost"],
        "models": fitted,
        "calibrators": calibrators,
        "calibrator_choice": bt.calibrator_choice,
        "models_long": models_long,
        "calibrators_long": calibrators_long,
        "calibrator_choice_long": long_bt["calibrator_choice"],
        "long_horizon": LONG_H,
        "theta": bt.theta,
        "feature_cols": feature_cols,
        "train_medians": train_medians,
        "train_start": str(TRAIN_START.date()),
        "train_end": str(train_end.date()),
        "fit_end": str(fit_idx.max().date()),
        "cal_start": str(cal_idx.min().date()) if len(cal_idx) else None,
        "n_train_rows": int(tr.sum()),
        "n_fit_rows": int(len(fit_idx)),
        "n_cal_rows": int(len(cal_idx)),
        "n_cal_events": int(labels.loc[cal_idx, "next_onset"][labels.loc[cal_idx, "y90"] == 1].nunique()) if len(cal_idx) else 0,
        "n_train_events": int(labels.loc[tr, "next_onset"][labels.loc[tr, "y90"] == 1].nunique()),
        "dataset_version": dataset_version(fetched),
        "boot_models": boot_models,
        "boot_models_long": boot_models_long,
        "selection": bt.selection,
        "horizons": list(HORIZONS),
        "label_convention": (
            "NBER monthly chronology (FRED USREC); onset = first day of the first "
            "recession month; targets are cumulative onset-within-h-days windows."
        ),
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    log.info("Saved production bundle -> %s", BUNDLE_PATH)

    # ------------------------------------------------------- persist results
    results = {
        "model_version": MODEL_VERSION,
        "created_at": bundle["created_at"],
        "selection": bt.selection,
        "calibrator_choice": bt.calibrator_choice,
        "theta": bt.theta,
        "metrics": bt.metrics,
        "horizon_metrics": bt.horizon_metrics,
        "long_model": {
            "metrics": long_bt["metrics"],
            "calibrator_choice": long_bt["calibrator_choice"],
            "events": long_bt["events"],
            "detect_threshold": long_bt["detect_threshold"],
        },
        "events": bt.events,
        "false_alarms": bt.false_alarms,
        "reliability": bt.reliability,
        "fold_info": bt.fold_info,
        "n_features": len(feature_cols),
        "train_rows": int(tr.sum()),
    }
    RESULTS_PATH.write_text(json.dumps(results, default=_json_default, indent=1))

    oos_out = bt.oos[
        ["fold", "eligible", "in_recession"]
        + [f"y{h}" for h in HORIZONS]
        + [c for c in bt.oos.columns if c.startswith(("p_raw_", "p_cal_"))]
        + [f"p{h}" for h in HORIZONS]
    ].copy()
    # served 1-year probability: calibrated long ensemble, floored at P90
    p_long = long_bt["oos"]["p_cal_long"].reindex(oos_out.index).to_numpy(dtype=float)
    p90 = oos_out["p90"].to_numpy(dtype=float)
    oos_out[f"p{LONG_H}"] = np.where(np.isnan(p_long), np.nan, np.maximum(p_long, p90))
    oos_out[f"y{LONG_H}"] = labels[f"y{LONG_H}"].reindex(oos_out.index)
    oos_out["eligible_long"] = labels["eligible_long"].reindex(oos_out.index)
    oos_out.index.name = "date"
    oos_out.to_parquet(OOS_PATH)

    write_model_card(bundle, results, cal)
    log.info("Training complete.")
    return {"bundle": bundle, "results": results}


def write_model_card(bundle: dict, results: dict, cal) -> None:
    m = results["metrics"]
    prod = bundle["production_model"]

    def _row(name):
        blk = m.get(name, {}).get("pooled_cal", {})
        return (
            f"| {name} | {blk.get('brier', float('nan')):.4f} | "
            f"{blk.get('log_loss', float('nan')):.4f} | {blk.get('roc_auc', float('nan')):.3f} | "
            f"{blk.get('pr_auc', float('nan')):.3f} | {blk.get('ece', float('nan')):.4f} |"
        )

    prod_events = results["events"].get(prod, [])
    events_lines = [
        f"| {e['onset']} | {e['max_p_90d_before']:.2f} | {e['lead_days'] if e['lead_days'] is not None else '—'} |"
        for e in prod_events
    ]
    alarms = results["false_alarms"].get(prod, [])
    card = f"""# Model Card — U.S. Recession Probability Model v{bundle['model_version']}

**Target.** Probability that the U.S. economy *enters* an NBER recession within
15/30/45/60/90 days, conditional on not currently being in one. Ground truth is
the NBER chronology via FRED `USREC`. Onset convention: the first calendar day
of the first recession month. NBER dating is monthly; sub-monthly horizon
distinctions come from the model's hazard term structure, not from the labels.

**Training period.** Weekly (Friday) observations {bundle['train_start']} to
{bundle['train_end']} — {bundle['n_train_rows']} rows, {bundle['n_train_events']}
recession onsets. Recession events observed: {', '.join(str(o.date()) for o in cal.onsets if o >= pd.Timestamp('1968-01-01'))}.

**Features.** {len(bundle['feature_cols'])} transformations across yield curve,
monetary policy, credit, equity markets, labor, real activity, housing,
consumer, inflation, and non-financial/high-frequency categories (see
`config/indicators.yaml`). Historical features are point-in-time: ALFRED
first-release values where vintages exist, and configured publication lags for
release timing everywhere.

**Methodology.** Base classifier for the 90-day onset window; probability
calibration selected out-of-sample among none/Platt/isotonic
({bundle['calibrator_choice']}); horizons mapped by a fitted hazard power law
P(h) = 1 - (1 - P90)^theta_h with monotone theta ({{
{', '.join(f"{h}: {t:.3f}" for h, t in bundle['theta'].items())} }}), which
enforces P15 <= P30 <= P45 <= P60 <= P90 structurally. Production model:
**{prod}**, selected by average rank across Brier, log loss, calibration
error, PR-AUC, event-detection rate and false alarms on pooled walk-forward
out-of-sample predictions. The production predictor repeats the exact
construction the walk-forward validated: model fitted on eligible history
through {bundle['fit_end']}, calibrator fitted on the purged tail
{bundle['cal_start']} → {bundle['train_end']} ({bundle['n_cal_events']}
recession event(s) in the calibration window).

**Out-of-sample performance** (pooled walk-forward OOS, calibrated inside each fold):

| Model | Brier | Log loss | ROC AUC | PR AUC | ECE |
|---|---|---|---|---|---|
{chr(10).join(_row(n) for n in ["A_yield_curve", "B_elastic_net", "C_grad_boost", "D_ensemble", "bl_constant", "bl_sahm", "bl_nfci"] if n in m)}

**1-year horizon.** A dedicated classifier (same architecture and validation,
purge widened to 455 days) serves the 365-day probability; it is floored at
P90 so the full horizon chain stays monotone. Out-of-sample (calibrated):
Brier {results['long_model']['metrics']['ensemble_1y']['brier']:.4f},
ROC AUC {results['long_model']['metrics']['ensemble_1y'].get('roc_auc', float('nan')):.3f},
base rate {results['long_model']['metrics']['ensemble_1y']['base_rate']:.3f};
{sum(e['detected'] for e in results['long_model']['events'])}/{len(results['long_model']['events'])}
recessions reached P1y >= {results['long_model']['detect_threshold']:.2f} in the prior year.

**Recession-by-recession (out-of-sample, production model)**

| Onset | Max P90 in prior 90d | Lead days (first P>=0.20) |
|---|---|---|
{chr(10).join(events_lines)}

**False alarms** (sustained P90 >= 0.30 for 28+ days, no onset within 270d):
{len(alarms)} episode(s) — {', '.join(f"{a['start']}..{a['end']} (max {a['max_p']:.2f})" for a in alarms) or 'none'}.

**Limitations.**
- Only ~8 usable U.S. recessions exist; uncertainty bands (block bootstrap)
  are wide and honest — treat point probabilities as ranges.
- NBER dating is monthly and announced with long delays; recent months could
  be re-labeled, and 15/30/45-day granularity is a modeling convention.
- Where ALFRED vintages don't exist (or without an API key), first-release
  values are approximated by later vintages: residual revision look-ahead is
  possible and flagged in Data Health.
- The 2020 COVID recession was an exogenous shock no macro-financial model
  anticipates at 90 days; treat that fold accordingly.
- Structural change (e.g., post-2020 labor dynamics) can degrade any model
  trained on 55 years of history.
"""
    CARD_PATH.write_text(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip bootstrap uncertainty")
    ap.add_argument("--offline", action="store_true", help="use cache only")
    ap.add_argument("--ttl-hours", type=float, default=12.0)
    args = ap.parse_args()
    out = train(fast=args.fast, ttl_hours=args.ttl_hours, offline=args.offline)
    sel = out["results"]["selection"]
    print(json.dumps({
        "production_model": sel["production_model"],
        "ranking": sel["ranking"],
        "theta": out["results"]["theta"],
        "pooled_brier": {
            k: v["pooled_cal"]["brier"] for k, v in out["results"]["metrics"].items()
        },
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
