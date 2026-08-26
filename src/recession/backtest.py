"""Walk-forward backtesting, calibration and model selection.

Validation design:

* **Expanding-window, recession-aware folds.** Test windows are chosen so
  every post-1978 recession falls in exactly one out-of-sample window, plus
  recession-free windows to price false alarms. Training always ends
  ``EMBARGO + horizon`` days before the test window starts, so no label
  window overlaps the test period and adjacent autocorrelated rows cannot
  leak.
* **Weekly rows.** Daily rows are massively autocorrelated; the model grid
  is weekly (Fridays), and event-aware weights stop any single recession
  from dominating the likelihood.
* **Calibration inside the time-series validation.** Each fold calibrates on
  an *inner, time-ordered split of its own training window*: the model is
  fitted on the first 75%, the calibrator on the (purged) last 25%, and the
  fold's test predictions run through that pair. No future data, and no
  mixing of score scales across eras. The calibration method (none / Platt /
  isotonic) is chosen by pooled out-of-sample Brier + log loss.
* **Selection is multi-criteria.** With a ~2.5% base rate, Brier score alone
  rewards a model that never predicts recessions. Production selection uses
  an average rank across calibration (Brier, ECE, log loss), discrimination
  (PR AUC), event detection, and false alarms — mirroring the stated
  priorities: calibration, Brier, robustness, lead time, false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .horizon import CALIBRATORS, HazardTermStructure, IdentityCalibrator, log_loss_w
from .labels import HORIZONS
from .models import build_models, training_weights
from .paths import get_logger

log = get_logger(__name__)

TRAIN_START = pd.Timestamp("1968-06-01")
EMBARGO_DAYS = 90          # gap between train end and test start
HORIZON_DAYS = 90          # base label window
INNER_CAL_FRAC = 0.25      # tail share of each training window used to calibrate
MIN_CAL_POS = 5            # positives required to fit a calibrator
DETECT_THRESHOLD = 0.20    # "the model warned" level for event detection
ALARM_THRESHOLD = 0.30
ALARM_MIN_DAYS = 28
ALARM_LOOKAHEAD_DAYS = 270

CONTENDERS = ("A_yield_curve", "B_elastic_net", "C_grad_boost", "D_ensemble")

FOLDS = [
    ("F1 1978-83 (1980, 1981-82 recessions)", "1978-01-01", "1983-12-31"),
    ("F2 1984-92 (1990-91 recession)",        "1984-01-01", "1992-12-31"),
    ("F3 1993-02 (2001 recession)",           "1993-01-01", "2002-12-31"),
    ("F4 2003-10 (Great Recession)",          "2003-01-01", "2010-12-31"),
    ("F5 2011-17 (no recession)",             "2011-01-01", "2017-12-31"),
    ("F6 2018-22 (COVID recession)",          "2018-01-01", "2022-12-31"),
    ("F7 2023-now (no recession)",            "2023-01-01", None),
]


@dataclass
class BacktestResult:
    oos: pd.DataFrame
    metrics: dict
    calibrator_choice: dict
    selection: dict
    theta: dict
    events: dict = field(default_factory=dict)          # model -> [event rows]
    false_alarms: dict = field(default_factory=dict)    # model -> [alarm rows]
    reliability: dict = field(default_factory=dict)
    horizon_metrics: dict = field(default_factory=dict)
    fold_info: list = field(default_factory=list)


# ------------------------------------------------------------------ metrics

def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(np.unique(p)) < 2:
        return float(abs(p.mean() - y.mean()))
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bins = np.digitize(p, qs[1:-1])
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def metric_block(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": log_loss_w(y, p),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        for thr in (0.2, 0.3, 0.5):
            pred = p >= thr
            tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            out[f"fpr@{thr}"] = float(fp / max((y == 0).sum(), 1))
            out[f"recall@{thr}"] = float(tp / max(tp + fn, 1))
    return out


# ----------------------------------------------------------------- backtest

def make_folds(index: pd.DatetimeIndex) -> list[dict]:
    last = index.max()
    folds = []
    for i, (name, start, end) in enumerate(FOLDS, start=1):
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end) if end else last
        if t0 > last:
            continue
        folds.append({
            "k": i, "name": name, "test_start": t0, "test_end": min(t1, last),
            "train_end": t0 - pd.Timedelta(days=EMBARGO_DAYS + HORIZON_DAYS),
        })
    return folds


def inner_split(train_idx: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Time-ordered fit/calibration split with a purge gap between them."""
    n = len(train_idx)
    cut = int(n * (1 - INNER_CAL_FRAC))
    fit_idx = train_idx[:cut]
    purge_until = fit_idx.max() + pd.Timedelta(days=EMBARGO_DAYS + HORIZON_DAYS)
    cal_idx = train_idx[train_idx > purge_until]
    return fit_idx, cal_idx


def fit_fold_calibrators(
    proto, X, y90, labels, fit_idx, cal_idx
) -> tuple[object, dict[str, object]]:
    """Fit the inner model on the fit split and one calibrator per method on
    the calibration split. Falls back to identity when positives are scarce."""
    w_fit = training_weights(y90.loc[fit_idx].to_numpy(), labels.loc[fit_idx, "next_onset"])
    inner = clone(proto).fit(X.loc[fit_idx], y90.loc[fit_idx].to_numpy(), sample_weight=w_fit)
    cals: dict[str, object] = {"none": IdentityCalibrator()}
    y_cal = y90.loc[cal_idx].to_numpy()
    if len(cal_idx) >= 30 and y_cal.sum() >= MIN_CAL_POS:
        p_cal_raw = inner.predict_proba(X.loc[cal_idx])[:, 1]
        for name, cls in CALIBRATORS.items():
            if name == "none":
                continue
            try:
                cals[name] = cls().fit(p_cal_raw, y_cal)
            except Exception as exc:  # pragma: no cover
                log.warning("Calibrator %s failed on fold split: %s", name, exc)
    return inner, cals


def run_backtest(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> BacktestResult:
    """X: weekly feature panel (train-mode); labels: aligned label frame."""
    feature_cols = feature_cols or list(X.columns)
    y90 = labels["y90"]
    eligible = labels["eligible"].astype(bool)
    folds = make_folds(X.index)

    models = build_models(feature_cols)
    base_names = list(models)
    oos = labels.copy()
    oos["fold"] = np.nan
    method_cols: dict[tuple[str, str], str] = {}

    for f in folds:
        tr_mask = (X.index >= TRAIN_START) & (X.index <= f["train_end"]) & eligible.to_numpy()
        te_mask = (X.index >= f["test_start"]) & (X.index <= f["test_end"])
        train_idx = X.index[tr_mask]
        if len(train_idx) < 100:
            log.warning("Fold %s skipped: only %d train rows", f["name"], len(train_idx))
            continue
        oos.loc[te_mask, "fold"] = f["k"]
        fit_idx, cal_idx = inner_split(train_idx)
        w_full = training_weights(y90.loc[train_idx].to_numpy(), labels.loc[train_idx, "next_onset"])

        for mname, proto in models.items():
            # full-train model: raw scores used later for the final calibrator
            full_model = clone(proto).fit(
                X.loc[train_idx], y90.loc[train_idx].to_numpy(), sample_weight=w_full
            )
            oos.loc[te_mask, f"p_raw_{mname}"] = full_model.predict_proba(X[te_mask])[:, 1]
            # inner model + calibrators: the honest walk-forward calibrated path
            inner, cals = fit_fold_calibrators(proto, X, y90, labels, fit_idx, cal_idx)
            p_inner = inner.predict_proba(X[te_mask])[:, 1]
            for cname, cal in cals.items():
                col = f"p_{cname}_{mname}"
                oos.loc[te_mask, col] = cal.transform(p_inner)
                method_cols[(mname, cname)] = col
        f["n_train"] = int(len(train_idx))
        f["n_train_pos"] = int(y90.loc[train_idx].sum())
        f["n_cal"] = int(len(cal_idx))
        f["n_cal_pos"] = int(y90.loc[cal_idx].sum())
        f["n_test_eligible"] = int((te_mask & eligible.to_numpy()).sum())
        f["n_test_pos"] = int(y90[te_mask & eligible.to_numpy()].sum())
        log.info("Fold %s done (train n=%d pos=%d, cal n=%d pos=%d)",
                 f["name"], f["n_train"], f["n_train_pos"], f["n_cal"], f["n_cal_pos"])

    oos = oos.dropna(subset=["fold"])
    el = oos["eligible"].astype(bool)

    # ---- choose the calibration method per model (pooled OOS Brier+logloss) --
    calibrator_choice: dict[str, str] = {}
    for mname in base_names:
        scores = {}
        for cname in ("none", "platt", "isotonic"):
            col = f"p_{cname}_{mname}"
            if col not in oos.columns:
                continue
            p = oos.loc[el, col].astype(float)
            filled = p.fillna(oos.loc[el, f"p_none_{mname}"].astype(float))
            y = oos.loc[el, "y90"].astype(int)
            scores[cname] = brier_score_loss(y, filled) + 0.25 * log_loss_w(y, filled)
        best = min(scores, key=scores.get)
        calibrator_choice[mname] = best
        pc = oos[f"p_{best}_{mname}"].astype(float)
        oos[f"p_cal_{mname}"] = pc.fillna(oos[f"p_none_{mname}"].astype(float))

    # ---- ensemble D: mean of calibrated B and C ----
    if "p_cal_B_elastic_net" in oos.columns and "p_cal_C_grad_boost" in oos.columns:
        oos["p_raw_D_ensemble"] = oos[["p_raw_B_elastic_net", "p_raw_C_grad_boost"]].mean(axis=1)
        oos["p_cal_D_ensemble"] = oos[["p_cal_B_elastic_net", "p_cal_C_grad_boost"]].mean(axis=1)
        calibrator_choice["D_ensemble"] = "members"

    # ---- metrics ----
    model_names = [m for m in base_names + ["D_ensemble"] if f"p_cal_{m}" in oos.columns]
    metrics: dict[str, dict] = {}
    for m in model_names:
        pm: dict = {
            "pooled_cal": metric_block(oos.loc[el, "y90"], oos.loc[el, f"p_cal_{m}"]),
            "pooled_raw": metric_block(oos.loc[el, "y90"], oos.loc[el, f"p_raw_{m}"]),
            "per_fold": [],
        }
        for f in folds:
            fm = el & (oos["fold"] == f["k"])
            if fm.sum() == 0:
                continue
            blk = metric_block(oos.loc[fm, "y90"], oos.loc[fm, f"p_cal_{m}"])
            blk["fold"] = f["name"]
            pm["per_fold"].append(blk)
        metrics[m] = pm

    # ---- event analysis for every contender ----
    events: dict[str, list] = {}
    false_alarms: dict[str, list] = {}
    for m in model_names:
        ev, fa = event_analysis(oos, f"p_cal_{m}")
        events[m], false_alarms[m] = ev, fa
        n_ev = max(len(ev), 1)
        metrics[m]["event"] = {
            "n_events": len(ev),
            "detect_rate": sum(e["detected"] for e in ev) / n_ev,
            "miss_rate": sum(e["max_p_365d_before"] < 0.10 for e in ev) / n_ev,
            "mean_lead_days": float(np.mean([e["lead_days"] for e in ev if e["lead_days"] is not None]))
            if any(e["lead_days"] is not None for e in ev) else None,
            "n_false_alarms": len(fa),
        }

    # ---- production selection: average rank across declared criteria ----
    contenders = [m for m in CONTENDERS if m in metrics]

    def _crit(m):
        pc, evb = metrics[m]["pooled_cal"], metrics[m]["event"]
        return {
            "brier": pc["brier"],
            "log_loss": pc["log_loss"],
            "ece": pc["ece"],
            "neg_pr_auc": -pc.get("pr_auc", 0.0),
            "neg_detect_rate": -evb["detect_rate"],
            "false_alarms": evb["n_false_alarms"],
        }

    crit_table = {m: _crit(m) for m in contenders}
    ranks = {m: 0.0 for m in contenders}
    for crit in next(iter(crit_table.values())):
        vals = pd.Series({m: crit_table[m][crit] for m in contenders})
        r = vals.rank(method="average")
        for m in contenders:
            ranks[m] += float(r[m])
    for m in ranks:
        ranks[m] /= len(next(iter(crit_table.values())))
    order = sorted(contenders, key=lambda m: (ranks[m], CONTENDERS.index(m)))
    production = order[0]
    selection = {
        "production_model": production,
        "ranking": order,
        "avg_rank": {m: round(ranks[m], 3) for m in contenders},
        "criteria": {m: {k: round(v, 5) for k, v in c.items()} for m, c in crit_table.items()},
        "criterion": (
            "average rank across Brier, log loss, ECE, PR-AUC, event detection "
            "rate and false alarms (pooled walk-forward OOS, calibrated); "
            "ties -> simpler model"
        ),
        "brier_vs_constant": (
            metrics.get("bl_constant", {}).get("pooled_cal", {}).get("brier", np.nan)
            - metrics[production]["pooled_cal"]["brier"]
        ),
        "brier_vs_yield_curve": (
            metrics.get("A_yield_curve", {}).get("pooled_cal", {}).get("brier", np.nan)
            - metrics[production]["pooled_cal"]["brier"]
        ),
    }

    # ---- hazard term structure fitted on clean calibrated OOS (folds >= 2) --
    score_mask = el & (oos["fold"] >= 2)
    ts = HazardTermStructure().fit(
        oos.loc[score_mask, f"p_cal_{production}"].to_numpy(), oos.loc[score_mask]
    )
    for h in HORIZONS:
        oos[f"p{h}"] = ts.term_structure(oos[f"p_cal_{production}"].to_numpy())[h]

    horizon_metrics = {}
    for h in HORIZONS:
        horizon_metrics[h] = metric_block(
            oos.loc[score_mask, f"y{h}"], oos.loc[score_mask, f"p{h}"]
        )

    # ---- reliability curve data ----
    reliability = {}
    for m in model_names:
        p = oos.loc[el, f"p_cal_{m}"].to_numpy()
        y = oos.loc[el, "y90"].to_numpy()
        qs = np.quantile(p, np.linspace(0, 1, 11))
        qs[0], qs[-1] = -np.inf, np.inf
        bins = np.digitize(p, qs[1:-1])
        rows = []
        for b in np.unique(bins):
            msk = bins == b
            rows.append({"pred": float(p[msk].mean()), "obs": float(y[msk].mean()),
                         "n": int(msk.sum())})
        reliability[m] = rows

    # keep the persisted frame tidy: drop per-method intermediate columns
    drop = [c for c in oos.columns if c.startswith(("p_none_", "p_platt_", "p_isotonic_"))]
    oos = oos.drop(columns=drop)

    return BacktestResult(
        oos=oos, metrics=metrics, calibrator_choice=calibrator_choice,
        selection=selection, theta=ts.theta_, events=events,
        false_alarms=false_alarms, reliability=reliability,
        horizon_metrics=horizon_metrics, fold_info=folds,
    )


LONG_H = 365
LONG_DETECT = 0.35  # base rate of the 1y target is ~5x the 90d one


def run_long_backtest(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Walk-forward backtest for the dedicated 1-year onset classifier.

    Same recession-aware folds and inner-split calibration as the 90-day
    model, with a purge widened to EMBARGO + 365 days so no 1-year label
    window overlaps a test period. Members are the production architecture
    (elastic net + gradient boosting); the served probability is their
    calibrated mean, floored at P90 downstream (pool-adjacent-violators
    between the two horizon estimates).
    """
    y = labels[f"y{LONG_H}"]
    el = labels["eligible_long"].astype(bool)
    folds = make_folds(X.index)
    protos = {k: v for k, v in build_models(feature_cols).items()
              if k in ("B_elastic_net", "C_grad_boost")}

    oos = labels[[f"y{LONG_H}", "eligible_long", "next_onset"]].copy()
    oos["fold"] = np.nan
    for f in folds:
        train_end = f["test_start"] - pd.Timedelta(days=EMBARGO_DAYS + LONG_H)
        tr = (X.index >= TRAIN_START) & (X.index <= train_end) & el.to_numpy()
        te = (X.index >= f["test_start"]) & (X.index <= f["test_end"])
        train_idx = X.index[tr]
        if len(train_idx) < 100:
            continue
        oos.loc[te, "fold"] = f["k"]
        fit_idx, cal_idx = inner_split(train_idx)
        for mname, proto in protos.items():
            inner, cals = fit_fold_calibrators(proto, X, y, labels, fit_idx, cal_idx)
            p_inner = inner.predict_proba(X[te])[:, 1]
            for cname, cal in cals.items():
                oos.loc[te, f"p_{cname}_{mname}"] = cal.transform(p_inner)
    oos = oos.dropna(subset=["fold"])
    elo = oos["eligible_long"].astype(bool)

    calibrator_choice: dict[str, str] = {}
    for mname in protos:
        scores = {}
        for cname in ("none", "platt", "isotonic"):
            col = f"p_{cname}_{mname}"
            if col not in oos.columns:
                continue
            p = oos.loc[elo, col].astype(float).fillna(oos.loc[elo, f"p_none_{mname}"].astype(float))
            yy = oos.loc[elo, f"y{LONG_H}"].astype(int)
            scores[cname] = brier_score_loss(yy, p) + 0.25 * log_loss_w(yy, p)
        best = min(scores, key=scores.get)
        calibrator_choice[mname] = best
        oos[f"p_cal_{mname}"] = (
            oos[f"p_{best}_{mname}"].astype(float).fillna(oos[f"p_none_{mname}"].astype(float))
        )
    oos["p_cal_long"] = oos[[f"p_cal_{m}" for m in protos]].mean(axis=1)

    metrics = {
        m: metric_block(oos.loc[elo, f"y{LONG_H}"], oos.loc[elo, f"p_cal_{m}"])
        for m in protos
    }
    metrics["ensemble_1y"] = metric_block(oos.loc[elo, f"y{LONG_H}"], oos.loc[elo, "p_cal_long"])

    # per-recession: max calibrated 1y probability in the 12 months pre-onset
    onsets = sorted(pd.DatetimeIndex(oos.loc[oos[f"y{LONG_H}"] == 1, "next_onset"].dropna().unique()))
    events = []
    for o in onsets:
        pre = oos.index[(oos.index < o) & (oos.index >= o - pd.Timedelta(days=365)) & elo]
        if len(pre) == 0:
            continue
        p_pre = oos.loc[pre, "p_cal_long"].astype(float)
        events.append({
            "onset": str(pd.Timestamp(o).date()),
            "max_p_1y_before": float(p_pre.max()),
            "detected": bool(p_pre.max() >= LONG_DETECT),
        })

    return {
        "oos": oos[["fold", "eligible_long", f"y{LONG_H}", "p_cal_long"]],
        "metrics": metrics,
        "calibrator_choice": calibrator_choice,
        "events": events,
        "detect_threshold": LONG_DETECT,
    }


def event_analysis(oos: pd.DataFrame, pcol: str) -> tuple[list, list]:
    """Per-recession detection stats + false-alarm episodes, on OOS rows."""
    el = oos["eligible"].astype(bool)
    onsets = sorted(pd.DatetimeIndex(oos.loc[oos["y90"] == 1, "next_onset"].dropna().unique()))
    events = []
    for o in onsets:
        pre = oos.index[(oos.index < o) & (oos.index >= o - pd.Timedelta(days=365)) & el]
        if len(pre) == 0:
            continue
        p_pre = oos.loc[pre, pcol].astype(float)
        win90 = p_pre[p_pre.index >= o - pd.Timedelta(days=90)]
        crossings = p_pre[p_pre >= DETECT_THRESHOLD]
        events.append({
            "onset": str(pd.Timestamp(o).date()),
            "max_p_90d_before": float(win90.max()) if len(win90) else np.nan,
            "max_p_365d_before": float(p_pre.max()),
            "first_cross": str(crossings.index.min().date()) if len(crossings) else None,
            "lead_days": int((o - crossings.index.min()).days) if len(crossings) else None,
            "detected": bool(p_pre.max() >= DETECT_THRESHOLD),
        })

    # false alarms: sustained high probability with no onset following
    onset_arr = pd.DatetimeIndex(onsets)
    sub = oos[el].sort_index()
    hot = sub[pcol].astype(float) >= ALARM_THRESHOLD
    false_alarms: list[dict] = []
    run_start = None
    prev_t = None
    for t, is_hot in hot.items():
        if is_hot and run_start is None:
            run_start = t
        broken = (not is_hot) or (prev_t is not None and (t - prev_t).days > 21)
        if run_start is not None and broken:
            _record_alarm(false_alarms, sub, pcol, run_start, prev_t, onset_arr)
            run_start = t if is_hot else None
        prev_t = t
    if run_start is not None:
        _record_alarm(false_alarms, sub, pcol, run_start, prev_t, onset_arr)
    return events, false_alarms


def _record_alarm(acc: list, sub: pd.DataFrame, pcol: str, start, end, onsets) -> None:
    if start is None or end is None or (end - start).days < ALARM_MIN_DAYS:
        return
    horizon_end = end + pd.Timedelta(days=ALARM_LOOKAHEAD_DAYS)
    followed = any((o > start) and (o <= horizon_end) for o in onsets)
    if not followed:
        seg = sub.loc[start:end, pcol].astype(float)
        acc.append({
            "start": str(pd.Timestamp(start).date()),
            "end": str(pd.Timestamp(end).date()),
            "days": int((end - start).days),
            "max_p": float(seg.max()),
        })
