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
  smoothed isotonic) is chosen by pooled out-of-sample Brier + log loss.
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

from .horizon import (
    CALIBRATORS, HazardTermStructure, IdentityCalibrator, PlattCalibrator, log_loss_w,
)
from .labels import HORIZONS
from .models import build_models, training_weights
from .paths import get_logger
from .variants import variant

log = get_logger(__name__)

TRAIN_START = pd.Timestamp("1968-06-01")
EMBARGO_DAYS = 90          # gap between train end and test start
HORIZON_DAYS = 90          # base label window
INNER_CAL_FRAC = 0.25      # tail share of each training window used to calibrate
MIN_CAL_POS = 5            # positive rows required to fit a calibrator
MIN_CAL_EVENTS = 2         # distinct onsets required to fit any calibrator
MIN_ISO_EVENTS = 3         # distinct onsets required for the isotonic one
# Event metrics use base-rate-relative thresholds: a well-scaled 90-day
# probability rarely reaches 0.20 when the base rate is ~3.5%, and absolute
# thresholds rewarded members whose scale was inflated. The absolute levels
# are kept for the dashboard bands and reported alongside.
DETECT_MULT = 3.0          # "the model warned": p >= 3x the OOS base rate
ALARM_MULT = 6.0           # sustained p >= 6x the base rate counts as an alarm
DETECT_THRESHOLD = 0.20    # absolute level, reported for comparison
ALARM_THRESHOLD = 0.30
ALARM_MIN_DAYS = 28
ALARM_LOOKAHEAD_DAYS = 365      # an alarm followed by an onset within a year is not false
EARLY_WARNING_DAYS = 270        # ...but 270-365 days ahead is reported as an early warning

CONTENDERS = ("A_yield_curve", "B_elastic_net", "C_grad_boost", "D_ensemble")
MEMBER_CODES = {"A": "A_yield_curve", "B": "B_elastic_net", "C": "C_grad_boost"}


def ensemble_members() -> list[str]:
    """Members of the ensemble D: experiment switch `ensemble`, e.g. "A,B,C"."""
    codes = [c.strip().upper() for c in str(variant("ensemble", "B,C")).split(",") if c.strip()]
    return [MEMBER_CODES[c] for c in codes]

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


def inner_split(
    train_idx: pd.DatetimeIndex, horizon_days: int = HORIZON_DAYS
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Time-ordered fit/calibration split with a purge gap between them. The
    purge must cover the label window of the target being calibrated (the
    1-year model needs 90 + 365 days, not the 90-day model's 180)."""
    n = len(train_idx)
    cut = int(n * (1 - INNER_CAL_FRAC))
    fit_idx = train_idx[:cut]
    purge_until = fit_idx.max() + pd.Timedelta(days=EMBARGO_DAYS + horizon_days)
    cal_idx = train_idx[train_idx > purge_until]
    return fit_idx, cal_idx


def n_events(y: np.ndarray, next_onset: pd.Series) -> int:
    """Distinct recession onsets among the positive rows."""
    pos = np.asarray(y, dtype=int) == 1
    return int(pd.Series(next_onset).astype("datetime64[ns]")[pos].nunique())


def prior_oos_calibrator(p_prior: np.ndarray, y_prior: np.ndarray, onsets_prior: pd.Series):
    """Platt calibrator fitted on the pooled raw out-of-sample scores of the
    folds before the current one (switch `calibration=prior_oos`). Scores
    from earlier folds are genuinely out of sample for their own models, so
    this uses several recessions instead of one purged tail. Identity when
    too few onsets are available or the fit carries no ranking information."""
    y_prior = np.asarray(y_prior, dtype=int)
    if len(y_prior) >= 30 and y_prior.sum() >= MIN_CAL_POS and n_events(y_prior, onsets_prior) >= MIN_CAL_EVENTS:
        cal = PlattCalibrator().fit(np.asarray(p_prior, dtype=float), y_prior)
        if cal.is_informative():
            return cal
    return IdentityCalibrator()


def fit_fold_calibrators(
    proto, X, y90, labels, fit_idx, cal_idx
) -> tuple[object, dict[str, object]]:
    """Fit the inner model on the fit split and one calibrator per method on
    the calibration split. Falls back to identity when the window holds
    fewer than MIN_CAL_EVENTS distinct onsets (one recession cannot establish
    a probability scale); a method whose fit carries no ranking information
    is omitted, so callers fall back to identity for it."""
    w_fit = training_weights(y90.loc[fit_idx].to_numpy(), labels.loc[fit_idx, "next_onset"])
    inner = clone(proto).fit(X.loc[fit_idx], y90.loc[fit_idx].to_numpy(), sample_weight=w_fit)
    cals: dict[str, object] = {"none": IdentityCalibrator()}
    y_cal = y90.loc[cal_idx].to_numpy()
    k_events = n_events(y_cal, labels.loc[cal_idx, "next_onset"]) if len(cal_idx) else 0
    if len(cal_idx) >= 30 and y_cal.sum() >= MIN_CAL_POS and k_events >= MIN_CAL_EVENTS:
        p_cal_raw = inner.predict_proba(X.loc[cal_idx])[:, 1]
        for name, cls in CALIBRATORS.items():
            if name == "none":
                continue
            if name == "isotonic_smooth" and k_events < MIN_ISO_EVENTS:
                continue  # a step function per onset is not a calibration
            try:
                cal = cls().fit(p_cal_raw, y_cal)
            except Exception as exc:  # pragma: no cover
                log.warning("Calibrator %s failed on fold split: %s", name, exc)
                continue
            if cal.is_informative():
                cals[name] = cal
            else:
                log.info("Calibrator %s rejected as uninformative on this split", name)
    elif len(cal_idx):
        log.info("Calibration window holds %d onset(s); serving raw scores", k_events)
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
                if not cal.covers(p_inner):
                    log.info("Calibrator %s/%s skipped on %s: test scores lie outside its fitted range",
                             mname, cname, f["name"])
                    continue
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

    calibrator_choice: dict[str, str] = {}
    calib_mode = str(variant("calibration", "inner"))
    if calib_mode == "prior_oos":
        # ---- served path: full-train model, calibrated on the raw OOS scores
        # of the folds before it (walk-forward: fold k never sees fold k) ----
        for mname in base_names:
            raw = f"p_raw_{mname}"
            oos[f"p_cal_{mname}"] = np.nan
            for f in folds:
                k = f["k"]
                rows = oos["fold"] == k
                prior = el & (oos["fold"] < k) & oos[raw].notna()
                cal = prior_oos_calibrator(
                    oos.loc[prior, raw].to_numpy(), oos.loc[prior, "y90"].to_numpy(),
                    oos.loc[prior, "next_onset"],
                )
                oos.loc[rows, f"p_cal_{mname}"] = cal.transform(oos.loc[rows, raw].to_numpy(dtype=float))
            calibrator_choice[mname] = "prior_oos"
    else:
        # ---- choose the calibration method per model (pooled OOS Brier+logloss) --
        for mname in base_names:
            scores = {}
            for cname in CALIBRATORS:
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

    # ---- ensemble D: plain mean of calibrated members (switch `ensemble`) ----
    members = ensemble_members()
    if all(f"p_cal_{m}" in oos.columns for m in members):
        oos["p_raw_D_ensemble"] = oos[[f"p_raw_{m}" for m in members]].mean(axis=1)
        oos["p_cal_D_ensemble"] = oos[[f"p_cal_{m}" for m in members]].mean(axis=1)
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

    # ---- hazard term structure ----
    # Production exponents: fitted on all calibrated OOS rows of folds >= 2
    # (theta_90 = 1 by construction). The OOS p{h} columns and the horizon
    # metrics use walk-forward exponents instead: fold k's sub-horizon
    # exponents come from folds 2..k-1 only, and before any prior fold
    # exists the constant-hazard default h/90 is used, so no horizon metric
    # is scored on rows its exponent was fitted on.
    score_mask = el & (oos["fold"] >= 2)
    ts = HazardTermStructure().fit(
        oos.loc[score_mask, f"p_cal_{production}"].to_numpy(), oos.loc[score_mask]
    )
    theta_walk_forward: dict[int, dict] = {}
    for h in HORIZONS:
        oos[f"p{h}"] = np.nan
    for f in folds:
        k = f["k"]
        rows = oos["fold"] == k
        prior = el & (oos["fold"] >= 2) & (oos["fold"] < k)
        ts_k = HazardTermStructure()
        if int(oos.loc[prior, "y90"].sum()) >= MIN_CAL_POS:
            ts_k.fit(oos.loc[prior, f"p_cal_{production}"].to_numpy(), oos.loc[prior])
        theta_walk_forward[k] = dict(ts_k.theta_)
        curve = ts_k.term_structure(oos.loc[rows, f"p_cal_{production}"].to_numpy())
        for h in HORIZONS:
            oos.loc[rows, f"p{h}"] = curve[h]

    horizon_metrics = {}
    for h in HORIZONS:
        horizon_metrics[h] = metric_block(
            oos.loc[score_mask, f"y{h}"], oos.loc[score_mask, f"p{h}"]
        )
    horizon_metrics["note"] = ("walk-forward exponents: fold k uses folds 2..k-1; "
                               "constant hazard h/90 where none exist")

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
    method_prefixes = tuple(f"p_{m}_" for m in CALIBRATORS) + ("p_prior_",)
    drop = [c for c in oos.columns if c.startswith(method_prefixes)]
    oos = oos.drop(columns=drop)

    selection["theta_walk_forward"] = theta_walk_forward
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
    calib_mode = str(variant("calibration", "inner"))
    for f in folds:
        train_end = f["test_start"] - pd.Timedelta(days=EMBARGO_DAYS + LONG_H)
        tr = (X.index >= TRAIN_START) & (X.index <= train_end) & el.to_numpy()
        te = (X.index >= f["test_start"]) & (X.index <= f["test_end"])
        train_idx = X.index[tr]
        if len(train_idx) < 100:
            continue
        oos.loc[te, "fold"] = f["k"]
        fit_idx, cal_idx = inner_split(train_idx, LONG_H)
        w_full = training_weights(y.loc[train_idx].to_numpy(), labels.loc[train_idx, "next_onset"])
        for mname, proto in protos.items():
            full_model = clone(proto).fit(X.loc[train_idx], y.loc[train_idx].to_numpy(), sample_weight=w_full)
            oos.loc[te, f"p_raw_{mname}"] = full_model.predict_proba(X[te])[:, 1]
            inner, cals = fit_fold_calibrators(proto, X, y, labels, fit_idx, cal_idx)
            p_inner = inner.predict_proba(X[te])[:, 1]
            for cname, cal in cals.items():
                if not cal.covers(p_inner):
                    continue
                oos.loc[te, f"p_{cname}_{mname}"] = cal.transform(p_inner)
    oos = oos.dropna(subset=["fold"])
    elo = oos["eligible_long"].astype(bool)

    calibrator_choice: dict[str, str] = {}
    if calib_mode == "prior_oos":
        for mname in protos:
            raw = f"p_raw_{mname}"
            oos[f"p_cal_{mname}"] = np.nan
            for f in folds:
                k = f["k"]
                rows = oos["fold"] == k
                prior = elo & (oos["fold"] < k) & oos[raw].notna()
                cal = prior_oos_calibrator(
                    oos.loc[prior, raw].to_numpy(), oos.loc[prior, f"y{LONG_H}"].to_numpy(),
                    oos.loc[prior, "next_onset"],
                )
                oos.loc[rows, f"p_cal_{mname}"] = cal.transform(oos.loc[rows, raw].to_numpy(dtype=float))
            calibrator_choice[mname] = "prior_oos"
    else:
        for mname in protos:
            scores = {}
            for cname in CALIBRATORS:
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

    keep = ["fold", "eligible_long", f"y{LONG_H}", "next_onset", "p_cal_long"]
    keep += [c for c in oos.columns if c.startswith(("p_raw_", "p_cal_")) and c not in keep]
    return {
        "oos": oos[keep],
        "metrics": metrics,
        "calibrator_choice": calibrator_choice,
        "events": events,
        "detect_threshold": LONG_DETECT,
    }


def event_thresholds(base_rate: float) -> tuple[float, float]:
    """(detection, alarm) thresholds relative to a base rate."""
    return DETECT_MULT * base_rate, ALARM_MULT * base_rate


def event_analysis(oos: pd.DataFrame, pcol: str) -> tuple[list, list]:
    """Per-recession detection stats + false-alarm episodes, on OOS rows.

    Thresholds are relative to the out-of-sample base rate of the target
    (see DETECT_MULT / ALARM_MULT); the absolute 0.20 level is reported too.
    """
    el = oos["eligible"].astype(bool)
    base = float(oos.loc[el, "y90"].astype(float).mean())
    det_thr, alarm_thr = event_thresholds(base)
    onsets = sorted(pd.DatetimeIndex(oos.loc[oos["y90"] == 1, "next_onset"].dropna().unique()))
    events = []
    for o in onsets:
        pre = oos.index[(oos.index < o) & (oos.index >= o - pd.Timedelta(days=365)) & el]
        if len(pre) == 0:
            continue
        p_pre = oos.loc[pre, pcol].astype(float)
        win90 = p_pre[p_pre.index >= o - pd.Timedelta(days=90)]
        crossings = p_pre[p_pre >= det_thr]
        events.append({
            "onset": str(pd.Timestamp(o).date()),
            "max_p_90d_before": float(win90.max()) if len(win90) else np.nan,
            "max_p_365d_before": float(p_pre.max()),
            "first_cross": str(crossings.index.min().date()) if len(crossings) else None,
            "lead_days": int((o - crossings.index.min()).days) if len(crossings) else None,
            "detected": bool(p_pre.max() >= det_thr),
            "detected_abs": bool(p_pre.max() >= DETECT_THRESHOLD),
            "detect_threshold": round(det_thr, 4),
        })

    # alarms: sustained high probability; false unless an onset follows
    onset_arr = pd.DatetimeIndex(onsets)
    sub = oos[el].sort_index()
    hot = sub[pcol].astype(float) >= alarm_thr
    alarms: list[dict] = []
    run_start = None
    prev_t = None
    for t, is_hot in hot.items():
        if is_hot and run_start is None:
            run_start = t
        broken = (not is_hot) or (prev_t is not None and (t - prev_t).days > 21)
        if run_start is not None and broken:
            _record_alarm(alarms, sub, pcol, run_start, prev_t, onset_arr, alarm_thr)
            run_start = t if is_hot else None
        prev_t = t
    if run_start is not None:
        _record_alarm(alarms, sub, pcol, run_start, prev_t, onset_arr, alarm_thr)
    false_alarms = [a for a in alarms if a["kind"] == "false"]
    return events, false_alarms


def _record_alarm(acc: list, sub: pd.DataFrame, pcol: str, start, end, onsets, thr) -> None:
    if start is None or end is None or (end - start).days < ALARM_MIN_DAYS:
        return
    nxt = [o for o in onsets if o > start]
    days_to_onset = int((nxt[0] - end).days) if nxt else None
    if days_to_onset is not None and days_to_onset <= EARLY_WARNING_DAYS:
        return  # a warning that an onset followed within the look-ahead
    kind = "early_warning" if (days_to_onset is not None and days_to_onset <= ALARM_LOOKAHEAD_DAYS) else "false"
    seg = sub.loc[start:end, pcol].astype(float)
    acc.append({
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "days": int((end - start).days),
        "max_p": float(seg.max()),
        "threshold": round(float(thr), 4),
        "days_to_next_onset": days_to_onset,
        "kind": kind,
    })
