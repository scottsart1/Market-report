"""Run one walk-forward experiment and record its out-of-sample metrics.

Every candidate change to the model (new series, new transforms, a modeling
switch) is judged here, on the same folds, the same purge and the same
calibration discipline as production. Nothing ships unless it improves the
out-of-sample numbers written by this script.

Examples:
  python scripts/experiment.py --name baseline
  python scripts/experiment.py --name no_credit --drop-categories credit
  python scripts/experiment.py --name only_v1 --indicators DGS10,SPREAD_10Y3M,...
  python scripts/experiment.py --name hgb_mono --variants hgb_monotone=1
  python scripts/experiment.py --name wide --exclude 'NEW_.*__z10y'

Results: experiments/<name>.json (metrics, selection, config) and
experiments/<name>.oos.parquet / <name>.long.parquet (the out-of-sample
predictions, for paired comparisons in scripts/compare.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from recession import MODEL_VERSION  # noqa: E402
from recession.backtest import run_backtest, run_long_backtest  # noqa: E402
from recession.config import load_config  # noqa: E402
from recession.variants import ENV_VAR, active_variants  # noqa: E402

EXP_DIR = ROOT / "experiments"


def select_features(cfg, columns: list[str], args) -> list[str]:
    specs = cfg.feature_specs()
    keep = []
    only = set(args.indicators.split(",")) if args.indicators else None
    drop_ind = set(args.drop_indicators.split(",")) if args.drop_indicators else set()
    drop_cat = set(args.drop_categories.split(",")) if args.drop_categories else set()
    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None
    for c in columns:
        ind = specs[c][0] if c in specs else None
        iid = ind.id if ind else c.split("__")[0]
        cat = ind.category if ind else ""
        if only is not None and iid not in only:
            continue
        if ind is not None and not ind.modeled and only is None:
            continue  # excluded by config unless named explicitly
        if iid in drop_ind or cat in drop_cat:
            continue
        if inc and not inc.search(c):
            continue
        if exc and exc.search(c):
            continue
        keep.append(c)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--indicators", help="comma-separated indicator ids to keep (others dropped)")
    ap.add_argument("--drop-indicators", help="comma-separated indicator ids to drop")
    ap.add_argument("--drop-categories", help="comma-separated categories to drop")
    ap.add_argument("--include", help="regex: keep only feature columns matching")
    ap.add_argument("--exclude", help="regex: drop feature columns matching")
    ap.add_argument("--variants", default="", help='modeling switches, "k=v;k2=v2" (see recession/variants.py)')
    ap.add_argument("--skip-long", action="store_true", help="skip the 1-year backtest")
    ap.add_argument("--online", action="store_true", help="allow fetching (default: cache only)")
    ap.add_argument("--ttl-hours", type=float, default=1e9)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.variants:
        os.environ[ENV_VAR] = args.variants
    # imported after the switches are set so nothing caches a default
    from recession.train import build_training_data

    t0 = time.time()
    cfg, fetched, cal, panel, Xw, labels = build_training_data(ttl_hours=args.ttl_hours, offline=not args.online)
    feature_cols = select_features(cfg, list(Xw.columns), args)
    if not feature_cols:
        raise SystemExit("No features left after selection")
    print(f"[{args.name}] {len(feature_cols)}/{Xw.shape[1]} features, {len(Xw)} weekly rows, "
          f"variants={active_variants() or '{}'}", flush=True)

    t1 = time.time()
    bt = run_backtest(Xw, labels, feature_cols)
    t2 = time.time()
    long_bt = None if args.skip_long else run_long_backtest(Xw, labels, feature_cols)
    t3 = time.time()

    EXP_DIR.mkdir(exist_ok=True)
    out = {
        "name": args.name,
        "note": args.note,
        "model_version": MODEL_VERSION,
        "variants": active_variants(),
        "selection_args": {k: v for k, v in vars(args).items()
                           if k in ("indicators", "drop_indicators", "drop_categories", "include", "exclude")},
        "n_features": len(feature_cols),
        "n_features_available": int(Xw.shape[1]),
        "features": feature_cols,
        "n_rows": int(len(Xw)),
        "panel_end": str(Xw.index.max().date()),
        "timing_s": {"data": round(t1 - t0, 1), "backtest90": round(t2 - t1, 1), "backtest365": round(t3 - t2, 1)},
        "metrics": bt.metrics,
        "selection": bt.selection,
        "calibrator_choice": bt.calibrator_choice,
        "theta": bt.theta,
        "horizon_metrics": bt.horizon_metrics,
        "events": bt.events,
        "false_alarms": bt.false_alarms,
        "fold_info": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in f.items()} for f in bt.fold_info],
        "long": None if long_bt is None else {k: v for k, v in long_bt.items() if k != "oos"},
    }
    (EXP_DIR / f"{args.name}.json").write_text(json.dumps(out, indent=1, default=str))
    bt.oos.to_parquet(EXP_DIR / f"{args.name}.oos.parquet")
    if long_bt is not None:
        long_bt["oos"].to_parquet(EXP_DIR / f"{args.name}.long.parquet")

    d = bt.metrics.get("D_ensemble", {}).get("pooled_cal", {})
    e = bt.metrics.get("D_ensemble", {}).get("event", {})
    line = (f"[{args.name}] D_ensemble brier={d.get('brier'):.4f} logloss={d.get('log_loss'):.4f} "
            f"auc={d.get('roc_auc'):.3f} pr_auc={d.get('pr_auc'):.3f} ece={d.get('ece'):.3f} "
            f"detect={e.get('detect_rate')} fas={e.get('n_false_alarms')} lead={e.get('mean_lead_days')} "
            f"prod={bt.selection.get('production_model')}")
    if long_bt is not None:
        l1 = long_bt["metrics"]["ensemble_1y"]
        line += f" | 1y brier={l1['brier']:.4f} logloss={l1['log_loss']:.4f} auc={l1.get('roc_auc'):.3f}"
    print(line, flush=True)
    print(f"[{args.name}] wrote {EXP_DIR / (args.name + '.json')} "
          f"({out['timing_s']['backtest90']}s + {out['timing_s']['backtest365']}s)")


if __name__ == "__main__":
    main()
