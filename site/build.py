"""Build the static snapshot dashboard (site/recession_monitor.html).

Runs the production prediction on fresh data, packages everything the page
needs as JSON, and injects it into site/template.html. Used interactively and
by the weekly auto-refresh routine.

Run:  python site/build.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from recession.paths import MODELS_DIR
from recession.predict import run_prediction


def build_payload(ttl_hours: float = 1.0, offline: bool = False) -> dict:
    out = run_prediction(ttl_hours=ttl_hours, offline=offline, save=True)
    pred = out["prediction"]

    oos = pd.read_parquet(MODELS_DIR / "oos_predictions.parquet")
    oos.index = pd.to_datetime(oos.index)
    results = json.loads((MODELS_DIR / "backtest_results.json").read_text())

    spans, start, prev = [], None, None
    for t, v in oos["in_recession"].astype(bool).items():
        if v and start is None:
            start = t
        if not v and start is not None:
            spans.append([str(start.date()), str(prev.date())])
            start = None
        prev = t
    if start is not None:
        spans.append([str(start.date()), str(prev.date())])

    hist_h = [h for h in (15, 30, 45, 60, 90, 365) if f"p{h}" in oos.columns]
    hist = {
        "dates": [str(d.date()) for d in oos.index],
        "horizons": hist_h,
        "spans": spans,
    }
    for h in hist_h:
        hist[f"p{h}"] = [None if pd.isna(v) else round(float(v), 5) for v in oos[f"p{h}"]]

    health = out["health"][
        ["name", "category", "tier", "source", "vintage_mode", "last_period", "days_stale", "status"]
    ].copy()
    health["last_period"] = pd.to_datetime(health["last_period"]).dt.strftime("%Y-%m-%d")

    mnames = ["A_yield_curve", "B_elastic_net", "C_grad_boost", "D_ensemble",
              "bl_constant", "bl_sahm", "bl_nfci"]
    metrics = []
    for m in mnames:
        if m not in results["metrics"]:
            continue
        b = results["metrics"][m]["pooled_cal"]
        e = results["metrics"][m].get("event", {})
        metrics.append({
            "model": m, "brier": b["brier"], "log_loss": b["log_loss"],
            "roc_auc": b.get("roc_auc"), "pr_auc": b.get("pr_auc"), "ece": b["ece"],
            "detect": e.get("detect_rate"), "fas": e.get("n_false_alarms"),
            "lead": e.get("mean_lead_days"),
        })

    prod = results["selection"]["production_model"]
    return {
        "pred": pred,
        "hist": hist,
        "categories": out["category_summary"].to_dict(orient="records"),
        "snapshot": out["snapshot"].replace({np.nan: None}).to_dict(orient="records"),
        "health": health.replace({np.nan: None}).to_dict(orient="records"),
        "metrics": metrics,
        "events": results["events"][prod],
        "false_alarms": results["false_alarms"][prod],
        "reliability": {m: results["reliability"][m]
                        for m in ("D_ensemble", "B_elastic_net", "C_grad_boost")
                        if m in results["reliability"]},
        "selection": results["selection"],
        "theta": results["theta"],
        "long_model": results.get("long_model"),
        "refresh": out["refresh"],
        "card": (MODELS_DIR / "model_card.md").read_text(),
    }


def main() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--ttl-hours", type=float, default=1.0)
    args = ap.parse_args()

    payload = build_payload(ttl_hours=args.ttl_hours, offline=args.offline)
    tpl = (ROOT / "site" / "template.html").read_text()
    assert "/*__DATA__*/" in tpl
    html = tpl.replace("/*__DATA__*/", json.dumps(payload, default=str))
    out_path = ROOT / "site" / "recession_monitor.html"
    out_path.write_text(html)
    p = payload["pred"]["probabilities"]
    print(json.dumps({
        "built": str(out_path),
        "data_date": payload["pred"]["data_date"],
        "probabilities_pct": {f"{k}d": round(100 * float(v), 2) for k, v in p.items()},
    }, indent=2))
    return out_path


if __name__ == "__main__":
    main()
