"""Compare recorded experiments side by side.

  python scripts/compare.py                      # every experiments/*.json
  python scripts/compare.py baseline wide_all    # named ones, first = reference
  python scripts/compare.py --model B_elastic_net

Prints a markdown table of pooled out-of-sample metrics for the chosen model
(default D_ensemble) and the 1-year ensemble, then per-fold Brier scores
against the reference experiment so a gain can be checked for consistency
across recessions rather than taken from one pooled number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"


def load(names: list[str]) -> list[dict]:
    paths = [EXP_DIR / f"{n}.json" for n in names] if names else sorted(EXP_DIR.glob("*.json"))
    out = []
    for p in paths:
        if not p.exists():
            sys.exit(f"missing experiment: {p}")
        out.append(json.loads(p.read_text()))
    return out


def fold_brier(name: str, model: str) -> pd.Series:
    p = EXP_DIR / f"{name}.oos.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    oos = pd.read_parquet(p)
    col = f"p_cal_{model}"
    sub = oos[oos["eligible"].astype(bool) & oos[col].notna()]
    return sub.groupby("fold").apply(lambda d: float(np.mean((d[col] - d["y90"]) ** 2)))


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def paired_bootstrap(ref: str, other: str, model: str, horizon: str = "90", n_boot: int = 2000,
                     block: int = 52, seed: int = 0) -> dict:
    """Moving-block bootstrap of the mean Brier difference (other - ref).

    Weekly rows are strongly autocorrelated and the positives cluster in ~8
    episodes, so a plain t-test overstates certainty; 1-year blocks keep
    whole episodes together. Returns the point difference, a 90% interval
    and the share of resamples in which `other` is better.
    """
    if horizon == "90":
        a = pd.read_parquet(EXP_DIR / f"{ref}.oos.parquet")
        b = pd.read_parquet(EXP_DIR / f"{other}.oos.parquet")
        col, ycol, elig = f"p_cal_{model}", "y90", "eligible"
    else:
        a = pd.read_parquet(EXP_DIR / f"{ref}.long.parquet")
        b = pd.read_parquet(EXP_DIR / f"{other}.long.parquet")
        col, ycol, elig = "p_cal_long", "y365", "eligible_long"
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    ok = a[elig].astype(bool) & a[col].notna() & b[col].notna()
    y = a.loc[ok, ycol].astype(float).to_numpy()
    d = (b.loc[ok, col].to_numpy() - y) ** 2 - (a.loc[ok, col].to_numpy() - y) ** 2
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(n - block, 1), size=(n_boot, int(np.ceil(n / block))))
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([np.arange(s, min(s + block, n)) for s in starts[i]])[:n]
        means[i] = d[pick].mean()
    return {"n": n, "diff": float(d.mean()), "ci90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
            "share_better": float((means < 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--model", default="D_ensemble")
    ap.add_argument("--paired", action="store_true",
                    help="block-bootstrap the Brier difference of each experiment vs the first")
    args = ap.parse_args()
    exps = load(args.names)
    m = args.model

    print(f"| experiment | feats | {m} Brier | logloss | AUC | PR-AUC | ECE | detect | FAs | lead d | 1y Brier | 1y logloss | 1y AUC | prod |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in exps:
        pc = e["metrics"].get(m, {}).get("pooled_cal", {})
        ev = e["metrics"].get(m, {}).get("event", {})
        l1 = (e.get("long") or {}).get("metrics", {}).get("ensemble_1y", {})
        print(f"| {e['name']} | {e['n_features']} | {fmt(pc.get('brier'))} | {fmt(pc.get('log_loss'))} | "
              f"{fmt(pc.get('roc_auc'), 3)} | {fmt(pc.get('pr_auc'), 3)} | {fmt(pc.get('ece'), 3)} | "
              f"{ev.get('detect_rate')} | {ev.get('n_false_alarms')} | {fmt(ev.get('mean_lead_days'), 0)} | "
              f"{fmt(l1.get('brier'))} | {fmt(l1.get('log_loss'))} | {fmt(l1.get('roc_auc'), 3)} | "
              f"{e['selection'].get('production_model')} |")

    ref = exps[0]["name"]
    base = fold_brier(ref, m)
    if len(exps) > 1 and len(base):
        print(f"\nPer-fold {m} Brier, change vs `{ref}` (negative = better):\n")
        folds = list(base.index)
        print("| experiment | " + " | ".join(f"F{int(f)}" for f in folds) + " | folds better |")
        print("|---|" + "---|" * (len(folds) + 1))
        for e in exps[1:]:
            fb = fold_brier(e["name"], m).reindex(folds)
            d = fb - base
            print(f"| {e['name']} | " + " | ".join(fmt(x) for x in d) +
                  f" | {int((d < 0).sum())}/{int(d.notna().sum())} |")

    if args.paired and len(exps) > 1:
        print(f"\nBlock-bootstrap Brier difference vs `{ref}` (negative = better; 52-week blocks):\n")
        print("| experiment | horizon | n rows | diff | 90% interval | share of resamples better |")
        print("|---|---|---|---|---|---|")
        for e in exps[1:]:
            for hz in ("90", "365"):
                try:
                    r = paired_bootstrap(ref, e["name"], m, hz)
                except FileNotFoundError:
                    continue
                print(f"| {e['name']} | {hz}d | {r['n']} | {r['diff']:+.5f} | "
                      f"[{r['ci90'][0]:+.5f}, {r['ci90'][1]:+.5f}] | {r['share_better']:.0%} |")


if __name__ == "__main__":
    main()
