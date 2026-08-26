# U.S. Recession Probability Dashboard

A local Streamlit dashboard that estimates the probability that the U.S.
economy **enters an NBER-style recession within the next 15, 30, 45, 60 and
90 days**, from a calibrated, backtested probabilistic model built on
point-in-time (ALFRED first-release) macro and financial data.

![status](https://img.shields.io/badge/data-FRED%2FALFRED-blue)

## Quick start

```bash
git clone <this repo>
cd Market-report
pip install -r requirements.txt

cp .env.example .env        # then put your FRED API key in .env (free:
                            # https://fred.stlouisfed.org/docs/api/api_key.html)

streamlit run app.py
```

A trained production model ships in `models/trained/`, so the dashboard works
immediately: it refreshes the latest data from FRED, runs the model, saves the
prediction locally and renders. To retrain from scratch (fetches full history,
runs the walk-forward backtest, calibration, bootstrap and model selection —
roughly 10–30 minutes depending on network):

```bash
python -m recession.train            # full training + backtest + artifacts
python -m recession.train --fast     # skip the bootstrap uncertainty pass
python -m recession.predict          # print the current prediction as JSON
python -m pytest tests/              # leakage/monotonicity/label test suite
```

Requires Python 3.11+.

### Without an API key

The app still works: it falls back to FRED's public CSV endpoint
(latest-vintage values) plus configured publication lags. Timing look-ahead is
still removed, but **revision look-ahead remains** for revision-prone series —
the dashboard flags this in Data Health, and the ICE BofA OAS extras are
disabled automatically (public-domain credit spreads substitute). With a key,
training history uses true ALFRED first-release values and actual release
dates.

## What the model is

| Piece | Choice | Why |
|---|---|---|
| Target | Onset of an NBER recession within 90 days, conditional on not being in one | NBER chronology is the ground truth; onset = first day of the first recession month (monthly precision is inherited, documented, and not overstated) |
| Modeling grid | Weekly (Fridays), 1968→present | Daily rows pretend independence that isn't there; weekly + event-aware weights treat each of the 8 recessions as the real unit of information |
| Point-in-time features | ALFRED first-release values + real release dates (fallback: configured publication lags) | A forecaster in June cannot see June payrolls, or 2009 revisions of 2008 data |
| Models | A: yield-curve probit · B: elastic-net logistic · C: histogram gradient boosting · D: calibrated ensemble · plus constant / Sahm-rule / financial-conditions baselines | Compare rather than assume; the production model is selected on out-of-sample Brier score |
| Validation | Expanding-window, recession-aware folds with a 180-day purge between train and test | Every post-1978 recession is genuinely out of sample exactly once; two recession-free folds price false alarms |
| Calibration | none vs Platt vs isotonic, chosen and fitted **only on out-of-sample predictions** | A "30%" must mean ≈30% historically |
| Horizons | Hazard power law `P(h) = 1 − (1 − P90)^θₕ`, θ fitted OOS per horizon, monotone by pool-adjacent-violators | Structural `P15 ≤ P30 ≤ P45 ≤ P60 ≤ P90` — no sorting of unrelated predictions; 15-day-only classifiers would rest on a handful of events |
| Uncertainty | 2-year moving-block bootstrap, 10–90% band | ~8 recessions of history ⇒ ranges, not decimals |
| Explanations | Feature-substitution attribution from the actual model + fixed plain-English templates from config | Never narrative-invented |

## Project layout

```
app.py                     Streamlit dashboard (7 sections + methodology)
config/indicators.yaml     Every indicator: series id, source, category, frequency,
                           publication lag, vintage availability, tier, license,
                           transforms with risk direction + plain-English label
src/recession/
    fred_client.py         FRED/ALFRED client: retries, throttling, caching,
                           keyless fallback, first-release vintages
    pipeline.py            Fetch → derived series → point-in-time panels → health
    features.py            Trailing-only transform library + as-of materialization
    labels.py              NBER onset convention, y15..y90, eligibility
    models.py              Models A–D + baselines (leak-safe sklearn pipelines)
    horizon.py             Calibrators + hazard term structure (monotone)
    backtest.py            Expanding-window backtest, OOS calibration, events
    uncertainty.py         Block-bootstrap bands
    explain.py             Substitution attribution + factor phrasing
    categories.py          0–100 category stress scores (diagnostic)
    predict.py             Production prediction + governance metadata
    history.py             SQLite prediction history
    train.py               End-to-end training CLI, artifacts, model card
models/trained/            production_bundle.joblib, backtest_results.json,
                           oos_predictions.parquet, model_card.md
data/                      raw/ (per-series cache) — gitignored, rebuilt from APIs
tests/                     leakage, labels, monotonicity, staleness, folds
```

## Data sources

All series come from FRED/ALFRED (Federal Reserve Bank of St. Louis), most of
them U.S.-government public-domain series (Treasury yields, BLS labor data,
Fed industrial production, Census housing, BEA income/consumption, NBER
chronology). Licensing-restricted series (ICE BofA OAS, S&P 500) are optional
extras with public-domain substitutes (Moody's Baa/Aaa spreads, Nasdaq
Composite for long-history equity behavior). Non-financial/high-frequency
candidates include weekly claims, heavy-truck sales, vehicle miles traveled
and the BTS freight index. See `config/indicators.yaml` for the full universe
with per-series notes.

## Honest limitations

- **~8 usable recessions.** Every metric, band and probability inherits this.
  The bootstrap band is a lower bound on true uncertainty.
- **NBER dating is monthly and late.** Sub-monthly horizons are a modeling
  convention (documented in-app); recent "no recession" labels are provisional.
- **Vintage coverage varies.** ALFRED vintages start in the 1990s–2000s for
  many series; earlier history uses earliest-vintage values (flagged).
- **2020 shows what exogenous shocks do** to any 90-day macro model.
- This is an analytical tool, not investment advice.
