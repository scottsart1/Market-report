# Model Card — U.S. Recession Probability Model v1.1.0

**Target.** Probability that the U.S. economy *enters* an NBER recession within
15/30/45/60/90 days, conditional on not currently being in one. Ground truth is
the NBER chronology via FRED `USREC`. Onset convention: the first calendar day
of the first recession month. NBER dating is monthly; sub-monthly horizon
distinctions come from the model's hazard term structure, not from the labels.

**Training period.** Weekly (Friday) observations 1968-06-01 to
2026-05-01 — 2654 rows, 8
recession onsets. Recession events observed: 1970-01-01, 1973-12-01, 1980-02-01, 1981-08-01, 1990-08-01, 2001-04-01, 2008-01-01, 2020-03-01.

**Features.** 85 transformations across yield curve,
monetary policy, credit, equity markets, labor, real activity, housing,
consumer, inflation, and non-financial/high-frequency categories (see
`config/indicators.yaml`). Historical features are point-in-time: ALFRED
first-release values where vintages exist, and configured publication lags for
release timing everywhere.

**Methodology.** Base classifier for the 90-day onset window; probability
calibration selected out-of-sample among none/Platt/isotonic
({'A_yield_curve': 'none', 'B_elastic_net': 'isotonic', 'C_grad_boost': 'none', 'bl_constant': 'isotonic', 'bl_sahm': 'platt', 'bl_nfci': 'platt', 'D_ensemble': 'members'}); horizons mapped by a fitted hazard power law
P(h) = 1 - (1 - P90)^theta_h with monotone theta ({
15: 0.073, 30: 0.165, 45: 0.247, 60: 0.330, 90: 0.495 }), which
enforces P15 <= P30 <= P45 <= P60 <= P90 structurally. Production model:
**D_ensemble**, selected by average rank across Brier, log loss, calibration
error, PR-AUC, event-detection rate and false alarms on pooled walk-forward
out-of-sample predictions. The production predictor repeats the exact
construction the walk-forward validated: model fitted on eligible history
through 2013-06-14, calibrator fitted on the purged tail
2013-12-13 → 2026-05-01 (1
recession event(s) in the calibration window).

**Out-of-sample performance** (pooled walk-forward OOS, calibrated inside each fold):

| Model | Brier | Log loss | ROC AUC | PR AUC | ECE |
|---|---|---|---|---|---|
| A_yield_curve | 0.0436 | 0.1931 | 0.823 | 0.146 | 0.0424 |
| B_elastic_net | 0.0501 | 0.2784 | 0.768 | 0.330 | 0.0759 |
| C_grad_boost | 0.0425 | 0.1520 | 0.899 | 0.242 | 0.0458 |
| D_ensemble | 0.0409 | 0.1507 | 0.875 | 0.325 | 0.0484 |
| bl_constant | 0.0531 | 0.2076 | 0.590 | 0.065 | 0.0924 |
| bl_sahm | 0.0644 | 0.2708 | 0.731 | 0.109 | 0.1047 |
| bl_nfci | 0.0434 | 0.1787 | 0.700 | 0.260 | 0.0616 |

**1-year horizon.** A dedicated classifier (same architecture and validation,
purge widened to 455 days) serves the 365-day probability; it is floored at
P90 so the full horizon chain stays monotone. Out-of-sample (calibrated):
Brier 0.1300,
ROC AUC 0.900,
base rate 0.140;
6/6
recessions reached P1y >= 0.35 in the prior year.

**Recession-by-recession (out-of-sample, production model)**

| Onset | Max P90 in prior 90d | Lead days (first P>=0.20) |
|---|---|---|
| 1980-02-01 | 0.97 | 364 |
| 1981-08-01 | 0.69 | 365 |
| 1990-08-01 | 0.03 | — |
| 2001-04-01 | 0.22 | 128 |
| 2008-01-01 | 0.73 | 361 |
| 2020-03-01 | 0.01 | — |

**False alarms** (sustained P90 >= 0.30 for 28+ days, no onset within 270d):
0 episode(s) — none.

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
