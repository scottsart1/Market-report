# Model Card — U.S. Recession Probability Model v2.0.0

**Target.** Probability that the U.S. economy *enters* an NBER recession within
15/30/45/60/90 days, conditional on not currently being in one. Ground truth is
the NBER chronology via FRED `USREC`. Onset convention: the first calendar day
of the first recession month. NBER dating is monthly; sub-monthly horizon
distinctions come from the model's hazard term structure, not from the labels.

**Training period.** Weekly (Friday) observations 1968-06-01 to
2025-06-27 — 2610 rows, 8
recession onsets. Recession events observed: 1970-01-01, 1973-12-01, 1980-02-01, 1981-08-01, 1990-08-01, 2001-04-01, 2008-01-01, 2020-03-01.

**Features.** 117 transformations across yield curve,
monetary policy, credit, equity markets, labor, real activity, housing,
consumer, inflation, business surveys, trade and breadth composites (see
`config/indicators.yaml` and `docs/v2_series_notes.md`); every member sees the same feature set (no series is routed to the boosted model alone). Historical features are
point-in-time: ALFRED first-release values where vintages exist
(75 of 107 downloaded series,
median first vintage 2010-07-01; earlier history and
the series without vintages — DGS10, DGS2, DGS5, DGS1, DTB3, FEDFUNDS, BAA, AAA, BAA10YM, CPFF, MORTGAGE30US, BAMLH0A0HYM2, NASDAQCOM, VIXCLS, WTISPLC, DFF, T1YFF, MICH, EMELASA, CP3M, CPN3M, DRIWCIL, SUBLPDCLIWXWFRNQ, DRSDCILM, SUBLPDMOSXWBNQ, EBP, GZ_SPREAD, MTSR133FMS, TWEXBMTH, TWEXBGSMTH, VXOCLS, USEPUNEWSINDXM —
use latest values), actual release dates for timing, chain-linked levels across
index rebasings, and configured publication lags everywhere else. Model-based
indices (NFCI, CFNAI, EBP) enter training only from their first real-time
publication. Modeling switches in force: defaults.

**Methodology.** Base classifier for the 90-day onset window; probability
calibration selected out-of-sample among none/Platt/smoothed isotonic
({'A_yield_curve': 'platt', 'B_elastic_net': 'none', 'C_grad_boost': 'none', 'bl_constant': 'none', 'bl_sahm': 'platt', 'bl_nfci': 'platt', 'D_ensemble': 'members'}; applied in production: {'A_yield_curve': 'none', 'B_elastic_net': 'none', 'C_grad_boost': 'none', 'bl_sahm': 'none', 'bl_nfci': 'none', 'bl_constant': 'none'}, 1-year {'B_elastic_net': 'none', 'C_grad_boost': 'none'} — a fit that carries no ranking information on the calibration window falls back to the raw score); horizons mapped by a fitted hazard power law
P(h) = 1 - (1 - P90)^theta_h with monotone theta ({
15: 0.063, 30: 0.145, 45: 0.218, 60: 0.330, 90: 1.000 }), which
enforces P15 <= P30 <= P45 <= P60 <= P90 structurally. Production model:
**C_grad_boost**, selected by average rank across Brier, log loss, calibration
error, PR-AUC, event-detection rate and false alarms on pooled walk-forward
out-of-sample predictions. The production predictor repeats the exact
construction the walk-forward validated: model fitted on eligible history
through 2012-10-26, calibrator fitted on the purged tail
2013-04-26 → 2025-06-27 (1
recession event(s) in the calibration window).

**Out-of-sample performance** (pooled walk-forward OOS, calibrated inside each fold):

| Model | Brier | Log loss | ROC AUC | PR AUC | ECE |
|---|---|---|---|---|---|
| A_yield_curve | 0.1451 | 0.5540 | 0.845 | 0.147 | 0.1954 |
| B_elastic_net | 0.0806 | 0.2842 | 0.863 | 0.344 | 0.1421 |
| C_grad_boost | 0.0493 | 0.1810 | 0.875 | 0.263 | 0.0536 |
| D_ensemble | 0.0561 | 0.1954 | 0.862 | 0.321 | 0.0840 |
| bl_constant | 0.0336 | 0.1529 | 0.654 | 0.068 | 0.0270 |
| bl_sahm | 0.1872 | 0.5779 | 0.761 | 0.084 | 0.3356 |
| bl_nfci | 0.4630 | 2.1579 | 0.595 | 0.041 | 0.5209 |

**Against the constant baseline.** The unconditional base rate scores Brier
0.0336 and log loss 0.1529 on the same rows, against
0.0493 and 0.1810 for the production model: the
model earns its keep through ranking (ROC AUC 0.875 vs 0.654),
detection and lead time, not through pooled proper scores, because the
class-balanced members overstate probabilities in calm years (expected
calibration error 0.0536 vs 0.0270). Read the level with the band
and the classification, not as a precise frequency.

**1-year horizon.** A dedicated classifier (same architecture and validation,
purge widened to 455 days) serves the 365-day probability; it is floored at
P90 so the full horizon chain stays monotone. Out-of-sample (calibrated):
Brier 0.0961,
ROC AUC 0.914,
base rate 0.143;
6/6
recessions reached P1y >= 0.35 in the prior year.

**Fold-by-fold (out-of-sample, production model).** Each recession since 1978
is held out exactly once, so these are the leave-one-recession-out results;
F5 and F7 contain no onset and price false alarms only.

| Fold | Rows | Onset weeks | Brier | Log loss | ROC AUC | Recall@0.2 | FPR@0.2 |
|---|---|---|---|---|---|---|---|
| F1 1978-83 (1980, 1981-82 recessions) | 218 | 25 | 0.1900 | 0.5972 | 0.836 | 0.80 | 0.34 |
| F2 1984-92 (1990-91 recession) | 434 | 13 | 0.0314 | 0.1325 | 0.802 | 0.00 | 0.02 |
| F3 1993-02 (2001 recession) | 487 | 13 | 0.0246 | 0.0877 | 1.000 | 0.00 | 0.00 |
| F4 2003-10 (Great Recession) | 340 | 13 | 0.0816 | 0.2279 | 0.956 | 1.00 | 0.17 |
| F5 2011-17 (no recession) | 365 | 0 | 0.0000 | 0.0002 | nan | nan | nan |
| F6 2018-22 (COVID recession) | 253 | 13 | 0.0516 | 0.3366 | 0.744 | 0.00 | 0.00 |
| F7 2023-now (no recession) | 130 | 0 | 0.0142 | 0.0760 | nan | nan | nan |

**Recession-by-recession (out-of-sample, production model)**

| Onset | Max P90 in prior 90d | Lead days (first P >= 0.10, three times the base rate) |
|---|---|---|
| 1980-02-01 | 0.99 | 357 |
| 1981-08-01 | 0.67 | 344 |
| 1990-08-01 | 0.08 | 334 |
| 2001-04-01 | 0.07 | — |
| 2008-01-01 | 0.92 | 361 |
| 2020-03-01 | 0.01 | — |

**False alarms** (sustained P90 >= 0.21, six times the base rate, for 28+ days, no onset within a year):
0 episode(s) — none.

**Uncertainty.** 10–90% band from 60 three-year
moving-block bootstrap replicates; each replicate refits the served members
(C_grad_boost) and is scored through the
production calibrators. The band prices estimation uncertainty only.

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
