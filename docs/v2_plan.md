# v2 plan: expanded universe, corrected modeling

Written 2026-10-13 from a research pass of seven variable scouts (263 series
verified against the FRED/ALFRED API), three independent modeling audits
(econometric, small-event ML, validation/leakage) and a completeness critic.
Every change below is judged by the walk-forward backtest through
`scripts/experiment.py`; the acceptance rule is fixed here, before any
result is looked at.

## Acceptance rule (pre-registered)

A modeling change or a batch of new features is adopted only if, against the
reference run of the same stage:

1. the 90-day ensemble's Brier score **and** log loss improve in at least 4
   of the 5 folds that contain a recession onset (F1 1978-83, F2 1984-92,
   F3 1993-02, F4 2003-10, F6 2018-22);
2. the 52-week block-bootstrap share of resamples in which it is better is
   at least 80% at 90 days (`scripts/compare.py --paired`);
3. no recession's maximum pre-onset probability falls below 3x the training
   base rate;
4. the 1-year ensemble is not worse by the same block-bootstrap test
   (share better >= 50%).

Folds F5 (2011-17) and F7 (2023-now) contain no onset: they are reported but
never decide a change. Bug fixes that remove a fitted parameter or a
look-ahead channel are adopted on correctness grounds and their effect is
reported, not gated. Feature batches are pre-specified blocks (below), not
chosen one series at a time on the backtest.

## Stage 1: correctness (adopted on correctness grounds)

| # | Finding (confirmed by) | Fix |
|---|---|---|
| 1 | Hazard exponent theta_90 = 0.495 fitted in-sample on the pooled OOS rows it is then scored on; served P90 can never exceed ~0.29 (all three audits) | theta_90 = 1 by construction; the 15-60 day exponents fitted walk-forward (fold k from folds < k) and reported as such |
| 2 | Transforms shift by row count: a missing month (Oct-2025 shutdown), quarterly pre-1978 sentiment, sparse series silently change a feature's horizon (validation) | Monthly/quarterly/weekly frames are reindexed to a regular period grid before any transform or derive |
| 3 | Backfill heuristic dates shutdown-delayed releases to period_end + lag, 1-3 months early (validation) | A row is backfill only if its release date is the series' first vintage date; late releases keep their true date; release dates persisted in the cache |
| 4 | SLOOS dated quarter-end + 40 days although it is released ~56 days before quarter end; Philly Fed surveys released before month end (econometric, validation, critic) | Negative publication lags allowed; configured lags validated against observed first-release gaps |
| 5 | 1-year model's inner calibration split purges 180 days, not 455 (econometric, validation) | `inner_split(..., horizon_days)` |
| 6 | Single-event isotonic calibrators degenerate (F2 maps all test rows to 0; production caps B at 4.7%) and the informativeness guard cannot see it (validation, econometric, ML) | A calibrator needs >= 2 distinct onsets in its window; isotonic replaced by Platt with slope shrunk toward 1 and linear-logit extrapolation; range checks |
| 7 | Elastic-net C chosen on an unpurged one-event tail, always returns the grid floor, and collapses the model when the grid is widened (ML, validation) | Fixed C = 0.01 |
| 8 | As-of join carries a dead series forward forever; splice offsets use the whole overlap (validation) | Staleness tolerance; splice offset fixed at the first overlap date |
| 9 | Provisional NBER labels: the last 12 months of "no recession" are not yet NBER-confirmed (validation) | 365-day label-maturity lag for training/calibration eligibility |

## Stage 2: modeling recipe (tested under the rule)

Variants, each switchable through `RECESSION_VARIANTS`:

- `class_balance=0`: event-equalised weights without class balancing (every
  recession still carries equal total weight; positives keep their natural
  share). Reproduced by two auditors: D Brier 0.0405 -> 0.0318, AUC 0.857 ->
  0.900.
- `hgb_monotone=1`: boosted member constrained to the config risk directions.
- `calibration=prior_oos`: fold k's calibrator fitted on the pooled raw OOS
  scores of folds < k (Platt, slope guard), production on all folds;
  `member_fit=full`: production members fitted on every eligible row
  through train_end instead of the first 75%.
- `ensemble=A,B,C`: the yield-curve probit (weighted like the others) as a
  third member; plain mean of calibrated members, no learned weights.
- Event metrics at base-rate-relative thresholds (detection 3x, alarm 6x,
  365-day look-ahead), folds >= 2 and leave-one-recession-out tables.

The 2020 event keeps full training weight; nothing is tuned on F7.

## Stage 3: variables

Routing: the elastic net keeps a curated core; everything new goes to the
monotone boosted model (the ML audit's noise-copy stress test showed it is
the member that tolerates a 2-3x larger universe). Every new transform
carries a risk direction. Batches:

- **Curve & policy**: DFF; spreads 2y-FF, 10y-FF, 3m-FF; near-term forward
  spread proxy 2.25*DGS2 - 1.25*DGS1 - DTB3; lagged 10y-3m (6m, 12m), its
  1-year minimum and inversion depth; FEDFUNDS chg24m and rise from 3-year
  low; curve-inversion breadth (share of seven pairs inverted, persistence).
- **Labor onset**: NSA claims 4-week average yoy (ICNSA), insured unemployment
  rate (IURSA, weekly Sahm analogue), job-loser rate U2RATE (+ Sahm gap),
  temporary layoffs LNS13023653, aggregate hours AWHI, construction
  employment USCONS, prime-age employment ratio LNS12300060 (replacing
  EMRATIO), layoff transition rate EMELASA (tier 2), a labor-onset rule count.
- **Surveys**: Philadelphia Fed current activity, new orders, employment
  (1968-, released before month end), PHILLY_ISM (ISM-weighted Philly
  components), regional survey z-mean (Philly, Empire, Dallas, NaN-skipping).
- **Credit**: paper-bill spread splice (CP3M -> CPN3M, 1971-), willingness-to-lend
  splice (1966-), SLOOS C&I demand, C&I loans BUSLOANS (credit-boom sign),
  business-loan delinquency DRBLACBS, card charge-offs CORCCACBS, excess bond
  premium EBP and GZ spread (Fed CSV, masked before first publication),
  NFCI credit sub-index change (masked before 2011), a credit-supply stress
  composite from raw components.
- **Housing/consumer/activity**: single-family permits, months' supply
  MSACSR, FHFA house prices (real, 1-year model), regional permit breadth,
  core capital-goods orders ADXTNO, consumer credit total TOTALSL (replacing
  REVOLSL, risk +1), unrevised federal receipts, industrial commodity PPI,
  broad dollar splice, VXO/VIX splice, policy-uncertainty news index (1900-).
- **Composites** (trailing-only, fixed membership of long-history inputs):
  BREADTH_STRESS, LABOR_ONSET_DIFF, CURVE_INV_BREADTH, HOUSING_BREADTH,
  CREDIT_SUPPLY_STRESS, COINC_BIG4 (coincident momentum).
- **Masks**: `real_time_from` per model-based index (NFCI family, CFNAI, EBP,
  GZ): values before first publication are NaN in training. First-release
  levels chain-linked across base-year/unit changes before growth
  transforms (interim fix; full vintage-window growth is future work).

Not added, with reasons recorded by the critic: OECD CLIs, BBK, ADS, KCFSI,
STLFSI, OFR FSI (two-sided or re-estimated model outputs); ICE BofA, S&P,
Conference Board, ISM, NAHB, MBA (licence); the EBP file's `est_prob`
column and Chauvet-Piger probabilities (model outputs, benchmark only).

## Stage 4: uncertainty, documentation, publication

Bootstrap bands through all members and their calibrators (3-year blocks);
model card with folds >= 2, leave-one-recession-out, per-fold first-release
coverage and calibrator ranges; methodology text on the page; adversarial
review of the diff for look-ahead before retraining and publishing.

## Stage 3 implementation notes and look-ahead review (2026-10-13)

What the full-universe build showed (143 enabled indicators, 225 feature
specs, 222 panel columns; 3380 weekly rows from 1962 as before):

- **Release timing verified against ALFRED.** For every series with
  vintages the median first-release lag matches period end + configured lag
  within a few days, including the negative lags: Philly Fed surveys
  (-11, observed 17 days after month start), Empire State (-15, 14), Dallas
  (-3/-2, 27) and the SLOOS family (-55, 34 days after quarter start).
- **Composite coverage columns** (`<ID>__coverage`) are diagnostics and are
  dropped before the panel, so the share of members present (which trends
  up over time) never becomes a predictor.
- **Splice `offset: auto`** measures the level step on the first four
  overlapping observations, i.e. values from a few months after each switch
  (1982, 1990, 1997, 2006). The step is a constant, so the only look-ahead
  is that constant during those months; all switches predate the first
  test fold's lessons by decades except the 2006 dollar splice.
- **Chain-linking** of first-release levels uses the latest vintage to
  locate rebasings; it changes the scale of the past, never its growth
  rates, and a rebasing is an event announced when it happens.
- **Model-based indices** (NFCI family, CFNAI, EBP, GZ) enter training from
  their first real-time publication only; EBP and the GZ spread are
  therefore 83-93% missing since 1968 and are routed to the boosted model
  only (`core: false`), as are the Dallas Fed surveys (2004-).
- **Discontinued splice legs** (CP3M, SUBLPDCLIWXWFRNQ, TWEXBMTH, VXOCLS)
  are flagged `discontinued` so Data Health reports them as archived rather
  than stale and the confidence grade ignores them.
- **HGB regularisation** is unchanged (depth 3, 250 rounds at 0.05, leaves
  of 40 weeks, L2 1.0, 70% feature subsampling, no early stopping so no
  inner split is consulted); the wide set reaches it only under the config's
  monotone directions when `hgb_monotone` is on.
- The `invshare2y` transform referenced by the 10Y-3M entry did not exist
  and was added (share of the trailing two years spent inverted).
