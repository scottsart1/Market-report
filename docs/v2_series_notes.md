# v2 series notes: what the Stage-3 universe adds to `config/indicators.yaml` and why

Companion to `docs/v2_plan.md` (Stage 3: variables). Drafted 2026-10-13 from
the scouts' API-verified candidate list (263 series), the three audits and the
completeness critic's verdicts. The additions were drafted as a separate file and have since been merged
into `config/indicators.yaml` (`replaces:` names the base id an entry
supersedes, and `composites:` is a new top-level list).

## 1. Counts

| batch (plan) | entries | of which parents with no transforms | feature columns |
|---|---|---|---|
| Curve & policy | 15 | 2 (MICH, M2SL) | 24 |
| Labor onset | 12 | 1 (ICNSA) | 20 |
| Surveys | 11 | 3 (SHCDFSA, DTCDFSA, IVCDFSA) | 18 |
| Credit | 16 | 7 (legs: CP3M, CPN3M, TB3MS, PAPER_BILL_PRE97/POST97, SUBLPDCLIWXWFRNQ, DRIWCIL) | 19 |
| Housing / consumer / activity / markets-uncertainty | 22 | 4 (USSTHPI, TWEXBMTH, TWEXBGSMTH, VXOCLS) | 38 |
| **Total indicator entries** | **76** (53 FRED, 2 url, 21 derived) | 17 | **119** |
| Composites | 7 (the plan's six plus REGIONAL_SURVEY_Z) | | 14 |

Merged with the 63 base entries the repo loader reports 146 indicators and
234 feature columns (214 tagged for the 90-day model, 189 for the 1-year
model); the curated core (elastic net) grows by six series and three
composites, everything else is routed to the monotone boosted model
(`core: false`).

Validation run: `python -c "from recession.config import load_config; load_config()"`
parses; a merged copy loads through `recession.config.load_config`; every
derive input, composite member and rule feature resolves to an id or feature
in the base file or the additions; transform codes are either implemented or
on the planned list (lagNm, minNy, invdepthNy, riseNy); categories are in
`CATEGORIES`; horizons are subsets of {90, 365}.

## 2. Conventions applied

* **Routing.** `core: false` on every entry unless named here as core:
  DFF (chg24m, rise3y), T1YFF, SPREAD_2Y_FFR, NTFS_PROXY, SPREAD_10Y3M_LAGS,
  ICNSA_4WK, and the composites BREADTH_STRESS, CURVE_INV_BREADTH,
  COINC_BIG4. These are the audit's curated curve/claims/breadth set (its
  15-feature list is an in-sample upper bound, so only the literature-prior
  members were promoted). The loader defaults `core` to true for the legacy
  universe, hence the explicit key on every new entry.
* **Horizons.** Short-window labor and coincident changes, credit-spread
  changes, implied volatility and survey changes are tagged `[90]`; curve
  lags at 12 months, housing levels, quarterly credit quality, consumer
  credit growth and the cumulative-tightening transforms are tagged `[365]`
  or both, following the audit's horizon profiles (claims 0.84 at 90d vs
  0.70 at 365d; 10y-3m level 0.86 vs 0.90; FEDFUNDS rise3y 0.81 vs 0.88).
* **Masks (`real_time_from`).** EBP and GZ_SPREAD 2016-06-01 (first FEDS
  Note publication; the file is re-estimated monthly and has no vintages;
  with the mask they inform the 2020 event only). NFCICREDIT 2012-07-05 and
  CFNAIDIFF 2012-06-25: the API-verified first ALFRED vintage of each series
  rather than the family dates suggested (NFCI 2011-05-27, CFNAI 2011-05),
  because the sub-index and the diffusion index were not on FRED/ALFRED until
  then and the API date is the verifiable, conservative choice; the cost is
  about one year of 2011-12 rows with no onset. EMELASA is **not** masked
  (judgement, see section 5). USEPUNEWSINDXM is not masked: it is revised as
  newspaper archives update but it is a word-count index, not a factor model
  with full-sample loadings.
* **Negative lags.** Philadelphia Fed BOS -11 for every component (the
  critic's single convention; the labor scout's 0 for NECDFSA is dropped),
  Empire State -15, Dallas manufacturing -3, Dallas services -2, SLOOS -55
  (FRED dates a survey at the quarter in which it is taken; the audit's
  observed median for DRTSCILM is -56).
* **vintage.** true only for revision-prone FRED series; false for market
  rates, unrevised surveys (SLOOS, Michigan expectations are the exception:
  MICH is true so that ALFRED release dates govern timing from 1999, as the
  base file does for UMCSENT), url sources and every derived entry (the base
  file sets true on some derived entries; this file follows the v2 rule).
* **Replacements.** LNS12300060 replaces EMRATIO (EMRATIO chg6m AUC 0.47,
  retirement drift); TOTALSL replaces REVOLSL (yoy sign-reversed, AUC 0.37,
  unit break in first releases; risk +1 on consumer-credit growth). The base
  ids are left fetched; the merge step decides whether to disable their
  transforms.
* **Licence.** All public domain except the CBOE VXO (fred_full, so
  VIX_SPLICED and CREDIT_SUPPLY_STRESS inherit fred_full) and the
  policy-uncertainty index (fred_full). No restricted series is added; the
  copyrighted Case-Shiller index is replaced by FHFA USSTHPI.

## 3. Additions

Transform notation: `code(sign, horizons)`; no horizons = both models.

| id | batch | source | freq | start | lag | vintage | real_time_from | core | tier | transforms (risk, horizons) | replaces | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFF | curve & policy | fred | d | 1954-07 | 1 | false |  | true | 1 | chg24m(+); rise3y(+) |  | parent of the fed-funds spreads; cumulative tightening (chg24m, rise3y) robust across eras |
| T1YFF | curve & policy | fred | d | 1962-01 | 1 | false |  | true | 1 | level(-); chg6m(-,90); inv_share1y(+) |  | near-term easing component of the curve, 1962, all eight recessions (critic top-20) |
| SPREAD_2Y_FFR | curve & policy | derived/spread | d | 1976-06 | 1 | false |  | true | 1 | level(-); chg3m(-,90); resteep(+,90) |  | market-implied two-year easing; strongest new curve feature (AUC90 0.89) |
| SPREAD_10Y_FFR | curve & policy | derived/spread | d | 1962-01 | 1 | false |  | false | 1 | level(-); inv_share1y(+) |  | long end vs policy rate; CURVE_INV_BREADTH member |
| SPREAD_3M_FFR | curve & policy | derived/spread | d | 1954-07 | 1 | false |  | false | 1 | level(-); avg1m(-,90) |  | bills vs funds: cuts priced / flight to bills, 1954; breadth member |
| SPREAD_5Y2Y | curve & policy | derived/spread | d | 1976-06 | 1 | false |  | false | 1 | level(-); inv_share1y(+) |  | belly slope, inverts earliest; kept only as a breadth member |
| SPREAD_2Y3M | curve & policy | derived/spread | d | 1976-06 | 1 | false |  | false | 1 | level(-); inv_share1y(+) |  | front-end slope; kept only as a breadth member |
| NTFS_PROXY | curve & policy | derived/ntfs | d | 1976-06 | 1 | false |  | true | 1 | level(-); chg6m(-,90) |  | Engstrom-Sharpe near-term forward spread proxy (corr 0.98 with GSW NTFS) |
| SPREAD_10Y3M_LAGS | curve & policy | derived/spread | d | 1962-01 | 1 | false |  | true | 1 | lag6m(-,90); lag12m(-,365); min1y(-); invdepth1y(+) |  | distributed-lag term spread: lag6m best single 90d predictor (AUC 0.915) |
| TAYLOR_GAP | curve & policy | derived/taylor_gap | m | ~1962 (10y u* warm-up) | 15 | false |  | false | 1 | level(+); chg12m(+,365) |  | policy stance vs a real-time Taylor rule (existing recipe; NROU rejected) |
| MICH | curve & policy | fred | m | 1978-01 | 2 | true |  | false | 1 | (parent only) |  | parent of the ex-ante real rate |
| REAL_FFR_EXANTE | curve & policy | derived/spread | m | 1978-01 | 3 | false |  | false | 1 | level(+); chg12m(+,365) |  | forward-looking real policy rate, 1978- |
| M2SL | curve & policy | fred | m | 1959-01 | 26 | true |  | false | 1 | (parent only) |  | parent of REAL_M2 |
| REAL_M2 | curve & policy | derived/deflate | m | 1959-01 | 26 | false |  | false | 2 | yoy(-); ann6m(-,90) |  | LEI money component until 2012; tier-2 test (false alarm 2023) |
| BORROW | curve & policy | fred | m | 1959-01 | 26 | false |  | false | 2 | pctile3y(+,90) |  | bank funding stress at the discount window; percentile only across regimes |
| ICNSA | labor onset | fred | w | 1967-01 | 5 | true |  | false | 1 | (parent only) |  | parent of the NSA 4-week average |
| ICNSA_4WK | labor onset | derived/avg4w | w | 1967-01 | 5 | false |  | true | 1 | yoy(+,90) |  | same-week yoy immune to SA-factor revisions; >10-15% rise before every onset since 1969 |
| IURSA | labor onset | fred | w | 1971-01 | 12 | true |  | false | 1 | chg3m(+,90); chg6m(+); pctile3y(+) |  | stationary weekly twin of the unemployment rate, 12-day lag |
| IURSA_SAHM | labor onset | derived/sahm | w | 1971-01 | 12 | false |  | false | 1 | level(+,90) |  | weekly Sahm analogue |
| U2RATE | labor onset | fred | m | 1967-01 | 7 | true |  | false | 1 | chg3m(+,90); chg6m(+); chg12m(+,365) |  | involuntary separations isolated from participation swings |
| U2RATE_SAHM | labor onset | derived/sahm | m | 1967-01 | 7 | false |  | false | 1 | level(+,90) |  | Sahm gap on job losers |
| LNS13023653 | labor onset | fred | m | 1967-01 | 7 | true |  | false | 1 | yoy(+,90); pct3m(+,90) |  | temporary layoffs, the first margin firms adjust (correct id; lag fixed to 7) |
| AWHI | labor onset | fred | m | 1964-01 | 7 | true |  | false | 1 | ann3m(-,90); ann6m(-); accel3m(-,90) |  | total labor input; hours cut before heads |
| USCONS | labor onset | fred | m | 1939-01 | 7 | true |  | false | 1 | ann6m(-); yoy(-); dd1y(-,90) |  | most rate-sensitive sector, 1939, vintages 1961 |
| LNS12300060 | labor onset | fred | m | 1948-01 | 7 | true |  | false | 1 | chg6m(-); chg12m(-,365) | EMRATIO | prime-age EPOP without retirement drift; swap for EMRATIO (AUC 0.47) |
| EMELASA | labor onset | fred | m | 1978-01 | 12 | false |  | false | 2 | avg3m(+,90); chg6m(+,90) |  | CPS layoff hazard to 1978; research series, tier 2 |
| CE16OV | labor onset | fred | m | 1948-01 | 7 | true |  | false | 1 | ann3m(-,90) |  | household employment turned before payrolls in 1990/2001/2007; rule input |
| GACDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | level(-); avg3m(-); chg6m(-,90) |  | only free survey covering all eight recessions, released before month end |
| NOCDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | level(-); avg3m(-); chg6m(-,90) |  | most leading PMI component; PHILLY_ISM input |
| NECDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | level(-); avg3m(-) |  | earliest labor-demand gauge each month; rule input |
| GAFDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | level(-); chg6m(-,90) |  | six-month expectations |
| SHCDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | (parent only) |  | PHILLY_ISM input only |
| DTCDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | (parent only) |  | PHILLY_ISM input only |
| IVCDFSA066MSFRBPHI | surveys | fred | m | 1968-05 | -11 | true |  | false | 1 | (parent only) |  | PHILLY_ISM input only |
| PHILLY_ISM | surveys | derived/wsum | m | 1968-05 | -11 | false |  | false | 1 | level(-); avg3m(-); chg6m(-,90) |  | licence-clean PMI proxy with fixed ISM weights, 1968 |
| GACDISA066MSFRBNY | surveys | fred | m | 2001-07 | -15 | true |  | false | 2 | level(-); avg3m(-,90) |  | second district for the z-mean (2001-07, covers 2008/2020 only) |
| BACTSAMFRBDAL | surveys | fred | m | 2004-06 | -3 | true |  | false | 2 | level(-) |  | third manufacturing district (2004) |
| TSSOSBACTSAMFRBDAL | surveys | fred | m | 2007-01 | -2 | true |  | false | 2 | level(-) |  | only free services survey with >1 recession (2007) |
| CP3M | credit | fred | m | 1971-04 | 3 | false |  | false | 1 | (parent only) |  | 1971-97 paper leg |
| CPN3M | credit | fred | m | 1997-01 | 3 | false |  | false | 1 | (parent only) |  | 1997- paper leg |
| TB3MS | credit | fred | m | 1934-01 | 3 | false |  | false | 1 | (parent only) |  | monthly bill for like-for-like paper-bill spread (unverified id) |
| PAPER_BILL_PRE97 | credit | derived/spread | m | 1971-04 | 3 | false |  | false | 1 | (parent only) |  | spread leg |
| PAPER_BILL_POST97 | credit | derived/spread | m | 1997-01 | 3 | false |  | false | 1 | (parent only) |  | spread leg |
| PAPER_BILL | credit | derived/splice | m | 1971-04 | 3 | false |  | false | 1 | level(+); chg3m(+,90) |  | classic paper-bill spread extended to seven recessions |
| SUBLPDCLIWXWFRNQ | credit | fred | q | 1966-07 | -55 | false |  | false | 1 | (parent only) |  | 1966-2017 willingness leg |
| DRIWCIL | credit | fred | q | 1982-04 | -55 | false |  | false | 1 | (parent only) |  | 1982- willingness leg |
| WILLING_LEND | credit | derived/splice | q | 1966-07 | -55 | false |  | false | 1 | level(-); chg12m(-) |  | only SLOOS question spanning all eight recessions |
| DRSDCILM | credit | fred | q | 1991-10 | -55 | false |  | false | 2 | level(-); chg12m(-) |  | firm-side loan demand (Bassett et al.) |
| BUSLOANS | credit | fred | m | 1947-01 | 12 | true |  | false | 1 | ann6m(+,365); yoy(+,365); accel3m(-,90) |  | late-cycle credit boom (+1) and credit stop (accel3m, -1) |
| DRBLACBS | credit | fred | q | 1987-01 | 56 | true |  | false | 2 | chg12m(+,365); chg6m(+) |  | business delinquencies lead onset by 2-4 quarters |
| CORCCACBS | credit | fred | q | 1985-01 | 56 | true |  | false | 2 | chg12m(+,365) |  | realised household losses, 1985 |
| EBP | credit | url | m | 1973-01 | 10 | false | 2016-06-01 | false | 2 | level(+); chg3m(+,90); z10y(+) |  | GZ excess bond premium; masked before the 2016 FEDS note |
| GZ_SPREAD | credit | url | m | 1973-01 | 10 | false | 2016-06-01 | false | 2 | level(+); chg6m(+,90) |  | duration-matched corporate spread; same file and mask |
| NFCICREDIT | credit | fred | w | 1971-01 | 5 | true | 2012-07-05 | false | 2 | chg6m(+) |  | the one NFCI sub-index kept (chg6m); masked before first vintage |
| PERMIT1 | housing/consumer/activity/markets | fred | m | 1960-01 | 18 | true |  | false | 1 | yoy(-); ann6m(-); dmax2y(-,365) |  | most forward-looking housing series (Leamer) |
| MSACSR | housing/consumer/activity/markets | fred | m | 1963-01 | 25 | true |  | false | 1 | avg3m(+); chg6m(+,90); z10y(+,365) |  | new-home inventory overhang, 1963 |
| USSTHPI | housing/consumer/activity/markets | fred | q | 1975-01 | 56 | true |  | false | 1 | (parent only) |  | public-domain house-price parent (Case-Shiller is copyrighted) |
| REAL_HPI | housing/consumer/activity/markets | derived/deflate | q | 1975-01 | 56 | false |  | false | 1 | yoy(-,365); ann6m(-,365) |  | real house-price growth, 1-year model only |
| PERMITNE | housing/consumer/activity/markets | fred | m | 1960-01 | 18 | true |  | false | 1 | yoy(-) |  | regional breadth input (quarter-weight rule) |
| PERMITMW | housing/consumer/activity/markets | fred | m | 1960-01 | 18 | true |  | false | 1 | yoy(-) |  | regional breadth input (quarter-weight rule) |
| PERMITS | housing/consumer/activity/markets | fred | m | 1960-01 | 18 | true |  | false | 1 | yoy(-) |  | regional breadth input (quarter-weight rule) |
| PERMITW | housing/consumer/activity/markets | fred | m | 1960-01 | 18 | true |  | false | 1 | yoy(-) |  | regional breadth input (quarter-weight rule) |
| A011RE1Q156NBEA | housing/consumer/activity/markets | fred | q | 1947-01 | 30 | true |  | false | 1 | chg12m(-,365); dmax2y(-,365) |  | residential investment share, Leamer precursor (8 of 10 recessions) |
| RHVRUSQ156N | housing/consumer/activity/markets | fred | q | 1956-01 | 28 | false |  | false | 2 | chg12m(+,365) |  | existing-stock excess supply (1988-90, 2005-07) |
| ADXTNO | housing/consumer/activity/markets | fred | m | 1992-02 | 26 | true |  | false | 2 | yoy(-); ann6m(-) |  | core durable orders ex transportation (1992, tier 2) |
| TOTALSL | housing/consumer/activity/markets | fred | m | 1943-01 | 37 | true |  | false | 1 | yoy(+,365); ann6m(+) | REVOLSL | economy-wide consumer credit, risk +1; swap for sign-reversed REVOLSL |
| DDURRA3M086SBEA | housing/consumer/activity/markets | fred | m | 1959-01 | 30 | true |  | false | 1 | yoy(-); ann6m(-,90) |  | postponable, credit-financed consumption |
| CFNAIDIFF | housing/consumer/activity/markets | fred | m | 1967-05 | 24 | true | 2012-06-25 | false | 2 | level(-,90); chg6m(-,90) |  | breadth of 85 CFNAI inputs; masked before first vintage |
| MTSR133FMS | housing/consumer/activity/markets | fred | m | 1980-10 | 12 | false |  | false | 2 | yoy(-) |  | unrevised withheld-tax read on nominal income, 12-day lag |
| WPU03T15M05 | housing/consumer/activity/markets | fred | m | 1967-01 | 14 | true |  | false | 1 | yoy(-); pct6m(-,90) |  | non-fuel industrial commodity demand, 1967 |
| TWEXBMTH | housing/consumer/activity/markets | fred | m | 1973-01 | 7 | false |  | false | 1 | (parent only) |  | dollar leg 1973-2019 |
| TWEXBGSMTH | housing/consumer/activity/markets | fred | m | 2006-01 | 7 | false |  | false | 1 | (parent only) |  | dollar leg 2006- |
| USD_BROAD | housing/consumer/activity/markets | derived/splice | m | 1973-01 | 7 | false |  | false | 1 | yoy(+); pct6m(+,90) |  | dollar appreciation tightens global conditions |
| VXOCLS | housing/consumer/activity/markets | fred | d | 1986-01 | 1 | false |  | false | 1 | (parent only) |  | VXO leg 1986-2021 |
| VIX_SPLICED | housing/consumer/activity/markets | derived/splice | d | 1986-01 | 1 | false |  | false | 2 | avg1m(+,90); pctile3y(+) |  | implied volatility through the 1987 crash and 1990 run-up |
| USEPUNEWSINDXM | housing/consumer/activity/markets | fred | m | 1900-01 | 3 | true |  | false | 2 | avg3m(+,90); pctile3y(+); chg12m(+,365) |  | only uncertainty measure spanning every fold (1900-) |

### 3a. Candidates on the critic's drop list that nevertheless appear above

* SHCDFSA, DTCDFSA, IVCDFSA (Philly shipments, delivery times, inventories):
  parents with **no transforms**; they exist only as PHILLY_ISM inputs, which
  is exactly the critic's recommendation.
* SPREAD_5Y2Y and SPREAD_2Y3M: the critic drops them as standalone columns
  but the plan's CURVE_INV_BREADTH needs "share of seven pairs inverted", and
  composite members must be features. They carry a level and an inv_share1y
  transform (boosted model only). If the merge supports composite-only
  features, drop their transforms from the standalone set.
* PERMITNE/MW/S/W and the three regional survey headline indexes: the critic
  wants them only as composite inputs; for the same reason they carry one
  transform each (boosted model only, tier 1/2).

### 3b. Critic "suspicious" items that were added, and how the concern was handled

| series_id | how the concern was handled |
|---|---|
| BORROW | added with pctile3y only (no level/z), lag 26, vintage false, tier 2. |
| EMELASA | added tier 2, vintage false, not masked (see section 5); cross-vintage chg6m check is an open item. |
| NFCICREDIT | added chg6m only, masked before its first ALFRED vintage 2012-07-05. |
| NECDFSA066MSFRBPHI | lag set to -11 like every other BOS component. |
| MICH | lag 2 (UMCSENT convention), vintage true so ALFRED release dates govern from 1999. |
| LNS13023653 | lag corrected to 7; this is the temporary-layoff series the audit meant (LNS13023621 is all job losers and is not added). |
| EBP | added via the url loader with est_prob never loaded, masked before 2016-06-01 as the plan requires (the critic would rather document than mask; the plan wins). |
| GACDISA066MSFRBNY | added tier 2 as a REGIONAL_SURVEY_Z member; the notes record that it covers 2008 and 2020 only. |

## 4. Composites

| id | kind | category | core | tier | n members/rules | transforms (risk, horizons) | members / rules |
|---|---|---|---|---|---|---|---|
| BREADTH_STRESS | breadth_pctile | composite | true | 1 | 25 | level(+); chg3m(+,90) | SPREAD_10Y3M__level, T1YFF__level, DFF__chg24m, BAA10YM__chg6m, BAA_AAA__level, PAPER_BILL__level, ICNSA_4WK__yoy, IURSA__chg6m, U2RATE__chg6m, LNS13023653__yoy, AWHI__ann6m, AWHMAN__chg6m, PAYEMS__ann6m, USCONS__ann6m, INDPRO__ann6m, CUMFNS__chg6m, W875RX1__ann6m, PERMIT1__yoy, MSACSR__chg6m, HOUST__yoy, NASDAQCOM__ret6m, UMCSENT__chg6m, DPCERA3M086SBEA__ann6m, GACDFSA066MSFRBPHI__avg3m, BUSLOANS__ann6m |
| LABOR_ONSET_DIFF | rule_share | composite | false | 1 | 12 | level(+,90); chg3m(+,90) | ICNSA_4WK__yoy > 0.1, IURSA__chg3m > 0.2, U2RATE_SAHM__level > 0.3, LNS13023653__yoy > 0.15, AWHI__ann3m < 0.0, AWHMAN__chg6m < -0.3, PAYEMS__ann3m < 0.005, TEMPHELPS__yoy < 0.0, USCONS__ann6m < 0.0, NECDFSA066MSFRBPHI__avg3m < 0.0, CE16OV__ann3m < 0.0, EMELASA__chg6m > 0.2 |
| CURVE_INV_BREADTH | inv_share | composite | true | 1 | 7 | level(+); avg12m(+,365) | SPREAD_10Y3M__level, SPREAD_10Y2Y__level, SPREAD_5Y2Y__level, SPREAD_2Y3M__level, T1YFF__level, SPREAD_3M_FFR__level, SPREAD_10Y_FFR__level |
| HOUSING_BREADTH | rule_share | composite | false | 1 | 11 | level(+); chg6m(+,365) | PERMIT1__yoy < 0.0, HOUST__yoy < 0.0, HSN1F__yoy < 0.0, MSACSR__z10y > 1.0, USCONS__ann6m < 0.0, PERMITNE__yoy < 0.0 (w 0.25), PERMITMW__yoy < 0.0 (w 0.25), PERMITS__yoy < 0.0 (w 0.25), PERMITW__yoy < 0.0 (w 0.25), REAL_HPI__yoy < 0.0, MTG_SPREAD__chg6m > 0.25 |
| CREDIT_SUPPLY_STRESS | breadth_pctile | composite | false | 1 | 9 | level(+); chg6m(+,90) | BAA_AAA__level, BAA10YM__chg6m, PAPER_BILL__level, MTG_SPREAD__level, WILLING_LEND__level, DRTSCILM__level, BUSLOANS__accel3m, EBP__level, VIX_SPLICED__avg1m |
| COINC_BIG4 | zmean | composite | true | 1 | 4 | level(+,90); chg3m(+,90) | PAYEMS__ann3m, INDPRO__ann3m, W875RX1__ann6m, CMRMTSPL__ann6m |
| REGIONAL_SURVEY_Z | zmean | survey | false | 2 | 4 | level(+); chg6m(+,90) | GACDFSA066MSFRBPHI__level, GACDISA066MSFRBNY__level, BACTSAMFRBDAL__level, TSSOSBACTSAMFRBDAL__level |

Kinds (to implement in `composites.py`): `breadth_pctile` = NaN-skipping mean
over members of the trailing 10-year percentile of the member oriented by its
risk sign; `rule_share` = weighted share of rules true (weight default 1;
NaN-skipping); `inv_share` = share of members below zero; `zmean` =
NaN-skipping mean of members' trailing 10-year z-scores times their risk
sign (so higher = more recessionary; the composite's own transforms carry
+1). Composites are computed on the weekly as-of panel after the member
features, so availability is inherited and nothing is fitted. BREADTH_STRESS
membership is frozen at 25 features all available by 1971; the critic's
`FEDFUNDS__chg24m` member is `DFF__chg24m` here (see section 5).

The plan's six composites are all present; REGIONAL_SURVEY_Z is a seventh
because the plan's survey batch asks for the regional z-mean and it is a
plain `zmean` composite. PHILLY_ISM is **not** a composite but a derived
series (`wsum`), because it needs ordinary transforms (level, avg3m, chg6m).

## 5. Fields I was unsure about

1. **TB3MS** is not in the scouts' verified list (the CP3M record pairs the
   paper rate with daily DTB3). It is the standard H.15 monthly 3-month bill
   (1934-); I used it so both paper legs are compared with a like-for-like
   monthly average instead of the daily bill rate as of the first of the
   month. Verify on first fetch. CPN3M is verified inside the CP3M record
   (1997-01 to 2026-10). TWEXBGSMTH is verified inside the TWEXBMTH record
   (2006-01, first vintage 2019-02-04). PERMITMW/S/W are verified inside the
   PERMITNE record.
2. **FEDFUNDS chg24m / rise3y live on DFF.** FEDFUNDS exists in the base file
   and this file never re-declares a base id, so the daily effective rate
   carries the cumulative-tightening features (periods_for_months('d', 24) =
   504, rise3y over 756 business days). If the merge can extend a base id's
   transforms, move them to FEDFUNDS (monthly averages are smoother than the
   month-end spikes of DFF pre-2008) and point BREADTH_STRESS at
   FEDFUNDS__chg24m.
3. **SPREAD_10Y3M_LAGS** is a derived twin of SPREAD_10Y3M for the same
   reason; fold lag6m/lag12m/min1y/invdepth1y into SPREAD_10Y3M if preferred.
4. **COINC_BIG4** uses W875RX1__ann6m and CMRMTSPL__ann6m because the base
   file has no ann3m on those two; add ann3m to them at merge time for the
   critic's exact recipe. The members must be chain-linked across
   first-release base changes (plan, Stage 3 masks) or the 1990 value is the
   INDPRO rebasing artefact.
5. **HOUSING_BREADTH** uses `MTG_SPREAD__chg6m > 0.25` because MTG_SPREAD has
   no z10y in the base file (the critic wrote z10y > 1); the four regional
   permit rules carry `weight: 0.25` so the region block counts once, an
   optional extension of the rule schema (treat a missing weight as 1).
6. **`wsum` derive recipe** (PHILLY_ISM) is the one recipe neither
   implemented nor on the planned list: value = sum_i weights[i] * v_i over
   aligned periods (NaN if any input is missing); weights
   0.30/0.25/0.20/0.15/0.10 are the pre-2008 ISM PMI weights.
7. **EMELASA mask.** Suggested real_time_from 2025-04-16 would leave no
   onset in its training history (every row would also fail the 365-day
   label maturity), i.e. a dashboard-only feature. It is a tabulation of
   unrevised CPS microdata with a fixed published method, not a factor model
   with full-sample loadings; its monthly republication is SA/matching
   maintenance, comparable to the SA-factor revisions of every pre-ALFRED CPS
   series in the base file. So no mask, tier 2, vintage false, boosted model
   only. If the cross-vintage chg6m comparison the critic asks for shows
   material drift, set real_time_from: "2025-04-16".
8. **Splice offsets are additive** (first four overlap observations). For
   USD_BROAD (index bases 1997=100 vs 2006=100) that scales pre-2006 growth
   rates by roughly level/(level+offset), about 10%; a multiplicative (log)
   splice option would remove it. VIX and paper-bill splices are in points
   and unaffected.
9. **Lags chosen without ALFRED evidence:** CP3M/CPN3M/TB3MS 3 (monthly
   H.15, like BAA); DRBLACBS/CORCCACBS 56 (true release ~55-60 days; the base
   DRALACBS 75 is stale per the audit); BUSLOANS 12 (H.8 monthly ~10);
   DDURRA3M086SBEA 30 (observed PCE-family gap; base entries use 28);
   MTSR133FMS 12; TWEX 7; USEPUNEWSINDXM 3; EMELASA 12; BORROW/M2SL 26.
10. **VIX duplication.** VIX_SPLICED__avg1m coexists with the base
    VIXCLS__avg1m (VIXCLS is a splice input, so `replaces` was not used).
    Drop the base transform once the splice is validated. Likewise IURSA is
    meant to take over CCSA's pct transforms (critic) but CCSA is untouched.
11. **DRSDCILM, WILLING_LEND** are `vintage: false` (unrevised surveys, lag
    -55) whereas the base DRTSCILM is `vintage: true`; either convention
    gives the same timing where ALFRED release dates exist (2010+).

## 6. Left out, with reasons

Every verified candidate that is not in the additions file. "Basis" says
whether the reason is the critic's redundancy verdict (keep/drop), the
critic's suspicious-list correction, or my own judgement within the plan's
batches (the plan's "not added" list is quoted where it applies).

| series_id | start | freq | scout priority | basis | reason |
|---|---|---|---|---|---|
| A261RX1Q020SBEA | 1947-01 | q | 3 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| A466RD3Q052SBEA | 1947-01 | q | 4 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| AAA10YM | 1953-04 | m | 3 | judgement | algebraically BAA10YM minus BAA_AAA, both already in the base file. |
| ABCOMP | 2001-01 | w | 4 | critic redundancy | Critic drop (keep CP3M + CPN3M paper-bill spread splice): CP quantities start 2001 and the ABCP market is a single-episode (2007) series that is structurally shrinking; the price (spread) version covers seven recessions. |
| ADS_INDEX | 1960-03 | d | 2 | critic suspicious | Critic suspicious: Proposed twice (ADS_PHIL by the activity scout, ADS_INDEX by the composite scout) from the same xlsx; the 'most current vintage' file is a two-sided Kalman smooth of claims, payrolls, IP, income, sales and GDP. Usable only through the Philly Fed vintage ... |
| ADS_PHIL | 1960-03 | d | 3 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| AHETPI | 1964-01 | m | 3 | judgement | critic keep (real wage via deflate) but outside the Stage-3 batches and sign-ambiguous; queue for a later block. |
| AIRRPMTSID11 | 2000-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| AISRSA | 1993-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| ALTSALES | 1976-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| AMTMNO | 1992-02 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| AMTMUO | 1992-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| ANFCI | 1971-01 | w | 2 | critic redundancy | Critic drop (keep NFCICREDIT): All NFCI sub-indexes are outputs of one weekly dynamic-factor model whose entire history is re-estimated (vintages 2011-2012); the headline NFCI is already in the model and shows the backcast symptom (AUC 0.977 pre-1990 vs 0.746 after). Risk overlaps ... |
| ANXAVS | 1992-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| AWCDFSA066MSFRBPHI | 1968-05 | m | 2 | critic redundancy | Critic drop (keep GACDFSA066MSFRBPHI, NOCDFSA066MSFRBPHI, NECDFSA066MSFRBPHI): Thirteen columns from one survey of ~100 Third-District manufacturers; general activity, new orders and employment carry the signal, the rest belong inside an ISM-weighted composite (0.30 new orders, 0.25 shipments, 0.20 employment, 0.15 deliveries, 0.10 ... |
| AWHNONAG | 1964-01 | m | 3 | critic redundancy | Critic drop (keep AWHI): Manufacturing hours index = AWHMAN (in model) x MANEMP; construction hours duplicate USCONS; the payrolls index is a nominal cousin of W875RX1; the private workweek is collinear with AWHMAN. One economy-wide labour-input series captures the intensive-margin ... |
| BABATOTALSAUS | 2004-07 | m | 3 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| BACDINA066MNFRBNY | 2004-09 | m | 4 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| BAHBATOTALSAUS | 2004-07 | m | 3 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| BBKMCOIX | 1960-01 | m | 4 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| BBKMLEIX | 1960-01 | m | 2 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| BCNSDODNS | 1945-10 | q | 3 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| BUSAPPWNSAUS | 2006-01 | w | 4 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| CANDH | 1967-03 | m | 2 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| CCNSA | 1967-01 | w | 3 | critic redundancy | Critic drop (keep ICNSA): CCSA is already in the model and continuing claims lag initial claims by 2-4 weeks; IURSA is the stationary, covered-employment-scaled version of CCSA and should replace CCSA's pct transforms rather than sit beside CCNSA. The existing avg4w derive and ... |
| CEFDFSA066MSFRBPHI | 1968-05 | m | 3 | critic redundancy | Critic drop (keep GACDFSA066MSFRBPHI, NOCDFSA066MSFRBPHI, NECDFSA066MSFRBPHI): Thirteen columns from one survey of ~100 Third-District manufacturers; general activity, new orders and employment carry the signal, the rest belong inside an ISM-weighted composite (0.30 new orders, 0.25 shipments, 0.20 employment, 0.15 deliveries, 0.10 ... |
| CES0500000035 | 1964-01 | m | 3 | critic redundancy | Critic drop (keep AWHI): Manufacturing hours index = AWHMAN (in model) x MANEMP; construction hours duplicate USCONS; the payrolls index is a nominal cousin of W875RX1; the private workweek is collinear with AWHMAN. One economy-wide labour-input series captures the intensive-margin ... |
| CES2000000034 | 1947-01 | m | 3 | critic redundancy | Critic drop (keep AWHI): Manufacturing hours index = AWHMAN (in model) x MANEMP; construction hours duplicate USCONS; the payrolls index is a nominal cousin of W875RX1; the private workweek is collinear with AWHMAN. One economy-wide labour-input series captures the intensive-margin ... |
| CES3000000034 | 1939-01 | m | 3 | critic redundancy | Critic drop (keep AWHI): Manufacturing hours index = AWHMAN (in model) x MANEMP; construction hours duplicate USCONS; the payrolls index is a nominal cousin of W875RX1; the private workweek is collinear with AWHMAN. One economy-wide labour-input series captures the intensive-margin ... |
| CFNAIMA3 | 1967-05 | m | 2 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| CFSBCACTIVITY | 2013-02 | m | 5 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| CHNLOLITOAASTSAM | 1992-05 | m | 3 | judgement | OECD CLI: two-sided filter, vintages 2018 (plan: not added). |
| CLEVELAND_YC_PROB | 1960-01 | m | 5 | judgement | recession-probability model output; benchmark only (plan). |
| CMDEBT | 1945-10 | q | 2 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| COMPOUT | 2001-01 | w | 3 | critic redundancy | Critic drop (keep CP3M + CPN3M paper-bill spread splice): CP quantities start 2001 and the ABCP market is a single-episode (2007) series that is structurally shrinking; the price (spread) version covers seven recessions. |
| COMPUTSA | 1968-01 | m | 4 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| CONSUMER | 1947-01 | m | 3 | critic redundancy | Critic drop (keep BUSLOANS): TOTBKCR is already in the model; consumer loans duplicate REVOLSL/TOTALSL, securities share and loan-to-deposit are second-order ratios with QE-era breaks, HELOCs start 1987 and decline secularly, real-estate loans overlap the housing credit candidates. |
| CORBLACBS | 1985-01 | q | 4 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| CORESTICKM159SFRBATL | 1968-01 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| CPATAX | 1947-01 | q | 2 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| CPIENGSL | 1957-01 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| CPROFIT | 1947-01 | q | 3 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| CSINFT02USM460S | 1978-01 | m | 5 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| CSUSHPISA | 1987-01 | m | 2 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| DAUTOSAAR | 1967-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| DCOILWTICO | 1986-01 | d | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| DEULOLITOAASTSAM | 1961-01 | m | 4 | judgement | OECD CLI: two-sided filter, vintages 2018 (plan: not added). |
| DFII10 | 2003-01 | d | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| DGDSRA3M086SBEA | 1959-01 | m | 2 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| DGORDER | 1992-02 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| DGS30 | 1977-02 | d | 4 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| DPSACBW027SBOG | 1973-01 | w | 3 | judgement | only useful inside a loan-to-deposit ratio the critic calls second-order with QE-era breaks. |
| DRCCLACBS | 1991-01 | q | 3 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| DRCLACBS | 1987-01 | q | 3 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| DRCRELEXFACBS | 1991-01 | q | 4 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| DRISCFLM | 1990-04 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| DRSFRMACBS | 1991-01 | q | 3 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| DRSREACBS | 1987-01 | q | 3 | critic redundancy | Critic drop (keep DRBLACBS): Eight quarterly credit-quality rates starting 1985-1991 with 50-75 day lags cover three recessions each; DRALACBS is already in the model. One business and one household realised-loss series are enough for the 1-year horizon. |
| DRTSCIS | 1990-04 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| DRTSCLCC | 1996-01 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| DRTSCREL | 1990-07 | q | 2 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| DSERRA3M086SBEA | 1959-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| DTB6 | 1958-12 | d | 4 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| DTWEXM | 1973-01 | d | 4 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| EMEQASA | 1978-01 | m | 3 | critic redundancy | Critic drop (keep EMELASA): The BLS gross-flow series start 1990 and are far noisier than the EM transition rate; quits are already represented by JTSQUL; JOLTS layoffs/hires start 2000-12 with a 40-day lag and the layoff rate was flat before the 2001 and 2008 onsets. |
| EMVOVERALLEMV | 1985-01 | m | 3 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| EPUMONETARY | 1985-01 | m | 4 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| EQ_BOND_REL12 | 1971-02 | d | 4 | judgement | priority 4, needs a new rel_ret12 recipe; NASDAQCOM return transforms already in the base. |
| EUANDH | 1967-03 | m | 3 | critic redundancy | Critic drop (keep none as standalone features; build the LABOR_ONSET_DIFF composite instead): The BLS diffusion index has vintages only from 2025-07 and the audit found payroll diffusion weak (AUC 0.53-0.56); the KC LMCI is a monthly re-estimated PCA starting 1992; EUANDH is a CFNAI sub-index (backcast, vintages 2011); Philly future employment is ... |
| EXPINF10YR | 1982-01 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| EXPINF1YR | 1982-01 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| EXP_SLOPE_10Y3M | 1990-01 | d | 3 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| FCOLKSAMFRBDAL | 2004-06 | m | 4 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| FGRECPT | 1947-01 | q | 3 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| FLEXCPIM159SFRBATL | 1968-01 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| FRBATLWGTUMHWG83O | 1983-01 | m | 4 | critic redundancy | Critic drop (keep AHETPI): The wage tracker is NSA, breaks in 1997 and has no vintages before 2023; unit labour costs are quarterly, revised heavily and 5 weeks late; the real-wage squeeze channel is carried by AHETPI deflated by CPIAUCSL with the existing recipe. |
| FRBKCLMCILA | 1992-01 | m | 3 | critic redundancy | Critic drop (keep none as standalone features; build the LABOR_ONSET_DIFF composite instead): The BLS diffusion index has vintages only from 2025-07 and the audit found payroll diffusion weak (AUC 0.53-0.56); the KC LMCI is a monthly re-estimated PCA starting 1992; EUANDH is a CFNAI sub-index (backcast, vintages 2011); Philly future employment is ... |
| FRBKCLMCIM | 1992-01 | m | 2 | critic redundancy | Critic drop (keep none as standalone features; build the LABOR_ONSET_DIFF composite instead): The BLS diffusion index has vintages only from 2025-07 and the audit found payroll diffusion weak (AUC 0.53-0.56); the KC LMCI is a monthly re-estimated PCA starting 1992; EUANDH is a CFNAI sub-index (backcast, vintages 2011); Philly future employment is ... |
| FWD3M3M | 1958-12 | d | 3 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| G7LOLITOAASTSAM | 1959-01 | m | 3 | judgement | OECD CLI: two-sided filter, vintages 2018 (plan: not added). |
| GABNDIF066MSFRBPHI | 2011-03 | m | 5 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| GAFDISA066MSFRBNY | 2001-07 | m | 3 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| GASREGW | 1990-08 | w | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| GDPC1 | 1947-01 | q | 2 | judgement | quarterly, coincident at best (critic); outside the batches. |
| GEPUCURRENT | 1997-01 | m | 4 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| GS10 | 1953-04 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| HHMSDODNS | 1945-10 | q | 3 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| HH_PAY_GAP | 1948-01 | m | 3 | judgement | priority 3; population-control breaks in CE16OV make the gap noisy; CE16OV enters through the LABOR_ONSET_DIFF rule instead. |
| HNFSEPUSSA | 1963-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| HNONWPDPI | 1946-10 | q | 3 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| HOUST1F | 1959-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| HPIPONM226S | 1991-01 | m | 4 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| IMPGSC1 | 1947-01 | q | 3 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| INVCMRMTSPL | 1967-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| INVEST | 1947-01 | m | 4 | critic redundancy | Critic drop (keep BUSLOANS): TOTBKCR is already in the model; consumer loans duplicate REVOLSL/TOTALSL, securities share and loan-to-deposit are second-order ratios with QE-era breaks, HELOCs start 1987 and decline secularly, real-estate loans overlap the housing credit candidates. |
| IPBUSEQ | 1947-01 | m | 2 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IPCONGD | 1939-01 | m | 3 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IPDCONGD | 1947-01 | m | 2 | judgement | same as IPMAN (volatile auto shutdowns); outside the batches. |
| IPDMAT | 1947-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IPG2211S | 1972-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IPMAN | 1972-01 | m | 2 | judgement | INDPRO and CUMFNS in the base; the critic prefers an IP_BREADTH composite (gap list), outside the batches. |
| IPMAT | 1939-01 | m | 3 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IPN3311A2RS | 1972-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| IR | 1982-09 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| JPNLOLITOAASTSAM | 1959-01 | m | 5 | judgement | OECD CLI: two-sided filter, vintages 2018 (plan: not added). |
| JTSHIR | 2000-12 | m | 3 | critic redundancy | Critic drop (keep EMELASA): The BLS gross-flow series start 1990 and are far noisier than the EM transition rate; quits are already represented by JTSQUL; JOLTS layoffs/hires start 2000-12 with a 40-day lag and the layoff rate was flat before the 2001 and 2008 onsets. |
| JTSLDR | 2000-12 | m | 3 | critic redundancy | Critic drop (keep EMELASA): The BLS gross-flow series start 1990 and are far noisier than the EM transition rate; quits are already represented by JTSQUL; JOLTS layoffs/hires start 2000-12 with a 40-day lag and the layoff rate was flat before the 2001 and 2008 onsets. |
| KCFSI | 1990-02 | m | 3 | critic redundancy | Critic drop (keep raw inputs): Four stress indices from three institutions, all re-estimated or redefined (STLFSI redefined four times, KCFSI inputs backcast in 2021, OFR dynamic-factor weights revised, KCRORO's first vintage is 2026-06-08 with 84 re-estimations since), starting 1990-2003. ... |
| KCRORO | 2003-05 | d | 4 | critic redundancy | Critic drop (keep raw inputs): Four stress indices from three institutions, all re-estimated or redefined (STLFSI redefined four times, KCFSI inputs backcast in 2021, OFR dynamic-factor weights revised, KCRORO's first vintage is 2026-06-08 with 84 re-estimations since), starting 1990-2003. ... |
| LB0000031Q020SBEA | 1947-01 | q | 3 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| LNS13023622 | 1967-01 | m | 3 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| LNS13023706 | 1967-01 | m | 3 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| LNS13025699 | 1967-01 | m | 2 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| LNS14000060 | 1948-01 | m | 4 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| LNS17100000 | 1990-02 | m | 3 | critic redundancy | Critic drop (keep EMELASA): The BLS gross-flow series start 1990 and are far noisier than the EM transition rate; quits are already represented by JTSQUL; JOLTS layoffs/hires start 2000-12 with a 40-day lag and the layoff rate was flat before the 2001 and 2008 onsets. |
| LNS17400000 | 1990-02 | m | 3 | critic redundancy | Critic drop (keep EMELASA): The BLS gross-flow series start 1990 and are far noisier than the EM transition rate; quits are already represented by JTSQUL; JOLTS layoffs/hires start 2000-12 with a 40-day lag and the layoff rate was flat before the 2001 and 2008 onsets. |
| M2REAL | 1959-01 | m | 2 | critic redundancy | Critic drop (keep M2SL deflated by CPIAUCSL): M2REAL has vintages only from 2014 and is the same information; velocity is quarterly and redundant with M2 plus nominal activity; reserves and the balance sheet are regime-dependent post-2008 with one recession of history; primary credit (2003+) duplicates ... |
| M2V | 1959-01 | q | 5 | critic redundancy | Critic drop (keep M2SL deflated by CPIAUCSL): M2REAL has vintages only from 2014 and is the same information; velocity is quarterly and redundant with M2 plus nominal activity; reserves and the balance sheet are regime-dependent post-2008 with one recession of history; primary credit (2003+) duplicates ... |
| MANEMP | 1939-01 | m | 2 | critic redundancy | Critic drop (keep AWHI): Manufacturing hours index = AWHMAN (in model) x MANEMP; construction hours duplicate USCONS; the payrolls index is a nominal cousin of W875RX1; the private workweek is collinear with AWHMAN. One economy-wide labour-input series captures the intensive-margin ... |
| MARTSSM44W72USS | 1992-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| MEDCPIM094SFRBCLE | 1982-12 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| MMMFFAQ027S | 1945-10 | q | 4 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| MNFCTRIRSA | 1992-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| MSPNHSUS | 1963-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| MTSO133FMS | 1980-10 | m | 4 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| MVATOTASSS | 1967-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| NCBCMDPMVCE | 1945-10 | q | 3 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| NECDISA066MSFRBNY | 2001-07 | m | 3 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| NEFDFSA066MSFRBPHI | 1968-05 | m | 3 | critic redundancy | Critic drop (keep none as standalone features; build the LABOR_ONSET_DIFF composite instead): The BLS diffusion index has vintages only from 2025-07 and the audit found payroll diffusion weak (AUC 0.53-0.56); the KC LMCI is a monthly re-estimated PCA starting 1992; EUANDH is a CFNAI sub-index (backcast, vintages 2011); Philly future employment is ... |
| NEIM156SFRBRIC | 1994-01 | m | 4 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| NFCILEVERAGE | 1971-01 | w | 2 | critic redundancy | Critic drop (keep NFCICREDIT): All NFCI sub-indexes are outputs of one weekly dynamic-factor model whose entire history is re-estimated (vintages 2011-2012); the headline NFCI is already in the model and shows the backcast symptom (AUC 0.977 pre-1990 vs 0.746 after). Risk overlaps ... |
| NFCINONFINLEVERAGE | 1971-01 | w | 1 | critic redundancy | Critic drop (keep NFCICREDIT): All NFCI sub-indexes are outputs of one weekly dynamic-factor model whose entire history is re-estimated (vintages 2011-2012); the headline NFCI is already in the model and shows the backcast symptom (AUC 0.977 pre-1990 vs 0.746 after). Risk overlaps ... |
| NFCIRISK | 1971-01 | w | 3 | critic redundancy | Critic drop (keep NFCICREDIT): All NFCI sub-indexes are outputs of one weekly dynamic-factor model whose entire history is re-estimated (vintages 2011-2012); the headline NFCI is already in the model and shows the backcast symptom (AUC 0.977 pre-1990 vs 0.746 after). Risk overlaps ... |
| NFCPATAX | 1947-01 | q | 3 | critic redundancy | Critic drop (keep GDPC1 advance estimate): Quarterly, 30-90 day lags, coincident at best; the audit screen puts corporate profits below 0.5 AUC, contradicting the scouts' tier-1 ratings; GDI arrives 60-90 days late with vintages from 2013. |
| NOCDISA066MSFRBNY | 2001-07 | m | 3 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| NOFDFSA066MSFRBPHI | 1968-05 | m | 3 | critic redundancy | Critic drop (keep GACDFSA066MSFRBPHI, NOCDFSA066MSFRBPHI, NECDFSA066MSFRBPHI): Thirteen columns from one survey of ~100 Third-District manufacturers; general activity, new orders and employment carry the signal, the rest belong inside an ISM-weighted composite (0.30 new orders, 0.25 shipments, 0.20 employment, 0.15 deliveries, 0.10 ... |
| NROU | 1949-01 | q | 3 | critic suspicious | Critic suspicious: Carries CBO projections to 2036 and is unnecessary: the repo's taylor_gap recipe already uses a trailing 10-year mean unemployment rate as u*. Adding it only creates a look-ahead channel. |
| OEHRENWBSHNO | 1945-10 | q | 4 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| OFR_FSI | 2000-01 | d | 4 | critic redundancy | Critic drop (keep raw inputs): Four stress indices from three institutions, all re-estimated or redefined (STLFSI redefined four times, KCFSI inputs backcast in 2021, OFR dynamic-factor weights revised, KCRORO's first vintage is 2026-06-08 with 84 re-estimations since), starting 1990-2003. ... |
| OPEN_PER_UNEMP | 2000-12 | m | 2 | judgement | JOLTS-based, 2000-12 start, 40-day lag, two recessions; JTSJOL and JTSQUL already in the base file. |
| PANDI | 1967-03 | m | 3 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| PCETRIM12M159SFRBDAL | 1978-01 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| PCOPPUSDM | 1992-01 | m | 3 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| PINDUINDEXM | 1992-01 | m | 4 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| PPCDFSA066MSFRBPHI | 1968-05 | m | 4 | critic redundancy | Critic drop (keep GACDFSA066MSFRBPHI, NOCDFSA066MSFRBPHI, NECDFSA066MSFRBPHI): Thirteen columns from one survey of ~100 Third-District manufacturers; general activity, new orders and employment carry the signal, the rest belong inside an ISM-weighted composite (0.30 new orders, 0.25 shipments, 0.20 employment, 0.15 deliveries, 0.10 ... |
| PPIACO | 1913-01 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| PRFI | 1947-01 | q | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| PROFIT_SHARE | 1947-01 | q | 3 | judgement | audit screen puts corporate profits below 0.5 AUC; critic: 1-year model at most; outside the batches. |
| PRRESCONS | 1993-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| QUSHAM770A | 1947-10 | q | 3 | judgement | critic keep for the 1-year model, but FRED copy ends 2025Q4 with a 170+ day irregular lag and 2016 vintages; outside the batches; queue with a release-date-driven lag. |
| QUSNAM770A | 1947-10 | q | 4 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| QUSPAM770A | 1947-10 | q | 3 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| RAILFRTCARLOADSD11 | 2000-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| RAILFRTINTERMODALD11 | 2000-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| REAINTRATREARAT10Y | 1982-01 | m | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| REALLN | 1947-01 | m | 3 | critic redundancy | Critic drop (keep BUSLOANS): TOTBKCR is already in the model; consumer loans duplicate REVOLSL/TOTALSL, securities share and loan-to-deposit are second-order ratios with QE-era breaks, HELOCs start 1987 and decline secularly, real-estate loans overlap the housing credit candidates. |
| RECPROUSM156N | 1967-06 | m | 5 | critic suspicious | Critic suspicious: Chauvet-Piger smoothed probabilities, CLEVELAND_YC_PROB and the EBP est_prob column are recession-probability model outputs (two-sided or in-sample fitted) and belong only in the dashboard comparison table; SPF_RECESS1 is a survey and could be a feature but ... |
| RHEACBW027SBOG | 1987-06 | w | 4 | critic redundancy | Critic drop (keep BUSLOANS): TOTBKCR is already in the model; consumer loans duplicate REVOLSL/TOTALSL, securities share and loan-to-deposit are second-order ratios with QE-era breaks, HELOCs start 1987 and decline secularly, real-estate loans overlap the housing credit candidates. |
| RHORUSQ156N | 1965-01 | q | 5 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| RRVRUSQ156N | 1956-01 | q | 4 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| RSBMGESD | 1992-01 | m | 4 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| RSFSDP | 1992-01 | m | 3 | critic redundancy | Critic drop (keep ADXTNO): NEWORDER, ISRATIO, CMRMTSPL, RSXFS/REAL_RETAIL, DPCERA3M086SBEA and TOTALSA already cover orders, inventories, sales, consumption and vehicles; the 1992-start Census M3/MARTS detail and the vehicle variants are near-duplicates with later vintages. |
| SFNEWSSENT | 1980-01 | d | 1 | judgement | non-FRED xlsx with an irregular calendar needing a new loader and business-day resampling; corpus re-runs; not in the plan batches. Strongest candidate for a later block. |
| SMS00000000000000021 | 1991-01 | m | 2 | critic redundancy | Critic drop (keep none as standalone features; build the LABOR_ONSET_DIFF composite instead): The BLS diffusion index has vintages only from 2025-07 and the audit found payroll diffusion weak (AUC 0.53-0.56); the KC LMCI is a monthly re-estimated PCA starting 1992; EUANDH is a CFNAI sub-index (backcast, vintages 2011); Philly future employment is ... |
| SOANDI | 1967-03 | m | 2 | critic redundancy | Critic drop (keep CFNAIDIFF): CFNAIMA3 equals the avg3m transform already applied to CFNAI; the sub-indexes sum to CFNAI and share its 2011 backcast; BBK is a two-sided dynamic factor with vintages from 2021; ADS was proposed twice (ADS_PHIL and ADS_INDEX are the same xlsx) and is a ... |
| SPF_RECESS1 | 1968-10 | q | 3 | judgement | survey of forecasters: confirming, not leading (critic); dashboard benchmark table. |
| SPREAD_30Y10Y | 1977-02 | d | 4 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| STLFSI4 | 1993-12 | w | 3 | critic redundancy | Critic drop (keep raw inputs): Four stress indices from three institutions, all re-estimated or redefined (STLFSI redefined four times, KCFSI inputs backcast in 2021, OFR dynamic-factor weights revised, KCRORO's first vintage is 2026-06-08 with 84 re-estimations since), starting 1990-2003. ... |
| SUBLPDMHSXWBNQ | 1991-07 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| SUBLPDMODXWBNQ | 1991-10 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| SUBLPDMOSXWBNQ | 1991-07 | q | 2 | judgement | critic keep, but 1991 start, overlaps DRTSCILM (base), and the plan credit batch lists demand and the willingness splice only; queue. |
| SUBLPDRCSC | 2013-10 | q | 3 | critic redundancy | Critic drop (keep DRIWCIL + SUBLPDCLIWXWFRNQ splice): DRTSCILM is already in the model and the small-firm and loan-spread questions correlate >0.95 with it; card standards have gaps before 2011; the household composite is covered by the willingness splice (the only SLOOS question spanning all eight recessions); ... |
| T10YFF | 1962-01 | d | 3 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| T10YIE | 2003-01 | d | 3 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| T5YIFR | 2003-01 | d | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| TB3SMFFM | 1954-07 | m | 3 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| TCU | 1967-01 | m | 4 | critic redundancy | Critic drop (keep IPMAN and IPDCONGD): INDPRO and CUMFNS are in the model; TCU correlates 0.98 with CUMFNS; business equipment duplicates NEWORDER; materials pairs are collinear; steel, electricity, rail and air start 1972-2000 with 49-75 day lags and weather/coal/port noise. |
| TDSP | 2005-01 | q | 3 | critic suspicious | Critic suspicious: Scout claims values are served from 1980-01-01 despite observation_start 2005. API-verified: the 1980-2004 rows are '.' placeholders (108 rows 1980-2006, only 8 numeric, first numeric 2005-01-01). The series covers only the 2008 and 2020 onsets; the 1980+ DSR ... |
| TERMCBAUTO48NS | 1972-02 | m | 4 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| THREEFYTP10 | 1990-01 | d | 3 | critic redundancy | Critic drop (keep T1YFF): Nine front-end/policy spreads proposed by three scouts and one auditor carry one signal (market-implied easing vs the policy rate). T1YFF covers all eight recessions; the audit's NTFS proxy (2.25*DGS2-1.25*DGS1-DTB3) duplicates it from 1976 and SPREAD_2Y3M ... |
| TNWBSHNO | 1945-10 | q | 4 | critic redundancy | Critic drop (keep QUSHAM770A): Ten quarterly Z.1/BIS stock variables with 73-170 day lags and vintages from 2010-2016 describe one slow credit-boom channel; keep the single ratio the literature names, and note the BIS copy on FRED ends 2025Q4 (last updated 2026-06-15). Net-worth drawdowns ... |
| TOTLL | 1973-01 | w | 4 | critic redundancy | Critic drop (keep BUSLOANS): TOTBKCR is already in the model; consumer loans duplicate REVOLSL/TOTALSL, securities share and loan-to-deposit are second-order ratios with QE-era breaks, HELOCs start 1987 and decline secularly, real-estate loans overlap the housing credit candidates. |
| TOTRESNS | 1959-01 | m | 4 | critic redundancy | Critic drop (keep M2SL deflated by CPIAUCSL): M2REAL has vintages only from 2014 and is the same information; velocity is quarterly and redundant with M2 plus nominal activity; reserves and the balance sheet are regime-dependent post-2008 with one recession of history; primary credit (2003+) duplicates ... |
| TRMMEANCPIM094SFRBCLE | 1982-12 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| U6RATE | 1994-01 | m | 3 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| UEMPLT5 | 1948-01 | m | 2 | judgement | audit screen weak (critic sides with the audits); outside the plan batches. |
| UEMPMEAN | 1948-01 | m | 4 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| ULCNFB | 1947-01 | q | 3 | critic redundancy | Critic drop (keep AHETPI): The wage tracker is NSA, breaks in 1997 and has no vintages before 2023; unit labour costs are quarterly, revised heavily and 5 weeks late; the real-wage squeeze channel is carried by AHETPI deflated by CPIAUCSL with the existing recipe. |
| UNDCONTSA | 1970-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| UNEMPLOY | 1948-01 | m | 3 | critic redundancy | Critic drop (keep U2RATE): Job-loser share, leaver share and permanent-loser level are algebraic cousins of the U-2 rate; UEMPMEAN is a lagging-index component (audit screen weak); U6RATE and NEI start 1994; prime-age UNRATE has >0.95 correlation of changes with UNRATE; UNEMPLOY is ... |
| UOCDFSA066MSFRBPHI | 1968-05 | m | 3 | critic redundancy | Critic drop (keep GACDFSA066MSFRBPHI, NOCDFSA066MSFRBPHI, NECDFSA066MSFRBPHI): Thirteen columns from one survey of ~100 Third-District manufacturers; general activity, new orders and employment carry the signal, the rest belong inside an ISM-weighted composite (0.30 new orders, 0.25 shipments, 0.20 employment, 0.15 deliveries, 0.10 ... |
| USALOLITOAASTSAM | 1955-01 | m | 3 | critic suspicious | Critic suspicious: OECD CLIs (also G7LOLITOAASTSAM, DEULOLITOAASTSAM, JPNLOLITOAASTSAM, CHNLOLITOAASTSAM) are built with two-sided filters, so the revised history leads the cycle by construction; vintages only from 2018-07. Verified live (updated 2026-09-15 through 2026-08), ... |
| USEPUINDXD | 1985-01 | d | 3 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| USEPUINDXM | 1985-01 | m | 2 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| USPHCI | 1979-01 | m | 2 | judgement | Philadelphia Fed coincident index is a smoothed single-factor model output with vintages from 2010; the state-breadth modality is logged as a gap for a later pass. |
| VNWOSAMFRBDAL | 2004-06 | m | 4 | critic redundancy | Critic drop (keep GACDISA066MSFRBNY, BACTSAMFRBDAL, TSSOSBACTSAMFRBDAL only as NaN-skipping inputs to REGIONAL_SURVEY_Z): Sub-components of surveys that start 2001-2013 cannot be validated across folds; the headline indexes feed the composite and the rest are collinear with them. |
| VXVCLS | 2007-12 | d | 4 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| W055RC1 | 1959-01 | m | 3 | critic redundancy | Critic drop (keep MTSR133FMS): Outlays are dominated by one-offs; quarterly NIPA receipts and monthly personal taxes are heavily revised interpolations of the same cash data; business applications start 2004 (two recessions) with a 2020 level shift. |
| WALCL | 2002-12 | w | 5 | critic redundancy | Critic drop (keep M2SL deflated by CPIAUCSL): M2REAL has vintages only from 2014 and is the same information; velocity is quarterly and redundant with M2 plus nominal activity; reserves and the balance sheet are regime-dependent post-2008 with one recession of history; primary credit (2003+) duplicates ... |
| WEI | 2008-01 | w | 2 | critic suspicious | Critic suspicious: Starts 2008; the factor model's 2008-2020 history is in-sample and vintages begin 2020-04. Live-dashboard value only; it cannot be validated across folds. |
| WLCFLPCL | 2002-12 | w | 4 | critic redundancy | Critic drop (keep M2SL deflated by CPIAUCSL): M2REAL has vintages only from 2014 and is the same information; velocity is quarterly and redundant with M2 plus nominal activity; reserves and the balance sheet are regime-dependent post-2008 with one recession of history; primary credit (2003+) duplicates ... |
| WLEMUINDXD | 1985-01 | d | 3 | critic redundancy | Critic drop (keep USEPUNEWSINDXM): Seven text-based uncertainty/volatility indexes from one provider (WLEMUINDXD was proposed by two scouts); only the historical news index spans the 1968-85 folds, the rest start 1985-2007 and are re-estimated as archives update. The VIX3M term ratio starts ... |
| WPSFD4131 | 1974-01 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| WPSFD49207 | 1947-04 | m | 2 | judgement | critic keep for PPI momentum; inflation is not a Stage-3 batch and CPILFESL infl_accel already carries momentum; queue. |
| WPSID61 | 1947-04 | m | 4 | critic redundancy | Critic drop (keep MICH): Twenty inflation/real-rate candidates for a model that already has core CPI yoy, infl_accel, core PCE and WTI. Median, trimmed, sticky, flexible and trimmed-PCE are five versions of underlying inflation; the Cleveland model series re-estimate their whole ... |
| WPU081 | 1926-01 | m | 3 | critic redundancy | Critic drop (keep PERMIT1, MSACSR, USSTHPI): PERMIT/HOUST/HSN1F are in the model; single-family starts correlate with PERMIT1; inventory level duplicates months' supply; three house-price indexes carry one signal and only FHFA is public domain (Case-Shiller is copyrighted); PRFI duplicates the GDP ... |
| WPU10 | 1926-01 | m | 3 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| WPUSI019011 | 1967-01 | m | 2 | judgement | optional per critic; WPU03T15M05 carries the industrial-commodity signal. |
| XTEXVA01USM664S | 1955-01 | m | 3 | critic redundancy | Critic drop (keep WPU03T15M05): IMF copper/materials start 1992 with 1980-91 served as missing and reach FRED 70+ days late; metals PPI is collinear with copper; the daily major-currencies dollar duplicates the broad monthly index at the weekly grid; exports and quarterly real imports add ... |
| XTIMVA01USM664S | 1955-01 | m | 2 | judgement | IMPORTS_M splice needs the unverified BOPTIMP leg and a multiplicative splice; OECD mirror reaches FRED 45-100 days late; outside the batches. |

## 7. Open questions for the orchestrator

1. Should SPREAD_5Y2Y__level / SPREAD_2Y3M__level and the four regional permit
   yoy columns exist only as composite inputs (a `composite_only` flag) or
   stay as boosted-model columns as drafted?
2. Fold the DFF cumulative-tightening and SPREAD_10Y3M_LAGS transforms into
   FEDFUNDS / SPREAD_10Y3M at merge time, or keep the twins?
3. NFCICREDIT / CFNAIDIFF masks: API first-vintage dates (2012) as drafted,
   or the family dates (2011-05)?
4. EMELASA: unmasked as drafted (with the cross-vintage check) or masked at
   2025-04-16?
5. The `wsum` recipe and the rule `weight` key are the two schema extensions
   this file relies on; both are a few lines.
6. Batches outside the plan that the critic keeps and this file leaves out
   (section 6, basis "judgement"): SFNEWSSENT, AHETPI real wage, WPSFD49207,
   IPMAN/IPDCONGD or an IP_BREADTH composite, GDPC1 advance, QUSHAM770A,
   SUBLPDMOSXWBNQ, XTIMVA01USM664S import splice, and the state-level breadth
   modality (50 state unemployment rates / claims / coincident indexes) the
   critic calls the largest unexploited source. They are the natural next
   pre-specified block.
