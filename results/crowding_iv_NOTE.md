> SUPERSEDED. This is the first pass, kept so the changes behind the revised numbers can be traced. The current numbers are in the files whose names end in `_v3` and in the ledger `D5_results_gate_v3.md`. The prediction gain is now +0.093 rather than +0.062, and the crowding instrument is reported as an attempted design rather than as suggestive causal support.

# D1 Crowding IV — Results Note
**Date**: 2026-06-26 | Revised after internal review

## Headline Number
2SLS β = 0.758, AR 95% CI (0.6490, 0.8659), reduced form negative, consistent with OLS 0.597.
β = 0.758 is a **point inside** the AR interval — not a precise estimate.
Present as: consistent-with-OLS / AR-bounded range (0.6490, 0.8659).

## First Stage — With Caveats
- Instrument: `Z1p_othersub` (same-day cross-subfield competitor proportion, log-scale)
- π₁ = -1.2925 (negative, as expected: crowding reduces upvotes for focal paper)
- KP F (primary-P) = **354.7** — numerically large, but PARTLY MECHANICAL.
  Z1p_othersub is a pure (cohort_day×subfield) aggregate (100% between-cell variance,
  0% within-cell variance). Median cell size = 1;
  52.6% of (cohort_day×subfield) cells are singletons
  (2975 papers, 26.2% of sample).
  For singleton cells the leave-ego-out in other subfields is effectively leave-ego-out,
  so F=354 should NOT be presented as clean exogenous first-stage strength.
- KP F without subfield FE = 580.3; companion (Z2+Z3) = 11.3

## OLS vs 2SLS (primary-P, cohort_day+subfield FE)
| Estimator | β on log_upvotes | Cluster SE | N |
|-----------|-------------------|------------|---|
| OLS       | 0.5966 | 0.0183 | 11236 |
| 2SLS (P)  | 0.7579 | 0.0550 | 11236 |
| 2SLS (NP) | 0.7610 | 0.0554 | 11336 |

## Anderson–Rubin 95% CI (weak-IV-robust, primary reportable result)
- AR CI (primary-P): **(0.6490, 0.8659)**
- Grid: β₀ ∈ [−300, +300], fine pass 200 001 points; CRV1 G/(G−1); χ²(1,5%) = 3.841
- CI is bounded. Report the interval, not the point.

## Identification — What the Variation Actually Is
Z1p_othersub varies **only across (cohort_day×subfield) cells**, never within them.
All first-stage and structural identification comes from cross-subfield-within-day variation.
This means:
1. The instrument cannot be separated from a (cohort_day×subfield) fixed effect.
2. Adding a day×subfield control would exactly absorb the instrument — the exclusion
   restriction is **structurally untestable** in this design.
3. The binding exclusion threat is a **subfield-day demand shock**: 'my topic is hot today'
   raises citations directly, independent of crowding. This threat cannot be controlled away.

## Reduced Form
- β on Z1p_othersub (log_citations ~ Z1p + X | cohort_day+subfield): **-0.9796**
- Sign: **negative** — consistent with crowding-suppresses-citations channel
- Companion (Z2_count): -0.0011;  Z3_blockbuster: -0.0347

## Reference Placebo
- Outcome: log(1+reference_count). RF: β_Z1p = 0.0294, t = 0.42 (N=11236)
- 2SLS: β = -0.0227, t = -0.42
- Result: **placebo passes** (t≈0.42) — supportive, not a composition artifact.
- **BUT LOW-POWERED**: OLS(log_ref ~ log_upvotes) ≈ -0.023 (t≈-1.5).
  Even if the demand channel biased log_citations, it need not move reference counts much.
  This placebo is not decisive against the subfield-demand threat.

## Covariate Balance on the Instrument
Method: OLS: covariate ~ Z1p_othersub | cohort_day+subfield, CRV1(cohort_day)
| Covariate | coef | SE | t |
|-----------|------|----|---|
| age_months | 0.0730 | 0.0290 | 2.514 ← IMBALANCE |
| max_hindex | -1.1150 | 1.2998 | -0.858 |
| n_authors | -6.1147 | 2.7767 | -2.202 ← IMBALANCE |
| has_github | -0.0634 | 0.0293 | -2.164 ← IMBALANCE |

**age_months imbalance (t≈2.5)**: papers in more-crowded subfield-day cells tend to be older.
n_authors and has_github also show t>2. This threatens exclusion if age/team-size affect
citation accrual independently of crowding.

## Hansen-J (Companion Over-ID, Z2+Z3)
- J stat = 1.0218, p = 0.3121, df = 1
- **Caveat**: Sargan J is non-cluster-robust and unreliable under weak IV. Treat as indicative.
- J does not reject (p > 0.05) — instruments not obviously inconsistent.

## What This Spec CAN and CANNOT Conclude (§6, honest version)
**CAN (suggestive)**: Using same-day cross-subfield crowding as an instrument, we find
suggestive evidence that marginal day-one HuggingFace attention (visibility) increases arXiv
citations for featured papers, conditional on day and subfield fixed effects.
2SLS β = 0.758, AR 95% CI (0.6490, 0.8659), consistent with OLS 0.597.
The result is weak-IV-robust. It is NOT a clean point identification.

**CANNOT**: identify the effect against a subfield-day demand shock (structurally untestable);
estimate an extensive-margin effect (featured vs not); claim a global ATE; generalise beyond
HF Daily Papers; or treat F=354 as unconditional instrument strength.

## Verdict
B3 is **suggestive, weak-IV-robust causal evidence consistent with the controlled association**.
It triangulates the Track B1 OLS association and the Track C/A findings.
It is NOT a clean causal point identification and should NOT be the project headline.
The project headline rests on Tracks C (audited prediction) + A (ACL companion);
B3 provides triangulating support.

## Spec Compliance
- ego_upvotes: NOT included as control ✓
- Z2/Z3/cohort_size: NOT in primary day-FE spec ✓
- cohort_size in companion: DROPPED (Z2_count = cohort_size-1 exactly; collinear → instrument absorbed) ✓
- cohort_day FE: NOT in companion spec ✓
- Full month×subfield interaction: NOT used (additive only) ✓
- Cluster at cohort_day: everywhere ✓
- Prestige flagged as INTERIM in all Spec P results ✓
- AR CI reported alongside 2SLS point ✓
- Homoskedastic F: NOT reported ✓
- Identification diagnostics: ADDED (internal review 2026-06-26) ✓