# D1 v3 — Crowding IV: attempted design, honest re-specification

Generated 2026-08-18T13:31:39 | script scripts/25_crowding_iv_v3.py | pyfixest 0.60.0 | N_base=11,344

## Verdict
VERDICT: the crowding IV is an attempted design, not informative causal evidence. Under the honest specification (release_month + dow + subfield_kw FE, no day FE) the first stage is modest (t=-5.6, KP-F=31) and the weak-IV-robust AR 95% CI (0.021, 0.921) is 0.90 log-points wide, barely excludes zero and includes the OLS estimate (0.508). The day-FE first stage (KP-F=413) is the within-day adding-up identity, not exogenous variation (2SLS beta 0.917 in singleton cells vs 0.619 in non-singleton cells); a same-topic crowding instrument with no adding-up link to own upvotes (own-subfield leave-one-out peer sum, no day FE) has no usable first stage (t=-0.55). Report as 'consistent with OLS but uninformative'; do not describe as suggestive causal support.

- Honest spec (month+dow+subfield_kw FE, leakage-free prestige, N=11,230): first-stage t=-5.56 (KP-F=30.9), 2SLS beta=0.474 (SE 0.215), AR 95% CI (0.021, 0.921), width 0.90 log-points.
- The AR interval excludes zero and includes the OLS coefficient (0.508); it is too wide to be informative about the size of the effect.
- Under day FE (v1 design) the first stage looks strong (t=-20.3, KP-F=413) but this is the within-day adding-up identity: 18% of papers sit in singleton (day x subfield) cells where Z1' is exactly log1p(day total - own upvotes); the 2SLS coefficient is 0.917 in singleton cells vs 0.619 in non-singleton cells (all-sample 0.713).
- A crowding instrument with no mechanical link to own upvotes (own-subfield leave-one-out peer sum) has a first stage (t=-3.80); the other-subfield paper COUNT under day FE has first-stage sign positive (+0.145), the wrong sign for a crowding story.

## (i) OLS, release_month + dow + subfield_kw FE (leakage-free prestige, log n_authors)
beta = 0.5081; SE clustered by cohort_day 0.0152 (G=674), by release_month 0.0351 (G=34); N=11,230.
No-prestige: 0.5103 (0.0152); day+subfield_kw FE: 0.5136 (0.0154); legacy v1 replication: 0.5966.

## (ii)–(iii) IV ladder (cluster = cohort_day unless stated; SE in parentheses; AR = cluster-robust Anderson–Rubin 95% CI)
| Spec | FE | Z | first-stage π (t) | KP-F | 2SLS β (SE) | AR 95% CI | OLS same sample | N |
|---|---|---|---|---|---|---|---|---|
| PRIMARY honest (tierB controls) | release_month + dow + subfield_kw | Z1p_kw | -0.107 (-5.6) | 30.9 | 0.474 (0.215) | (0.021, 0.921) | 0.508 | 11,230 |
| honest, cluster=month | release_month + dow + subfield_kw | Z1p_kw | -0.107 (-4.7) | 22.0 | 0.474 (0.258) | (-0.022, 1.098) | 0.508 | 11,230 |
| honest, no prestige | release_month + dow + subfield_kw | Z1p_kw | -0.107 (-5.6) | 31.5 | 0.450 (0.214) | (-0.003, 0.892) | 0.510 | 11,344 |
| honest, COUNT instrument | release_month + dow + subfield_kw | Zc_kw | -0.178 (-8.2) | 67.3 | 0.317 (0.167) | (-0.016, 0.657) | 0.508 | 11,230 |
| honest, own-subfield LOO peers | release_month + dow + subfield_kw | Zown_loo_kw | -0.006 (-0.5) | 0.3 | 1.917 (3.261) | (-inf, +inf) | 0.491 | 9,221 |
| day FE (v1 design, kw taxonomy) | cohort_day + subfield_kw | Z1p_kw | -1.159 (-20.3) | 413.0 | 0.713 (0.050) | (0.615, 0.814) | 0.514 | 11,220 |
| day FE, singleton cells | cohort_day + subfield_kw | Z1p_kw | -3.772 (-9.9) | 98.4 | 0.917 (0.070) | — | 0.611 | 1,907 |
| day FE, non-singleton cells | cohort_day + subfield_kw | Z1p_kw | -1.168 (-18.3) | 336.4 | 0.619 (0.057) | — | 0.495 | 9,221 |
| day FE, own-subfield LOO peers | cohort_day + subfield_kw | Zown_loo_kw | -0.052 (-3.8) | 14.5 | 0.839 (0.276) | — | 0.495 | 9,221 |
| day FE, COUNT instrument | cohort_day + subfield_kw | Zc_kw | +0.145 (+2.0) | 4.2 | 1.074 (0.770) | — | 0.514 | 11,220 |
| LEGACY v1 primary-P replication | cohort_day + subfield | Z1p_othersub | -1.292 (-18.8) | 354.7 | 0.758 (0.055) | (0.650, 0.867) | 0.597 | 11,236 |
| LEGACY, honest FE | release_month + dow + subfield | Z1p_othersub | -0.085 (-4.8) | 23.1 | 0.389 (0.277) | (-0.217, 0.973) | 0.588 | 11,246 |

AR CI implementation: test inversion of the cluster-robust t² of Z in (Y − β0·D) ~ Z + X | FE, closed-form via FWL residuals with
pyfixest small-sample factors; closed form vs pyfixest at a check point differ by 1.82e-08.

## Reflection diagnostics (uniform taxonomy)
- (day × subfield_kw) cells: 4,568; singleton cells 44.2% of cells, 17.9% of papers; within-cell variance share of Z1'_kw = 0.0000.
- Within-day corr(Z1'_kw, log upvotes): all -0.323; singleton -0.623; non-singleton -0.300. Within-month corr: -0.082.
- Balance of Z1'_kw under honest FE: age_months t=-0.75; log_n_authors t=-2.75 (imbalance); has_github t=+1.50; log1p_max_prior_papers_true t=+1.35; max_years_active t=+0.01

## What may be said
- OLS association (β≈0.51 log-points per log-upvote, month+dow+subfield_kw FE) is precise and robust to clustering level.
- The IV is an attempted design: honest first stage t=-5.6; AR interval (0.021, 0.921) — consistent with OLS but uninformative; the day-FE F≈413 is mechanical (reflection); own-subfield LOO peer sum first stage t=-3.80 (day FE) / -0.55 (honest FE), both with wide 2SLS intervals.
- Not to be described as causal support, suggestive or otherwise.
