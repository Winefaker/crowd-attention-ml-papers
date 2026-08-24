# association_v3 — NOTE (definitions, caveats, headline numbers)

Script: `scripts/27_association_v3.py` · Ledger: `results/association_v3.json` · Tables: `association_v3_tables.md` ·
Dashboard file: `data/processed/overunder_v3.csv` · Seed 20260818 · runtime 381.8 s.

## What this is
Part I of the project (the proposal's question) rebuilt on the final frame (`analysis_final.csv`, n = 11,344; citations 5–40 months old).
It supersedes the June-11 v2 scripts 12/14/15. Differences from v2 that matter:
1. **Subfield** = `subfield_kw`, one keyword rule set applied to every paper (13 levels). The legacy `subfield` mixed arXiv-category labels
   (only for the most-upvoted 27.5 %) with keyword labels and therefore encoded "upvotes ≥ 17"; it appears only as a robustness row.
2. **Prestige** = Tier B leakage-free (max prior papers and years active of first/last author *before* the paper). The 2026-measured
   h-index (max/last author) is kept as a separately flagged, **leaky** control (M5/M6, driver logits) — it is downstream of the outcome.
   `author_max_appear` (48 % look-ahead) is replaced by the prior-only count (`max_appear_prior`); the look-ahead version is fit once only to size the leak.
3. All SEs are **cluster-robust by release month** (34 clusters); the NB2 with month FE **converges** (v2's M4 did not).
4. The never-trending comparison uses **exact matching** with a balance-preserving design and CIs, and the "under-rated papers sit at the
   91st percentile" claim is replaced by the honest bottom-tertile-attention comparison.

## Definitions
* Attention: `log_upvotes = log(1 + upvotes)`; upvotes are cumulative at scrape (2026-06-05 / 06-11), **not day-one**.
* Impact: Semantic Scholar `citation_count` at 2026-06-11 (113 outcomes title-match-repaired; robustness row drops them).
* IRR per doubling of upvotes = exp(β · ln 2) for Poisson/NB; for log1p-OLS the analogous quantity is 2^β on (1 + citations).
* Ladder: M0 raw; M1 + log age; M2 + subfield_kw FE; M3 + release-month FE + day-of-week FE; M4 + Tier B prestige (+ missing flag) [main];
  M5 + log1p max/last h-index (leaky); M6 + log authors, title words/colon, log abstract chars, GitHub, 10 kw flags.
* Poisson-QMLE: `pyfixest.fepois`, FE absorbed, CRV1 by month. NB2: `statsmodels.negativebinomial` with dummies, Poisson-GLM start values,
  Newton (then BFGS/NM fallback), cluster-robust by month. OLS: `pyfixest.feols` on log1p citations, CRV1.
* Hierarchical: `MixedLM` log1p citations with month + dow FE, log age, Tier B prestige as fixed effects; random intercept + random slope
  of log_upvotes by subfield_kw (REML); LR test on ML fits. A Poisson GLMM was **not** fit (statsmodels only offers a variational-Bayes
  Poisson mixed GLM without ML/cluster-robust inference); the count-scale estimate comes from the FE ladder.
* Selection: treated = top tertile of raw upvotes (≥ 22), control = bottom tertile (≤ 9), middle dropped;
  PS logit on Tier B prestige + log age + subfield_kw + release_month; 1:1 NN with replacement, caliper 0.2 SD of logit PS; ATT-IPW
  stabilised/normalised/trimmed p99; CEM exact on month × subfield_kw × Tier B quintile; 300 bootstrap reps each.
* Over/under-rated: residual percentiles from `log_upvotes ~ log_age + subfield_kw + release_month` and the same for `log_citations`;
  overrated = attention pct ≥ 2/3 & impact pct ≤ 1/3; underrated = reverse. Same definition as the dashboard explorer.
* Control comparison: control subfield = same keyword rules on the title, else arXiv category → nearest label; h-index bins = pooled quintiles;
  exact matching on month × subfield × h-bin with ATT weights; percentile = position of each trending paper's citation count among
  same-release-month background papers (mid-rank ties).

## Headline numbers (M4, main spec)
* Poisson-QMLE β = 1.0992 (SE 0.0608), IRR per doubling **2.142** [1.972, 2.327]; month-block bootstrap CI [1.997, 2.318].
* NB2 β = 0.7997 (SE 0.0291), IRR per doubling **1.741** [1.673, 1.811] (converged: True); bootstrap CI [1.677, 1.817].
* log1p-OLS elasticity **0.6335** (SE 0.0332) → ×1.551 (1+citations) per doubling; bootstrap CI [0.5807, 0.713].
* E-values: Poisson 3.706 (CI bound 3.356); NB2 2.877 (CI bound 2.734).
* Placebo permutation: Poisson β mean 0.05 (SD 0.0555); real β is 18.9 SDs above; reference-count placebo outcome = 9% of the citation elasticity.
* MixedLM: fixed slope 0.6253, between-subfield slope SD 0.031, slopes [0.594, 0.672].
* Selection ATT (log pts): naive 1.295; PS-matched 1.277 [1.1854, 1.3674]; IPW 1.264 [1.2082, 1.3294]; CEM 1.241 [1.1512, 1.2937].
* Trending premium (CEM month × subfield × h-bin): 1.1 log pts, ×3.0 [2.73, 3.18] (naive ×3.43).
* Bottom-tertile-attention trending papers sit at the 70% percentile of month-matched background
  (CI [0.6963, 0.7125]); the outcome-selected "under-rated" group sits at 98% (circular by construction).
* Placebo with month × subfield interacted FE: Poisson placebo mean 0.0041 (SD 0.0664) vs real 1.0896.
* Over/under-rated counts: {'neutral': 10151, 'overrated': 632, 'underrated': 561}. Prestige asymmetry (median, over vs under vs neutral): Tier B prior papers
  11.0 / 14.0 / 12.0
  (diff over−under -3.0 CI [-6.0, -1.0]); 2026 h-index (leaky)
  14.0 / 20.0 / 18.0 (diff -6.0 CI [-8.0, -4.0]).
  Leakage-free head-to-head logit: Tier B prior papers OR 0.788 per SD for being over- rather than under-rated
  (CI [0.692, 0.897]). Title-only TF-IDF themes separate the groups weakly (CV AUC 0.631).
* 69 of the 632 over-rated papers have 0 recorded citations (19 at ≥ 12 months) — some are S2 matching failures (e.g. Differential Transformer), i.e. impact-side measurement error.

## Important caveats (carry into the report)
1. **Estimator disagreement is real and informative.** Poisson-QMLE (mean-scale, dominated by the heavy right tail) gives a larger IRR than
   NB2, and log1p-OLS (geometric-mean elasticity) is smaller still. The squared-term row shows the log-log relation is convex — the
   attention elasticity is larger among highly-upvoted papers — so "the" IRR depends on how papers are weighted. Report all three; quote
   NB2 or OLS as the typical-paper effect and Poisson as the mean-citations effect.
2. **Association, not causation.** Upvotes are cumulative at scrape and citations at a single snapshot; reverse causality (fame → later
   upvotes) cannot be excluded. Matching/IPW/CEM adjust only for observed prestige/field/time. The E-value is a sensitivity summary.
3. **h-index is measured in 2026** (after the outcome). M5/M6 and the driver logits that include it are flagged; the M4 estimate barely
   moves when it is added, so the leak does not drive the attention coefficient — but the *prestige asymmetry* using h-index mixes crowd
   error with a citation-side Matthew effect; the Tier B version is the leakage-free one.
4. **Control sample** = first-day(s)-of-month cs.CL/CV/LG submissions with S2 outcomes (n = 3,280), popularity-blind but not a random month
   sample; its subfield labels are title/category-based (coarser than the trending labels), and only 65% of trending papers find an exact cell.
5. **Exposure heterogeneity**: 2025 papers have 5–17 months of exposure; log age + month FE absorb most of it (see cohort rows), and the
   over/under labels are residualised on age, field and month.
6. Numbers differ from the June-11 (v2) write-up because of the taxonomy, prestige, month-FE and robust-SE changes above — v2's numbers
   are not wrong on their own frame but should be replaced by these throughout.
