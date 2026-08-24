# D3 v3 — Audited incremental prediction (PROJECT HEADLINE), audit-response run

Generated: 2026-08-18T13:46:54 | script: scripts/26_prediction_v3.py | seed=1626 | runtime 381s
Python 3.12.13, scikit-learn 1.9.0, pandas 3.0.3

## Headline (pre-specified: logistic, P_tierB leakage-free baseline, within-quarter top-decile label, month-clustered CI)

- Controls-only AUC **0.7234** [0.7042, 0.7438] → +attention AUC **0.8159** [0.8013, 0.8322]
- **ΔAUC = +0.0925**, 95% CI month-clustered [+0.0719, +0.1092] (i.i.d. [+0.0758, +0.1103])
- PR-AUC 0.2550 → 0.3679 (**ΔPR-AUC +0.1129** [+0.0911, +0.1336])
- Precision@top-decile (k=632) 0.2801 → 0.3892 (**Δ +0.1092** [+0.0654, +0.1441]); ΔP@100 +0.13 [+0.05, +0.17] (noisy: 100-draw binomial)
- Brier 0.0841 → 0.0763
- **log upvotes alone: AUC 0.8001** [0.7789, 0.8200], PR-AUC 0.3305; controls add +0.0158 AUC on top of upvotes [+0.0092, +0.0250]
- HGB (equal prominence): controls-only 0.7307 → +attention 0.8083, ΔAUC +0.0776 [+0.0500, +0.1011]; ΔPR-AUC +0.0889; ΔP@top-decile +0.0839; upvotes-only 0.7805
- n_train=5,019 (release_year ≤ 2024), n_test=6,325 (2025), base rate test 0.100

Pre-specified headline (logistic, P_tierB leakage-free baseline, within-quarter top-decile label): adding cumulative HF attention raises test AUC from 0.723 to 0.816 (dAUC +0.093, month-clustered 95% CI [+0.072, +0.109]); PR-AUC 0.255 -> 0.368; precision@top-decile 0.280 -> 0.389. log_upvotes alone gives AUC 0.800; controls add +0.016 AUC on top of upvotes (CI [+0.009, +0.025]). HGB: dAUC +0.078 [+0.050, +0.101]. Predictive, not causal; upvotes are cumulative at collection.

## What changed vs v1 (scripts/26_prediction.py) and why
1. `subfield` → `subfield_kw` (uniform keyword taxonomy). Legacy label source was keyed on an upvote-ordered arXiv fetch and predicted the label with AUC 0.659 on its own (Table 6); it inflated the controls-only baseline. Legacy kept only as a flagged robustness row.
2. P_tierB (leakage-free prior-paper count + years active) is the definitive baseline; P_interim/P_none are brackets. Bool/first-year/raw Tier-B columns are not features (v1 hook would have ingested them).
3. Headline model pre-specified as logistic; HGB re-tuned with early stopping (v1 HGB was under-tuned: +attention AUC below upvotes-alone).
4. Added: upvotes-only row, PR-AUC, precision@top-decile, Brier, month-clustered CIs, comments ablation, within-month + influential labels, backward-2023 test, drop-age, no-launch-months, per-quarter AUC, provenance and accrual evidence.
5. No hard-coded n / dates; regressors (ridge/HGB-reg on q_i) dropped in v3 (not part of the audited claim).

Model notes: the HGB "log upvotes only" row (AUC 0.780) sits below the model-free AUC of log upvotes
(0.800; = the logistic upvotes-only row) because a boosted one-feature model is not monotone in its input;
the model-free number is the one to quote for "upvotes alone". HGB tuning selected the largest l2_regularization in the grid
(1000) in 32% of fits — CV-AUC differences across the upper l2 values are small
and the early-stopped HGB does not beat the logistic here; both are reported.

## Robustness (see prediction_v3_tables.md Table 3)
- Primary: within-quarter label, test 2025 (n=6,325): logistic ΔAUC +0.093 [+0.072, +0.109]; hgb ΔAUC +0.078 [+0.050, +0.101]
- Mature K=12 subset (age ≥ 12 mo; label re-ranked within quarter) (n=2,717): logistic ΔAUC +0.083 [+0.043, +0.124]; hgb ΔAUC +0.066 [+0.013, +0.124]
- Within-month label (n=6,325): logistic ΔAUC +0.097 [+0.081, +0.111]; hgb ΔAUC +0.093 [+0.068, +0.117]
- Top-decile influential citations (within quarter) (n=6,325): logistic ΔAUC +0.091 [+0.071, +0.105]; hgb ΔAUC +0.075 [+0.054, +0.093]
- Backward: train 2024–25 → test 2023 (within-quarter label) (n=1,602): logistic ΔAUC +0.125 [+0.093, +0.161]; hgb ΔAUC +0.120 [+0.092, +0.162]
- Backward: train 2024–25 → test 2023 (within-month label) (n=1,602): logistic ΔAUC +0.131 [+0.101, +0.162]; hgb ΔAUC +0.119 [+0.093, +0.152]
- Drop age_months from controls (n=6,325): logistic ΔAUC +0.092 [+0.072, +0.109]; hgb ΔAUC +0.082 [+0.053, +0.107]
- Train excludes launch-era months (≤ 2023-06) (n=6,325): logistic ΔAUC +0.093 [+0.071, +0.110]; hgb ΔAUC +0.084 [+0.059, +0.106]
- FLAGGED: legacy mixed-taxonomy `subfield` instead of subfield_kw (n=6,325): logistic ΔAUC +0.067 [+0.046, +0.084]; hgb ΔAUC +0.066 [+0.039, +0.089]
- FLAGGED: v2 leaky (max_hindex + P_interim baseline) (n=6,325): logistic ΔAUC +0.090 [+0.069, +0.106]; hgb ΔAUC +0.072 [+0.043, +0.096]

Per-quarter AUC of log upvotes alone in 2025: 2025Q1 0.835, 2025Q2 0.759, 2025Q3 0.828, 2025Q4 0.795.
Raw AUC of log upvotes: test-2025 0.800; backward test-2023 0.735; mature K=12 0.789.

## Leakage audit (verdict: PASS/DISCLOSED — no outcome leakage in headline rows)
- split_temporal: **PASS** — train=release_year<=2024 (n=5,019), test=release_year=2025 (n=6,325); years and quarters disjoint (asserted).
- labels_same_snapshot_caveat: **DISCLOSED** — All citation counts are one 2026-06 snapshot; train labels use longer exposure windows than test labels. Within-cohort ranking, the K=12 mature row and the backward-2023 test address this; a true forward simulation would need historical snapshots.
- scalers_fit_on_train_only: **PASS** — SimpleImputer/StandardScaler/OneHotEncoder inside a Pipeline fitted on train only; test transformed with frozen objects.
- hyperparameters_tuned_on_train_only: **PASS** — GridSearchCV with GroupKFold(5) grouped by release_month on train; one evaluation on test.
- headline_model_prespecified: **PASS** — headline model=logistic, branch=P_tierB, row=+attention, CI=month_cluster; no test-set-based selection of model or branch.
- target_is_label_only: **PASS** — citation_count/influential_citations used only to build labels; FORBIDDEN set asserted for every feature matrix.
- within_group_ranks: **PASS** — upvote_rank_within_month and all labels are within-group ranks; release_month/quarter never enter as features.
- forbidden_columns_absent: **PASS** — check_features() raised on any forbidden column; max_hindex only in v2_leaky_replication, legacy `subfield` only in legacy_subfield.
- tierB_columns_whitelisted: **PASS** — P_tierB uses exactly log1p_max_prior_papers_true + max_years_active; tierB_resolved, *_first_year and raw/winsorised counts excluded; missing Tier-B values: train 82, test 16 (median-imputed).
- label_provenance_uniform: **PASS** — subfield_kw comes from one rule set for every paper; source flag (ai_keywords vs fallback) predicts the label with AUC 0.518 in test. The legacy `subfield` provenance flag (arXiv fetched) had AUC 0.659 — that leak is now confined to the flagged legacy_subfield row.
- attention_is_cumulative_at_collection: **DISCLOSED** — upvotes are the cumulative HF count at collection (June 2026), not a day-one measurement; n_trend_days is 1 for essentially all papers so 'peak' == snapshot. Evidence that late accrual is small relative to cross-paper variation: flat monthly medians after the launch era and near-zero Spearman(upvotes, age_months) within 2024 and 2025.
- age_months_support: **DISCLOSED** — age_months is a deterministic function of release date and does not overlap across the split; the drop_age robustness row shows the sensitivity.

Attention accrual evidence: Spearman(upvotes, age_months) within year — 2023: -0.501, 2024: +0.126, 2025: +0.031 (2023 reflects the launch-era ramp; months ≤ 2023-06 have medians 2023-05=2, 2023-06=7). Monthly medians afterwards range 10–21.

## CAN / CANNOT
**CAN:** say that cumulative HF attention (upvotes at collection) adds honest out-of-sample incremental predictive value for within-cohort citation rank, forward in time and backward onto a mature cohort, over a leakage-checked controls-only baseline — and that a single column (log upvotes) carries essentially all of it.
**CANNOT:** claim causation; claim day-one/early attention (no vote timestamps; cumulative counts); generalise beyond HF Daily Papers; treat the paired bootstrap as covering training-set variability.

## Files
- results/prediction_v3.json — all metrics, CIs, deltas, tuning, audit
- results/prediction_v3_tables.md — paste-ready tables
- results/prediction_v3_scores.csv — per-test-row labels + predicted probabilities (columns `<experiment>__<model>__<branch>__<row>`; forward-2025 and backward-2023 rows stacked, `split` column)
