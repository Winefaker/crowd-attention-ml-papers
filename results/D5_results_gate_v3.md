# D5 v3 — Results gate & numbers ledger (audit-response pipeline)

**Generated:** 2026-08-24T14:51:33 by `scripts/28_results_gate_v3.py` from `results/prediction_v3.json` (2026-08-18T13:46:54), `results/crowding_iv_v3.json` (2026-08-18T13:31:39), `results/association_v3.json` (scripts/27_association_v3.py).
**Rule:** the report, dashboard and deck may cite only numbers in this ledger; each row gives n, CI and the exact script + JSON key. Re-run this script after any upstream re-run.

## A. Per-result verdict (v3)

| Result | Verdict | One-line judgment |
|---|---|---|
| **D3 v3 prediction (HEADLINE)** | **MEANINGFUL — predictive, leakage-checked** | Logistic (pre-specified), P_tierB leakage-free baseline: AUC 0.723 → 0.816, ΔAUC +0.093 [+0.072, +0.109] (month-clustered); HGB ΔAUC +0.078 [+0.050, +0.101]. log upvotes alone AUC 0.800. Robust across labels, backward-2023 test, mature subset. Predictive, not causal. |
| **D1 v3 crowding IV** | **ATTEMPTED DESIGN — not informative causal evidence** | Honest spec (month+dow+subfield_kw FE): 2SLS β 0.474 (SE 0.215), AR 95% CI (0.021, 0.921), first-stage t -5.6; day-FE F=413 is the within-day adding-up identity (reflection). Report only as 'consistent with OLS but uninformative'. |
| **Part I association (v3, other workstream)** | **MEANINGFUL — association, not causation** | M4 (subfield_kw + month + dow FE, Tier B prestige): Poisson-QMLE IRR per doubling 2.14; NB2 1.74; log1p-OLS elasticity 0.633. See §C5 (numbers owned by `27_association_v3.py`). |
| Feasibility checks and the deferred preprint policy design | unchanged from the first pass | Clean nulls, and a design specified but not run. |

## B. Identification posture (v3)

- **The clean evidence is predictive (D3 v3).** Cumulative HF attention (upvotes at collection, June 2026) adds out-of-sample incremental predictive value for within-cohort citation rank, forward in time (train ≤2024 → test 2025) and backward onto the mature 2023 cohort, over a leakage-checked controls-only baseline that uses a uniform subfield taxonomy and leakage-free Tier-B prestige. A single column (log upvotes) carries essentially all of it.
- **The crowding IV is an attempted design.** Under day FE the instrument is a transform of the paper's own upvotes (reflection); without day FE it is modest and its weak-IV-robust interval is uninformative. It provides no causal information beyond the OLS association.
- **Part I (association ladder, selection adjustment, over/under-rated, never-trending comparison)** is reported as association with sensitivity analysis (E-values, placebo), never as causal.

## C. Numbers ledger

### C1. D3 v3 headline — `results/prediction_v3.json` (n_train=5,019, n_test=6,325, top-decile k=632, bootstrap 2000 reps seed 1626; CIs month-clustered unless stated)

| # | Quantity | Value | 95% CI | n | Source (script → JSON key) |
|---|---|---|---|---|---|
| 1 | Logistic (headline): controls-only AUC (P_tierB) | 0.7234 | [0.7042, 0.7438] | 6325 | 26_prediction_v3.py → headline.by_model.logistic.controls_only_auc |
| 2 | Logistic (headline): +attention AUC | 0.8159 | [0.8013, 0.8322] | 6325 | headline.by_model.logistic.attention_auc |
| 3 | **Logistic (headline): ΔAUC (+attention − controls)** | **+0.0925** | [+0.0719, +0.1092] (i.i.d. [+0.0758, +0.1103]) | 6325 | headline.by_model.logistic.delta_auc / delta_auc_ci_month |
| 4 | Logistic (headline): log-upvotes-only AUC | 0.8001 | [0.7789, 0.8200] | 6325 | headline.by_model.logistic.upvotes_only_auc |
| 5 | Logistic (headline): (+attention) − (upvotes only) ΔAUC | +0.0158 | [+0.0092, +0.0250] | 6325 | headline.by_model.logistic.attention_minus_upvotes_only_auc |
| 6 | Logistic (headline): PR-AUC controls → +attention | 0.2550 → 0.3679 | see ΔPR-AUC | 6325 | headline.by_model.logistic.controls_only_pr_auc / attention_pr_auc |
| 7 | Logistic (headline): ΔPR-AUC | +0.1129 | [+0.0911, +0.1336] | 6325 | headline.by_model.logistic.delta_pr_auc |
| 8 | Logistic (headline): precision@top-decile controls → +attention (k=632) | 0.2801 → 0.3892 | see Δ | 6325 | headline.by_model.logistic.controls_only_p_at_k / attention_p_at_k |
| 9 | Logistic (headline): Δprecision@top-decile | +0.1092 | [+0.0654, +0.1441] | 6325 | headline.by_model.logistic.delta_p_at_k |
| 10 | Logistic (headline): ΔP@100 (NOISY — 100-draw binomial; do not headline) | +0.13 | [+0.05, +0.17] | 6325 | headline.by_model.logistic.delta_p_at_100 |
| 11 | Logistic (headline): Brier controls → +attention | 0.0841 → 0.0763 | — | 6325 | headline.by_model.logistic.controls_only_brier / attention_brier |
| 12 | HGB: controls-only AUC (P_tierB) | 0.7307 | [0.7115, 0.7512] | 6325 | 26_prediction_v3.py → headline.by_model.hgb.controls_only_auc |
| 13 | HGB: +attention AUC | 0.8083 | [0.7883, 0.8283] | 6325 | headline.by_model.hgb.attention_auc |
| 14 | **HGB: ΔAUC (+attention − controls)** | **+0.0776** | [+0.0500, +0.1011] (i.i.d. [+0.0587, +0.0981]) | 6325 | headline.by_model.hgb.delta_auc / delta_auc_ci_month |
| 15 | HGB: log-upvotes-only AUC | 0.7805 | [0.7589, 0.8035] | 6325 | headline.by_model.hgb.upvotes_only_auc |
| 16 | HGB: (+attention) − (upvotes only) ΔAUC | +0.0279 | [+0.0168, +0.0364] | 6325 | headline.by_model.hgb.attention_minus_upvotes_only_auc |
| 17 | HGB: PR-AUC controls → +attention | 0.2581 → 0.3471 | see ΔPR-AUC | 6325 | headline.by_model.hgb.controls_only_pr_auc / attention_pr_auc |
| 18 | HGB: ΔPR-AUC | +0.0889 | [+0.0509, +0.1252] | 6325 | headline.by_model.hgb.delta_pr_auc |
| 19 | HGB: precision@top-decile controls → +attention (k=632) | 0.3101 → 0.3940 | see Δ | 6325 | headline.by_model.hgb.controls_only_p_at_k / attention_p_at_k |
| 20 | HGB: Δprecision@top-decile | +0.0839 | [+0.0445, +0.1267] | 6325 | headline.by_model.hgb.delta_p_at_k |
| 21 | HGB: ΔP@100 (NOISY — 100-draw binomial; do not headline) | +0.12 | [+0.04, +0.20] | 6325 | headline.by_model.hgb.delta_p_at_100 |
| 22 | HGB: Brier controls → +attention | 0.0833 → 0.0771 | — | 6325 | headline.by_model.hgb.controls_only_brier / attention_brier |
| 23 | Model-free AUC of log upvotes (test 2025) | 0.8001 | = logistic upvotes-only row | 6325 | raw_attention_auc.test_2025_log_upvotes |
| 24 | Base rate (top decile) train / test | 0.101 / 0.100 | — | 5,019/6,325 | experiments.forward_yq.base_rate_train / base_rate_test |
| 25 | +attention+comments ΔAUC (logistic / HGB) | +0.0938 / +0.0746 | [+0.0742, +0.1095] (logistic) | 6325 | experiments.forward_yq.models.<model>.main.rows['P_tierB|+attention+comments'].delta_vs_controls_only |

### C2. D3 v3 prestige bracket and robustness rows (ΔAUC = +attention − controls-only; month-clustered CI)

| # | Row | n_test | Logistic ΔAUC [CI] | HGB ΔAUC [CI] | Logistic ctrl→+att AUC | Upvotes-only AUC | Source (JSON key) |
|---|---|---|---|---|---|---|---|
| 26 | Branch P_tierB (DEFINITIVE) | 6,325 | +0.0925 [+0.0719, +0.1092] | +0.0776 [+0.0500, +0.1011] | 0.7234 → 0.8159 | 0.8001 | experiments.forward_yq.models.*.main.rows['P_tierB|+attention'] |
| 27 | Branch P_interim (ROBUSTNESS) | 6,325 | +0.0938 [+0.0730, +0.1106] | +0.0827 [+0.0539, +0.1067] | 0.7216 → 0.8154 | 0.8001 | experiments.forward_yq.models.*.main.rows['P_interim|+attention'] |
| 28 | Branch P_none (ROBUSTNESS) | 6,325 | +0.0936 [+0.0728, +0.1103] | +0.0799 [+0.0518, +0.1030] | 0.7217 → 0.8153 | 0.8001 | experiments.forward_yq.models.*.main.rows['P_none|+attention'] |
| 29 | Mature K=12 test subset (age ≥ 12 mo; label re-ranked within quarter) | 2,717 | +0.0834 [+0.0434, +0.1235] | +0.0655 [+0.0132, +0.1241] | 0.7237 → 0.8071 | 0.7888 | experiments.forward_yq.models.*.mature_k12.rows['P_tierB|+attention'] |
| 30 | Within-month label | 6,325 | +0.0974 [+0.0810, +0.1109] | +0.0929 [+0.0675, +0.1172] | 0.7147 → 0.8121 | 0.7959 | experiments.forward_ym.models.*.main.rows['P_tierB|+attention'] |
| 31 | Top-decile influential citations | 6,325 | +0.0907 [+0.0706, +0.1050] | +0.0749 [+0.0540, +0.0927] | 0.7106 → 0.8013 | 0.7811 | experiments.forward_yinf.models.*.main.rows['P_tierB|+attention'] |
| 32 | Backward test: train 2024–25 → test 2023 (within-quarter label) | 1,602 | +0.1246 [+0.0927, +0.1606] | +0.1204 [+0.0923, +0.1617] | 0.6729 → 0.7975 | 0.7345 | experiments.backward_yq.models.*.main.rows['P_tierB|+attention'] |
| 33 | Backward test (within-month label) | 1,602 | +0.1305 [+0.1011, +0.1620] | +0.1190 [+0.0933, +0.1515] | 0.6729 → 0.8035 | — | experiments.backward_ym.models.*.main.rows['P_tierB|+attention'] |
| 34 | Drop age_months from controls | 6,325 | +0.0924 [+0.0716, +0.1094] | +0.0816 [+0.0528, +0.1070] | 0.7232 → 0.8156 | — | experiments.drop_age.models.*.main.rows['P_tierB|+attention'] |
| 35 | Train excludes launch-era months (≤ 2023-06) | 6,325 | +0.0934 [+0.0715, +0.1103] | +0.0839 [+0.0586, +0.1056] | 0.7231 → 0.8165 | — | experiments.no_launch_months.models.*.main.rows['P_tierB|+attention'] |
| 36 | FLAGGED: legacy mixed-taxonomy `subfield` control (v1 baseline leak) | 6,325 | +0.0672 [+0.0465, +0.0836] | +0.0662 [+0.0387, +0.0890] | 0.7481 → 0.8153 | — | experiments.legacy_subfield.models.*.main.rows['P_tierB|+attention'] |
| 37 | FLAGGED: v2 leaky (max_hindex + P_interim) | 6,325 | +0.0897 [+0.0695, +0.1061] | +0.0716 [+0.0434, +0.0957] | 0.7296 → 0.8193 | — | experiments.v2_leaky_replication.models.*.main.rows['P_interim|+attention'] |

Per-quarter AUC of log upvotes alone (test 2025; `per_quarter_upvotes_only_2025`): 2025Q1 0.835 (n=1,318), 2025Q2 0.759 (n=1,749), 2025Q3 0.828 (n=1,406), 2025Q4 0.795 (n=1,852).

### C3. D3 v3 leakage / measurement evidence (`prediction_v3.json → audit`)

| # | Quantity | Value | Source (JSON key) |
|---|---|---|---|
| 38 | Audit verdict | PASS/DISCLOSED — no outcome leakage in headline rows | headline.audit_verdict |
| 39 | subfield_kw: AUC(y \| source=ai_keywords), test | 0.518 (share ai_keywords 0.921) | audit.label_provenance_uniform.evidence.subfield_kw |
| 40 | legacy subfield: AUC(y \| arXiv-fetched flag), test; median upvotes flag=1/0 | 0.659; 33.0 / 8.0 | audit.label_provenance_uniform.evidence.legacy_subfield |
| 41 | Spearman(upvotes, age_months) within 2023 / 2024 / 2025 | -0.501 / +0.126 / +0.031 | audit.attention_is_cumulative_at_collection.spearman_upvotes_age_by_year |
| 42 | Monthly median upvotes, 2023-07 → 2025-12 (range) | 10–21 | audit.attention_is_cumulative_at_collection.monthly_median_upvotes |
| 43 | Share of papers with n_trend_days = 1 | 0.9999 | audit.attention_is_cumulative_at_collection.n_trend_days_share_eq1 |
| 44 | age_months support train / test (no overlap) | 17.3–39.8 / 5.4–17.2 | audit.age_months_support |
| 45 | n_train / n_test / n_mature(K=12) / n_test_backward | 5,019 / 6,325 / 2,717 / 1,602 | meta.* |

### C4. D1 v3 crowding IV — `results/crowding_iv_v3.json` (cluster = cohort_day unless stated)

| # | Quantity | Value | 95% CI / SE | N | Source (JSON key) |
|---|---|---|---|---|---|
| 46 | OLS β (log_citations on log_upvotes), month+dow+subfield_kw FE, Tier-B prestige | 0.5081 | SE day 0.0152 (G=674); SE month 0.0351 (G=34) | 11,230 | ols.honest_FE_tierB |
| 47 | **PRIMARY honest 2SLS** (Z1'_kw, month+dow+subfield_kw FE): 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.4744; π -0.107 (t -5.6), F 30.9; RF -0.051 (t -2.0) | SE 0.2149; AR (0.021, 0.921) | 11,230 | primary_honest |
| 48 | honest 2SLS, cluster = release_month: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.4744; π -0.107 (t -4.7), F 22.0; RF -0.051 (t -1.9) | SE 0.2579; AR (-0.022, 1.098) | 11,230 | primary_honest_month_cluster |
| 49 | honest 2SLS, no prestige: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.4504; π -0.107 (t -5.6), F 31.5; RF -0.048 (t -1.9) | SE 0.2140; AR (-0.003, 0.892) | 11,344 | primary_honest_noprestige |
| 50 | honest FE, other-subfield paper COUNT instrument: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.3167; π -0.178 (t -8.2), F 67.3; RF -0.056 (t -1.9) | SE 0.1667; AR (-0.016, 0.657) | 11,230 | honest_count_instrument |
| 51 | honest FE, own-subfield leave-one-out peer sum (null first stage): 2SLS β; first-stage π (t), KP-F; RF β (t) | β 1.9165; π -0.006 (t -0.5), F 0.3; RF -0.012 (t -0.9) | SE 3.2615; AR (-inf, +inf) | 9,221 | honest_own_subfield_loo |
| 52 | day+subfield_kw FE (v1 design; reflection): 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.7132; π -1.159 (t -20.3), F 413.0; RF -0.827 (t -12.4) | SE 0.0504; AR (0.615, 0.814) | 11,220 | dayfe_Z1p_kw |
| 53 | day FE, singleton cells: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.9166; π -3.772 (t -9.9), F 98.4; RF -3.458 (t -10.6) | SE 0.0702; AR — | 1,907 | dayfe_singleton_cells |
| 54 | day FE, non-singleton cells: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.6189; π -1.168 (t -18.3), F 336.4; RF -0.723 (t -9.5) | SE 0.0570; AR — | 9,221 | dayfe_nonsingleton_cells |
| 55 | day FE, own-subfield LOO peers: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.8390; π -0.052 (t -3.8), F 14.5; RF -0.043 (t -2.8) | SE 0.2756; AR — | 9,221 | dayfe_own_subfield_loo |
| 56 | day FE, COUNT instrument (wrong-sign first stage): 2SLS β; first-stage π (t), KP-F; RF β (t) | β 1.0740; π +0.145 (t +2.0), F 4.2; RF +0.155 (t +1.3) | SE 0.7701; AR — | 11,220 | dayfe_count_instrument |
| 57 | LEGACY v1 primary-P replication (Z1p_othersub, legacy subfield + prestige): 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.7579; π -1.292 (t -18.8), F 354.7; RF -0.980 (t -11.3) | SE 0.0550; AR (0.650, 0.867) | 11,236 | legacy_v1_primary_P |
| 58 | LEGACY instrument under honest FE: 2SLS β; first-stage π (t), KP-F; RF β (t) | β 0.3888; π -0.085 (t -4.8), F 23.1; RF -0.033 (t -1.3) | SE 0.2771; AR (-0.217, 0.973) | 11,246 | legacy_v1_honest_FE |
| 59 | (day × subfield_kw) singleton cells: share of cells / of papers; within-cell variance share of Z1'_kw | 0.442 / 0.179; 0.0000 | — | 11,230 | diagnostics |
| 60 | Within-day corr(Z1'_kw, log upvotes): all / singleton / non-singleton | -0.323 / -0.623 / -0.300 | — | — | diagnostics.reflection_within_day |
| 61 | Balance t-stats on Z1'_kw (honest FE): age_months, log_n_authors, has_github, log1p_max_prior_papers_true, max_years_active | -0.75, -2.75, +1.50, +1.35, +0.01 | — | — | diagnostics.balance_on_Z1p_kw_honest_FE |
| 62 | IV verdict string | see `verdict.string` | — | — | verdict |

### C5. Part I association — `results/association_v3.json` (owned by `scripts/27_association_v3.py`; values copied at generation time — re-run this script if that file changes)

| # | Quantity | Value | 95% CI | n | Source (JSON key) |
|---|---|---|---|---|---|
| 63 | M4 Poisson-QMLE β; IRR per doubling | 1.0992; 2.142 | β SE 0.0608; IRR CI [1.972, 2.327]; block-bootstrap [1.997, 2.318] | 11344 | ladder.M4_tierB_prestige.poisson_qmle; bootstrap_M4 |
| 64 | M4 NB2 β; IRR per doubling | 0.7997; 1.741 | β SE 0.0291; IRR CI [1.673, 1.811]; bootstrap [1.677, 1.817] | 11344 | ladder.M4_tierB_prestige.nb2 |
| 65 | M4 log1p-OLS elasticity | 0.6335 | SE 0.0332; bootstrap [0.5807, 0.7130] | 11344 | ladder.M4_tierB_prestige.ols_log1p |
| 66 | E-values (Poisson / NB2 / OLS-ratio), point (CI bound) | 3.71 (3.36) / 2.88 (2.73) / 2.48 (2.33) | — | — | e_value_M4 |
| 67 | Selection ATT (log pts): naive / PS-match / IPW / CEM | 1.295 / 1.277 / 1.264 / 1.241 | PS [1.185, 1.367]; IPW [1.208, 1.329]; CEM [1.151, 1.294] | — | selection_adjustment |
| 68 | Trending premium vs never-trending background: naive ratio / CEM ratio | 3.43 / 3.00 | CEM [2.73, 3.18] | — | control_comparison.trending_premium |
| 69 | Bottom-tertile-attention trending papers: mean percentile in same-month background (honest replacement for the '91st percentile' line); CEM ratio vs background | 0.704; ×1.64 | [0.696, 0.713] | 3752 | control_comparison.low_attention_vs_background.bottom_tertile_attention_residual |
| 70 | (context) outcome-selected 'under-rated' group percentile — circular, DO NOT CITE as a finding | 0.979 | — | 556 | control_comparison.low_attention_vs_background.underrated_label_OUTCOME_SELECTED |
| 71 | Over / under / neutral counts | 632 / 561 / 10151 | — | 11344 | over_under.counts |
| 72 | MixedLM fixed slope of log upvotes; between-subfield slope SD; slope range | 0.6253; 0.031; [0.594, 0.672] | slope CI [0.5940, 0.6567] | 11344 | hierarchical_mixedlm |

### C6. Sample facts

| # | Quantity | Value | Source |
|---|---|---|---|
| 73 | Analysis sample | 11,344 papers × 65 cols (`analysis_final.csv`, rebuilt with subfield_kw + Tier B) | 24_assemble.py; prediction_v3.json meta.n_all |
| 74 | Release-year split | 2023 = 1,602 / 2024 = 3,417 / 2025 = 6,325 | meta.n_* |
| 75 | subfield_kw levels; source shares | 13; ai_keywords 0.921, title_summary 0.074, none 0.005 | audit.label_provenance_uniform.evidence.subfield_kw |
| 76 | Software | Python 3.12.13, scikit-learn 1.9.0, pandas 3.0.3, pyfixest 0.60.0 | meta |

## D. The report / dashboard / deck MUST NOT claim

1. MUST NOT call the attention→citation link **causal** on the strength of D3 (predictive) or of the association ladder (E-values are sensitivity summaries, not identification).
2. MUST NOT describe upvotes as **'early', 'day-one' or 'peak' attention** — say **'cumulative upvotes at collection (June 2026)'**; n_trend_days = 1 for essentially all papers, so 'peak' == snapshot. The evidence that late accrual is small (flat monthly medians; Spearman(upvotes, age) ≈ 0 within 2024–25) may be cited (C3).
3. MUST NOT present the crowding IV as **'suggestive causal support'**, 'weak-IV-robust causal evidence' or 'triangulation'. It is an **attempted design**: the day-FE first stage is the within-day adding-up identity; the honest AR interval is uninformative; the own-subfield LOO instrument has no first stage (C4).
4. MUST NOT cite the v1 IV numbers (β 0.758, AR (0.649, 0.866), KP-F 355) as results — only as the **legacy replication** row explaining what changed (C4 legacy rows).
5. MUST NOT say Tier B prestige **'can only widen'** the attention lift — it was an empirical question; report the P_tierB / P_interim / P_none bracket as found (C2).
6. MUST NOT use the **'91st percentile'** line for under-rated papers — it conditions on the outcome; use the bottom-tertile-attention comparison (C5) instead.
7. MUST NOT call the v1 controls-only baseline **'strong'** without noting that a single column (log upvotes) has higher AUC than the whole controls-only model and that controls add only a small increment given upvotes (C1).
8. MUST NOT cite the **legacy `subfield`** results as the main specification — the label source was keyed on an upvote-ordered arXiv fetch (AUC(y|flag) in C3); it appears only as a flagged robustness row.
9. MUST NOT headline **P@100 / ΔP@100** — report precision@top-decile and PR-AUC with CIs; P@100 is a 100-draw binomial.
10. MUST NOT report the **HGB** as 'the' headline or the logistic as 'the' headline without the other: both are reported with equal prominence; the pre-specified headline model is the logistic.
11. MUST NOT present the paired bootstrap CIs as covering training-set variability or model selection (they are conditional on the fitted models).
12. MUST NOT present D4 (ACL DiD) as a result, report a 14-day RD estimate or a B2 featured-vs-submitted LATE, or generalise beyond HF Daily Papers (unchanged from v1 gate).
13. MUST NOT call the June-11 v2 analysis 'prior work' — it is Part I of the same project, re-estimated in `27_association_v3.py`.
14. MUST NOT quote a single 'n' for the whole paper: prediction uses 5,019/6,325 (+1,602 backward), the IV honest spec N is in C4, association N in C5.

## E. Cross-result consistency (v3)

- Direction agrees everywhere: OLS β≈0.51 (month+dow+subfield_kw FE), honest 2SLS β 0.47 with AR (0.021, 0.921) (uninformative but same sign), prediction ΔAUC +0.093 (logistic) / +0.078 (HGB).
- Fixing the subfield leak moved the controls-only baseline down (legacy 0.748 → uniform 0.723, logistic) and the lift up (legacy ΔAUC +0.067 → +0.093); the v1 headline (+0.062, HGB, P_interim, legacy subfield) was conservative for the wrong reasons (under-tuned HGB + leaky baseline).
- Prestige branches barely move the lift (C2): leakage-free Tier B prestige does not absorb the attention signal.
- Backward-2023 test (1,602 papers, ≥29 months exposure) gives the largest lift; per-quarter upvotes-only AUCs in 2025 show no decline in the youngest quarter — the signal is not just early visibility.

## F. Regeneration

`23_crowding_cohort_v3.py` → `25_crowding_iv_v3.py`; `26_prediction_v3.py`; (`27_association_v3.py`, other workstream); then `28_results_gate_v3.py` to rebuild this ledger. Figures agent: ROC/PR curves from `results/prediction_v3_scores.csv` (columns `<experiment>__<model>__<branch>__<row>`, `split` = forward_test_2025 / backward_test_2023); AUC ladder from `prediction_v3.json`.