> SUPERSEDED. This is the first pass, kept so the changes behind the revised numbers can be traced. The current numbers are in the files whose names end in `_v3` and in the ledger `D5_results_gate_v3.md`. The prediction gain is now +0.093 rather than +0.062, and the crowding instrument is reported as an attempted design rather than as suggestive causal support.

# D3 Prediction — Audited ΔAUC (PROJECT HEADLINE)

Generated: 2026-06-27  |  Spec: the pre-specified plan  |  seed=1626

## Headline

**Conservative ΔAUC (HGB, P_interim baseline):**
`ΔAUC = +0.06222  95%CI = [+0.04708, +0.07730]`

- Controls-only AUC: **0.73384**
- +Attention AUC:    **0.79606**
- ΔP@100:           **+0.14000**
- n_train=5,019 / n_test=6,325

**v2 leaky contrast (max_hindex+P_interim, context only):**
Controls=0.75745 → +Attention=0.81209  (ΔAUC=+0.05464)

---

## Per-Baseline Brackets (HGB Classifier)

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC | 95%CI |
|---|---|---|---|---|
| P_interim (FLAGGED anachronistic) | 0.7338 | 0.7961 | +0.0622 | [+0.0471, +0.0773] |
| P_none (no prestige) | 0.7223 | 0.7959 | +0.0736 | [+0.0591, +0.0889] |
| P_tierB (PENDING) | — | — | — | — |
| v2 leaky [CONTEXT ONLY] | 0.7574 | 0.8121 | +0.0546 | (not headline) |

## Per-Baseline Brackets (Logistic)

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC |
|---|---|---|---|
| P_interim | 0.7089 | 0.8054 | +0.0965 |
| P_none | 0.7085 | 0.8053 | +0.0967 |

---

## Robustness Row (K=12 mature test, first_trend_date ≤ 2025-06-01, n=2,365)

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC |
|---|---|---|---|
| P_interim | 0.7473 | 0.8015 | +0.0542 |
| P_none | 0.7351 | 0.7989 | +0.0638 |

---

## Leakage Audit Verdict

**PASS — no leakage found in headline rows**

- Split: temporal (train≤2024, test=2025) — PASS
- Scalers/encoders fit on train only (inside Pipeline) — PASS
- Hyperparameters tuned on train only (GroupKFold by release_month) — PASS
- Target (citation_count) never entered feature matrix — PASS
- upvote_rank_within_month: within-group rank, no cross-split contamination — PASS
- q_i (within-quarter percentile): train/test quarters disjoint — PASS
- Forbidden columns asserted absent from all headline feature matrices — PASS
- max_hindex used ONLY in explicitly-flagged v2_leaky_replication row — PASS
- P_interim baseline flagged provisional (anachronistic productivity) — documented
- P_tierB baseline: PENDING (re-run hook in scripts/26_prediction.py)

---

## Audited ΔAUC vs v2 Leaky +0.05

v2 leaky setup (max_hindex + no forward-in-time split) obtained ΔAUC ≈ +0.05.
Audited ΔAUC (clean temporal split, no h-index): **+0.06222**
(P_interim conservative baseline).

---

## CAN / CANNOT

**CAN:** Determine whether early/peak HF attention (upvotes) adds honest out-of-sample
incremental predictive value for within-quarter citation rank, forward in time
(train≤2024 → test 2025), over a leakage-audited controls-only baseline.

**CANNOT:**
- Claim causation
- Claim "day-one" signal (upvotes = peak, no day-level timestamps)
- Generalize beyond HF Daily Papers
- P_interim baseline is provisional (anachronistic) — flagged until P_tierB available

---

## One-line Verdict

HF attention adds modest incremental predictive value (ΔAUC=+0.0622) over the P_interim controls-only baseline in a clean forward-in-time test.
