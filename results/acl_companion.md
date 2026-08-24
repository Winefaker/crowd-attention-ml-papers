> SUPERSEDED. This is the first pass, kept so the changes behind the revised numbers can be traced. The current numbers are in the files whose names end in `_v3` and in the ledger `D5_results_gate_v3.md`. The prediction gain is now +0.093 rather than +0.062, and the crowding instrument is reported as an attempted design rather than as suggestive causal support.

# Track A — A clean companion experiment: the early-arXiving neighbour question (documented, deferred)

**Status: DOCUMENTED DEFERRAL** (a planned exit). Rigorously specified, not executed. Reasons under "Why deferred."

## The question
Does posting a preprint *early* — before peer review concludes — causally raise a paper's downstream impact? This is the clean-identification **neighbour** of our headline question (crowd attention → citations): a different treatment (timing of disclosure, not volume of attention) on a different population (peer-reviewed *CL papers, not HF-featured papers). We specify it rigorously and defer execution.

## The natural experiment
The ACL/ARR *anonymity-period* policy prohibited authors from posting or advertising a non-anonymous preprint from one month before the submission deadline through the end of review (adopted Oct 2017, effective for 2018 venues). On ~15 Jan 2024 the ACL **repealed** it, effective the ARR 15 Feb 2024 cycle: authors became free to post non-anonymous preprints at any time. The repeal exogenously **lifts** a constraint on *whether/when* a \*CL paper is arXiv'd early, for a reason unrelated to the individual paper's quality.

Sources: ARR anonymity policy update (aclrollingreview.org/anonymity); ACL 2022 chairs' blog on commitment/anonymity; NAACL 2018 chairs' blog on the preprint policy; ACL Policies admin-wiki. (Exact adoption first-bound deadline needs ACL admin-wiki confirmation — admin-wiki returned a 504 at probe time; does not affect the design.)

## Treatment and control
- Treated = papers in ACL-Anthology venues subject to the policy (ACL/EMNLP/NAACL/EACL/Findings, via ARR).
- Control = a comparable venue never subject to the embargo. **Only CVPR is a clean vision control; NeurIPS/ICLR are LLM/NLP-saturated in 2023-2025 and partly inherit the same shock as the treated group, so they are poor controls for this window.**
- Cohorts span ~2 submission cycles before and after Feb 2024.

## Estimand and equations
Let `E_i = 1[arXiv-v1 date ≤ venue submission deadline]` (early-arXived), `Post_t = 1[cycle ≥ Feb 2024]`, `Treated_v = 1[venue ∈ *CL]`, `δ_t` cycle FE, `X_i` controls (subfield, team size, prior author output).
- **First stage / policy bite (behavioral DiD):** `E_{ivt} = α + γ(Treated_v·Post_t) + λ Treated_v + δ_t + Γ X_i + u` — γ = repeal-induced jump in the early-arXiving rate (expect γ>0).
- **Reduced-form impact (DiD):** `Y_{ivt} = α + θ(Treated_v·Post_t) + λ Treated_v + δ_t + Γ X_i + ε`, `Y` = citations within a fixed K-month window (or within-cycle citation percentile). θ = ITT effect of the regime change on \*CL paper impact.
- **LATE (Wald-DiD):** `β = θ / γ` — effect of early-arXiving on impact for **policy-compliers**.

## Identifying assumption and its test
Parallel trends: absent the repeal, mean impact for \*CL and control venues would evolve in parallel given `δ_t` and `X`; plus exclusion (the repeal affects \*CL impact *only* through early-arXiving). **Test** via an event-study with leads/lags, jointly testing pre-period coefficients = 0, plus a pre-period placebo cutoff. **Our prior is the pre-test would fail**: 2023-2025 is the post-ChatGPT LLM shock — a differential field-level trend hitting \*CL far harder than vision venues, breaking the parallel counterfactual.

**Scope of the failure.** This kills the cross-venue **citation** DiD (the reduced form / θ) — the estimand we actually care about. It does NOT kill the within-\*CL **first stage** (γ): the jump in the early-arXiving *rate* at the repeal, estimable cheaply via an interrupted-time-series / RD-in-time on \*CL alone, is not confounded by the LLM citation shock. So the "policy bite" is identifiable; the *impact* effect is not. A first-stage-only result would confirm the policy changed behavior but say nothing about impact, which is why it is low-value and still requires the new ACL↔arXiv corpus.

## Threats register (Track A)
1. **Differential LLM-era field shock (dominant)** — violates parallel trends; \*CL ≫ CV citation/volume drift in 2023-2025.
2. **Fuzzy, staggered treatment date** — ARR feeds multiple venues; control calendars differ (CVPR ~Nov, NeurIPS ~May); no sharp common cutoff.
3. **Bundled co-treatment** — the repeal also added anonymity incentives (awards, borderline-acceptance priority) → exclusion violation.
4. **Endogenous compliance** — who early-arXivs post-repeal is self-selected; policy is a weak instrument; β identifies only a complier LATE.
5. **Right-censoring** — short citation exposure for 2024-2025 cohorts; mitigate with a fixed K-month outcome + ≥K-month exposure restriction.
6. **Linkage selection** — ACL→arXiv→S2 fuzzy match ~80-90%; missingness (never-arXived/rejected) is non-random.
7. **Prior art (a neighbour, not a full answer)** — arXiv:2306.13891 (Elazar et al., TMLR 2023) estimates early-arXiving's effect on **paper *acceptance*** (not citations) via within-venue matching + a negative-outcome-control design (ICLR 2018-2022): the naive lift of ~9.8-10.0% shrinks to ~2% after adjustment (<4% in 7/9 specs; statistically insignificant in 4/9). It studies a *different outcome* and *predates* the Jan-2024 repeal, so it is a neighbour result, not a full answer to our citation question. Notably, that team — with full policy awareness — deliberately did NOT use the cross-venue policy DiD.

## What execution would require (bounded pipeline, if ever authorized)
1. ACL Anthology XML dump → treated/control venue+title+year (offline, free).
2. Per-cycle venue submission-deadline table (~12 hand-coded rows).
3. One S2 batch pass for `externalIds`+`citationCount`+`publicationDate` (~10⁴ calls, hours; reuse the `scripts/22_*` S2 collector + checkpoint harness).
4. arXiv v1 date from the Kaggle snapshot (offline, as in Spike 1) → compute `E_i`.
5. statsmodels event-study DiD + Wald-DiD + placebo (~150 lines).
Estimate: ~1 day collection + ~½ day analysis.

## Why deferred (honest)
The project headline is already locked and verified (D3 audited prediction ΔAUC +0.062 [+0.047, +0.077]); Track A is an explicitly **non-load-bearing companion** for a *neighbour* question. The build is a genuine new corpus (ACL↔arXiv↔S2) — the overrun risk the plan was designed to avoid. Decisively, the **parallel-trends assumption is not defensible during the 2023-2025 LLM shock**, so execution would most likely yield a confounded estimate requiring walk-back, and the published prior work already provides a more carefully identified *neighbour* result (early-arXiving → **acceptance**, small/null on ICLR 2018-2022 — a different outcome, predating the repeal). Reporting Track A as a rigorously specified design rather than a run is the more honest and higher-quality choice.

## CAN / CANNOT
- **CAN** (if executed with defensible trends): the effect of early-preprinting for \*CL papers under exogenous policy variation.
- **CANNOT**: anything about HF crowd attention or featuring (different treatment, different population). Labelled throughout as the clean-ID **neighbour**, triangulating — not the main claim.
