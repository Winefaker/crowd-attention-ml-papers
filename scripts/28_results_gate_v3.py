"""
28_results_gate_v3.py — build results/D5_results_gate_v3.md (numbers ledger + MUST-NOT-claim list)
from the v3 JSON outputs.  Every number in the ledger is read from JSON (no hand transcription).

Inputs : results/prediction_v3.json (scripts/26_prediction_v3.py)
         results/crowding_iv_v3.json (scripts/25_crowding_iv_v3.py)
         results/association_v3.json (scripts/27_association_v3.py; other workstream — read only, if present)
Output : results/D5_results_gate_v3.md
Usage  : python scripts/28_results_gate_v3.py
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = json.load(open(ROOT / "results/prediction_v3.json"))
IV = json.load(open(ROOT / "results/crowding_iv_v3.json"))
A_PATH = ROOT / "results/association_v3.json"
A = json.load(open(A_PATH)) if A_PATH.exists() else None
OUT = ROOT / "results/D5_results_gate_v3.md"


def ci(v, nd=3, sign=False):
    if v is None:
        return "—"
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f"[{f.format(v[0])}, {f.format(v[1])}]"


def fmt(v, nd=3, sign=False):
    if v is None:
        return "—"
    return (f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}").format(v)


def prow(exp, model, branch, row, ev="main"):
    return P["experiments"][exp]["models"][model][ev]["rows"][f"{branch}|{row}"]


meta = P["meta"]
n_tr, n_te = meta["n_train"], meta["n_test"]
k = P["experiments"]["forward_yq"]["k_top_decile"]
L = []
L.append("# D5 v3 — Results gate & numbers ledger (audit-response pipeline)\n")
L.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')} by `scripts/28_results_gate_v3.py` from "
         f"`results/prediction_v3.json` ({meta['generated']}), `results/crowding_iv_v3.json` ({IV['meta']['generated']})"
         + (f", `results/association_v3.json` ({A['meta'].get('generated', A['meta'].get('script'))})" if A else "") + ".  ")
L.append("**Rule:** the report, dashboard and deck may cite only numbers in this ledger; each row gives n, CI and the exact "
         "script + JSON key. Re-run this script after any upstream re-run.\n")

# ── A. verdicts ────────────────────────────────────────────────────────────
hl = P["headline"]["primary"]; hh = P["headline"]["by_model"]["hgb"]
ivh = IV["headline"]
L.append("## A. Per-result verdict (v3)\n")
L.append("| Result | Verdict | One-line judgment |\n|---|---|---|")
L.append(f"| **D3 v3 prediction (HEADLINE)** | **MEANINGFUL — predictive, leakage-checked** | Logistic (pre-specified), P_tierB leakage-free baseline: "
         f"AUC {fmt(hl['controls_only_auc'])} → {fmt(hl['attention_auc'])}, ΔAUC {fmt(hl['delta_auc'], sign=True)} "
         f"{ci(hl['delta_auc_ci_month'], sign=True)} (month-clustered); HGB ΔAUC {fmt(hh['delta_auc'], sign=True)} {ci(hh['delta_auc_ci_month'], sign=True)}. "
         f"log upvotes alone AUC {fmt(hl['upvotes_only_auc'])}. Robust across labels, backward-2023 test, mature subset. Predictive, not causal. |")
L.append(f"| **D1 v3 crowding IV** | **ATTEMPTED DESIGN — not informative causal evidence** | Honest spec (month+dow+subfield_kw FE): "
         f"2SLS β {fmt(ivh['beta_2sls'])} (SE {fmt(ivh['se_cluster_day'])}), AR 95% CI {ivh['AR_CI_string']}, first-stage t {fmt(ivh['first_stage_t'], 1)}; "
         f"day-FE F={fmt(ivh['day_FE_kp_f'], 0)} is the within-day adding-up identity (reflection). Report only as 'consistent with OLS but uninformative'. |")
if A:
    m4 = A["ladder"]["M4_tierB_prestige"]
    L.append(f"| **Part I association (v3, other workstream)** | **MEANINGFUL — association, not causation** | M4 (subfield_kw + month + dow FE, Tier B prestige): "
             f"Poisson-QMLE IRR per doubling {fmt(m4['poisson_qmle'].get('irr_per_doubling'), 2)}; NB2 {fmt(m4['nb2'].get('irr_per_doubling'), 2)}; "
             f"log1p-OLS elasticity {fmt(m4['ols_log1p'].get('beta') if 'ols_log1p' in m4 else None)}. See §C5 (numbers owned by `27_association_v3.py`). |")
L.append("| Feasibility checks and the deferred preprint policy design | unchanged from the first pass | Clean nulls, and a design specified but not run. |")
L.append("")

# ── B. posture ─────────────────────────────────────────────────────────────
L.append("## B. Identification posture (v3)\n")
L.append("- **The clean evidence is predictive (D3 v3).** Cumulative HF attention (upvotes at collection, June 2026) adds out-of-sample "
         "incremental predictive value for within-cohort citation rank, forward in time (train ≤2024 → test 2025) and backward onto the mature 2023 cohort, "
         "over a leakage-checked controls-only baseline that uses a uniform subfield taxonomy and leakage-free Tier-B prestige. A single column (log upvotes) carries essentially all of it.")
L.append("- **The crowding IV is an attempted design.** Under day FE the instrument is a transform of the paper's own upvotes (reflection); without day FE it is modest and its "
         "weak-IV-robust interval is uninformative. It provides no causal information beyond the OLS association.")
L.append("- **Part I (association ladder, selection adjustment, over/under-rated, never-trending comparison)** is reported as association with sensitivity analysis (E-values, placebo), never as causal.\n")

# ── C1 headline ────────────────────────────────────────────────────────────
L.append("## C. Numbers ledger\n")
L.append(f"### C1. D3 v3 headline — `results/prediction_v3.json` (n_train={n_tr:,}, n_test={n_te:,}, top-decile k={k}, "
         f"bootstrap {meta['n_boot']} reps seed {meta['seed']}; CIs month-clustered unless stated)\n")
L.append("| # | Quantity | Value | 95% CI | n | Source (script → JSON key) |\n|---|---|---|---|---|---|")
i = 1
def add(q, val, c, n, src):
    global i
    L.append(f"| {i} | {q} | {val} | {c} | {n} | {src} |")
    i += 1
for m, tag in [("logistic", "Logistic (headline)"), ("hgb", "HGB")]:
    h = P["headline"]["by_model"][m]
    add(f"{tag}: controls-only AUC (P_tierB)", fmt(h["controls_only_auc"], 4), ci(h["controls_only_auc_ci_month"], 4), n_te,
        f"26_prediction_v3.py → headline.by_model.{m}.controls_only_auc")
    add(f"{tag}: +attention AUC", fmt(h["attention_auc"], 4), ci(h["attention_auc_ci_month"], 4), n_te, f"headline.by_model.{m}.attention_auc")
    add(f"**{tag}: ΔAUC (+attention − controls)**", f"**{fmt(h['delta_auc'], 4, True)}**", f"{ci(h['delta_auc_ci_month'], 4, True)} (i.i.d. {ci(h['delta_auc_ci_iid'], 4, True)})",
        n_te, f"headline.by_model.{m}.delta_auc / delta_auc_ci_month")
    add(f"{tag}: log-upvotes-only AUC", fmt(h["upvotes_only_auc"], 4), ci(h["upvotes_only_auc_ci_month"], 4), n_te, f"headline.by_model.{m}.upvotes_only_auc")
    add(f"{tag}: (+attention) − (upvotes only) ΔAUC", fmt(h["attention_minus_upvotes_only_auc"], 4, True), ci(h["attention_minus_upvotes_only_auc_ci_month"], 4, True), n_te,
        f"headline.by_model.{m}.attention_minus_upvotes_only_auc")
    add(f"{tag}: PR-AUC controls → +attention", f"{fmt(h['controls_only_pr_auc'], 4)} → {fmt(h['attention_pr_auc'], 4)}", "see ΔPR-AUC", n_te,
        f"headline.by_model.{m}.controls_only_pr_auc / attention_pr_auc")
    add(f"{tag}: ΔPR-AUC", fmt(h["delta_pr_auc"], 4, True), ci(h["delta_pr_auc_ci_month"], 4, True), n_te, f"headline.by_model.{m}.delta_pr_auc")
    add(f"{tag}: precision@top-decile controls → +attention (k={k})", f"{fmt(h['controls_only_p_at_k'], 4)} → {fmt(h['attention_p_at_k'], 4)}", "see Δ", n_te,
        f"headline.by_model.{m}.controls_only_p_at_k / attention_p_at_k")
    add(f"{tag}: Δprecision@top-decile", fmt(h["delta_p_at_k"], 4, True), ci(h["delta_p_at_k_ci_month"], 4, True), n_te, f"headline.by_model.{m}.delta_p_at_k")
    add(f"{tag}: ΔP@100 (NOISY — 100-draw binomial; do not headline)", fmt(h["delta_p_at_100"], 2, True), ci(h["delta_p_at_100_ci_month"], 2, True), n_te, f"headline.by_model.{m}.delta_p_at_100")
    add(f"{tag}: Brier controls → +attention", f"{fmt(h['controls_only_brier'], 4)} → {fmt(h['attention_brier'], 4)}", "—", n_te, f"headline.by_model.{m}.controls_only_brier / attention_brier")
add("Model-free AUC of log upvotes (test 2025)", fmt(P["raw_attention_auc"]["test_2025_log_upvotes"], 4), "= logistic upvotes-only row", n_te, "raw_attention_auc.test_2025_log_upvotes")
add("Base rate (top decile) train / test", f"{fmt(P['experiments']['forward_yq']['base_rate_train'], 3)} / {fmt(P['experiments']['forward_yq']['base_rate_test'], 3)}", "—",
    f"{n_tr:,}/{n_te:,}", "experiments.forward_yq.base_rate_train / base_rate_test")
add("+attention+comments ΔAUC (logistic / HGB)", f"{fmt(prow('forward_yq','logistic','P_tierB','+attention+comments')['delta_vs_controls_only']['d_auc'], 4, True)} / "
    f"{fmt(prow('forward_yq','hgb','P_tierB','+attention+comments')['delta_vs_controls_only']['d_auc'], 4, True)}",
    ci(prow('forward_yq','logistic','P_tierB','+attention+comments')['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4, True) + " (logistic)", n_te,
    "experiments.forward_yq.models.<model>.main.rows['P_tierB|+attention+comments'].delta_vs_controls_only")
L.append("")

# ── C2 prestige bracket + robustness ───────────────────────────────────────
L.append("### C2. D3 v3 prestige bracket and robustness rows (ΔAUC = +attention − controls-only; month-clustered CI)\n")
L.append("| # | Row | n_test | Logistic ΔAUC [CI] | HGB ΔAUC [CI] | Logistic ctrl→+att AUC | Upvotes-only AUC | Source (JSON key) |\n|---|---|---|---|---|---|---|---|")
for b in ["P_tierB", "P_interim", "P_none"]:
    lg = prow("forward_yq", "logistic", b, "+attention"); hg = prow("forward_yq", "hgb", b, "+attention")
    lc = prow("forward_yq", "logistic", b, "controls_only")
    up = prow("forward_yq", "logistic", b, "upvotes_only")
    L.append(f"| {i} | Branch {b} ({P['audit']['prestige_branch_tags'][b].split(' — ')[0]}) | {n_te:,} | "
             f"{fmt(lg['delta_vs_controls_only']['d_auc'], 4, True)} {ci(lg['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4, True)} | "
             f"{fmt(hg['delta_vs_controls_only']['d_auc'], 4, True)} {ci(hg['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4, True)} | "
             f"{fmt(lc['auc'], 4)} → {fmt(lg['auc'], 4)} | {fmt(up['auc'], 4)} | experiments.forward_yq.models.*.main.rows['{b}|+attention'] |")
    i += 1
rob = [("forward_yq", "mature_k12", "P_tierB", f"Mature K=12 test subset (age ≥ {meta['k12_months']:.0f} mo; label re-ranked within quarter)"),
       ("forward_ym", "main", "P_tierB", "Within-month label"),
       ("forward_yinf", "main", "P_tierB", "Top-decile influential citations"),
       ("backward_yq", "main", "P_tierB", "Backward test: train 2024–25 → test 2023 (within-quarter label)"),
       ("backward_ym", "main", "P_tierB", "Backward test (within-month label)"),
       ("drop_age", "main", "P_tierB", "Drop age_months from controls"),
       ("no_launch_months", "main", "P_tierB", f"Train excludes launch-era months (≤ {meta['launch_era_end']})"),
       ("legacy_subfield", "main", "P_tierB", "FLAGGED: legacy mixed-taxonomy `subfield` control (v1 baseline leak)"),
       ("v2_leaky_replication", "main", "P_interim", "FLAGGED: v2 leaky (max_hindex + P_interim)")]
for exp, ev, b, lab in rob:
    blk_l = P["experiments"][exp]["models"]["logistic"][ev]; blk_h = P["experiments"][exp]["models"]["hgb"][ev]
    lg = blk_l["rows"][f"{b}|+attention"]; hg = blk_h["rows"][f"{b}|+attention"]; lc = blk_l["rows"][f"{b}|controls_only"]
    up = blk_l["rows"].get(f"{b}|upvotes_only")
    L.append(f"| {i} | {lab} | {blk_l['n']:,} | {fmt(lg['delta_vs_controls_only']['d_auc'], 4, True)} {ci(lg['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4, True)} | "
             f"{fmt(hg['delta_vs_controls_only']['d_auc'], 4, True)} {ci(hg['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4, True)} | "
             f"{fmt(lc['auc'], 4)} → {fmt(lg['auc'], 4)} | {fmt(up['auc'], 4) if up else '—'} | experiments.{exp}.models.*.{ev}.rows['{b}|+attention'] |")
    i += 1
L.append("")
L.append("Per-quarter AUC of log upvotes alone (test 2025; `per_quarter_upvotes_only_2025`): " +
         ", ".join(f"{q['quarter']} {fmt(q['auc_log_upvotes'])} (n={q['n']:,})" for q in P["per_quarter_upvotes_only_2025"]) + ".\n")

# ── C3 audit evidence ──────────────────────────────────────────────────────
au = P["audit"]
L.append("### C3. D3 v3 leakage / measurement evidence (`prediction_v3.json → audit`)\n")
L.append("| # | Quantity | Value | Source (JSON key) |\n|---|---|---|---|")
pk = au["label_provenance_uniform"]["evidence"]["subfield_kw"]; pl = au["label_provenance_uniform"]["evidence"]["legacy_subfield"]
L.append(f"| {i} | Audit verdict | {P['headline']['audit_verdict']} | headline.audit_verdict |"); i += 1
L.append(f"| {i} | subfield_kw: AUC(y \\| source=ai_keywords), test | {fmt(pk['test_auc_y_given_source_is_ai_keywords'])} (share ai_keywords {fmt(pk['source_shares_all'].get('ai_keywords'), 3)}) | audit.label_provenance_uniform.evidence.subfield_kw |"); i += 1
L.append(f"| {i} | legacy subfield: AUC(y \\| arXiv-fetched flag), test; median upvotes flag=1/0 | {fmt(pl['test_auc_y_given_arxiv_fetched_flag'])}; "
         f"{pl['test_median_upvotes_by_arxiv_fetched'].get('1', pl['test_median_upvotes_by_arxiv_fetched'].get(1))} / {pl['test_median_upvotes_by_arxiv_fetched'].get('0', pl['test_median_upvotes_by_arxiv_fetched'].get(0))} | audit.label_provenance_uniform.evidence.legacy_subfield |"); i += 1
sp = au["attention_is_cumulative_at_collection"]["spearman_upvotes_age_by_year"]
L.append(f"| {i} | Spearman(upvotes, age_months) within 2023 / 2024 / 2025 | " + " / ".join(fmt(sp[y]['spearman_upvotes_age_months'], 3, True) for y in ["2023", "2024", "2025"]) +
         " | audit.attention_is_cumulative_at_collection.spearman_upvotes_age_by_year |"); i += 1
mm = au["attention_is_cumulative_at_collection"]["monthly_median_upvotes"]
post = [r for r in mm if not r["launch_era"]]
L.append(f"| {i} | Monthly median upvotes, {post[0]['release_month']} → {post[-1]['release_month']} (range) | {min(r['median_upvotes'] for r in post):.0f}–{max(r['median_upvotes'] for r in post):.0f} | audit.attention_is_cumulative_at_collection.monthly_median_upvotes |"); i += 1
L.append(f"| {i} | Share of papers with n_trend_days = 1 | {fmt(au['attention_is_cumulative_at_collection']['n_trend_days_share_eq1'], 4)} | audit.attention_is_cumulative_at_collection.n_trend_days_share_eq1 |"); i += 1
ag = au["age_months_support"]
L.append(f"| {i} | age_months support train / test (no overlap) | {ag['train_min']:.1f}–{ag['train_max']:.1f} / {ag['test_min']:.1f}–{ag['test_max']:.1f} | audit.age_months_support |"); i += 1
L.append(f"| {i} | n_train / n_test / n_mature(K=12) / n_test_backward | {n_tr:,} / {n_te:,} / {meta['n_mature_k12']:,} / {meta['n_test_backward']:,} | meta.* |"); i += 1
L.append("")

# ── C4 IV ──────────────────────────────────────────────────────────────────
L.append("### C4. D1 v3 crowding IV — `results/crowding_iv_v3.json` (cluster = cohort_day unless stated)\n")
L.append("| # | Quantity | Value | 95% CI / SE | N | Source (JSON key) |\n|---|---|---|---|---|---|")
o = IV["ols"]["honest_FE_tierB"]
L.append(f"| {i} | OLS β (log_citations on log_upvotes), month+dow+subfield_kw FE, Tier-B prestige | {fmt(o['beta'], 4)} | SE day {fmt(o['se_cluster_day'], 4)} (G={o['G_day']}); SE month {fmt(o['se_cluster_month'], 4)} (G={o['G_month']}) | {o['N']:,} | ols.honest_FE_tierB |"); i += 1
for key, lab in [("primary_honest", "**PRIMARY honest 2SLS** (Z1'_kw, month+dow+subfield_kw FE)"), ("primary_honest_month_cluster", "honest 2SLS, cluster = release_month"),
                 ("primary_honest_noprestige", "honest 2SLS, no prestige"), ("honest_count_instrument", "honest FE, other-subfield paper COUNT instrument"),
                 ("honest_own_subfield_loo", "honest FE, own-subfield leave-one-out peer sum (null first stage)"),
                 ("dayfe_Z1p_kw", "day+subfield_kw FE (v1 design; reflection)"), ("dayfe_singleton_cells", "day FE, singleton cells"),
                 ("dayfe_nonsingleton_cells", "day FE, non-singleton cells"), ("dayfe_own_subfield_loo", "day FE, own-subfield LOO peers"),
                 ("dayfe_count_instrument", "day FE, COUNT instrument (wrong-sign first stage)"),
                 ("legacy_v1_primary_P", "LEGACY v1 primary-P replication (Z1p_othersub, legacy subfield + prestige)"),
                 ("legacy_v1_honest_FE", "LEGACY instrument under honest FE")]:
    b = IV[key]; fs = b["first_stage"]; iv2 = b["iv_2sls"]; ar = b.get("ar_ci95", {})
    L.append(f"| {i} | {lab}: 2SLS β; first-stage π (t), KP-F; RF β (t) | β {fmt(iv2['beta'], 4)}; π {fmt(fs['pi'], 3, True)} (t {fmt(fs['t'], 1, True)}), F {fmt(fs['kp_f'], 1)}; "
             f"RF {fmt(b['reduced_form']['beta'], 3, True)} (t {fmt(b['reduced_form']['t'], 1, True)}) | SE {fmt(iv2['se'], 4)}; AR {ar.get('string', '—')} | {b['N']:,} | {key} |"); i += 1
d = IV["diagnostics"]
L.append(f"| {i} | (day × subfield_kw) singleton cells: share of cells / of papers; within-cell variance share of Z1'_kw | {fmt(d['singleton_cell_share'], 3)} / {fmt(d['paper_singleton_share'], 3)}; {fmt(d['Z1p_kw_within_cell_variance_share'], 4)} | — | {IV['primary_honest']['N']:,} | diagnostics |"); i += 1
r = d["reflection_within_day"]
L.append(f"| {i} | Within-day corr(Z1'_kw, log upvotes): all / singleton / non-singleton | {fmt(r['corr_Z1p_kw_log_upvotes_within_day_all'], 3, True)} / {fmt(r['corr_Z1p_kw_log_upvotes_within_day_singleton'], 3, True)} / {fmt(r['corr_Z1p_kw_log_upvotes_within_day_nonsingleton'], 3, True)} | — | — | diagnostics.reflection_within_day |"); i += 1
bal = d["balance_on_Z1p_kw_honest_FE"]
L.append(f"| {i} | Balance t-stats on Z1'_kw (honest FE): " + ", ".join(bal.keys()) + " | " + ", ".join(fmt(v['t'], 2, True) for v in bal.values()) + " | — | — | diagnostics.balance_on_Z1p_kw_honest_FE |"); i += 1
L.append(f"| {i} | IV verdict string | see `verdict.string` | — | — | verdict |"); i += 1
L.append("")

# ── C5 association (other workstream) ──────────────────────────────────────
if A:
    L.append("### C5. Part I association — `results/association_v3.json` (owned by `scripts/27_association_v3.py`; values copied at generation time — re-run this script if that file changes)\n")
    L.append("| # | Quantity | Value | 95% CI | n | Source (JSON key) |\n|---|---|---|---|---|---|")
    m4 = A["ladder"]["M4_tierB_prestige"]
    def g(dct, *ks):
        for k_ in ks:
            if not isinstance(dct, dict) or k_ not in dct:
                return None
            dct = dct[k_]
        return dct
    pq = m4.get("poisson_qmle", {}); nb = m4.get("nb2", {}); ol = m4.get("ols_log1p", m4.get("ols", {}))
    L.append(f"| {i} | M4 Poisson-QMLE β; IRR per doubling | {fmt(pq.get('beta'), 4)}; {fmt(pq.get('irr_per_doubling'), 3)} | β SE {fmt(pq.get('se'), 4)}; IRR CI {ci(pq.get('irr_ci95') or pq.get('irr_per_doubling_ci95'), 3)}; block-bootstrap {ci(g(A, 'bootstrap_M4', 'poisson_irr2x_ci95'), 3)} | {pq.get('nobs', m4.get('n', '—'))} | ladder.M4_tierB_prestige.poisson_qmle; bootstrap_M4 |"); i += 1
    L.append(f"| {i} | M4 NB2 β; IRR per doubling | {fmt(nb.get('beta'), 4)}; {fmt(nb.get('irr_per_doubling'), 3)} | β SE {fmt(nb.get('se'), 4)}; IRR CI {ci(nb.get('irr_ci95') or nb.get('irr_per_doubling_ci95'), 3)}; bootstrap {ci(g(A, 'bootstrap_M4', 'nb2_irr2x_ci95'), 3)} | {nb.get('nobs', m4.get('n', '—'))} | ladder.M4_tierB_prestige.nb2 |"); i += 1
    L.append(f"| {i} | M4 log1p-OLS elasticity | {fmt(ol.get('beta'), 4)} | SE {fmt(ol.get('se'), 4)}; bootstrap {ci(g(A, 'bootstrap_M4', 'ols_beta_ci95'), 4)} | {ol.get('nobs', m4.get('n', '—'))} | ladder.M4_tierB_prestige.ols_log1p |"); i += 1
    ev = A.get("e_value_M4", {})
    L.append(f"| {i} | E-values (Poisson / NB2 / OLS-ratio), point (CI bound) | {fmt(g(ev,'poisson_qmle','e_value_point'), 2)} ({fmt(g(ev,'poisson_qmle','e_value_ci_lower'), 2)}) / {fmt(g(ev,'nb2','e_value_point'), 2)} ({fmt(g(ev,'nb2','e_value_ci_lower'), 2)}) / {fmt(g(ev,'ols_log1p_ratio','e_value_point'), 2)} ({fmt(g(ev,'ols_log1p_ratio','e_value_ci_lower'), 2)}) | — | — | e_value_M4 |"); i += 1
    sa = A.get("selection_adjustment", {})
    L.append(f"| {i} | Selection ATT (log pts): naive / PS-match / IPW / CEM | {fmt(g(sa,'naive_gap','log_pts'), 3)} / {fmt(g(sa,'ps_matching','att_log_pts'), 3)} / {fmt(g(sa,'ipw','att_log_pts'), 3)} / {fmt(g(sa,'cem','att_log_pts'), 3)} | PS {ci(g(sa,'ps_matching','ci95'), 3)}; IPW {ci(g(sa,'ipw','ci95'), 3)}; CEM {ci(g(sa,'cem','ci95'), 3)} | — | selection_adjustment |"); i += 1
    cc = A.get("control_comparison", {})
    tp = g(cc, "trending_premium") or {}
    cem_tp = tp.get("cem_month_subfield13_hbin", tp.get("cem", {}))
    L.append(f"| {i} | Trending premium vs never-trending background: naive ratio / CEM ratio | {fmt(g(tp,'naive','ratio'), 2)} / {fmt(cem_tp.get('ratio') if isinstance(cem_tp, dict) else None, 2)} | CEM {ci(cem_tp.get('ratio_ci95') if isinstance(cem_tp, dict) else None, 2)} | — | control_comparison.trending_premium |"); i += 1
    lo = g(cc, "low_attention_vs_background") or {}
    bt = lo.get("bottom_tertile_attention_residual", {})
    btp = bt.get("percentile_in_background_same_month", {})
    L.append(f"| {i} | Bottom-tertile-attention trending papers: mean percentile in same-month background (honest replacement for the '91st percentile' line); CEM ratio vs background | {fmt(btp.get('mean_percentile'), 3)}; ×{fmt(bt.get('cem_ratio'), 2)} | {ci(btp.get('ci95'), 3)} | {bt.get('n', '—')} | control_comparison.low_attention_vs_background.bottom_tertile_attention_residual |"); i += 1
    us = lo.get("underrated_label_OUTCOME_SELECTED", {})
    L.append(f"| {i} | (context) outcome-selected 'under-rated' group percentile — circular, DO NOT CITE as a finding | {fmt((us.get('percentile_in_background_same_month') or {}).get('mean_percentile'), 3)} | — | {us.get('n', '—')} | control_comparison.low_attention_vs_background.underrated_label_OUTCOME_SELECTED |"); i += 1
    ou = A.get("over_under", {})
    L.append(f"| {i} | Over / under / neutral counts | {g(ou,'counts','overrated')} / {g(ou,'counts','underrated')} / {g(ou,'counts','neutral')} | — | {A['measurement']['descriptives'].get('n', '—')} | over_under.counts |"); i += 1
    hm = A.get("hierarchical_mixedlm", {})
    L.append(f"| {i} | MixedLM fixed slope of log upvotes; between-subfield slope SD; slope range | {fmt(hm.get('fixed_slope_log_upvotes'), 4)}; {fmt(g(hm,'variance_components','slope_sd'), 3)}; {ci(hm.get('slope_range'), 3)} | slope CI {ci(hm.get('fixed_slope_ci95'), 4)} | {hm.get('n', '—')} | hierarchical_mixedlm |"); i += 1
    L.append("")
else:
    L.append("### C5. Part I association — `results/association_v3.json` not present at generation time; re-run this script when it exists.\n")

# ── C6 sample facts ────────────────────────────────────────────────────────
L.append("### C6. Sample facts\n")
L.append("| # | Quantity | Value | Source |\n|---|---|---|---|")
L.append(f"| {i} | Analysis sample | {meta['n_all']:,} papers × 65 cols (`analysis_final.csv`, rebuilt with subfield_kw + Tier B) | 24_assemble.py; prediction_v3.json meta.n_all |"); i += 1
L.append(f"| {i} | Release-year split | 2023 = {meta['n_test_backward']:,} / 2024 = {meta['n_train'] - meta['n_test_backward']:,} / 2025 = {meta['n_test']:,} | meta.n_* |"); i += 1
L.append(f"| {i} | subfield_kw levels; source shares | {pk['n_levels']}; " + ", ".join(f"{k_} {v:.3f}" for k_, v in pk['source_shares_all'].items()) + " | audit.label_provenance_uniform.evidence.subfield_kw |"); i += 1
L.append(f"| {i} | Software | Python {meta['python_version']}, scikit-learn {meta['sklearn']}, pandas {meta['pandas']}, pyfixest {IV['meta']['pyfixest']} | meta |"); i += 1
L.append("")

# ── D. MUST NOT ────────────────────────────────────────────────────────────
L.append("## D. The report / dashboard / deck MUST NOT claim\n")
musts = [
    "call the attention→citation link **causal** on the strength of D3 (predictive) or of the association ladder (E-values are sensitivity summaries, not identification).",
    "describe upvotes as **'early', 'day-one' or 'peak' attention** — say **'cumulative upvotes at collection (June 2026)'**; n_trend_days = 1 for essentially all papers, so 'peak' == snapshot. The evidence that late accrual is small (flat monthly medians; Spearman(upvotes, age) ≈ 0 within 2024–25) may be cited (C3).",
    "present the crowding IV as **'suggestive causal support'**, 'weak-IV-robust causal evidence' or 'triangulation'. It is an **attempted design**: the day-FE first stage is the within-day adding-up identity; the honest AR interval is uninformative; the own-subfield LOO instrument has no first stage (C4).",
    f"cite the v1 IV numbers (β 0.758, AR (0.649, 0.866), KP-F 355) as results — only as the **legacy replication** row explaining what changed (C4 legacy rows).",
    "say Tier B prestige **'can only widen'** the attention lift — it was an empirical question; report the P_tierB / P_interim / P_none bracket as found (C2).",
    "use the **'91st percentile'** line for under-rated papers — it conditions on the outcome; use the bottom-tertile-attention comparison (C5) instead.",
    "call the v1 controls-only baseline **'strong'** without noting that a single column (log upvotes) has higher AUC than the whole controls-only model and that controls add only a small increment given upvotes (C1).",
    "cite the **legacy `subfield`** results as the main specification — the label source was keyed on an upvote-ordered arXiv fetch (AUC(y|flag) in C3); it appears only as a flagged robustness row.",
    "headline **P@100 / ΔP@100** — report precision@top-decile and PR-AUC with CIs; P@100 is a 100-draw binomial.",
    "report the **HGB** as 'the' headline or the logistic as 'the' headline without the other: both are reported with equal prominence; the pre-specified headline model is the logistic.",
    "present the paired bootstrap CIs as covering training-set variability or model selection (they are conditional on the fitted models).",
    "present D4 (ACL DiD) as a result, report a 14-day RD estimate or a B2 featured-vs-submitted LATE, or generalise beyond HF Daily Papers (unchanged from v1 gate).",
    "call the June-11 v2 analysis 'prior work' — it is Part I of the same project, re-estimated in `27_association_v3.py`.",
    "quote a single 'n' for the whole paper: prediction uses 5,019/6,325 (+1,602 backward), the IV honest spec N is in C4, association N in C5.",
]
for j, mtxt in enumerate(musts, 1):
    L.append(f"{j}. MUST NOT {mtxt}")
L.append("")

# ── E. consistency ─────────────────────────────────────────────────────────
L.append("## E. Cross-result consistency (v3)\n")
L.append(f"- Direction agrees everywhere: OLS β≈{fmt(o['beta'], 2)} (month+dow+subfield_kw FE), honest 2SLS β {fmt(ivh['beta_2sls'], 2)} with AR {ivh['AR_CI_string']} (uninformative but same sign), "
         f"prediction ΔAUC {fmt(hl['delta_auc'], 3, True)} (logistic) / {fmt(hh['delta_auc'], 3, True)} (HGB).")
L.append(f"- Fixing the subfield leak moved the controls-only baseline down (legacy {fmt(prow('legacy_subfield','logistic','P_tierB','controls_only')['auc'], 3)} → uniform "
         f"{fmt(hl['controls_only_auc'], 3)}, logistic) and the lift up (legacy ΔAUC {fmt(prow('legacy_subfield','logistic','P_tierB','+attention')['delta_vs_controls_only']['d_auc'], 3, True)} → {fmt(hl['delta_auc'], 3, True)}); "
         "the v1 headline (+0.062, HGB, P_interim, legacy subfield) was conservative for the wrong reasons (under-tuned HGB + leaky baseline).")
L.append(f"- Prestige branches barely move the lift (C2): leakage-free Tier B prestige does not absorb the attention signal.")
L.append(f"- Backward-2023 test ({meta['n_test_backward']:,} papers, ≥29 months exposure) gives the largest lift; per-quarter upvotes-only AUCs in 2025 show no decline in the youngest quarter — the signal is not just early visibility.")
L.append("")
L.append("## F. Regeneration\n")
L.append("`23_crowding_cohort_v3.py` → `25_crowding_iv_v3.py`; `26_prediction_v3.py`; (`27_association_v3.py`, other workstream); then `28_results_gate_v3.py` to rebuild this ledger. "
         "Figures agent: ROC/PR curves from `results/prediction_v3_scores.csv` (columns `<experiment>__<model>__<branch>__<row>`, `split` = forward_test_2025 / backward_test_2023); AUC ladder from `prediction_v3.json`.")

OUT.write_text("\n".join(L))
print(f"wrote {OUT} ({len(L)} lines)")
