#!/usr/bin/env python
"""Build the project dashboard as ONE self contained HTML file.

Reads only rebuilt results (results/*_v3.json, prediction_v3_scores.csv,
data/processed/overunder_v3.csv, analysis_final.csv, and the read-only control
sample under Project/data/raw) and writes dashboard.html.

Every number shown in the page is either taken verbatim from
results/D5_results_gate_v3.md (via the JSON keys named there) or computed
here from the embedded data (curves, CCDFs, densities). No external requests:
vanilla JS + inline SVG/canvas, no fonts, no images.

Run:  .venv/bin/python dashboard/build_dashboard.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
AUG = BASE
REPRO = AUG
OUT_HTML = AUG / "index.html"
HERE = Path(__file__).resolve().parent

F_OU = REPRO / "data/processed/overunder_v3.csv"
F_AN = REPRO / "data/processed/analysis_final.csv"
F_ASSOC = REPRO / "results/association_v3.json"
F_PRED = REPRO / "results/prediction_v3.json"
F_IV = REPRO / "results/crowding_iv_v3.json"
F_SCORES = REPRO / "results/prediction_v3_scores.csv"
F_CTRL_META = BASE / "data/raw/arxiv_control.csv"      # read only
F_CTRL_S2 = BASE / "data/raw/arxiv_control_s2.csv"     # read only
SNAPSHOT = pd.Timestamp("2026-06-11")  # citation snapshot used for control ages (not shown in text)


def r(x, d=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), d)


# ----------------------------------------------------------------------------
# 1. Papers payload
# ----------------------------------------------------------------------------
ou = pd.read_csv(F_OU, dtype={"arxiv_id_clean": str})
assert len(ou) == 11344, len(ou)
SUBFIELDS = sorted(ou.subfield_kw.unique().tolist())
sf_idx = {s: i for i, s in enumerate(SUBFIELDS)}
LBL = {"underrated": 0, "neutral": 1, "overrated": 2}


def clean_title(t):
    t = "" if pd.isna(t) else str(t)
    return " ".join(t.split())


rows = []
for rec in ou.itertuples(index=False):
    rows.append([
        rec.arxiv_id_clean,
        clean_title(rec.title),
        sf_idx[rec.subfield_kw],
        str(rec.release_month),
        int(rec.upvotes),
        int(rec.citation_count),
        r(rec.age_months, 1),
        r(rec.attention_resid_pct, 4),
        r(rec.impact_resid_pct, 4),
        LBL[rec.label],
        None if pd.isna(rec.max_hindex) else int(rec.max_hindex),
        None if pd.isna(rec.max_prior_papers_true_w99) else int(round(rec.max_prior_papers_true_w99)),
    ])
COUNTS = ou.label.value_counts().to_dict()
YEARS = sorted(ou.release_year.unique().tolist())
n_by_year = ou.release_year.value_counts().sort_index().to_dict()

# ----------------------------------------------------------------------------
# 2. Evidence arrays
# ----------------------------------------------------------------------------
A = json.load(open(F_ASSOC))
P = json.load(open(F_PRED))
IV = json.load(open(F_IV))

# (a) dose response deciles
dr = [{"d": d["decile"], "n": d["n"], "up": d["median_upvotes"], "umin": d["upvotes_min"], "umax": d["upvotes_max"],
       "raw": d["mean_log1p_cites"], "raw_ci": d["ci95"], "adj": d["mean_log1p_cites_adjusted"],
       "adj_ci": d["ci95_adjusted"], "med": d["median_cites"], "top": d["share_top_decile_cites_in_quarter"]}
      for d in A["dose_response_deciles"]["rows"]]
spearman_raw = A["dose_response_deciles"]["spearman_upvotes_cites"]

# (b) spec ladder
LADDER_LABELS = {
    "M0_raw": "Raw",
    "M1_age": "+ log age",
    "M2_subfield": "+ subfield FE",
    "M3_month_dow": "+ release month and day of week FE",
    "M4_tierB_prestige": "+ leakage free prestige (main)",
    "M5_plus_hindex_LEAKY": "+ h-index measured today (leaky)",
    "M6_plus_text": "+ text and format controls",
}
ladder = []
for k, lab in LADDER_LABELS.items():
    v = A["ladder"][k]
    ladder.append({"k": k, "label": lab, "main": k == "M4_tierB_prestige",
                   "poisson": [v["poisson_qmle"]["irr_per_doubling"]] + v["poisson_qmle"]["irr_per_doubling_ci95"],
                   "nb2": [v["nb2"]["irr_per_doubling"]] + v["nb2"]["irr_per_doubling_ci95"],
                   "ols": [v["ols_log1p"]["ratio_1p_cites_per_doubling"]] + v["ols_log1p"]["ratio_1p_cites_per_doubling_ci95"]})
M4 = A["ladder"]["M4_tierB_prestige"]
BOOT = A["bootstrap_M4"]
EVAL = A["e_value_M4"]

# (c) ROC + PR curves from per test row scores (forward 2025)
sc = pd.read_csv(F_SCORES, dtype={"arxiv_id_clean": str})
sc = sc[sc.split == "forward_test_2025"].copy()
assert len(sc) == 6325, len(sc)
y = sc.y_q.values.astype(int)


def roc_pr(score, npts=160):
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P_ = ys.sum()
    N_ = len(ys) - P_
    tpr = tp / P_
    fpr = fp / N_
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tpr
    # AUC (trapezoid) and average precision (step)
    auc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    idx = np.unique(np.linspace(0, len(ys) - 1, npts).astype(int))
    roc = [[0.0, 0.0]] + [[r(fpr[i], 4), r(tpr[i], 4)] for i in idx]
    pr = [[r(rec[i], 4), r(prec[i], 4)] for i in idx if i >= 20]
    return roc, pr, auc, ap


CURVES = {}
for model in ["logistic", "hgb"]:
    CURVES[model] = {}
    for row, key in [("controls_only", "controls_only"), ("upvotes_only", "upvotes_only"), ("+attention", "attention")]:
        col = f"forward_yq__{model}__P_tierB__{row}"
        roc, pr, auc, ap = roc_pr(sc[col].values)
        CURVES[model][key] = {"roc": roc, "pr": pr, "auc": r(auc, 4), "ap": r(ap, 4)}
# ledger AUCs for the legend (cite the ledger values, not the recomputed ones)
HL = P["headline"]["by_model"]
for model in ["logistic", "hgb"]:
    for key, jk in [("controls_only", "controls_only"), ("upvotes_only", "upvotes_only"), ("attention", "attention")]:
        CURVES[model][key]["auc_ledger"] = r(HL[model][f"{jk}_auc"], 4)
        CURVES[model][key]["pr_ledger"] = r(HL[model][f"{jk}_pr_auc"], 4)
        # sanity: recomputed AUC within 0.002 of ledger
        assert abs(CURVES[model][key]["auc"] - CURVES[model][key]["auc_ledger"]) < 0.002, (model, key)
base_rate_test = P["experiments"]["forward_yq"]["base_rate_test"]

# (d) delta AUC forest rows (ledger C2)
EX = P["experiments"]


def drow(exp, sub, branch, label, n_key="n", flagged=False, headline=False):
    out = {"label": label, "flagged": flagged, "headline": headline}
    for m in ["logistic", "hgb"]:
        blk = EX[exp]["models"][m][sub]
        rr = blk["rows"][f"{branch}|+attention"]
        d = rr["delta_vs_controls_only"]
        out[m] = [r(d["d_auc"], 4)] + [r(x, 4) for x in d["ci"]["month_cluster"]["auc"]]
        out["n"] = blk["n"]
    return out


FOREST = [
    drow("forward_yq", "main", "P_tierB", "Headline: leakage free prestige, forward test 2025", headline=True),
    drow("forward_yq", "main", "P_interim", "Prestige: career count measured today"),
    drow("forward_yq", "main", "P_none", "No prestige controls"),
    drow("forward_yq", "mature_k12", "P_tierB", "Mature subset (age at least 12 months)"),
    drow("forward_ym", "main", "P_tierB", "Within month label"),
    drow("forward_yinf", "main", "P_tierB", "Top decile influential citations"),
    drow("backward_yq", "main", "P_tierB", "Backward test: train 2024 and 2025, test 2023"),
    drow("backward_ym", "main", "P_tierB", "Backward test, within month label"),
    drow("drop_age", "main", "P_tierB", "Drop age from controls"),
    drow("no_launch_months", "main", "P_tierB", "Train excludes launch era months"),
    drow("legacy_subfield", "main", "P_tierB", "Flagged: legacy mixed taxonomy subfield control", flagged=True),
    drow("v2_leaky_replication", "main", "P_interim", "Flagged: h-index measured today (leaky)", flagged=True),
]

# (e) per subfield mixed model slopes
LMM = A["hierarchical_mixedlm"]
slopes = [{"sf": k, "slope": v["slope"], "ci": v["ci95"], "n": v["n"]} for k, v in LMM["subfield_slopes"].items()]
slopes.sort(key=lambda d: -d["slope"])

# (f) misjudged: prestige medians + drivers (leakage free driver model)
PA = A["over_under"]["prestige_asymmetry"]
prestige = []
for key, lab, leaky in [("max_prior_papers_true_w99", "Prior papers, first or last author (before the paper)", False),
                        ("max_years_active", "Years active before the paper", False),
                        ("max_hindex", "Max author h-index (today, leaky)", True),
                        ("last_author_hindex", "Last author h-index (today, leaky)", True)]:
    v = PA[key]
    prestige.append({"key": key, "label": lab, "leaky": leaky,
                     "over": [v["overrated"]["median"]] + v["overrated"]["ci95"],
                     "under": [v["underrated"]["median"]] + v["underrated"]["ci95"],
                     "neutral": [v["neutral"]["median"]] + v["neutral"]["ci95"],
                     "n_over": v["overrated"]["n"], "n_under": v["underrated"]["n"], "n_neutral": v["neutral"]["n"],
                     "diff": v["diff_median_over_minus_under"]})
DRV = A["over_under"]["logit_drivers"]
NICE = {"tierB_prior (per SD)": "Prior papers (per SD)", "tierB_yrs (per SD)": "Years active (per SD)",
        "kw_efficient": "Keyword: efficient", "kw_survey": "Keyword: survey", "has_github": "Has Github repo",
        "kw_agent": "Keyword: agent", "kw_reasoning": "Keyword: reasoning", "kw_benchmark": "Keyword: benchmark",
        "title_n_words (per SD)": "Title length (per SD)", "log_abs_chars (per SD)": "Abstract length (per SD)",
        "kw_rl": "Keyword: RL", "kw_diffusion": "Keyword: diffusion", "log_n_authors (per SD)": "Number of authors (per SD)",
        "title_has_colon": "Title has a colon", "kw_multimodal": "Keyword: multimodal", "kw_llm": "Keyword: LLM",
        "kw_scaling": "Keyword: scaling"}


def drivers(key, k=8):
    rows_ = [x for x in DRV[key]["rows"] if not x["feature"].startswith("subfield ")][:k]
    return [{"f": NICE.get(x["feature"], x["feature"]), "or": x["or"], "ci": x["or_ci95"], "p": x["p"]} for x in rows_]


DRIVERS = {"over": drivers("overrated_vs_rest_NOLEAK"), "under": drivers("underrated_vs_rest_NOLEAK"),
           "n_over": DRV["overrated_vs_rest_NOLEAK"]["n_events"], "n_under": DRV["underrated_vs_rest_NOLEAK"]["n_events"]}

# (g) CCDF: trending vs never trending vs bottom tertile attention trending
an = pd.read_csv(F_AN, dtype={"arxiv_id_clean": str}, low_memory=False)
meta = pd.read_csv(F_CTRL_META, dtype={"arxiv_id_clean": str})
s2 = pd.read_csv(F_CTRL_S2, dtype={"arxiv_id_clean": str})
c = meta.merge(s2, on="arxiv_id_clean", how="inner", validate="1:1")
c = c[c.ss_found == 1].copy()
rel = pd.to_datetime(c.published_v1, errors="coerce", utc=True).dt.tz_localize(None)
c["age_months"] = (SNAPSHOT - rel).dt.days / 30.44
c = c[(c.age_months >= 5) & (c.age_months <= 40)].copy()
c["citation_count"] = pd.to_numeric(c.citation_count, errors="coerce")
c = c[c.citation_count.notna()].copy()
N_CTRL = int(len(c))
assert N_CTRL == A["control_comparison"]["control_sample_facts"]["n_analysis_5_40_months"], N_CTRL
GRID = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 5000]


def ccdf(vals):
    vals = np.asarray(vals, dtype=float)
    return [[g, r((vals >= g).mean(), 4)] for g in GRID]


low_att = ou[ou.attention_resid_pct <= 1 / 3]
CCDF = {"trending": ccdf(an.citation_count.values), "control": ccdf(c.citation_count.values),
        "low_att": ccdf(low_att.citation_count.values),
        "n_trending": int(len(an)), "n_control": N_CTRL, "n_low_att": int(len(low_att)),
        "median_trending": float(an.citation_count.median()), "median_control": float(c.citation_count.median()),
        "median_low_att": float(low_att.citation_count.median())}
CC = A["control_comparison"]
PREM = CC["trending_premium"]["cem_month_subfield13_hbin"]
LOWATT = CC["low_attention_vs_background"]["bottom_tertile_attention_residual"]

# (h) crowding IV coefficient plot
ivrows = [
    {"label": "OLS, month + day of week + subfield FE, leakage free prestige", "b": IV["ols"]["honest_FE_tierB"]["beta"],
     "lo": IV["ols"]["honest_FE_tierB"]["beta"] - 1.96 * IV["ols"]["honest_FE_tierB"]["se_cluster_month"],
     "hi": IV["ols"]["honest_FE_tierB"]["beta"] + 1.96 * IV["ols"]["honest_FE_tierB"]["se_cluster_month"],
     "n": IV["ols"]["honest_FE_tierB"]["N"], "kind": "ols", "note": "95% CI, clustered by release month"},
    {"label": "2SLS, same fixed effects, Anderson Rubin CI", "b": IV["primary_honest"]["iv_2sls"]["beta"],
     "lo": IV["primary_honest"]["ar_ci95"]["lower"], "hi": IV["primary_honest"]["ar_ci95"]["upper"],
     "n": IV["primary_honest"]["N"], "kind": "iv", "note": f"first stage t {IV['primary_honest']['first_stage']['t']:.1f}, KP F {IV['primary_honest']['first_stage']['kp_f']:.1f}"},
    {"label": "2SLS with day FE (mechanical first stage), Anderson Rubin CI", "b": IV["dayfe_Z1p_kw"]["iv_2sls"]["beta"],
     "lo": IV["dayfe_Z1p_kw"]["ar_ci95"]["lower"], "hi": IV["dayfe_Z1p_kw"]["ar_ci95"]["upper"],
     "n": IV["dayfe_Z1p_kw"]["N"], "kind": "mech", "note": f"first stage t {IV['dayfe_Z1p_kw']['first_stage']['t']:.1f}, KP F {IV['dayfe_Z1p_kw']['first_stage']['kp_f']:.0f}, adding up identity"},
]
for x in ivrows:
    for k in ("b", "lo", "hi"):
        x[k] = r(x[k], 4)

# ----------------------------------------------------------------------------
# 3. Headline numbers (ledger values, cited by value)
# ----------------------------------------------------------------------------
LG = HL["logistic"]
HG = HL["hgb"]
NUM = {
    "n_all": 11344, "n_by_year": {int(k): int(v) for k, v in n_by_year.items()},
    "n_train": P["meta"]["n_train"], "n_test": P["meta"]["n_test"], "n_mature": P["meta"]["n_mature_k12"],
    "n_backward": P["meta"]["n_test_backward"], "k_top_decile": LG["k_top_decile"],
    "lg_ctrl_auc": r(LG["controls_only_auc"], 3), "lg_att_auc": r(LG["attention_auc"], 3),
    "lg_dauc": r(LG["delta_auc"], 3), "lg_dauc_ci": [r(x, 3) for x in LG["delta_auc_ci_month"]],
    "lg_up_auc": r(LG["upvotes_only_auc"], 3), "lg_up_auc_ci": [r(x, 3) for x in LG["upvotes_only_auc_ci_month"]],
    "lg_dpr": r(LG["delta_pr_auc"], 3), "lg_dpr_ci": [r(x, 3) for x in LG["delta_pr_auc_ci_month"]],
    "lg_pr_ctrl": r(LG["controls_only_pr_auc"], 3), "lg_pr_att": r(LG["attention_pr_auc"], 3),
    "lg_pk_ctrl": r(LG["controls_only_p_at_k"], 3), "lg_pk_att": r(LG["attention_p_at_k"], 3),
    "lg_dpk": r(LG["delta_p_at_k"], 3), "lg_dpk_ci": [r(x, 3) for x in LG["delta_p_at_k_ci_month"]],
    "lg_att_minus_up": r(LG["attention_minus_upvotes_only_auc"], 3), "lg_att_minus_up_ci": [r(x, 3) for x in LG["attention_minus_upvotes_only_auc_ci_month"]],
    "hg_ctrl_auc": r(HG["controls_only_auc"], 3), "hg_att_auc": r(HG["attention_auc"], 3),
    "hg_dauc": r(HG["delta_auc"], 3), "hg_dauc_ci": [r(x, 3) for x in HG["delta_auc_ci_month"]],
    "hg_dpk": r(HG["delta_p_at_k"], 3), "hg_dpk_ci": [r(x, 3) for x in HG["delta_p_at_k_ci_month"]],
    "nb2_irr": r(M4["nb2"]["irr_per_doubling"], 2), "nb2_irr_ci": [r(x, 2) for x in M4["nb2"]["irr_per_doubling_ci95"]],
    "nb2_boot_ci": [r(x, 2) for x in BOOT["nb2_irr2x_ci95"]],
    "pois_irr": r(M4["poisson_qmle"]["irr_per_doubling"], 2), "pois_irr_ci": [r(x, 2) for x in M4["poisson_qmle"]["irr_per_doubling_ci95"]],
    "ols_ratio": r(M4["ols_log1p"]["ratio_1p_cites_per_doubling"], 2), "ols_ratio_ci": [r(x, 2) for x in M4["ols_log1p"]["ratio_1p_cites_per_doubling_ci95"]],
    "ols_elast": r(M4["ols_log1p"]["beta"], 3),
    "e_nb2": r(EVAL["nb2"]["e_value_point"], 2), "e_nb2_lo": r(EVAL["nb2"]["e_value_ci_lower"], 2),
    "e_pois": r(EVAL["poisson_qmle"]["e_value_point"], 2), "e_ols": r(EVAL["ols_log1p_ratio"]["e_value_point"], 2),
    "prem_ratio": r(PREM["ratio"], 2), "prem_ci": [r(x, 2) for x in PREM["ratio_ci95"]],
    "prem_naive": r(CC["trending_premium"]["naive"]["ratio"], 2), "n_ctrl": N_CTRL,
    "prem_nt": PREM["n_trending_matched"], "prem_nc": PREM["n_control_matched"],
    "lowatt_pct": r(LOWATT["percentile_in_background_same_month"]["mean_percentile"], 3),
    "lowatt_pct_ci": [r(x, 3) for x in LOWATT["percentile_in_background_same_month"]["ci95"]],
    "lowatt_ratio": r(LOWATT["cem_ratio"], 2), "lowatt_n": LOWATT["n"],
    "n_over": int(COUNTS["overrated"]), "n_under": int(COUNTS["underrated"]), "n_neutral": int(COUNTS["neutral"]),
    "spearman_resid": A["over_under"]["spearman_attention_impact_resid"], "spearman_raw": spearman_raw,
    "resid_r2": A["over_under"]["resid_model_r2"],
    "pp_over": PA["max_prior_papers_true_w99"]["overrated"]["median"], "pp_over_ci": PA["max_prior_papers_true_w99"]["overrated"]["ci95"],
    "pp_under": PA["max_prior_papers_true_w99"]["underrated"]["median"], "pp_under_ci": PA["max_prior_papers_true_w99"]["underrated"]["ci95"],
    "pp_n_over": PA["max_prior_papers_true_w99"]["overrated"]["n"], "pp_n_under": PA["max_prior_papers_true_w99"]["underrated"]["n"],
    "h_over": PA["max_hindex"]["overrated"]["median"], "h_over_ci": PA["max_hindex"]["overrated"]["ci95"],
    "h_under": PA["max_hindex"]["underrated"]["median"], "h_under_ci": PA["max_hindex"]["underrated"]["ci95"],
    "lmm_slope": r(LMM["fixed_slope_log_upvotes"], 3), "lmm_slope_ci": [r(x, 3) for x in LMM["fixed_slope_ci95"]],
    "lmm_sd": LMM["variance_components"]["slope_sd"], "lmm_range": LMM["slope_range"],
    "iv_ols": r(IV["ols"]["honest_FE_tierB"]["beta"], 3), "iv_b": r(IV["primary_honest"]["iv_2sls"]["beta"], 3),
    "iv_se": r(IV["primary_honest"]["iv_2sls"]["se"], 3), "iv_ar": IV["primary_honest"]["ar_ci95"]["string"],
    "iv_n": IV["primary_honest"]["N"], "iv_t": r(IV["primary_honest"]["first_stage"]["t"], 1),
    "iv_f": r(IV["primary_honest"]["first_stage"]["kp_f"], 1),
    "iv_day_b": r(IV["dayfe_Z1p_kw"]["iv_2sls"]["beta"], 3), "iv_day_f": r(IV["dayfe_Z1p_kw"]["first_stage"]["kp_f"], 0),
    "iv_day_ar": IV["dayfe_Z1p_kw"]["ar_ci95"]["string"],
    # Spearman(upvotes, age) within release year, printed as three signed values (2023, 2024, 2025)
    "sp_up_age": [round(float(v["spearman_upvotes_age_months"]), 2) for _, v in sorted(P["audit"]["attention_is_cumulative_at_collection"]["spearman_upvotes_age_by_year"].items())],
    "n_over_zero": A["over_under"]["overrated_zero_citation_records"]["n_overrated_with_0_cites"],
    "n_over_zero_12": A["over_under"]["overrated_zero_citation_records"]["n_overrated_with_0_cites_age_ge_12"],
    "sel_naive": r(A["selection_adjustment"]["naive_gap"]["ratio"], 2), "sel_cem": r(A["selection_adjustment"]["cem"]["ratio"], 2),
    "sel_ipw": r(A["selection_adjustment"]["ipw"]["ratio"], 2), "sel_ps": r(A["selection_adjustment"]["ps_matching"]["ratio"], 2),
}

DATA_JS = {
    "SF": SUBFIELDS, "P": rows, "YEARS": YEARS, "COUNTS": {"under": NUM["n_under"], "over": NUM["n_over"], "neutral": NUM["n_neutral"]},
    "DR": dr, "LADDER": ladder, "CURVES": CURVES, "FOREST": FOREST, "SLOPES": slopes, "PRESTIGE": prestige,
    "DRIVERS": DRIVERS, "CCDF": CCDF, "IVROWS": ivrows, "NUM": NUM, "BASE_RATE": r(base_rate_test, 3),
}


def js_json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


# ----------------------------------------------------------------------------
# 4. HTML
# ----------------------------------------------------------------------------
TEMPLATE = (HERE / "template.html").read_text(encoding="utf-8")
html = TEMPLATE.replace("/*__DATA__*/", "const D=" + js_json(DATA_JS) + ";")
# simple {{num:key}} substitution for headline text
import re


DEC2 = {"nb2_irr", "nb2_irr_ci", "nb2_boot_ci", "pois_irr", "pois_irr_ci", "ols_ratio", "ols_ratio_ci", "prem_ratio", "prem_ci",
        "prem_naive", "e_nb2", "e_nb2_lo", "e_pois", "e_ols", "lowatt_ratio", "sel_naive", "sel_cem", "sel_ipw", "sel_ps"}
SIGNED = {"lg_dauc", "lg_dauc_ci", "hg_dauc", "hg_dauc_ci", "lg_dpk", "lg_dpk_ci", "hg_dpk", "hg_dpk_ci", "lg_dpr", "lg_dpr_ci",
          "lg_att_minus_up", "lg_att_minus_up_ci"}
DEC1 = {"iv_t", "iv_f"}
DEC0 = {"iv_day_f"}


def fmt(key, v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if key in DEC2:
            d = 2
        elif key in DEC1:
            d = 1
        elif key in DEC0:
            d = 0
        elif key in ("lmm_sd", "spearman_resid", "spearman_raw", "lmm_range", "sp_up_age", "resid_r2"):
            d = 3
        elif key.startswith(("pp_", "h_")):
            return f"{v:g}"
        else:
            d = 3
        s = f"{v:.{d}f}"
        if key in SIGNED and v > 0:
            s = "+" + s
        return s
    return str(v)


def sub_num(m):
    key = m.group(1)
    if key not in NUM:
        raise KeyError(key)
    v = NUM[key]
    if key == "sp_up_age":
        return ", ".join(f"{x:+.2f}" for x in v)
    if isinstance(v, list):
        return "[" + ", ".join(fmt(key, x) for x in v) + "]"
    if isinstance(v, dict):
        return ", ".join(fmt(key, x) for x in v.values())
    return fmt(key, v)


html = re.sub(r"\{\{num:([a-z0-9_]+)\}\}", sub_num, html)
html = re.sub(r"\{\{sf_count\}\}", str(len(SUBFIELDS)), html)
leftover = re.findall(r"\{\{[^}]+\}\}", html)
assert not leftover, leftover
OUT_HTML.write_text(html, encoding="utf-8")
size = OUT_HTML.stat().st_size
print(f"wrote {OUT_HTML} ({size/1e6:.2f} MB), papers={len(rows)}, subfields={len(SUBFIELDS)}")
assert size < 4_000_000, size
