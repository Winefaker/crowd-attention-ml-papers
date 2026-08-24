#!/usr/bin/env python
"""
27_association_v3.py — Part I ("association") analysis, rebuilt (v3, 2026-08).

Question (proposal, May 2026): does community attention (HF Daily Papers upvotes)
predict research impact (Semantic Scholar citations) of arXiv ML papers after
controlling for author prestige, subfield and age — and which papers does the
crowd over-/under-rate?

This ONE script supersedes Project/scripts/12_models_v2.py, 14_text_themes.py and
15_control_analysis.py. It runs on the final analysis frame
(data/processed/analysis_final.csv, n = 11,344) with

  * the uniform 13-level keyword taxonomy `subfield_kw` (the legacy `subfield`
    mixes two taxonomies keyed on upvotes; kept only as a robustness row);
  * the leakage-free Tier B prestige (prior papers / years active of the
    first/last author BEFORE the paper) as the main prestige control; the
    today-measured h-index (2026-06-11) is a flagged, leaky control;
  * a prior-only author_max_appear (diagnostics/scratch_data/author_appear_prior.csv)
    instead of the 48 %-look-ahead column in analysis_final.

Blocks
  1. count-model ladder M0–M6: Poisson-QMLE with FE (pyfixest.fepois, CRV1 by
     release_month), NB2 (statsmodels, month-clustered robust SE, Poisson starts),
     log1p-OLS elasticity (feols, CRV1); month-cluster block bootstrap for M4;
     E-value; within-cell permutation placebo (+ reference-count placebo outcome);
     MixedLM random intercept + random slope by subfield_kw; dose-response deciles.
  2. selection adjustment (high vs low attention tertile): PS 1:1 NN matching with
     caliper, ATT-IPW (normalised/stabilised, trimmed), CEM exact matching;
     balance tables + bootstrap CIs.
  3. never-trending control comparison (exact matching month × subfield × h-bin;
     bootstrap CI; honest low-attention percentile).
  4. over-/under-rated residual analysis (labels, cluster-robust logit drivers,
     prestige asymmetry with bootstrap CIs, TF-IDF+L1 title themes, examples)
     -> data/processed/overunder_v3.csv for the dashboard.
  5. measurement evidence (monthly medians, Spearman(upvotes, age), zero shares).

Outputs: results/association_v3.json, results/association_v3_tables.md,
         results/association_v3_NOTE.md, data/processed/overunder_v3.csv

Run:  python scripts/27_association_v3.py
No network access. Deterministic (SEED = 20260818).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 20260818
T0 = time.time()

BASE = Path(__file__).resolve().parents[1]
AUG = BASE
ROOT = AUG
DATA = ROOT / "data/processed/analysis_final.csv"
PAPERS_V2 = BASE / "data/processed/papers_v2_text.csv"            # read-only (text)
CTRL_META = BASE / "data/raw/arxiv_control.csv"              # read-only
CTRL_S2 = BASE / "data/raw/arxiv_control_s2.csv"             # read-only
APPEAR_PRIOR = AUG / "data/processed/author_appear_prior.csv"
OUT_JSON = ROOT / "results/association_v3.json"
OUT_TABLES = ROOT / "results/association_v3_tables.md"
OUT_NOTE = ROOT / "results/association_v3_NOTE.md"
OUT_OU = ROOT / "data/processed/overunder_v3.csv"

SNAPSHOT = pd.Timestamp("2026-06-11")   # S2 citation snapshot used for age & control
HF_SCRAPE = "2026-06-05 (2024–25 papers) / 2026-06-11 (2023 papers)"
N_BOOT = 500
N_PLACEBO = 300
N_BOOT_MATCH = 300
if os.environ.get("ASSOC_QUICK"):      # smoke-test mode (not for reported numbers)
    N_BOOT, N_PLACEBO, N_BOOT_MATCH = 10, 5, 5

RES: dict = {"meta": {
    "script": "scripts/27_association_v3.py", "seed": SEED,
    "data": str(DATA.relative_to(ROOT)), "snapshot_citations": str(SNAPSHOT.date()),
    "hf_upvote_scrape": HF_SCRAPE,
    "subfield_variable": "subfield_kw (uniform 13-level keyword taxonomy)",
    "prestige_main": "Tier B leakage-free: log1p(max prior papers of first/last author) + max years active",
    "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "n_boot_match": N_BOOT_MATCH,
}}
KW = ["kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning", "kw_benchmark", "kw_survey",
      "kw_efficient", "kw_multimodal", "kw_rl", "kw_scaling"]

# ---- keyword rules (identical to scripts/30_subfield_uniform.py / v2 04_merge) ----
KEYWORD_RULES = [
    ("Multimodal", r"multimodal|vision-language|vision language|mllm|vlm|image-text|video-language"),
    ("Vision/Image-Gen", r"diffusion|image generation|text-to-image|video generation|gaussian splatting|nerf|3d generation|super-resolution"),
    ("Agents", r"\bagent|agentic|tool use|tool-use|llm agent|multi-agent|autonomous"),
    ("RAG/Retrieval", r"retrieval-augmented|\brag\b|retrieval|dense retrieval|reranking"),
    ("Reasoning/RL", r"reasoning|chain-of-thought|chain of thought|reinforcement learning|\brlhf\b|reward model|preference optimization|\bgrpo\b|\bppo\b"),
    ("Efficiency/Systems", r"quantization|efficient|kv cache|inference|distillation|pruning|moe|mixture-of-experts|long context|flash"),
    ("Vision-Perception", r"object detection|segmentation|depth|pose|tracking|optical flow|point cloud"),
    ("Speech/Audio", r"speech|audio|asr|text-to-speech|\btts\b|music|voice"),
    ("Robotics/Embodied", r"robot|embodied|manipulation|navigation|locomotion|sim-to-real"),
    ("Benchmark/Eval", r"benchmark|evaluation|dataset|leaderboard"),
    ("Code/Math", r"\bcode\b|program synthesis|coding|software|theorem|mathematical reasoning"),
    ("LLM-core", r"large language model|\bllm\b|language model|pretraining|fine-tuning|instruction tuning|transformer|attention|scaling"),
]
RULES = [(lab, re.compile(pat)) for lab, pat in KEYWORD_RULES]
CAT_TO_KW = {  # nearest keyword label for an arXiv primary category (control fallback)
    "cs.CV": "Vision-Perception", "eess.IV": "Vision-Perception", "cs.GR": "Vision/Image-Gen",
    "cs.MM": "Multimodal",
    "cs.CL": "LLM-core", "cs.LG": "LLM-core", "stat.ML": "LLM-core", "cs.AI": "LLM-core",
    "cs.IR": "RAG/Retrieval", "cs.RO": "Robotics/Embodied", "cs.SD": "Speech/Audio",
    "eess.AS": "Speech/Audio", "cs.SE": "Code/Math", "cs.PL": "Code/Math", "cs.MA": "Agents",
    "cs.DC": "Efficiency/Systems", "cs.AR": "Efficiency/Systems",
}
FAMILY = {"Vision/Image-Gen": "Vision", "Vision-Perception": "Vision", "Multimodal": "Vision",
          "LLM-core": "Language/ML", "RAG/Retrieval": "Language/ML", "Reasoning/RL": "Language/ML",
          "Agents": "Language/ML", "Code/Math": "Language/ML", "Benchmark/Eval": "Language/ML",
          "Efficiency/Systems": "Language/ML", "Speech/Audio": "Speech", "Robotics/Embodied": "Robotics",
          "Other": "Other"}


def kw_label(text):
    t = (text if isinstance(text, str) else "").lower()
    for lab, rx in RULES:
        if rx.search(t):
            return lab
    return None


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def fnum(x, nd=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(round(float(x), nd))


def ci_pct(a, lo=2.5, hi=97.5):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return [fnum(np.percentile(a, lo), 4), fnum(np.percentile(a, hi), 4)]


# =============================================================================
# 0. LOAD
# =============================================================================
def load():
    df = pd.read_csv(DATA, dtype={"arxiv_id_clean": str})
    assert len(df) == 11344 and df.arxiv_id_clean.is_unique
    p = pd.read_csv(PAPERS_V2, dtype={"arxiv_id_clean": str}, low_memory=False)
    p = p[["arxiv_id_clean", "title", "ai_keywords", "citation_repaired"]]
    df = df.merge(p, on="arxiv_id_clean", how="left", validate="1:1")
    ap = pd.read_csv(APPEAR_PRIOR, dtype={"arxiv_id_clean": str})
    df = df.merge(ap[["arxiv_id_clean", "max_appear_prior"]], on="arxiv_id_clean", how="left", validate="1:1")
    assert df.max_appear_prior.notna().all() and df.title.notna().all()

    df["cites"] = df.citation_count.round().astype(int)
    df["infl"] = df.influential_citations.round().astype(int)
    df["log_age"] = np.log(df.age_months)
    df["month_id"] = pd.factorize(df.release_month, sort=True)[0]
    # Tier B (leakage-free) prestige
    df["tierB_miss"] = df.log1p_max_prior_papers_true.isna().astype(int)
    df["tierB_prior"] = df.log1p_max_prior_papers_true.fillna(df.log1p_max_prior_papers_true.median())
    df["tierB_yrs"] = df.max_years_active.fillna(df.max_years_active.median())
    # today-measured h-index (LEAKY: measured 2026-06-11, after the outcome)
    df["h_miss"] = df.max_hindex.isna().astype(int)
    df["log_max_h"] = np.log1p(df.max_hindex).fillna(np.log1p(df.max_hindex).median())
    df["log_last_h"] = np.log1p(df.last_author_hindex).fillna(np.log1p(df.last_author_hindex).median())
    # text / format
    df["log_n_authors"] = np.log(df.n_authors.clip(lower=1).fillna(df.n_authors.median()))
    df["log_abs_chars"] = np.log(df.abstract_n_chars.clip(lower=1))
    df["log_appear_prior"] = np.log1p(df.max_appear_prior)
    df["log_appear_lookahead"] = np.log1p(df.author_max_appear)
    df["log_comments"] = np.log1p(df.num_comments)
    df["log_upvotes_sq"] = df.log_upvotes ** 2
    return df


# =============================================================================
# 1. COUNT-MODEL LADDER
# =============================================================================
X_TEXT = ["log_n_authors", "title_n_words", "title_has_colon", "log_abs_chars", "has_github"] + KW
SPECS = {
    "M0_raw":            dict(x=[], fe=[]),
    "M1_age":            dict(x=["log_age"], fe=[]),
    "M2_subfield":       dict(x=["log_age"], fe=["subfield_kw"]),
    "M3_month_dow":      dict(x=["log_age"], fe=["subfield_kw", "release_month", "dow"]),
    "M4_tierB_prestige": dict(x=["log_age", "tierB_prior", "tierB_yrs", "tierB_miss"],
                              fe=["subfield_kw", "release_month", "dow"]),
    "M5_plus_hindex_LEAKY": dict(x=["log_age", "tierB_prior", "tierB_yrs", "tierB_miss",
                                    "log_max_h", "log_last_h", "h_miss"],
                                 fe=["subfield_kw", "release_month", "dow"]),
    "M6_plus_text":      dict(x=["log_age", "tierB_prior", "tierB_yrs", "tierB_miss",
                                 "log_max_h", "log_last_h", "h_miss"] + X_TEXT,
                              fe=["subfield_kw", "release_month", "dow"]),
}
SPEC_LABELS = {
    "M0_raw": "M0 raw", "M1_age": "M1 + log age", "M2_subfield": "M2 + subfield_kw FE",
    "M3_month_dow": "M3 + release-month FE + day-of-week FE",
    "M4_tierB_prestige": "M4 + Tier B prestige (leakage-free) [main]",
    "M5_plus_hindex_LEAKY": "M5 + today-measured h-index (LEAKY control)",
    "M6_plus_text": "M6 + text/format controls (authors, title, abstract, GitHub, kw flags)",
}


def _fml(y, x, fe, treat="log_upvotes", sm_style=False):
    rhs = [treat] + list(x)
    if sm_style:
        rhs += [f"C({f})" for f in fe]
        return f"{y} ~ " + " + ".join(rhs)
    f = f"{y} ~ " + " + ".join(rhs)
    if fe:
        f += " | " + " + ".join(fe)
    return f


def fit_poisson(df, y, x, fe, treat="log_upvotes"):
    warnings.simplefilter("ignore")   # a lazily-imported dependency resets the warning filters on first fit
    m = pf.fepois(_fml(y, x, fe, treat), data=df, vcov={"CRV1": "release_month"})
    b, se = float(m.coef()[treat]), float(m.se()[treat])
    return dict(beta=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se], nobs=int(m._N),
                converged=bool(m.convergence), model=m)


def fit_ols(df, y, x, fe, treat="log_upvotes"):
    warnings.simplefilter("ignore")
    m = pf.feols(_fml(y, x, fe, treat), data=df, vcov={"CRV1": "release_month"})
    b, se = float(m.coef()[treat]), float(m.se()[treat])
    r2 = float(m._r2) if hasattr(m, "_r2") else None
    return dict(beta=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se], nobs=int(m._N), r2=r2, model=m)


def fit_nb(df, y, x, fe, treat="log_upvotes", start=None):
    f = _fml(y, x, fe, treat, sm_style=True)
    mod = smf.negativebinomial(f, data=df)
    if start is None:
        po = smf.glm(f, data=df, family=sm.families.Poisson()).fit()
        start = np.append(po.params.values, 1.0)
    groups = df["month_id"].values
    res, method_used = None, None
    for method, kw in [("newton", dict(maxiter=300)), ("bfgs", dict(maxiter=3000)), ("nm", dict(maxiter=20000))]:
        try:
            r = mod.fit(start_params=start, method=method, disp=0, cov_type="cluster",
                        cov_kwds={"groups": groups}, **kw)
            if r.mle_retvals.get("converged", False) and np.isfinite(r.bse[treat]):
                res, method_used = r, method
                break
            if res is None:
                res, method_used = r, method
            start = r.params.values
        except Exception as e:  # pragma: no cover
            log(f"    NB {method} failed: {e}")
    b, se = float(res.params[treat]), float(res.bse[treat])
    return dict(beta=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se], nobs=int(res.nobs),
                alpha=float(res.params["alpha"]), converged=bool(res.mle_retvals.get("converged", False)),
                method=method_used, model=res)


def pack(r, kind):
    out = {"beta": fnum(r["beta"], 4), "se": fnum(r["se"], 4),
           "ci95": [fnum(r["ci"][0], 4), fnum(r["ci"][1], 4)], "nobs": r["nobs"]}
    if kind in ("poisson", "nb"):
        out["irr_per_doubling"] = fnum(np.exp(r["beta"] * np.log(2)), 3)
        out["irr_per_doubling_ci95"] = [fnum(np.exp(r["ci"][0] * np.log(2)), 3), fnum(np.exp(r["ci"][1] * np.log(2)), 3)]
        out["converged"] = r.get("converged")
    if kind == "nb":
        out["alpha"] = fnum(r["alpha"], 3)
        out["optimizer"] = r.get("method")
    if kind == "ols":
        out["ratio_1p_cites_per_doubling"] = fnum(2 ** r["beta"], 3)
        out["ratio_1p_cites_per_doubling_ci95"] = [fnum(2 ** r["ci"][0], 3), fnum(2 ** r["ci"][1], 3)]
        out["r2"] = fnum(r.get("r2"), 3)
    return out


def ladder(df):
    log("1. Count-model ladder")
    out = {}
    for name, sp in SPECS.items():
        po = fit_poisson(df, "cites", sp["x"], sp["fe"])
        nb = fit_nb(df, "cites", sp["x"], sp["fe"])
        ol = fit_ols(df, "log_citations", sp["x"], sp["fe"])
        out[name] = {"label": SPEC_LABELS[name], "controls": sp["x"], "fixed_effects": sp["fe"],
                     "poisson_qmle": pack(po, "poisson"), "nb2": pack(nb, "nb"), "ols_log1p": pack(ol, "ols")}
        log(f"  {name:22s} Pois {po['beta']:.3f}({po['se']:.3f}) IRR2x {np.exp(po['beta']*np.log(2)):.2f} | "
            f"NB {nb['beta']:.3f}({nb['se']:.3f}) conv={nb['converged']} | OLS {ol['beta']:.3f}({ol['se']:.3f})")
    RES["ladder"] = out

    # ---- robustness rows at M4 controls -------------------------------------
    log("  robustness rows")
    sp = SPECS["M4_tierB_prestige"]
    rob = {}

    def add(name, d, y_p="cites", y_o="log_citations", x=None, fe=None, treat="log_upvotes", note=""):
        x = sp["x"] if x is None else x
        fe = sp["fe"] if fe is None else fe
        po = fit_poisson(d, y_p, x, fe, treat)
        nb = fit_nb(d, y_p, x, fe, treat)
        ol = fit_ols(d, y_o, x, fe, treat)
        rob[name] = {"note": note, "n": int(len(d)), "poisson_qmle": pack(po, "poisson"),
                     "nb2": pack(nb, "nb"), "ols_log1p": pack(ol, "ols")}
        log(f"    {name:28s} Pois {po['beta']:.3f} NB {nb['beta']:.3f} OLS {ol['beta']:.3f} (n={len(d)})")

    add("M4_legacy_subfield_taxonomy", df, fe=["subfield", "release_month", "dow"],
        note="legacy 26-level `subfield` (two taxonomies keyed on upvotes) instead of subfield_kw")
    add("M4_no_log_age", df, x=[c for c in sp["x"] if c != "log_age"],
        note="month FE only, no within-month log-age term")
    add("M4_plus_appear_prior", df, x=sp["x"] + ["log_appear_prior"],
        note="+ log1p prior-only author HF appearances (v1 'recurrence prestige', look-ahead removed)")
    add("M4_plus_appear_LOOKAHEAD", df, x=sp["x"] + ["log_appear_lookahead"],
        note="+ log1p author_max_appear as stored (48% look-ahead) — shown only to quantify the leak")
    add("M4_plus_hindex_LEAKY_only", df, x=sp["x"] + ["log_max_h", "log_last_h", "h_miss"],
        note="same as M5")
    add("M4_plus_comments", df, x=sp["x"] + ["log_comments"],
        note="+ log1p HF comments (second attention signal, cumulative at scrape)")
    add("M4_influential_citations", df, y_p="infl", y_o="log_infl",
        note="outcome = influential citations (S2)")
    add("M4_age_ge_12mo", df[df.age_months >= 12], note="papers with >= 12 months exposure")
    add("M4_cohort_2023", df[df.release_year == 2023], note="mature 2023 cohort only")
    add("M4_cohort_2024", df[df.release_year == 2024], note="2024 cohort only")
    add("M4_cohort_2025", df[df.release_year == 2025], note="2025 cohort only (young)")
    add("M4_drop_repaired", df[df.citation_repaired == 0], note="drop 113 title-match-repaired outcomes")
    add("M4_upvote_rank_in_month", df, treat="upvote_rank_month",
        note="treatment = within-release-month percentile rank of upvotes (0..1); beta = bottom-to-top gap")
    add("M4_ai_keywords_only_labels", df[df.subfield_kw_source == "ai_keywords"],
        note="papers whose subfield_kw came from ai_keywords (drop title/summary fallback + Other)")
    RES["ladder_robustness"] = rob

    # convexity check
    po = pf.fepois(_fml("cites", sp["x"] + ["log_upvotes_sq"], sp["fe"]), data=df, vcov={"CRV1": "release_month"})
    ol = pf.feols(_fml("log_citations", sp["x"] + ["log_upvotes_sq"], sp["fe"]), data=df, vcov={"CRV1": "release_month"})
    nb = fit_nb(df, "cites", sp["x"] + ["log_upvotes_sq"], sp["fe"])
    RES["ladder_convexity"] = {
        "note": "M4 + log_upvotes^2. Positive squared term = elasticity rises with attention; Poisson-QMLE "
                "weights papers by their expected count, so it is dominated by the top tail and returns a "
                "larger average IRR than NB2 (which down-weights high counts) or log-OLS (geometric mean).",
        "poisson": {"beta_lin": fnum(po.coef()["log_upvotes"], 4), "beta_sq": fnum(po.coef()["log_upvotes_sq"], 4),
                    "se_sq": fnum(po.se()["log_upvotes_sq"], 4)},
        "nb2": {"beta_lin": fnum(nb["model"].params["log_upvotes"], 4), "beta_sq": fnum(nb["model"].params["log_upvotes_sq"], 4),
                "se_sq": fnum(nb["model"].bse["log_upvotes_sq"], 4)},
        "ols": {"beta_lin": fnum(ol.coef()["log_upvotes"], 4), "beta_sq": fnum(ol.coef()["log_upvotes_sq"], 4),
                "se_sq": fnum(ol.se()["log_upvotes_sq"], 4)},
    }
    # slope by attention band (Poisson vs OLS)
    bands = {}
    for lab, lo, hi in [("upvotes<10", 0, 10), ("10-29", 10, 30), (">=30", 30, 10 ** 9)]:
        d = df[(df.upvotes >= lo) & (df.upvotes < hi)]
        po = fit_poisson(d, "cites", ["log_age", "tierB_prior", "tierB_yrs", "tierB_miss"], ["subfield_kw", "release_month"])
        ol = fit_ols(d, "log_citations", ["log_age", "tierB_prior", "tierB_yrs", "tierB_miss"], ["subfield_kw", "release_month"])
        bands[lab] = {"n": int(len(d)), "poisson_beta": fnum(po["beta"], 3), "ols_beta": fnum(ol["beta"], 3)}
    RES["ladder_convexity"]["slope_by_upvote_band"] = bands


def bootstrap_evalue_placebo(df):
    log("1b. Month-cluster block bootstrap (M4)")
    sp = SPECS["M4_tierB_prestige"]
    rng = np.random.default_rng(SEED)
    months = np.sort(df.release_month.unique())
    bp, bn, bo = [], [], []
    groups = {m: g for m, g in df.groupby("release_month")}
    for i in range(N_BOOT):
        draw = rng.choice(months, size=len(months), replace=True)
        parts = []
        for j, m in enumerate(draw):
            g = groups[m].copy()
            g["release_month"] = f"{m}_b{j}"   # keep the block's FE identity distinct
            g["month_id"] = j
            parts.append(g)
        d = pd.concat(parts, ignore_index=True)
        try:
            bp.append(fit_poisson(d, "cites", sp["x"], sp["fe"])["beta"])
            bo.append(fit_ols(d, "log_citations", sp["x"], sp["fe"])["beta"])
            nb = fit_nb(d, "cites", sp["x"], sp["fe"])
            bn.append(nb["beta"] if nb["converged"] else np.nan)
        except Exception as e:  # pragma: no cover
            log(f"   boot rep {i} failed: {e}")
        if (i + 1) % 100 == 0:
            log(f"   boot {i + 1}/{N_BOOT}")
    bn_arr = np.array(bn, dtype=float)
    RES["bootstrap_M4"] = {
        "note": "block bootstrap over release months (34 clusters), percentile 95% CI; each rep refits M4",
        "n_reps": N_BOOT,
        "poisson_beta_ci95": ci_pct(bp), "poisson_irr2x_ci95": [fnum(np.exp(v * np.log(2)), 3) for v in ci_pct(bp)],
        "poisson_beta_sd": fnum(np.std(bp), 4),
        "nb2_beta_ci95": ci_pct(bn_arr), "nb2_irr2x_ci95": [fnum(np.exp(v * np.log(2)), 3) for v in ci_pct(bn_arr)],
        "nb2_reps_converged": int(np.isfinite(bn_arr).sum()),
        "ols_beta_ci95": ci_pct(bo), "ols_beta_sd": fnum(np.std(bo), 4),
    }
    log(f"   Poisson CI {RES['bootstrap_M4']['poisson_beta_ci95']}, NB CI {RES['bootstrap_M4']['nb2_beta_ci95']}, OLS CI {RES['bootstrap_M4']['ols_beta_ci95']}")

    # ---- E-value -------------------------------------------------------------
    def ev(rr):
        rr = float(rr)
        if rr < 1:
            rr = 1 / rr
        return rr + np.sqrt(rr * (rr - 1))
    evs = {}
    for kind in ["poisson_qmle", "nb2"]:
        r = RES["ladder"]["M4_tierB_prestige"][kind]
        evs[kind] = {"irr_per_doubling": r["irr_per_doubling"], "e_value_point": fnum(ev(r["irr_per_doubling"]), 3),
                     "irr_ci_lower": r["irr_per_doubling_ci95"][0], "e_value_ci_lower": fnum(ev(r["irr_per_doubling_ci95"][0]), 3)}
    r = RES["ladder"]["M4_tierB_prestige"]["ols_log1p"]
    evs["ols_log1p_ratio"] = {"ratio_per_doubling": r["ratio_1p_cites_per_doubling"],
                              "e_value_point": fnum(ev(r["ratio_1p_cites_per_doubling"]), 3),
                              "e_value_ci_lower": fnum(ev(r["ratio_1p_cites_per_doubling_ci95"][0]), 3)}
    evs["note"] = ("VanderWeele–Ding E-value on the risk-ratio scale (IRR per doubling of upvotes). Minimum strength "
                   "of association an unmeasured confounder needs with BOTH attention and citations to explain "
                   "the estimate away. Not a causal claim — a sensitivity summary.")
    RES["e_value_M4"] = evs

    # ---- placebo: permute upvotes within month × subfield cells ---------------
    log("1c. Placebo permutations")
    real_p = RES["ladder"]["M4_tierB_prestige"]["poisson_qmle"]["beta"]
    real_o = RES["ladder"]["M4_tierB_prestige"]["ols_log1p"]["beta"]
    d = df.copy()
    pb, ob = [], []
    cell = d.groupby(["release_month", "subfield_kw"]).ngroup().values
    for i in range(N_PLACEBO):
        perm = d.log_upvotes.values.copy()
        for c in np.unique(cell):
            idx = np.where(cell == c)[0]
            perm[idx] = perm[rng.permutation(idx)]
        d["log_upvotes"] = perm
        pb.append(fit_poisson(d, "cites", sp["x"], sp["fe"])["beta"])
        ob.append(fit_ols(d, "log_citations", sp["x"], sp["fe"])["beta"])
    pb, ob = np.array(pb), np.array(ob)
    RES["placebo_permutation_M4"] = {
        "note": "log_upvotes permuted within release_month × subfield_kw cells; M4 refit each time",
        "n_perm": N_PLACEBO,
        "poisson": {"placebo_beta_mean": fnum(pb.mean(), 4), "placebo_beta_sd": fnum(pb.std(), 4),
                    "placebo_beta_ci95": ci_pct(pb), "placebo_beta_max": fnum(pb.max(), 4),
                    "placebo_irr2x_range": [fnum(np.exp(pb.min() * np.log(2)), 3), fnum(np.exp(pb.max() * np.log(2)), 3)],
                    "real_beta": fnum(real_p, 4), "real_irr2x": fnum(np.exp(real_p * np.log(2)), 3),
                    "share_placebo_ge_real": fnum((pb >= real_p).mean(), 4),
                    "z_real_vs_placebo": fnum((real_p - pb.mean()) / pb.std(), 1)},
        "ols": {"placebo_beta_mean": fnum(ob.mean(), 4), "placebo_beta_sd": fnum(ob.std(), 4),
                "placebo_beta_ci95": ci_pct(ob), "real_beta": fnum(real_o, 4),
                "share_placebo_ge_real": fnum((ob >= real_o).mean(), 4),
                "z_real_vs_placebo": fnum((real_o - ob.mean()) / ob.std(), 1)},
    }
    log(f"   placebo Poisson beta mean {pb.mean():.4f} sd {pb.std():.4f} max {pb.max():.4f}; real {real_p:.3f}")
    # variant with month × subfield INTERACTED FE: absorbs the cell means that a within-cell permutation preserves,
    # so the placebo distribution should be centred at 0 (the additive-FE placebo is centred slightly above 0)
    fe_cell = ["release_month^subfield_kw", "dow"]
    d = df.copy()
    real_pc = fit_poisson(d, "cites", sp["x"], fe_cell)["beta"]
    real_oc = fit_ols(d, "log_citations", sp["x"], fe_cell)["beta"]
    pbc, obc = [], []
    for i in range(min(N_PLACEBO, 100)):
        perm = d.log_upvotes.values.copy()
        for c in np.unique(cell):
            idx = np.where(cell == c)[0]
            perm[idx] = perm[rng.permutation(idx)]
        d["log_upvotes"] = perm
        pbc.append(fit_poisson(d, "cites", sp["x"], fe_cell)["beta"])
        obc.append(fit_ols(d, "log_citations", sp["x"], fe_cell)["beta"])
    pbc, obc = np.array(pbc), np.array(obc)
    RES["placebo_permutation_M4"]["additive_fe_note"] = (
        "With additive month + subfield FE the placebo mean is slightly positive because permuting within month × subfield "
        "cells preserves cell-mean upvotes, and cell means of upvotes and citations co-move beyond the additive FE. "
        "The cell-FE variant below (release_month × subfield_kw interacted FE) removes this and is centred at ~0.")
    RES["placebo_permutation_M4"]["cell_fe_variant"] = {
        "n_perm": int(len(pbc)), "fixed_effects": fe_cell,
        "poisson": {"real_beta": fnum(real_pc, 4), "real_irr2x": fnum(np.exp(real_pc * np.log(2)), 3), "placebo_beta_mean": fnum(pbc.mean(), 4),
                    "placebo_beta_sd": fnum(pbc.std(), 4), "placebo_beta_ci95": ci_pct(pbc), "share_placebo_ge_real": fnum((pbc >= real_pc).mean(), 4)},
        "ols": {"real_beta": fnum(real_oc, 4), "placebo_beta_mean": fnum(obc.mean(), 4), "placebo_beta_sd": fnum(obc.std(), 4),
                "placebo_beta_ci95": ci_pct(obc), "share_placebo_ge_real": fnum((obc >= real_oc).mean(), 4)}}
    log(f"   cell-FE placebo: Poisson mean {pbc.mean():.4f} sd {pbc.std():.4f} (real {real_pc:.3f}); OLS mean {obc.mean():.4f} (real {real_oc:.3f})")

    # ---- placebo OUTCOME: reference count (fixed at submission) ---------------
    d = df[df.reference_count > 0].copy()
    d["refs"] = d.reference_count.round().astype(int)
    d["log_refs"] = np.log(d.refs)
    po = fit_poisson(d, "refs", sp["x"], sp["fe"])
    ol = fit_ols(d, "log_refs", sp["x"], sp["fe"])
    RES["placebo_outcome_reference_count_M4"] = {
        "note": "reference count is fixed at submission; attention should not 'predict' it if controls are adequate. "
                "n excludes 771 papers with reference_count = 0 (S2 parse gaps).",
        "n": int(len(d)),
        "poisson": pack(po, "poisson"), "ols_log": pack(ol, "ols"),
        "ratio_to_citation_effect_ols": fnum(ol["beta"] / real_o, 3),
    }
    log(f"   reference-count placebo: OLS beta {ol['beta']:.4f} (se {ol['se']:.4f}) = {ol['beta']/real_o:.1%} of citation elasticity")


def hierarchical(df):
    log("1d. Hierarchical MixedLM (random intercept + slope by subfield_kw)")
    f = "log_citations ~ log_upvotes + log_age + tierB_prior + tierB_yrs + tierB_miss + C(release_month) + C(dow)"
    m_slope = smf.mixedlm(f, df, groups=df["subfield_kw"], re_formula="~log_upvotes").fit(method="lbfgs", maxiter=1000, reml=True)
    m_int = smf.mixedlm(f, df, groups=df["subfield_kw"]).fit(method="lbfgs", maxiter=1000, reml=True)
    # ML fits for the LR test of the random slope
    ml_slope = smf.mixedlm(f, df, groups=df["subfield_kw"], re_formula="~log_upvotes").fit(method="lbfgs", maxiter=1000, reml=False)
    ml_int = smf.mixedlm(f, df, groups=df["subfield_kw"]).fit(method="lbfgs", maxiter=1000, reml=False)
    lr = 2 * (ml_slope.llf - ml_int.llf)
    # boundary-corrected p (50:50 mixture of chi2_1 and chi2_2)
    p_lr = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
    fe = float(m_slope.fe_params["log_upvotes"])
    fe_se = float(m_slope.bse_fe["log_upvotes"])
    cov = m_slope.cov_re
    slopes = {}
    for g, re in m_slope.random_effects.items():
        cv = m_slope.random_effects_cov[g]
        s = fe + float(re["log_upvotes"])
        se = float(np.sqrt(cv.loc["log_upvotes", "log_upvotes"] + fe_se ** 2))
        slopes[g] = {"slope": fnum(s, 3), "se_approx": fnum(se, 3), "ci95": [fnum(s - 1.96 * se, 3), fnum(s + 1.96 * se, 3)],
                     "intercept_re": fnum(re["Group"], 3), "n": int((df.subfield_kw == g).sum())}
    slopes = dict(sorted(slopes.items(), key=lambda kv: -kv[1]["slope"]))
    RES["hierarchical_mixedlm"] = {
        "spec": f + "  | groups = subfield_kw, re_formula = ~log_upvotes (REML)",
        "why_not_poisson_glmm": ("statsmodels has no maximum-likelihood Poisson/NB GLMM with correlated random "
                                 "intercept+slope and cluster-robust SEs (only a variational-Bayes PoissonBayesMixedGLM); "
                                 "the count-scale association is therefore estimated with FE Poisson/NB (ladder) and the "
                                 "hierarchical/heterogeneity part with this linear mixed model on log1p citations, as "
                                 "in the June-11 analysis but with the uniform taxonomy and month FE."),
        "converged": bool(m_slope.converged), "n": int(m_slope.nobs), "n_groups": int(len(slopes)),
        "fixed_slope_log_upvotes": fnum(fe, 4), "fixed_slope_se": fnum(fe_se, 4),
        "fixed_slope_ci95": [fnum(fe - 1.96 * fe_se, 4), fnum(fe + 1.96 * fe_se, 4)],
        "fixed_log_age": fnum(m_slope.fe_params["log_age"], 3),
        "fixed_tierB_prior": fnum(m_slope.fe_params["tierB_prior"], 4),
        "variance_components": {"intercept_var": fnum(cov.iloc[0, 0], 5), "slope_var": fnum(cov.iloc[1, 1], 6),
                                "intercept_slope_cov": fnum(cov.iloc[0, 1], 6), "slope_sd": fnum(np.sqrt(max(cov.iloc[1, 1], 0)), 4),
                                "intercept_sd": fnum(np.sqrt(max(cov.iloc[0, 0], 0)), 4), "residual_var": fnum(m_slope.scale, 4),
                                "icc_intercept_only_model": fnum(m_int.cov_re.iloc[0, 0] / (m_int.cov_re.iloc[0, 0] + m_int.scale), 4)},
        "lr_test_random_slope": {"lr_stat": fnum(lr, 3), "p_boundary_mixture": fnum(p_lr, 4),
                                 "note": "ML fits; 50:50 chi2(1)/chi2(2) mixture for the boundary"},
        "subfield_slopes": slopes,
        "slope_range": [min(v["slope"] for v in slopes.values()), max(v["slope"] for v in slopes.values())],
    }
    log(f"   fixed slope {fe:.3f} (se {fe_se:.3f}); slope SD {np.sqrt(cov.iloc[1,1]):.3f}; range {RES['hierarchical_mixedlm']['slope_range']}; LR p {p_lr:.3f}")


def dose_response(df):
    log("1e. Dose-response deciles")
    d = df.copy()
    d["dec"] = pd.qcut(d.upvotes.rank(method="first"), 10, labels=False) + 1
    # age/field/month-adjusted log citations (residual + grand mean)
    adj = smf.ols("log_citations ~ log_age + C(subfield_kw) + C(release_month)", data=d).fit()
    d["log_cit_adj"] = adj.resid + d.log_citations.mean()
    rows = []
    for k, g in d.groupby("dec"):
        n = len(g)
        m, s = g.log_citations.mean(), g.log_citations.std() / np.sqrt(n)
        ma, sa = g.log_cit_adj.mean(), g.log_cit_adj.std() / np.sqrt(n)
        rows.append({"decile": int(k), "n": int(n), "upvotes_min": int(g.upvotes.min()), "upvotes_max": int(g.upvotes.max()),
                     "median_upvotes": fnum(g.upvotes.median(), 1),
                     "mean_log1p_cites": fnum(m, 3), "ci95": [fnum(m - 1.96 * s, 3), fnum(m + 1.96 * s, 3)],
                     "geo_mean_1p_cites": fnum(np.exp(m), 1),
                     "median_cites": fnum(g.citation_count.median(), 1),
                     "mean_log1p_cites_adjusted": fnum(ma, 3), "ci95_adjusted": [fnum(ma - 1.96 * sa, 3), fnum(ma + 1.96 * sa, 3)],
                     "share_zero_cites": fnum((g.citation_count == 0).mean(), 3),
                     "share_top_decile_cites_in_quarter": None})
    # top-decile-in-quarter share
    d["rq"] = d.release_month.str[:4] + "Q" + ((d.release_month.str[5:7].astype(int) - 1) // 3 + 1).astype(str)
    d["top10q"] = d.groupby("rq").citation_count.rank(pct=True) >= 0.9
    for r in rows:
        r["share_top_decile_cites_in_quarter"] = fnum(d.loc[d.dec == r["decile"], "top10q"].mean(), 3)
    RES["dose_response_deciles"] = {"note": "deciles of raw upvotes (ties split by order); adjusted = residual of "
                                            "log1p cites on log age + subfield_kw + month FE, plus grand mean",
                                    "rows": rows,
                                    "spearman_upvotes_cites": fnum(stats.spearmanr(df.upvotes, df.citation_count)[0], 3)}


# =============================================================================
# 2. SELECTION ADJUSTMENT (high vs low attention tertile)
# =============================================================================
def smd(x_t, x_c, w_t=None, w_c=None):
    w_t = np.ones(len(x_t)) if w_t is None else np.asarray(w_t, float)
    w_c = np.ones(len(x_c)) if w_c is None else np.asarray(w_c, float)
    mt, mc = np.average(x_t, weights=w_t), np.average(x_c, weights=w_c)
    vt = np.average((x_t - mt) ** 2, weights=w_t)
    vc = np.average((x_c - mc) ** 2, weights=w_c)
    s = np.sqrt((vt + vc) / 2)
    return 0.0 if s == 0 else (mt - mc) / s


def selection(df):
    log("2. Selection adjustment: high vs low attention tertile")
    rng = np.random.default_rng(SEED + 1)
    q_lo, q_hi = df.upvotes.quantile(1 / 3), df.upvotes.quantile(2 / 3)
    d = df[(df.upvotes >= q_hi) | (df.upvotes <= q_lo)].copy().reset_index(drop=True)
    d["treat"] = (d.upvotes >= q_hi).astype(int)
    d["prest_q"] = pd.qcut(d.max_prior_papers_true.rank(method="first"), 5, labels=False)
    d["prest_q"] = d.prest_q.fillna(-1).astype(int)  # missing Tier B -> own bin
    y = d.log_citations.values
    t = d.treat.values
    naive = y[t == 1].mean() - y[t == 0].mean()

    cont = ["tierB_prior", "tierB_yrs", "log_age"]
    sf_d = pd.get_dummies(d.subfield_kw, prefix="sf").astype(float)
    mo_d = pd.get_dummies(d.release_month, prefix="m").astype(float)
    Xps = pd.concat([d[cont + ["tierB_miss"]], sf_d.iloc[:, 1:], mo_d.iloc[:, 1:]], axis=1).astype(float)
    bal_vars = pd.concat([d[cont + ["tierB_miss"]], sf_d], axis=1)

    def fit_ps(X, tt):
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1e4, max_iter=5000).fit(sc.transform(X), tt)
        p = lr.predict_proba(sc.transform(X))[:, 1]
        return np.clip(p, 1e-4, 1 - 1e-4)

    def match_att(ps, yy, tt, caliper_sd=0.2, return_idx=False):
        lps = np.log(ps / (1 - ps))
        it, ic = np.where(tt == 1)[0], np.where(tt == 0)[0]
        nn = NearestNeighbors(n_neighbors=1).fit(lps[ic, None])
        dist, j = nn.kneighbors(lps[it, None])
        cal = caliper_sd * lps.std()
        ok = dist[:, 0] <= cal
        mi, mj = it[ok], ic[j[ok, 0]]
        att = float((yy[mi] - yy[mj]).mean())
        if return_idx:
            return att, mi, mj
        return att

    def ipw_att(ps, yy, tt, trim=0.99):
        w = np.where(tt == 1, 1.0, ps / (1 - ps))
        # stabilise (constant factor P(T=0)/P(T=1)) — irrelevant after normalisation, kept for transparency
        w = np.where(tt == 1, w, w * (1 - tt.mean()) / tt.mean())
        cap = np.quantile(w[tt == 0], trim)
        w = np.where(tt == 0, np.minimum(w, cap), w)
        return float(np.average(yy[tt == 1]) - np.average(yy[tt == 0], weights=w[tt == 0])), w

    def cem_att(cells, yy, tt):
        dd = pd.DataFrame({"cell": cells, "y": yy, "t": tt})
        g = dd.groupby("cell").agg(nt=("t", "sum"), nc=("t", lambda s: (s == 0).sum()))
        good = g[(g.nt > 0) & (g.nc > 0)].index
        dd = dd[dd.cell.isin(good)]
        # ATT weights: controls weighted to the treated cell distribution
        w = np.where(dd.t == 1, 1.0, dd.cell.map(g.nt / g.nc).values)
        att = float(np.average(dd.y[dd.t == 1]) - np.average(dd.y[dd.t == 0], weights=w[dd.t.values == 0]))
        return att, int((dd.t == 1).sum()), int((dd.t == 0).sum()), int(len(good))

    ps = fit_ps(Xps, t)
    # variant: PS additionally on the 2026-measured h-index (as v2 did) — LEAKY, for comparison only
    Xps_h = pd.concat([Xps, d[["log_max_h", "log_last_h", "h_miss"]]], axis=1).astype(float)
    ps_h = fit_ps(Xps_h, t)
    att_m_h = match_att(ps_h, y, t)
    att_w_h = ipw_att(ps_h, y, t)[0]
    att_m, mi, mj = match_att(ps, y, t, return_idx=True)
    att_w, w = ipw_att(ps, y, t)
    cells = d.release_month + "|" + d.subfield_kw + "|" + d.prest_q.astype(str)
    att_c, n_t_c, n_c_c, n_cells = cem_att(cells.values, y, t)

    # balance table
    it, ic = np.where(t == 1)[0], np.where(t == 0)[0]
    balance = []
    for v in bal_vars.columns:
        x = bal_vars[v].values.astype(float)
        wc_m = pd.Series(mj).value_counts()
        balance.append({"covariate": v,
                        "smd_before": fnum(smd(x[it], x[ic]), 3),
                        "smd_after_matching": fnum(smd(x[mi], x[wc_m.index.values], None, wc_m.values), 3),
                        "smd_after_ipw": fnum(smd(x[it], x[ic], None, w[ic]), 3)})
    # month balance summary
    mo_smds = []
    for v in mo_d.columns:
        x = mo_d[v].values
        wc_m = pd.Series(mj).value_counts()
        mo_smds.append([abs(smd(x[it], x[ic])), abs(smd(x[mi], x[wc_m.index.values], None, wc_m.values)), abs(smd(x[it], x[ic], None, w[ic]))])
    mo_smds = np.array(mo_smds)

    # bootstrap (paper-level, stratified by treatment; PS refit + rematch each rep)
    bm, bw, bc, bnv, bmh, bwh = [], [], [], [], [], []
    for i in range(N_BOOT_MATCH):
        idx = np.concatenate([rng.choice(it, len(it)), rng.choice(ic, len(ic))])
        Xb, yb, tb = Xps.iloc[idx].values, y[idx], t[idx]
        psb = fit_ps(Xb, tb)
        bm.append(match_att(psb, yb, tb))
        bw.append(ipw_att(psb, yb, tb)[0])
        psbh = fit_ps(Xps_h.iloc[idx].values, tb)
        bmh.append(match_att(psbh, yb, tb))
        bwh.append(ipw_att(psbh, yb, tb)[0])
        bc.append(cem_att(cells.values[idx], yb, tb)[0])
        bnv.append(yb[tb == 1].mean() - yb[tb == 0].mean())
        if (i + 1) % 100 == 0:
            log(f"   match boot {i + 1}/{N_BOOT_MATCH}")

    RES["selection_adjustment"] = {
        "design": {"treated": f"upvotes >= {q_hi:.0f} (top tertile)", "control": f"upvotes <= {q_lo:.0f} (bottom tertile)",
                   "treated_threshold": fnum(q_hi, 1), "control_threshold": fnum(q_lo, 1),
                   "middle_tertile_dropped": True, "n_treated": int(t.sum()), "n_control": int((t == 0).sum()),
                   "outcome": "log1p citations", "ps_model": "logit(treat) ~ Tier B prior papers (log1p) + years active + "
                                                             "Tier B-missing flag + log age + subfield_kw dummies + release_month dummies",
                   "matching": "1:1 nearest neighbour on logit(PS), with replacement, caliper 0.2 SD of logit(PS)",
                   "ipw": "ATT weights ps/(1-ps) for controls, stabilised by P(T=0)/P(T=1), trimmed at the 99th pct, normalised (Hajek)",
                   "cem": "exact cells release_month × subfield_kw × Tier B prior-papers quintile (missing = own bin)",
                   "bootstrap": f"{N_BOOT_MATCH} paper-level reps stratified by treatment; PS refit + rematch each rep"},
        "naive_gap": {"log_pts": fnum(naive, 3), "ci95": ci_pct(bnv), "ratio": fnum(np.exp(naive), 2)},
        "ps_matching": {"att_log_pts": fnum(att_m, 3), "ci95": ci_pct(bm), "ratio": fnum(np.exp(att_m), 2),
                        "n_treated_matched": int(len(mi)), "share_treated_within_caliper": fnum(len(mi) / t.sum(), 3),
                        "n_unique_controls_used": int(len(np.unique(mj)))},
        "ipw": {"att_log_pts": fnum(att_w, 3), "ci95": ci_pct(bw), "ratio": fnum(np.exp(att_w), 2),
                "ess_controls": fnum(w[ic].sum() ** 2 / (w[ic] ** 2).sum(), 1), "max_control_weight": fnum(w[ic].max(), 2)},
        "cem": {"att_log_pts": fnum(att_c, 3), "ci95": ci_pct(bc), "ratio": fnum(np.exp(att_c), 2),
                "n_treated_matched": n_t_c, "n_controls_matched": n_c_c, "n_cells": n_cells,
                "share_treated_matched": fnum(n_t_c / t.sum(), 3)},
        "variant_ps_with_hindex_LEAKY": {
            "note": "PS also includes log1p max/last author h-index measured 2026-06-11 (as the June-11 v2 analysis did). "
                    "h-index is downstream of citations, so this over-adjusts; shown to explain the gap to v2's 1.15 / 0.75.",
            "ps_matching_att": fnum(att_m_h, 3), "ps_matching_ci95": ci_pct(bmh),
            "ipw_att": fnum(att_w_h, 3), "ipw_ci95": ci_pct(bwh)},
        "balance": balance,
        "balance_month_dummies_max_abs_smd": {"before": fnum(mo_smds[:, 0].max(), 3), "after_matching": fnum(mo_smds[:, 1].max(), 3),
                                              "after_ipw": fnum(mo_smds[:, 2].max(), 3)},
        "balance_max_abs_smd_all_listed": {"before": fnum(max(abs(b["smd_before"]) for b in balance), 3),
                                           "after_matching": fnum(max(abs(b["smd_after_matching"]) for b in balance), 3),
                                           "after_ipw": fnum(max(abs(b["smd_after_ipw"]) for b in balance), 3)},
    }
    s = RES["selection_adjustment"]
    log(f"   naive {naive:.3f}; matched {att_m:.3f} {s['ps_matching']['ci95']}; IPW {att_w:.3f} {s['ipw']['ci95']}; CEM {att_c:.3f} {s['cem']['ci95']}")


# =============================================================================
# 3. NEVER-TRENDING CONTROL COMPARISON
# =============================================================================
def control_comparison(df, ou):
    log("3. Never-trending control comparison")
    rng = np.random.default_rng(SEED + 2)
    meta = pd.read_csv(CTRL_META, dtype={"arxiv_id_clean": str})
    s2 = pd.read_csv(CTRL_S2, dtype={"arxiv_id_clean": str})
    c = meta.merge(s2, on="arxiv_id_clean", how="inner", validate="1:1")
    n_raw = len(c)
    c = c[c.ss_found == 1].copy()
    n_found = len(c)
    rel = pd.to_datetime(c.published_v1, errors="coerce", utc=True).dt.tz_localize(None)
    c["age_months"] = (SNAPSHOT - rel).dt.days / 30.44
    c["release_month"] = rel.dt.to_period("M").astype(str)
    c["day_of_month"] = rel.dt.day
    c["dow_ctrl"] = rel.dt.day_name()
    facts = {
        "construction": ("Project/scripts/10: for each month × stratum {cs.CL, cs.CV, cs.LG} the first ~40 arXiv submissions "
                         "by submittedDate ascending (i.e. the first day(s) of the month), de-duplicated across strata, "
                         "minus ids that ever trended on HF; S2 citations/h-index fetched 2026-06-11."),
        "n_after_dedup_and_hf_removal": int(n_raw), "n_s2_found": int(n_found),
        "share_submitted_day1": fnum((c.day_of_month == 1).mean(), 3),
        "share_submitted_day1to3": fnum((c.day_of_month <= 3).mean(), 3),
        "share_weekend_submissions": fnum(c.dow_ctrl.isin(["Saturday", "Sunday"]).mean(), 3),
        "strata_counts": {k: int(v) for k, v in c.strat_cat.value_counts().items()},
        "primary_category_top": {k: int(v) for k, v in c.primary_category.value_counts().head(8).items()},
        "share_primary_outside_stratum": fnum((c.primary_category != c.strat_cat).mean(), 3),
        "months": [str(c.release_month.min()), str(c.release_month.max())],
    }
    c = c[(c.age_months >= 5) & (c.age_months <= 40)].copy()
    c["citation_count"] = pd.to_numeric(c.citation_count, errors="coerce")
    c = c[c.citation_count.notna()].copy()
    c["log_citations"] = np.log1p(c.citation_count)
    facts["n_analysis_5_40_months"] = int(len(c))
    facts["citations"] = {"median": fnum(c.citation_count.median(), 1), "p75": fnum(c.citation_count.quantile(.75), 1),
                          "p90": fnum(c.citation_count.quantile(.9), 1), "share_zero": fnum((c.citation_count == 0).mean(), 3),
                          "share_ge100": fnum((c.citation_count >= 100).mean(), 3), "median_max_hindex": fnum(c.max_hindex.median(), 1),
                          "median_age_months": fnum(c.age_months.median(), 1)}
    # subfield for controls: same keyword rules on the TITLE (only text available), else category-mapped
    c["sf_title"] = c.title.map(kw_label)
    c["subfield_ctrl"] = c.sf_title.fillna(c.primary_category.map(CAT_TO_KW)).fillna("Other")
    c["sf_source"] = np.where(c.sf_title.notna(), "title_keyword_rules", np.where(c.primary_category.map(CAT_TO_KW).notna(), "arxiv_category_map", "none"))
    facts["control_subfield_source"] = {k: int(v) for k, v in c.sf_source.value_counts().items()}
    facts["control_subfield_counts"] = {k: int(v) for k, v in c.subfield_ctrl.value_counts().items()}
    facts["control_subfield_note"] = ("Controls have no ai_keywords/abstract text, only titles: the SAME 13-level keyword "
                                      "rule set was applied to titles (labels ~53%); the remainder is mapped from the arXiv "
                                      "primary category to the nearest keyword label (cs.CV/eess.IV -> Vision-Perception, "
                                      "cs.CL/cs.LG/stat.ML/cs.AI -> LLM-core, cs.RO -> Robotics/Embodied, cs.SD/eess.AS -> "
                                      "Speech/Audio, cs.IR -> RAG/Retrieval, cs.SE -> Code/Math, else Other). On trending "
                                      "papers the title-only rule agrees with subfield_kw for ~64% of the labelled ones, so "
                                      "the subfield match is coarse; a 4-family variant is also reported.")
    c["family"] = c.subfield_ctrl.map(FAMILY).fillna("Other")

    t = df[["arxiv_id_clean", "citation_count", "log_citations", "release_month", "age_months", "max_hindex", "subfield_kw", "upvotes"]].copy()
    t = t.merge(ou[["arxiv_id_clean", "attention_resid_pct", "impact_resid_pct", "label"]], on="arxiv_id_clean")
    t["family"] = t.subfield_kw.map(FAMILY)
    t = t.rename(columns={"subfield_kw": "sf"})
    c = c.rename(columns={"subfield_ctrl": "sf"})
    both = pd.concat([t.assign(trend=1), c[["arxiv_id_clean", "citation_count", "log_citations", "release_month", "age_months", "max_hindex", "sf", "family"]].assign(trend=0)],
                     ignore_index=True)
    both = both[both.max_hindex.notna()].copy()
    edges = np.unique(np.quantile(both.max_hindex, [0, .2, .4, .6, .8, 1.0]))
    both["hbin"] = pd.cut(both.max_hindex, bins=edges, include_lowest=True, labels=False)
    trend_med = float(both.loc[both.trend == 1, "citation_count"].median())
    facts["trending"] = {"n": int((both.trend == 1).sum()), "median_cites": trend_med,
                         "p90": fnum(both.loc[both.trend == 1, "citation_count"].quantile(.9), 1),
                         "share_zero": fnum((both.loc[both.trend == 1, "citation_count"] == 0).mean(), 3),
                         "share_ge100": fnum((both.loc[both.trend == 1, "citation_count"] >= 100).mean(), 3),
                         "median_max_hindex": fnum(both.loc[both.trend == 1, "max_hindex"].median(), 1)}
    facts["hindex_bin_edges"] = [fnum(e, 1) for e in edges]

    Y = both.log_citations.values
    TR = both.trend.values.astype(int)
    CELLS = {}
    for keys in [("release_month", "sf", "hbin"), ("release_month", "family", "hbin"), ("release_month", "hbin")]:
        CELLS[keys] = both.groupby(list(keys), sort=False).ngroup().values

    def cem_arr(cell, y, tr, sel=None):
        """exact-cell ATT-weighted gap; sel = boolean over rows selecting which trending rows count as treated"""
        if sel is not None:
            keep = (tr == 0) | sel
            cell, y, tr = cell[keep], y[keep], tr[keep]
        ncell = cell.max() + 1
        nt = np.bincount(cell[tr == 1], minlength=ncell)
        nc = np.bincount(cell[tr == 0], minlength=ncell)
        good = (nt > 0) & (nc > 0)
        kt = good[cell] & (tr == 1)
        kc = good[cell] & (tr == 0)
        w = (nt / np.maximum(nc, 1))[cell[kc]]
        gap = float(y[kt].mean() - np.average(y[kc], weights=w))
        return gap, int(kt.sum()), int(kc.sum()), int(good.sum())

    IT, IC = np.where(TR == 1)[0], np.where(TR == 0)[0]

    def cem(dd, keys, mask_t=None):
        sel = None if mask_t is None else mask_t.reindex(both.index).fillna(False).values.astype(bool)
        return cem_arr(CELLS[tuple(keys)], Y, TR, sel)

    def boot_cem(dd, keys, mask_t=None, reps=N_BOOT):
        cell = CELLS[tuple(keys)]
        sel = None if mask_t is None else mask_t.reindex(both.index).fillna(False).values.astype(bool)
        out = []
        for _ in range(reps):
            idx = np.concatenate([rng.choice(IT, len(IT)), rng.choice(IC, len(IC))])
            out.append(cem_arr(cell[idx], Y[idx], TR[idx], None if sel is None else sel[idx])[0])
        return ci_pct(out)

    naive = float(both.loc[both.trend == 1, "log_citations"].mean() - both.loc[both.trend == 0, "log_citations"].mean())
    # naive bootstrap
    it, ic = np.where(both.trend.values == 1)[0], np.where(both.trend.values == 0)[0]
    nv = [both.log_citations.values[rng.choice(it, len(it))].mean() - both.log_citations.values[rng.choice(ic, len(ic))].mean() for _ in range(N_BOOT)]
    prem = {"naive": {"gap_log_pts": fnum(naive, 3), "ci95": ci_pct(nv), "ratio": fnum(np.exp(naive), 2),
                      "median_ratio": fnum(trend_med / facts["citations"]["median"], 2)}}
    for name, keys, note in [
        ("cem_month_subfield13_hbin", ["release_month", "sf", "hbin"], "exact: release_month × 13-level subfield × h-index quintile [main]"),
        ("cem_month_family_hbin", ["release_month", "family", "hbin"], "exact: release_month × 4-family × h-index quintile"),
        ("cem_month_hbin", ["release_month", "hbin"], "exact: release_month × h-index quintile (no field)"),
    ]:
        gap, nt, nc, ncell = cem(both, keys)
        prem[name] = {"note": note, "gap_log_pts": fnum(gap, 3), "ci95": boot_cem(both, keys), "ratio": fnum(np.exp(gap), 2),
                      "ratio_ci95": None, "n_trending_matched": nt, "n_control_matched": nc, "n_cells": ncell,
                      "share_trending_matched": fnum(nt / (both.trend == 1).sum(), 3)}
        prem[name]["ratio_ci95"] = [fnum(np.exp(v), 2) for v in prem[name]["ci95"]]
        log(f"   {name}: gap {gap:.3f} (x{np.exp(gap):.2f}) matched {nt}/{(both.trend==1).sum()}")
    # regression version with cluster by month
    both["sf_c"] = both.sf.astype("category")
    reg = pf.feols("log_citations ~ trend | release_month + sf + hbin", data=both, vcov={"CRV1": "release_month"})
    prem["ols_fe_month_subfield_hbin"] = {"gap_log_pts": fnum(reg.coef()["trend"], 3),
                                          "ci95": [fnum(reg.coef()["trend"] - 1.96 * reg.se()["trend"], 3), fnum(reg.coef()["trend"] + 1.96 * reg.se()["trend"], 3)],
                                          "ratio": fnum(np.exp(reg.coef()["trend"]), 2), "n": int(reg._N)}

    # ---- honest low-attention comparison ---------------------------------------
    tt = both[both.trend == 1]
    low_res = tt.attention_resid_pct <= 1 / 3
    q_lo = df.upvotes.quantile(1 / 3)
    low_raw = tt.upvotes <= q_lo
    under = tt.label == "underrated"
    top_res = tt.attention_resid_pct >= 2 / 3

    def pct_in_background(sub, keys):
        """mean percentile of each trending paper's citation count within same-cell background papers (mid-rank ties)"""
        bg = both[both.trend == 0]
        bg_groups = {(k if isinstance(k, tuple) else (k,)): np.sort(g.citation_count.values) for k, g in bg.groupby(keys)}
        pcts = []
        kv = sub[keys].values
        cv = sub.citation_count.values
        for i in range(len(sub)):
            arr = bg_groups.get(tuple(kv[i]))
            if arr is None or len(arr) < 5:
                continue
            lo = np.searchsorted(arr, cv[i], side="left")
            hi = np.searchsorted(arr, cv[i], side="right")
            pcts.append((lo + 0.5 * (hi - lo)) / len(arr))
        pcts = np.array(pcts)
        bs = [np.mean(rng.choice(pcts, len(pcts))) for _ in range(N_BOOT)] if len(pcts) else [np.nan]
        return {"mean_percentile": fnum(pcts.mean(), 3), "ci95": ci_pct(bs), "median_percentile": fnum(np.median(pcts), 3),
                "share_above_background_median": fnum((pcts > 0.5).mean(), 3), "n_used": int(len(pcts))}

    lowatt = {}
    for gname, mask, desc in [
        ("bottom_tertile_attention_residual", low_res, "trending papers in the bottom tertile of the age/field/month-adjusted attention residual, regardless of citations [honest]"),
        ("bottom_tertile_raw_upvotes", low_raw, f"trending papers with upvotes <= {q_lo:.0f} (bottom tertile of raw upvotes), regardless of citations"),
        ("top_tertile_attention_residual", top_res, "trending papers in the top tertile of the adjusted attention residual (context)"),
        ("underrated_label_OUTCOME_SELECTED", under, "v2-style 'under-rated' = bottom-tertile attention AND top-tertile citation residual — selected on the outcome; shown only to document why the 91st-percentile claim was circular"),
    ]:
        sub = tt[mask.values]
        gap, nt, nc, ncell = cem(both, ["release_month", "sf", "hbin"], mask_t=pd.Series(mask.values, index=tt.index))
        ci = boot_cem(both, ["release_month", "sf", "hbin"], mask_t=pd.Series(mask.values, index=tt.index), reps=N_BOOT)
        lowatt[gname] = {"desc": desc, "n": int(len(sub)), "median_cites": fnum(sub.citation_count.median(), 1),
                         "median_upvotes": fnum(sub.upvotes.median(), 1),
                         "cem_gap_vs_background_log_pts": fnum(gap, 3), "cem_gap_ci95": ci, "cem_ratio": fnum(np.exp(gap), 2),
                         "n_matched": nt,
                         "percentile_in_background_same_month": pct_in_background(sub, ["release_month"]),
                         "percentile_in_background_same_month_family": pct_in_background(sub, ["release_month", "family"])}
        log(f"   {gname}: n={len(sub)} gap {gap:.3f} pct(month) {lowatt[gname]['percentile_in_background_same_month']['mean_percentile']}")
    # v2's crude number for reference (pooled, un-age-matched)
    bg_all = both.loc[both.trend == 0, "citation_count"].values
    und = tt.loc[under.values, "citation_count"].values
    lowatt["underrated_label_OUTCOME_SELECTED"]["v2_style_pooled_percentile_no_age_match"] = fnum((bg_all[None, :] < und[:, None]).mean(), 3)
    RES["control_comparison"] = {"control_sample_facts": facts, "trending_premium": prem, "low_attention_vs_background": lowatt}


# =============================================================================
# 4. OVER-/UNDER-RATED
# =============================================================================
def over_under(df):
    log("4. Over-/under-rated residual analysis")
    rng = np.random.default_rng(SEED + 3)
    d = df.copy()
    d["title"] = d.title.str.replace(r"\s+", " ", regex=True).str.strip()
    att = smf.ols("log_upvotes ~ log_age + C(subfield_kw) + C(release_month)", data=d).fit()
    imp = smf.ols("log_citations ~ log_age + C(subfield_kw) + C(release_month)", data=d).fit()
    d["attention_resid"] = att.resid
    d["impact_resid"] = imp.resid
    d["attention_resid_pct"] = d.attention_resid.rank(pct=True)
    d["impact_resid_pct"] = d.impact_resid.rank(pct=True)
    d["label"] = "neutral"
    d.loc[(d.attention_resid_pct >= 2 / 3) & (d.impact_resid_pct <= 1 / 3), "label"] = "overrated"
    d.loc[(d.attention_resid_pct <= 1 / 3) & (d.impact_resid_pct >= 2 / 3), "label"] = "underrated"
    d["resid_gap"] = d.attention_resid_pct - d.impact_resid_pct
    counts = d.label.value_counts().to_dict()
    out = {"definition": ("attention residual = log1p upvotes minus OLS fit on log age + subfield_kw FE + release_month FE; "
                          "impact residual = same for log1p citations; percentile ranks over all 11,344 papers. "
                          "overrated = attention pct >= 2/3 AND impact pct <= 1/3; underrated = attention pct <= 1/3 AND impact pct >= 2/3; else neutral."),
           "resid_model_r2": {"attention": fnum(att.rsquared, 3), "impact": fnum(imp.rsquared, 3)},
           "counts": {k: int(v) for k, v in counts.items()},
           "spearman_attention_impact_resid": fnum(stats.spearmanr(d.attention_resid, d.impact_resid)[0], 3)}

    # ---- CSV for the dashboard ------------------------------------------------
    keep = ["arxiv_id_clean", "title", "subfield_kw", "release_month", "release_year", "upvotes", "citation_count",
            "influential_citations", "age_months", "attention_resid_pct", "impact_resid_pct", "label",
            "max_prior_papers_true_w99", "max_years_active", "tierB_resolved", "max_hindex", "last_author_hindex",
            "n_authors", "has_github", "num_comments", "citation_repaired", "ai_keywords"]
    d[keep].to_csv(OUT_OU, index=False)
    log(f"   wrote {OUT_OU} ({len(d)} rows) counts={counts}")

    # ---- logit drivers (cluster by month) --------------------------------------
    cont = ["tierB_prior", "tierB_yrs", "log_max_h", "log_last_h", "log_n_authors", "title_n_words", "log_abs_chars"]
    bins = ["title_has_colon", "has_github"] + KW   # missing-flags excluded (perfect separation in the labelled groups)
    z = d.copy()
    for cn in cont:
        z[cn + "_z"] = (z[cn] - z[cn].mean()) / z[cn].std()
    sf_ref = d.subfield_kw.value_counts().idxmax()
    sf_levels = [s for s in sorted(d.subfield_kw.unique()) if s != sf_ref]
    for s in sf_levels:
        z["sf_" + re.sub(r"[^A-Za-z0-9]", "_", s)] = (z.subfield_kw == s).astype(int)
    sf_cols = ["sf_" + re.sub(r"[^A-Za-z0-9]", "_", s) for s in sf_levels]
    feats = [cn + "_z" for cn in cont] + bins + sf_cols
    feat_desc = {**{cn + "_z": f"{cn} (per SD)" for cn in cont}, **{b: b for b in bins}, **{c: f"subfield {s} (vs {sf_ref})" for c, s in zip(sf_cols, sf_levels)}}
    X = sm.add_constant(z[feats].astype(float))
    drivers = {}
    LEAKY = ["log_max_h_z", "log_last_h_z"]
    for name, yv, sub, drop in [("overrated_vs_rest", (z.label == "overrated").astype(int), np.ones(len(z), bool), []),
                                ("underrated_vs_rest", (z.label == "underrated").astype(int), np.ones(len(z), bool), []),
                                ("overrated_vs_underrated", (z.label == "overrated").astype(int), (z.label != "neutral").values, []),
                                ("overrated_vs_rest_NOLEAK", (z.label == "overrated").astype(int), np.ones(len(z), bool), LEAKY),
                                ("underrated_vs_rest_NOLEAK", (z.label == "underrated").astype(int), np.ones(len(z), bool), LEAKY),
                                ("overrated_vs_underrated_NOLEAK", (z.label == "overrated").astype(int), (z.label != "neutral").values, LEAKY)]:
        Xs, ys, gs = X.loc[sub, [c for c in X.columns if c not in drop]], yv[sub], z.month_id.values[sub]
        # drop constant / near-empty columns in the subsample
        keepc = [c for c in Xs.columns if c == "const" or (Xs[c].nunique() > 2) or (Xs[c].nunique() == 2 and Xs[c].sum() >= 5)]
        Xs = Xs[keepc]
        m = sm.Logit(ys, Xs).fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": gs})
        rows = []
        for cn in keepc:
            if cn == "const":
                continue
            b, se = float(m.params[cn]), float(m.bse[cn])
            rows.append({"feature": feat_desc.get(cn, cn), "or": fnum(np.exp(b), 3), "or_ci95": [fnum(np.exp(b - 1.96 * se), 3), fnum(np.exp(b + 1.96 * se), 3)],
                         "beta": fnum(b, 3), "se": fnum(se, 3), "p": fnum(m.pvalues[cn], 4)})
        rows_sorted = sorted(rows, key=lambda r: (r["p"] if r["p"] is not None else 1.0))
        drivers[name] = {"n": int(len(ys)), "n_events": int(ys.sum()), "pseudo_r2": fnum(m.prsquared, 3), "converged": bool(m.mle_retvals.get("converged", True)),
                         "note": "statsmodels Logit, cluster-robust by release_month; continuous features standardised (OR per SD); "
                                 + ("h-index features EXCLUDED (leakage-free driver model)" if drop else
                                    "h-index features are 2026-measured (LEAKY, Matthew effect on the outcome side)"),
                         "rows": rows_sorted}
        log(f"   {name}: n={len(ys)} events={ys.sum()} top: " + ", ".join(f"{r['feature']}({r['or']})" for r in rows_sorted[:5]))
    out["logit_drivers"] = drivers

    # ---- prestige asymmetry with bootstrap CIs ---------------------------------
    def med_ci(x, reps=1000):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        bs = [np.median(rng.choice(x, len(x))) for _ in range(reps)]
        return {"median": fnum(np.median(x), 1), "ci95": [fnum(v, 1) for v in ci_pct(bs)], "n": int(len(x))}
    asym = {}
    for var, lab in [("max_prior_papers_true_w99", "Tier B max prior papers of first/last author (w99, leakage-free)"),
                     ("max_years_active", "Tier B max years active before the paper"),
                     ("max_hindex", "max author h-index (2026-measured, LEAKY)"),
                     ("last_author_hindex", "last-author h-index (2026-measured, LEAKY)")]:
        asym[var] = {"label": lab, **{g: med_ci(d.loc[d.label == g, var]) for g in ["overrated", "underrated", "neutral"]}}
        # difference in medians over - under
        a, b = d.loc[d.label == "overrated", var].dropna().values, d.loc[d.label == "underrated", var].dropna().values
        diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(1000)]
        asym[var]["diff_median_over_minus_under"] = {"est": fnum(np.median(a) - np.median(b), 1), "ci95": [fnum(v, 1) for v in ci_pct(diffs)]}
        asym[var]["mannwhitney_p_over_vs_under"] = fnum(stats.mannwhitneyu(a, b).pvalue, 4)
    out["prestige_asymmetry"] = asym

    # ---- text themes: TF-IDF + L1 logistic on titles ---------------------------
    lab = d[d.label != "neutral"].copy()
    yv = (lab.label == "overrated").astype(int).values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, stop_words="english", sublinear_tf=True)
    Xt = vec.fit_transform(lab.title.str.lower())
    terms = np.array(vec.get_feature_names_out())
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=2.0, class_weight="balanced", random_state=SEED, max_iter=5000)
    cv_auc = cross_val_score(lr, Xt, yv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
    lr.fit(Xt, yv)
    coef = lr.coef_[0]
    order = np.argsort(coef)
    out["title_themes"] = {"note": "TF-IDF (1-2 grams, min_df 5, english stop words) on TITLES only; L1 logistic overrated(1) vs underrated(0), balanced; "
                                   "5-fold CV AUC reported as the (weak) separability of the two groups by title alone",
                           "n_overrated": int(yv.sum()), "n_underrated": int((1 - yv).sum()), "vocab": int(len(terms)),
                           "cv_auc_mean": fnum(cv_auc.mean(), 3), "cv_auc_sd": fnum(cv_auc.std(), 3),
                           "n_nonzero_terms": int((coef != 0).sum()),
                           "overrated_terms": [{"term": str(terms[i]), "coef": fnum(coef[i], 3)} for i in order[::-1][:15] if coef[i] > 0],
                           "underrated_terms": [{"term": str(terms[i]), "coef": fnum(coef[i], 3)} for i in order[:15] if coef[i] < 0]}
    # secondary: title + HF ai_keywords (as the June-11 14_text_themes.py used title+keywords+summary)
    txt2 = (lab.title.fillna("") + ". " + lab.ai_keywords.fillna("")).str.lower()
    vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=8, max_df=0.5, stop_words="english", sublinear_tf=True)
    X2 = vec2.fit_transform(txt2)
    terms2 = np.array(vec2.get_feature_names_out())
    lr2 = LogisticRegression(penalty="l1", solver="liblinear", C=2.0, class_weight="balanced", random_state=SEED, max_iter=5000)
    cv2 = cross_val_score(lr2, X2, yv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
    lr2.fit(X2, yv)
    c2 = lr2.coef_[0]
    o2 = np.argsort(c2)
    out["title_keyword_themes_secondary"] = {"note": "same model on title + HF ai_keywords (secondary)", "vocab": int(len(terms2)),
                                             "cv_auc_mean": fnum(cv2.mean(), 3), "cv_auc_sd": fnum(cv2.std(), 3),
                                             "overrated_terms": [{"term": str(terms2[i]), "coef": fnum(c2[i], 3)} for i in o2[::-1][:15] if c2[i] > 0],
                                             "underrated_terms": [{"term": str(terms2[i]), "coef": fnum(c2[i], 3)} for i in o2[:15] if c2[i] < 0]}
    log(f"   title themes CV AUC {cv_auc.mean():.3f}; over: " + ", ".join(t['term'] for t in out['title_themes']['overrated_terms'][:8]))
    log(f"   under: " + ", ".join(t['term'] for t in out['title_themes']['underrated_terms'][:8]))

    # ---- examples --------------------------------------------------------------
    ex = {}
    out["examples_note"] = ("chosen by largest |attention pct - impact pct| among papers with age >= 12 months AND citation_count >= 1; "
                            "zero-citation records at >= 12 months are excluded because several are Semantic Scholar matching failures "
                            "(e.g. 2410.05258 'Differential Transformer', 182 upvotes, 0 recorded citations) rather than genuinely uncited papers.")
    out["overrated_zero_citation_records"] = {"n_overrated_with_0_cites": int(((d.label == "overrated") & (d.citation_count == 0)).sum()),
                                              "n_overrated_with_0_cites_age_ge_12": int(((d.label == "overrated") & (d.citation_count == 0) & (d.age_months >= 12)).sum()),
                                              "note": "some of these are S2 match failures, i.e. part of the 'over-rated' set is measurement error on the impact side"}
    for g, asc in [("overrated", False), ("underrated", True)]:
        s = d[(d.label == g) & (d.age_months >= 12) & (d.citation_count >= 1)].sort_values("resid_gap", ascending=asc).head(5)
        ex[g] = [{"arxiv_id": r.arxiv_id_clean, "title": r.title, "upvotes": int(r.upvotes), "citations": int(r.citation_count),
                  "subfield_kw": r.subfield_kw, "release_month": r.release_month, "age_months": fnum(r.age_months, 1),
                  "max_prior_papers_true_w99": fnum(r.max_prior_papers_true_w99, 0), "max_hindex": fnum(r.max_hindex, 0),
                  "attention_pct": fnum(r.attention_resid_pct, 3), "impact_pct": fnum(r.impact_resid_pct, 3)} for r in s.itertuples()]
    out["examples_age_ge_12mo"] = ex
    # subfield composition of the groups
    comp = pd.crosstab(d.subfield_kw, d.label, normalize="columns")
    out["subfield_share_by_label"] = {sf: {lab_: fnum(v, 3) for lab_, v in row.items()} for sf, row in comp.iterrows()}
    RES["over_under"] = out
    return d[["arxiv_id_clean", "attention_resid_pct", "impact_resid_pct", "label"]]


# =============================================================================
# 5. MEASUREMENT EVIDENCE
# =============================================================================
def measurement(df):
    log("5. Measurement evidence")
    m = df.groupby("release_month").agg(n=("upvotes", "size"), median_upvotes=("upvotes", "median"),
                                        p90_upvotes=("upvotes", lambda s: s.quantile(.9)),
                                        median_cites=("citation_count", "median"), share_zero_cites=("citation_count", lambda s: (s == 0).mean()),
                                        median_age=("age_months", "median"))
    sp = {}
    for yr in [2023, 2024, 2025]:
        s = df[df.release_year == yr]
        r, p = stats.spearmanr(s.upvotes, s.age_months)
        sp[str(yr)] = {"n": int(len(s)), "spearman_upvotes_age": fnum(r, 3), "p": fnum(p, 4)}
    r_all, _ = stats.spearmanr(df.upvotes, df.age_months)
    RES["measurement"] = {
        "snapshot_statement": ("Upvotes are cumulative HF counts at scrape time (2026-06-05 for 2024–25 papers, 2026-06-11 "
                               "for 2023 papers), not day-one counts; citations are Semantic Scholar counts at the 2026-06-11 "
                               "snapshot; age_months = months from the HF published_at date to 2026-06-11 (5.4–39.8)."),
        "monthly": {k: {"n": int(v.n), "median_upvotes": fnum(v.median_upvotes, 1), "p90_upvotes": fnum(v.p90_upvotes, 1),
                        "median_cites": fnum(v.median_cites, 1), "share_zero_cites": fnum(v.share_zero_cites, 3),
                        "median_age_months": fnum(v.median_age, 1)} for k, v in m.iterrows()},
        "median_upvotes_range_2023H2_on": [fnum(m.loc["2023-07":].median_upvotes.min(), 1), fnum(m.loc["2023-07":].median_upvotes.max(), 1)],
        "spearman_upvotes_age_by_year": sp, "spearman_upvotes_age_all": fnum(r_all, 3),
        "share_zero_cites_by_year": {str(y): fnum(v, 3) for y, v in df.groupby("release_year").citation_count.apply(lambda s: (s == 0).mean()).items()},
        "median_cites_by_year": {str(y): fnum(v, 1) for y, v in df.groupby("release_year").citation_count.median().items()},
        "median_age_by_year": {str(y): fnum(v, 1) for y, v in df.groupby("release_year").age_months.median().items()},
        "n_by_year": {str(y): int(v) for y, v in df.release_year.value_counts().sort_index().items()},
        "descriptives": {"n": int(len(df)), "median_upvotes": fnum(df.upvotes.median(), 1), "median_cites": fnum(df.citation_count.median(), 1),
                         "share_zero_cites": fnum((df.citation_count == 0).mean(), 3), "spearman_upvotes_cites": fnum(stats.spearmanr(df.upvotes, df.citation_count)[0], 3),
                         "n_repaired_outcomes": int(df.citation_repaired.sum()), "tierB_resolved_share": fnum(df.tierB_resolved.mean(), 3),
                         "author_max_appear_lookahead_share": fnum(1 - (df.max_appear_prior.sum() / max(df.author_max_appear.sum(), 1)), 3)},
    }
    log(f"   Spearman(upvotes, age): " + ", ".join(f"{k}: {v['spearman_upvotes_age']}" for k, v in sp.items()))


# =============================================================================
# 6. TABLES + NOTE
# =============================================================================
def md_tables():
    L = []
    L.append("# association_v3 — paste-ready tables\n")
    L.append(f"Generated by `scripts/27_association_v3.py` (seed {SEED}); n = 11,344; all numbers computed, none hand-entered.\n")
    # ladder
    L.append("## T1. Count-model ladder — coefficient on log(1+upvotes) (cluster-robust SE by release month, 34 clusters)\n")
    L.append("| Spec | Poisson-QMLE β (SE) | IRR per 2× upvotes [95% CI] | NB2 β (SE) | IRR per 2× [95% CI] | NB conv. | log1p-OLS β (SE) | ×(1+cites) per 2× | n |")
    L.append("|---|---:|---:|---:|---:|:-:|---:|---:|---:|")
    for k, v in RES["ladder"].items():
        p, nb, o = v["poisson_qmle"], v["nb2"], v["ols_log1p"]
        L.append(f"| {v['label']} | {p['beta']:.3f} ({p['se']:.3f}) | {p['irr_per_doubling']:.2f} [{p['irr_per_doubling_ci95'][0]:.2f}, {p['irr_per_doubling_ci95'][1]:.2f}] | "
                 f"{nb['beta']:.3f} ({nb['se']:.3f}) | {nb['irr_per_doubling']:.2f} [{nb['irr_per_doubling_ci95'][0]:.2f}, {nb['irr_per_doubling_ci95'][1]:.2f}] | "
                 f"{'yes' if nb['converged'] else 'NO'} | {o['beta']:.3f} ({o['se']:.3f}) | {o['ratio_1p_cites_per_doubling']:.2f} | {p['nobs']:,} |")
    b = RES["bootstrap_M4"]
    L.append(f"\nM4 month-block bootstrap ({b['n_reps']} reps) 95% CI: Poisson β {b['poisson_beta_ci95']} (IRR {b['poisson_irr2x_ci95']}); "
             f"NB2 β {b['nb2_beta_ci95']} (IRR {b['nb2_irr2x_ci95']}); OLS β {b['ols_beta_ci95']}.")
    e = RES["e_value_M4"]
    L.append(f"E-value (M4): Poisson IRR {e['poisson_qmle']['irr_per_doubling']} → E = {e['poisson_qmle']['e_value_point']} (CI-bound {e['poisson_qmle']['e_value_ci_lower']}); "
             f"NB2 IRR {e['nb2']['irr_per_doubling']} → E = {e['nb2']['e_value_point']} (CI-bound {e['nb2']['e_value_ci_lower']}).")
    pl = RES["placebo_permutation_M4"]["poisson"]
    L.append(f"Placebo (upvotes permuted within month × subfield, {RES['placebo_permutation_M4']['n_perm']} reps): Poisson β mean {pl['placebo_beta_mean']} "
             f"(SD {pl['placebo_beta_sd']}, max {pl['placebo_beta_max']}); real β {pl['real_beta']} = {pl['z_real_vs_placebo']} placebo-SDs above; share ≥ real: {pl['share_placebo_ge_real']}.")
    pc = RES["placebo_permutation_M4"]["cell_fe_variant"]
    L.append(f"Placebo with month × subfield interacted FE ({pc['n_perm']} reps): Poisson placebo mean {pc['poisson']['placebo_beta_mean']} (SD {pc['poisson']['placebo_beta_sd']}) vs real {pc['poisson']['real_beta']}; "
             f"OLS placebo mean {pc['ols']['placebo_beta_mean']} (SD {pc['ols']['placebo_beta_sd']}) vs real {pc['ols']['real_beta']}.")
    po = RES["placebo_outcome_reference_count_M4"]
    L.append(f"Placebo outcome (reference count, n={po['n']:,}): OLS β {po['ols_log']['beta']} (SE {po['ols_log']['se']}) = {po['ratio_to_citation_effect_ols']:.0%} of the citation elasticity; Poisson β {po['poisson']['beta']} (SE {po['poisson']['se']}).\n")
    cx = RES["ladder_convexity"]
    L.append(f"Convexity (M4 + log_upvotes²): Poisson β_sq {cx['poisson']['beta_sq']} (SE {cx['poisson']['se_sq']}); NB2 {cx['nb2']['beta_sq']} (SE {cx['nb2']['se_sq']}); OLS {cx['ols']['beta_sq']} (SE {cx['ols']['se_sq']}). "
             "Slope by upvote band (Poisson / OLS): " + "; ".join(f"{k}: {v['poisson_beta']} / {v['ols_beta']} (n={v['n']:,})" for k, v in cx["slope_by_upvote_band"].items()) + "\n")

    L.append("## T1b. Robustness rows (M4 controls unless stated)\n")
    L.append("| Row | Poisson β (SE) | IRR 2× | NB2 β (SE) | OLS β (SE) | n | note |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for k, v in RES["ladder_robustness"].items():
        p, nb, o = v["poisson_qmle"], v["nb2"], v["ols_log1p"]
        L.append(f"| {k} | {p['beta']:.3f} ({p['se']:.3f}) | {p['irr_per_doubling']:.2f} | {nb['beta']:.3f} ({nb['se']:.3f}){'' if nb['converged'] else ' [not conv.]'} | {o['beta']:.3f} ({o['se']:.3f}) | {v['n']:,} | {v['note']} |")

    h = RES["hierarchical_mixedlm"]
    L.append("\n## T2. Hierarchical model (MixedLM, log1p citations; random intercept + random slope of log upvotes by subfield_kw; month + dow FE, log age, Tier B prestige)\n")
    L.append(f"Fixed slope {h['fixed_slope_log_upvotes']} (SE {h['fixed_slope_se']}, 95% CI {h['fixed_slope_ci95']}); intercept SD {h['variance_components']['intercept_sd']}, "
             f"slope SD {h['variance_components']['slope_sd']}, residual var {h['variance_components']['residual_var']}; LR test for random slope: {h['lr_test_random_slope']['lr_stat']} (p {h['lr_test_random_slope']['p_boundary_mixture']}).\n")
    L.append("| subfield_kw | n | slope | approx 95% CI |")
    L.append("|---|---:|---:|---|")
    for g, v in h["subfield_slopes"].items():
        L.append(f"| {g} | {v['n']:,} | {v['slope']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")

    L.append("\n## T3. Dose–response: deciles of upvotes → citations\n")
    L.append("| decile | upvotes range | n | mean log1p cites [95% CI] | geo-mean (1+cites) | median cites | adjusted mean log1p [95% CI] | share top-decile-in-quarter |")
    L.append("|---:|---|---:|---|---:|---:|---|---:|")
    for r in RES["dose_response_deciles"]["rows"]:
        L.append(f"| {r['decile']} | {r['upvotes_min']}–{r['upvotes_max']} | {r['n']:,} | {r['mean_log1p_cites']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | {r['geo_mean_1p_cites']:.1f} | {r['median_cites']:.0f} | "
                 f"{r['mean_log1p_cites_adjusted']:.2f} [{r['ci95_adjusted'][0]:.2f}, {r['ci95_adjusted'][1]:.2f}] | {r['share_top_decile_cites_in_quarter']:.3f} |")

    s = RES["selection_adjustment"]
    L.append(f"\n## T4. Selection adjustment — top vs bottom attention tertile (n_treated {s['design']['n_treated']:,}, n_control {s['design']['n_control']:,}; outcome log1p citations)\n")
    L.append("| Estimator | ATT (log pts) | 95% CI | ratio | matched / support |")
    L.append("|---|---:|---|---:|---|")
    L.append(f"| Naive gap | {s['naive_gap']['log_pts']:.3f} | {s['naive_gap']['ci95']} | ×{s['naive_gap']['ratio']:.2f} | — |")
    L.append(f"| PS 1:1 NN matching (caliper 0.2 SD) | {s['ps_matching']['att_log_pts']:.3f} | {s['ps_matching']['ci95']} | ×{s['ps_matching']['ratio']:.2f} | {s['ps_matching']['n_treated_matched']:,} treated matched ({s['ps_matching']['share_treated_within_caliper']:.1%}), {s['ps_matching']['n_unique_controls_used']:,} unique controls |")
    L.append(f"| IPW (ATT, stabilised, trimmed p99) | {s['ipw']['att_log_pts']:.3f} | {s['ipw']['ci95']} | ×{s['ipw']['ratio']:.2f} | control ESS {s['ipw']['ess_controls']}, max w {s['ipw']['max_control_weight']} |")
    L.append(f"| CEM exact (month × subfield_kw × Tier B quintile) | {s['cem']['att_log_pts']:.3f} | {s['cem']['ci95']} | ×{s['cem']['ratio']:.2f} | {s['cem']['n_treated_matched']:,} treated ({s['cem']['share_treated_matched']:.1%}) in {s['cem']['n_cells']} cells |")
    vh = s["variant_ps_with_hindex_LEAKY"]
    L.append(f"| (variant, LEAKY) PS matching incl. 2026 h-index | {vh['ps_matching_att']:.3f} | {vh['ps_matching_ci95']} | ×{np.exp(vh['ps_matching_att']):.2f} | comparison with v2 only |")
    L.append(f"| (variant, LEAKY) IPW incl. 2026 h-index | {vh['ipw_att']:.3f} | {vh['ipw_ci95']} | ×{np.exp(vh['ipw_att']):.2f} | comparison with v2 only |")
    L.append("\n**Balance (standardised mean differences):**\n")
    L.append("| covariate | before | after matching | after IPW |")
    L.append("|---|---:|---:|---:|")
    for r in s["balance"]:
        L.append(f"| {r['covariate']} | {r['smd_before']:+.3f} | {r['smd_after_matching']:+.3f} | {r['smd_after_ipw']:+.3f} |")
    mm = s["balance_month_dummies_max_abs_smd"]
    L.append(f"| max abs SMD over 34 month dummies | {mm['before']:.3f} | {mm['after_matching']:.3f} | {mm['after_ipw']:.3f} |")

    c = RES["control_comparison"]
    f = c["control_sample_facts"]
    L.append(f"\n## T5. Trending vs never-trending background (control n = {f['n_analysis_5_40_months']:,} of {f['n_s2_found']:,} S2-matched; ages 5–40 months)\n")
    L.append("| Estimator | gap (log pts) | 95% CI | ratio [CI] | trending matched |")
    L.append("|---|---:|---|---|---|")
    p = c["trending_premium"]
    L.append(f"| Naive (all trending vs all controls) | {p['naive']['gap_log_pts']:.3f} | {p['naive']['ci95']} | ×{p['naive']['ratio']:.2f} (median ratio ×{p['naive']['median_ratio']:.1f}) | — |")
    for k in ["cem_month_subfield13_hbin", "cem_month_family_hbin", "cem_month_hbin"]:
        v = p[k]
        L.append(f"| {v['note']} | {v['gap_log_pts']:.3f} | {v['ci95']} | ×{v['ratio']:.2f} {v['ratio_ci95']} | {v['n_trending_matched']:,} ({v['share_trending_matched']:.1%}) in {v['n_cells']} cells |")
    v = p["ols_fe_month_subfield_hbin"]
    L.append(f"| OLS with month + subfield + h-bin FE (CRV1 month) | {v['gap_log_pts']:.3f} | {v['ci95']} | ×{v['ratio']:.2f} | n={v['n']:,} |")
    L.append("\n**Where do low-attention trending papers sit in the background? (percentile within same-release-month background papers)**\n")
    L.append("| group | n | median upvotes | median cites | CEM gap vs background [CI] | ratio | mean percentile in month-matched background [CI] | same month × family |")
    L.append("|---|---:|---:|---:|---|---:|---|---|")
    for k, v in c["low_attention_vs_background"].items():
        pm, pf_ = v["percentile_in_background_same_month"], v["percentile_in_background_same_month_family"]
        L.append(f"| {k} | {v['n']:,} | {v['median_upvotes']} | {v['median_cites']} | {v['cem_gap_vs_background_log_pts']:.3f} {v['cem_gap_ci95']} | ×{v['cem_ratio']:.2f} | {pm['mean_percentile']:.3f} {pm['ci95']} | {pf_['mean_percentile']:.3f} {pf_['ci95']} |")

    ou = RES["over_under"]
    L.append(f"\n## T6. Over-/under-rated — counts {ou['counts']}\n")
    for name in ["overrated_vs_rest", "underrated_vs_rest", "overrated_vs_underrated", "overrated_vs_underrated_NOLEAK"]:
        dr = ou["logit_drivers"][name]
        L.append(f"\n**{name}** (n={dr['n']:,}, events={dr['n_events']:,}, pseudo-R² {dr['pseudo_r2']}; OR per SD for continuous; cluster-robust by month)\n")
        L.append("| feature | OR | 95% CI | p |")
        L.append("|---|---:|---|---:|")
        for r in dr["rows"][:14]:
            L.append(f"| {r['feature']} | {r['or']:.2f} | [{r['or_ci95'][0]:.2f}, {r['or_ci95'][1]:.2f}] | {r['p']:.3f} |")
    L.append("\n**Prestige asymmetry (medians with bootstrap 95% CI)**\n")
    L.append("| measure | overrated | underrated | neutral | diff (over − under) [CI] | MWU p |")
    L.append("|---|---|---|---|---|---:|")
    for var, v in ou["prestige_asymmetry"].items():
        L.append(f"| {v['label']} | {v['overrated']['median']} {v['overrated']['ci95']} | {v['underrated']['median']} {v['underrated']['ci95']} | {v['neutral']['median']} {v['neutral']['ci95']} | "
                 f"{v['diff_median_over_minus_under']['est']} {v['diff_median_over_minus_under']['ci95']} | {v['mannwhitney_p_over_vs_under']} |")
    th = ou["title_themes"]
    L.append(f"\n**Title themes** (TF-IDF + L1 logistic, 5-fold CV AUC {th['cv_auc_mean']} ± {th['cv_auc_sd']}): overrated ← " +
             ", ".join(t['term'] for t in th['overrated_terms']) + " | underrated ← " + ", ".join(t['term'] for t in th['underrated_terms']) + "\n")
    L.append("\n**Examples (age ≥ 12 months, ≥ 1 citation, largest residual gap)** — " + ou["examples_note"] + "\n")
    L.append("| group | arXiv id | title | subfield_kw | upvotes | citations | age (mo) | Tier B prior papers | max h |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for g, rows in ou["examples_age_ge_12mo"].items():
        for r in rows:
            L.append(f"| {g} | {r['arxiv_id']} | {r['title'][:80]} | {r['subfield_kw']} | {r['upvotes']} | {r['citations']} | {r['age_months']} | {r['max_prior_papers_true_w99']} | {r['max_hindex']} |")

    me = RES["measurement"]
    L.append("\n## T7. Measurement evidence\n")
    L.append(me["snapshot_statement"] + "\n")
    L.append("| year | n | median age (mo) | median cites | share zero cites | Spearman(upvotes, age) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for y in ["2023", "2024", "2025"]:
        L.append(f"| {y} | {me['n_by_year'][y]:,} | {me['median_age_by_year'][y]} | {me['median_cites_by_year'][y]} | {me['share_zero_cites_by_year'][y]:.3f} | {me['spearman_upvotes_age_by_year'][y]['spearman_upvotes_age']:+.3f} |")
    L.append(f"\nMonthly median upvotes from 2023-07 on: {me['median_upvotes_range_2023H2_on'][0]}–{me['median_upvotes_range_2023H2_on'][1]} (full series in JSON `measurement.monthly`).\n")
    OUT_TABLES.write_text("\n".join(L))


def md_note():
    lad = RES["ladder"]
    m4p, m4n, m4o = lad["M4_tierB_prestige"]["poisson_qmle"], lad["M4_tierB_prestige"]["nb2"], lad["M4_tierB_prestige"]["ols_log1p"]
    s = RES["selection_adjustment"]
    c = RES["control_comparison"]
    ou = RES["over_under"]
    N = f"""# association_v3 — NOTE (definitions, caveats, headline numbers)

Script: `scripts/27_association_v3.py` · Ledger: `results/association_v3.json` · Tables: `association_v3_tables.md` ·
Dashboard file: `data/processed/overunder_v3.csv` · Seed {SEED} · runtime {RES['meta'].get('runtime_sec', '?')} s.

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
* Selection: treated = top tertile of raw upvotes (≥ {s['design']['treated_threshold']:.0f}), control = bottom tertile (≤ {s['design']['control_threshold']:.0f}), middle dropped;
  PS logit on Tier B prestige + log age + subfield_kw + release_month; 1:1 NN with replacement, caliper 0.2 SD of logit PS; ATT-IPW
  stabilised/normalised/trimmed p99; CEM exact on month × subfield_kw × Tier B quintile; {N_BOOT_MATCH} bootstrap reps each.
* Over/under-rated: residual percentiles from `log_upvotes ~ log_age + subfield_kw + release_month` and the same for `log_citations`;
  overrated = attention pct ≥ 2/3 & impact pct ≤ 1/3; underrated = reverse. Same definition as the dashboard explorer.
* Control comparison: control subfield = same keyword rules on the title, else arXiv category → nearest label; h-index bins = pooled quintiles;
  exact matching on month × subfield × h-bin with ATT weights; percentile = position of each trending paper's citation count among
  same-release-month background papers (mid-rank ties).

## Headline numbers (M4, main spec)
* Poisson-QMLE β = {m4p['beta']} (SE {m4p['se']}), IRR per doubling **{m4p['irr_per_doubling']}** {m4p['irr_per_doubling_ci95']}; month-block bootstrap CI {RES['bootstrap_M4']['poisson_irr2x_ci95']}.
* NB2 β = {m4n['beta']} (SE {m4n['se']}), IRR per doubling **{m4n['irr_per_doubling']}** {m4n['irr_per_doubling_ci95']} (converged: {m4n['converged']}); bootstrap CI {RES['bootstrap_M4']['nb2_irr2x_ci95']}.
* log1p-OLS elasticity **{m4o['beta']}** (SE {m4o['se']}) → ×{m4o['ratio_1p_cites_per_doubling']} (1+citations) per doubling; bootstrap CI {RES['bootstrap_M4']['ols_beta_ci95']}.
* E-values: Poisson {RES['e_value_M4']['poisson_qmle']['e_value_point']} (CI bound {RES['e_value_M4']['poisson_qmle']['e_value_ci_lower']}); NB2 {RES['e_value_M4']['nb2']['e_value_point']} (CI bound {RES['e_value_M4']['nb2']['e_value_ci_lower']}).
* Placebo permutation: Poisson β mean {RES['placebo_permutation_M4']['poisson']['placebo_beta_mean']} (SD {RES['placebo_permutation_M4']['poisson']['placebo_beta_sd']}); real β is {RES['placebo_permutation_M4']['poisson']['z_real_vs_placebo']} SDs above; reference-count placebo outcome = {RES['placebo_outcome_reference_count_M4']['ratio_to_citation_effect_ols']:.0%} of the citation elasticity.
* MixedLM: fixed slope {RES['hierarchical_mixedlm']['fixed_slope_log_upvotes']}, between-subfield slope SD {RES['hierarchical_mixedlm']['variance_components']['slope_sd']}, slopes {RES['hierarchical_mixedlm']['slope_range']}.
* Selection ATT (log pts): naive {s['naive_gap']['log_pts']}; PS-matched {s['ps_matching']['att_log_pts']} {s['ps_matching']['ci95']}; IPW {s['ipw']['att_log_pts']} {s['ipw']['ci95']}; CEM {s['cem']['att_log_pts']} {s['cem']['ci95']}.
* Trending premium (CEM month × subfield × h-bin): {c['trending_premium']['cem_month_subfield13_hbin']['gap_log_pts']} log pts, ×{c['trending_premium']['cem_month_subfield13_hbin']['ratio']} {c['trending_premium']['cem_month_subfield13_hbin']['ratio_ci95']} (naive ×{c['trending_premium']['naive']['ratio']}).
* Bottom-tertile-attention trending papers sit at the {c['low_attention_vs_background']['bottom_tertile_attention_residual']['percentile_in_background_same_month']['mean_percentile']:.0%} percentile of month-matched background
  (CI {c['low_attention_vs_background']['bottom_tertile_attention_residual']['percentile_in_background_same_month']['ci95']}); the outcome-selected "under-rated" group sits at {c['low_attention_vs_background']['underrated_label_OUTCOME_SELECTED']['percentile_in_background_same_month']['mean_percentile']:.0%} (circular by construction).
* Placebo with month × subfield interacted FE: Poisson placebo mean {RES['placebo_permutation_M4']['cell_fe_variant']['poisson']['placebo_beta_mean']} (SD {RES['placebo_permutation_M4']['cell_fe_variant']['poisson']['placebo_beta_sd']}) vs real {RES['placebo_permutation_M4']['cell_fe_variant']['poisson']['real_beta']}.
* Over/under-rated counts: {ou['counts']}. Prestige asymmetry (median, over vs under vs neutral): Tier B prior papers
  {ou['prestige_asymmetry']['max_prior_papers_true_w99']['overrated']['median']} / {ou['prestige_asymmetry']['max_prior_papers_true_w99']['underrated']['median']} / {ou['prestige_asymmetry']['max_prior_papers_true_w99']['neutral']['median']}
  (diff over−under {ou['prestige_asymmetry']['max_prior_papers_true_w99']['diff_median_over_minus_under']['est']} CI {ou['prestige_asymmetry']['max_prior_papers_true_w99']['diff_median_over_minus_under']['ci95']}); 2026 h-index (leaky)
  {ou['prestige_asymmetry']['max_hindex']['overrated']['median']} / {ou['prestige_asymmetry']['max_hindex']['underrated']['median']} / {ou['prestige_asymmetry']['max_hindex']['neutral']['median']} (diff {ou['prestige_asymmetry']['max_hindex']['diff_median_over_minus_under']['est']} CI {ou['prestige_asymmetry']['max_hindex']['diff_median_over_minus_under']['ci95']}).
  Leakage-free head-to-head logit: Tier B prior papers OR {[r for r in ou['logit_drivers']['overrated_vs_underrated_NOLEAK']['rows'] if r['feature'].startswith('tierB_prior')][0]['or']} per SD for being over- rather than under-rated
  (CI {[r for r in ou['logit_drivers']['overrated_vs_underrated_NOLEAK']['rows'] if r['feature'].startswith('tierB_prior')][0]['or_ci95']}). Title-only TF-IDF themes separate the groups weakly (CV AUC {ou['title_themes']['cv_auc_mean']}).
* {ou['overrated_zero_citation_records']['n_overrated_with_0_cites']} of the {ou['counts']['overrated']} over-rated papers have 0 recorded citations ({ou['overrated_zero_citation_records']['n_overrated_with_0_cites_age_ge_12']} at ≥ 12 months) — some are S2 matching failures (e.g. Differential Transformer), i.e. impact-side measurement error.

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
4. **Control sample** = first-day(s)-of-month cs.CL/CV/LG submissions with S2 outcomes (n = {c['control_sample_facts']['n_analysis_5_40_months']:,}), popularity-blind but not a random month
   sample; its subfield labels are title/category-based (coarser than the trending labels), and only {c['trending_premium']['cem_month_subfield13_hbin']['share_trending_matched']:.0%} of trending papers find an exact cell.
5. **Exposure heterogeneity**: 2025 papers have 5–17 months of exposure; log age + month FE absorb most of it (see cohort rows), and the
   over/under labels are residualised on age, field and month.
6. Numbers differ from the June-11 (v2) write-up because of the taxonomy, prestige, month-FE and robust-SE changes above — v2's numbers
   are not wrong on their own frame but should be replaced by these throughout.
"""
    OUT_NOTE.write_text(N)


# =============================================================================
def main():
    df = load()
    df["upvote_rank_month"] = df.groupby("release_month").upvotes.rank(pct=True)
    df["log_infl"] = np.log1p(df.infl)
    log(f"loaded n={len(df)}")
    measurement(df)
    ladder(df)
    bootstrap_evalue_placebo(df)
    hierarchical(df)
    dose_response(df)
    selection(df)
    ou = over_under(df)
    control_comparison(df, ou)
    RES["meta"]["runtime_sec"] = round(time.time() - T0, 1)
    OUT_JSON.write_text(json.dumps(RES, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    md_tables()
    md_note()
    log(f"wrote {OUT_JSON}, {OUT_TABLES}, {OUT_NOTE}")


if __name__ == "__main__":
    main()
