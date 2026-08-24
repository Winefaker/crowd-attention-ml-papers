"""
12_models_v2.py
---------------
v2 statistical analysis (paper-grade). Builds on 05_models.py with:

  (1) a SPECIFICATION LADDER for beta1 (log upvotes):
        M0 bivariate NB
        M1 + age + subfield FE
        M2 + v1 prestige proxy (HF recurrence) + n_authors + github
        M3 + REAL prestige (max & last-author h-index)        <- main spec
        M4 month FE instead of log(age)                        <- robustness
  (2) E-value sensitivity analysis for unmeasured confounding
  (3) placebo outcome: reference_count (fixed at submission; attention must not
      "predict" it if controls are adequate)
  (4) random slopes: does the attention-citation slope vary by subfield?
  (5) matching + IPW with the h-index prestige
  (6) robustness: drop repaired outcomes; 2023 mature cohort only;
      within-month attention rank as alternative treatment
  (7) keyword-subfield vs arXiv-category agreement (D6 validation)
  (8) age+field-adjusted overrated/underrated labels + logistic drivers (with h-index)

Outputs: data/processed/model_results_v2.json, data/processed/papers_scored_v2.csv
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RES = {}


def load():
    df = pd.read_csv(os.path.join(PROC, "papers_v2.csv"), dtype={"arxiv_id_clean": str})
    df = df[df["citation_count"].notna() & df["upvotes"].notna()].copy()
    df = df[(df["age_months"] >= 5) & (df["age_months"] <= 40)].copy()
    df["log_age"] = np.log(df["age_months"].clip(lower=1))
    df["log_n_authors"] = df["log_n_authors"].fillna(df["log_n_authors"].median())
    global MISS_TERM
    MISS_TERM = ""
    for c in ["log_max_hindex", "log_last_hindex"]:
        miss = df[c].isna()
        df[c + "_miss"] = miss.astype(int)
        df[c] = df[c].fillna(df[c].median())
        # only keep the indicator if it actually varies (else: singular design matrix)
        if 0 < miss.sum() < len(df):
            MISS_TERM = " + log_max_hindex_miss"
    vc = df["subfield"].value_counts()
    df["subfield_grp"] = np.where(df["subfield"].isin(vc[vc >= 30].index), df["subfield"], "Other")
    # within-month attention rank (alternative treatment robust to upvote inflation)
    df["upvote_rank_month"] = df.groupby("release_month")["upvotes"].rank(pct=True)
    return df.reset_index(drop=True)


def spec_ladder(df):
    print("\n=== (1) Specification ladder: NB citation_count ~ log_upvotes + ... ===")
    specs = {
        "M0_bivariate": "citation_count ~ log_upvotes",
        "M1_age_field": "citation_count ~ log_upvotes + log_age + C(subfield_grp)",
        "M2_v1_proxy":  ("citation_count ~ log_upvotes + log_age + C(subfield_grp)"
                         " + log_author_max_appear + log_n_authors + has_github"),
        "M3_hindex":    ("citation_count ~ log_upvotes + log_age + C(subfield_grp)"
                         " + log_max_hindex + log_last_hindex" + MISS_TERM +
                         " + log_n_authors + has_github"),
        "M4_monthFE":   ("citation_count ~ log_upvotes + C(release_month) + C(subfield_grp)"
                         " + log_max_hindex + log_last_hindex" + MISS_TERM +
                         " + log_n_authors + has_github"),
    }
    out = {}
    for name, f in specs.items():
        try:
            m = smf.negativebinomial(f, data=df).fit(disp=0, maxiter=200)
            b = float(m.params["log_upvotes"])
            ci = m.conf_int().loc["log_upvotes"].tolist()
            out[name] = {"beta": b, "ci": [float(ci[0]), float(ci[1])],
                         "p": float(m.pvalues["log_upvotes"]),
                         "irr_per_doubling": float(2 ** b),
                         "converged": bool(m.mle_retvals.get("converged", True)),
                         "nobs": int(m.nobs)}
            print(f"  {name:13s} beta={b:.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  "
                  f"2x-upvotes->x{2**b:.2f} cites  (conv={out[name]['converged']})")
        except Exception as e:
            out[name] = {"error": str(e)}
            print(f"  {name}: FAILED {e}")
    RES["spec_ladder"] = out

    # main-spec OLS elasticity + bootstrap
    f_main = ("log_citations ~ log_upvotes + log_age + C(subfield_grp) + log_max_hindex"
              " + log_last_hindex" + MISS_TERM + " + log_n_authors + has_github")
    m_ols = smf.ols(f_main, data=df).fit(cov_type="HC1")
    m_no = smf.ols(f_main.replace("log_upvotes + ", ""), data=df).fit()
    RES["ols_main"] = {
        "beta": float(m_ols.params["log_upvotes"]),
        "se_hc1": float(m_ols.bse["log_upvotes"]),
        "r2": float(m_ols.rsquared), "r2_without_upvotes": float(m_no.rsquared),
    }
    print(f"  OLS main: beta={RES['ols_main']['beta']:.3f} (HC1 se {RES['ols_main']['se_hc1']:.3f}); "
          f"R2 {m_no.rsquared:.3f} -> {m_ols.rsquared:.3f} with upvotes")

    rng = np.random.default_rng(7)
    betas = []
    n = len(df)
    for _ in range(500):
        s = df.iloc[rng.integers(0, n, n)]
        try:
            betas.append(smf.ols(f_main, data=s).fit().params["log_upvotes"])
        except Exception:
            pass
    lo, hi = np.percentile(betas, [2.5, 97.5])
    RES["bootstrap_main"] = {"beta_mean": float(np.mean(betas)),
                             "ci95": [float(lo), float(hi)], "n_boot": len(betas)}
    print(f"  bootstrap (500): beta={np.mean(betas):.3f} CI=[{lo:.3f},{hi:.3f}]")
    return f_main


def e_value(df):
    print("\n=== (2) E-value for unmeasured confounding ===")
    s = RES["spec_ladder"].get("M3_hindex", {})
    if "beta" not in s:
        return
    def ev(rr):
        return rr + np.sqrt(rr * (rr - 1)) if rr > 1 else None
    rr_point = s["irr_per_doubling"]
    rr_lo = float(2 ** s["ci"][0])
    RES["e_value"] = {"rr_per_doubling": rr_point, "e_value_point": float(ev(rr_point)),
                      "rr_ci_lower": rr_lo, "e_value_ci_lower": float(ev(rr_lo))}
    print(f"  IRR per doubling = {rr_point:.2f} -> E-value {ev(rr_point):.2f} "
          f"(CI-lower bound E-value {ev(rr_lo):.2f})")
    print("  i.e. an unmeasured confounder must be associated with BOTH attention and")
    print(f"  citations by RR >= {ev(rr_lo):.2f} to push the CI to the null.")


def placebo(df):
    print("\n=== (3) Placebo outcome: reference_count (set before any attention) ===")
    d = df[df["reference_count"].notna() & (df["reference_count"] > 0)]
    f = ("np.log(reference_count) ~ log_upvotes + log_age + C(subfield_grp) + log_max_hindex"
         " + log_last_hindex" + MISS_TERM + " + log_n_authors + has_github")
    m = smf.ols(f, data=d).fit(cov_type="HC1")
    b, se = float(m.params["log_upvotes"]), float(m.bse["log_upvotes"])
    RES["placebo"] = {"beta": b, "se": se, "n": int(m.nobs),
                      "note": "log reference count on attention + controls; expect ~0"}
    print(f"  beta(log_upvotes) on log(references) = {b:.4f} (se {se:.4f}, n={int(m.nobs)})")
    cit_beta = RES["ols_main"]["beta"]
    print(f"  vs {cit_beta:.3f} for citations -> placebo is {abs(b)/cit_beta*100:.0f}% of the real effect")


def random_slopes(df):
    print("\n=== (4) Random slopes: attention effect by subfield (MixedLM) ===")
    try:
        d = df.copy()
        md = smf.mixedlm("log_citations ~ log_upvotes + log_age + log_max_hindex"
                         " + log_last_hindex" + MISS_TERM + " + log_n_authors + has_github",
                         d, groups=d["subfield_grp"], re_formula="~log_upvotes")
        m = md.fit(method="lbfgs", maxiter=300)
        fe = float(m.fe_params["log_upvotes"])
        re_sd = float(np.sqrt(max(m.cov_re.iloc[1, 1], 0)))
        slopes = {g: float(fe + re.iloc[1]) for g, re in m.random_effects.items()}
        RES["random_slopes"] = {"fixed_beta": fe, "slope_sd": re_sd,
                                "subfield_slopes": dict(sorted(slopes.items(), key=lambda x: -x[1]))}
        print(f"  fixed beta={fe:.3f}, between-subfield slope SD={re_sd:.3f}")
        top = sorted(slopes.items(), key=lambda x: -x[1])
        print("  steepest:", ", ".join(f"{k} {v:.2f}" for k, v in top[:3]))
        print("  flattest:", ", ".join(f"{k} {v:.2f}" for k, v in top[-3:]))
    except Exception as e:
        RES["random_slopes"] = {"error": str(e)}
        print("  failed:", e)


def matching_ipw(df):
    print("\n=== (5) Matching + IPW with h-index prestige ===")
    d = df.copy()
    hi, lo = d["upvotes"].quantile(2/3), d["upvotes"].quantile(1/3)
    d = d[(d["upvotes"] >= hi) | (d["upvotes"] <= lo)].copy()
    d["treat"] = (d["upvotes"] >= hi).astype(int)
    naive = d.loc[d.treat == 1, "log_citations"].mean() - d.loc[d.treat == 0, "log_citations"].mean()

    X = pd.concat([d[["log_max_hindex", "log_last_hindex", "log_n_authors", "log_age", "has_github"]],
                   pd.get_dummies(d["subfield_grp"], prefix="sf", drop_first=True)], axis=1).astype(float)
    Xs = StandardScaler().fit_transform(X)
    ps = LogisticRegression(max_iter=1000).fit(Xs, d["treat"]).predict_proba(Xs)[:, 1]
    d = d.reset_index(drop=True); d["ps"] = np.clip(ps, 0.02, 0.98)

    tr, co = d[d.treat == 1].reset_index(drop=True), d[d.treat == 0].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=1).fit(co[["ps"]].values)
    _, idx = nn.kneighbors(tr[["ps"]].values)
    att_match = float((tr["log_citations"].values - co.iloc[idx.flatten()]["log_citations"].values).mean())

    w = np.where(d.treat == 1, 1.0, d.ps / (1 - d.ps))  # ATT weights
    att_ipw = float(np.average(d.loc[d.treat == 1, "log_citations"]) -
                    np.average(d.loc[d.treat == 0, "log_citations"], weights=w[d.treat == 0]))
    RES["selection_v2"] = {"naive_gap": float(naive), "matched_att": att_match,
                           "ipw_att": att_ipw, "n_treated": int(tr.shape[0])}
    print(f"  naive={naive:.3f}  matched={att_match:.3f}  IPW={att_ipw:.3f} "
          f"(share surviving: {att_match/naive*100:.0f}% / {att_ipw/naive*100:.0f}%)")


def robustness(df, f_main):
    print("\n=== (6) Robustness ===")
    out = {}
    sub = df[df["citation_repaired"] == 0]
    b = smf.ols(f_main, data=sub).fit().params["log_upvotes"]
    out["drop_repaired"] = {"beta": float(b), "n": len(sub)}
    sub = df[df["release_year"] == 2023]
    b = smf.ols(f_main, data=sub).fit().params["log_upvotes"]
    out["cohort_2023_mature"] = {"beta": float(b), "n": len(sub)}
    sub = df[df["age_months"] >= 12]
    b = smf.ols(f_main, data=sub).fit().params["log_upvotes"]
    out["age_ge_12mo"] = {"beta": float(b), "n": len(sub)}
    f_rank = f_main.replace("log_upvotes", "upvote_rank_month")
    b = smf.ols(f_rank, data=df).fit().params["upvote_rank_month"]
    out["within_month_rank"] = {"beta_rank01": float(b), "n": len(df),
                                "note": "treatment = upvote percentile within release month"}
    RES["robustness"] = out
    for k, v in out.items():
        bb = v.get("beta", v.get("beta_rank01"))
        print(f"  {k:22s} beta={bb:.3f} (n={v['n']})")


def subfield_validation(df):
    print("\n=== (7) Keyword-subfield vs arXiv-category agreement ===")
    have = df[df["primary_category"].notna()].copy()
    if len(have) < 100:
        RES["subfield_validation"] = {"n_overlap": int(len(have)),
                                      "note": "insufficient arXiv categories collected"}
        print(f"  only {len(have)} papers have arXiv categories; skipping")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m4", os.path.join(os.path.dirname(__file__), "04_merge_and_features.py"))
    m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)
    have["sf_from_kw"] = have.apply(lambda r: m4.coarse_subfield(None, None, r.get("ai_keywords")), axis=1)
    # The two taxonomies carve the space differently (a multimodal paper is
    # legitimately cs.CV OR cs.CL), so the honest test is COMPATIBILITY: is the
    # keyword label consistent with the official primary category?
    COMPAT = {
        "Multimodal": {"cs.CV", "cs.CL", "cs.LG", "cs.AI", "cs.MM"},
        "Vision/Image-Gen": {"cs.CV", "cs.GR", "cs.LG", "eess.IV", "cs.MM"},
        "Vision-Perception": {"cs.CV", "cs.RO", "eess.IV", "cs.LG"},
        "Agents": {"cs.AI", "cs.CL", "cs.LG", "cs.MA", "cs.SE", "cs.HC"},
        "RAG/Retrieval": {"cs.CL", "cs.IR", "cs.AI", "cs.LG"},
        "Reasoning/RL": {"cs.CL", "cs.LG", "cs.AI"},
        "Efficiency/Systems": {"cs.LG", "cs.CL", "cs.DC", "cs.AR", "cs.PF", "cs.AI", "cs.CV"},
        "Speech/Audio": {"cs.SD", "eess.AS", "cs.CL", "cs.MM"},
        "Robotics/Embodied": {"cs.RO", "cs.CV", "cs.AI", "cs.LG"},
        "Benchmark/Eval": {"cs.CL", "cs.CV", "cs.LG", "cs.AI", "cs.SE"},
        "Code/Math": {"cs.SE", "cs.CL", "cs.PL", "cs.LG", "cs.AI", "cs.LO"},
        "LLM-core": {"cs.CL", "cs.LG", "cs.AI"},
    }
    test = have[have["sf_from_kw"].isin(COMPAT.keys())].copy()
    test["compatible"] = test.apply(
        lambda r: r["primary_category"] in COMPAT[r["sf_from_kw"]], axis=1)
    compat = float(test["compatible"].mean()) if len(test) else np.nan
    # also report the raw confusion for the appendix
    conf = (test.groupby(["sf_from_kw", "primary_category"]).size()
            .sort_values(ascending=False).head(15))
    RES["subfield_validation"] = {
        "n_overlap": int(len(have)), "n_testable": int(len(test)),
        "compatibility_rate": compat,
        "top_pairs": [{"kw": a, "cat": b, "n": int(n)} for (a, b), n in conf.items()],
    }
    print(f"  n={len(test)} testable; keyword-subfield compatible with arXiv primary "
          f"category in {compat:.1%} of papers")


def over_under(df):
    print("\n=== (8) Overrated / underrated v2 ===")
    d = df.copy()
    att_m = smf.ols("log_upvotes ~ log_age + C(subfield_grp)", data=d).fit()
    imp_m = smf.ols("log_citations ~ log_age + C(subfield_grp)", data=d).fit()
    d["attention_pct"] = (d["log_upvotes"] - att_m.fittedvalues).rank(pct=True)
    d["impact_pct"] = (d["log_citations"] - imp_m.fittedvalues).rank(pct=True)
    d["overrated"] = ((d["attention_pct"] >= 0.66) & (d["impact_pct"] <= 0.34)).astype(int)
    d["underrated"] = ((d["attention_pct"] <= 0.34) & (d["impact_pct"] >= 0.66)).astype(int)
    RES["over_under_v2"] = {"n_overrated": int(d.overrated.sum()),
                            "n_underrated": int(d.underrated.sum())}
    print(f"  overrated={d.overrated.sum()}  underrated={d.underrated.sum()}")

    feats = ["title_n_words", "title_has_colon", "log_n_authors", "log_max_hindex",
             "log_last_hindex", "has_github", "abstract_n_chars", "n_trend_days"] + \
            [c for c in d.columns if c.startswith("kw_")]
    X = StandardScaler().fit_transform(d[feats].fillna(0).astype(float))
    for lab in ["overrated", "underrated"]:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, d[lab])
        coefs = sorted(zip(feats, lr.coef_[0]), key=lambda x: -abs(x[1]))
        RES.setdefault("logit_drivers_v2", {})[lab] = [
            {"feature": f, "coef": float(c)} for f, c in coefs[:10]]
        print(f"  {lab}: " + ", ".join(f"{f}({c:+.2f})" for f, c in coefs[:5]))

    keep = ["arxiv_id_clean", "title", "subfield_grp", "upvotes", "citation_count",
            "influential_citations", "age_months", "max_hindex", "last_author_hindex",
            "has_github", "num_comments", "n_trend_days", "attention_pct", "impact_pct",
            "overrated", "underrated", "release_month", "release_year", "citation_repaired",
            "ai_keywords", "hf_summary"]
    d[[c for c in keep if c in d.columns]].to_csv(
        os.path.join(PROC, "papers_scored_v2.csv"), index=False)
    print("  saved papers_scored_v2.csv")


def descriptives(df):
    RES["descriptives_v2"] = {
        "n_papers": int(len(df)),
        "release_range": [str(df["release_month"].min()), str(df["release_month"].max())],
        "median_upvotes": float(df["upvotes"].median()),
        "median_citations": float(df["citation_count"].median()),
        "spearman": float(df[["upvotes", "citation_count"]].corr(method="spearman").iloc[0, 1]),
        "n_repaired": int(df["citation_repaired"].sum()),
        "median_max_hindex": float(df["max_hindex"].median()),
        "snapshot": "2026-06-11",
    }
    print("=== Descriptives v2 ===")
    for k, v in RES["descriptives_v2"].items():
        print(f"  {k}: {v}")


def main():
    df = load()
    print(f"Analysis sample v2: {len(df)} papers")
    descriptives(df)
    f_main = spec_ladder(df)
    e_value(df)
    placebo(df)
    random_slopes(df)
    matching_ipw(df)
    robustness(df, f_main)
    subfield_validation(df)
    over_under(df)
    with open(os.path.join(PROC, "model_results_v2.json"), "w") as f:
        json.dump(RES, f, indent=2)
    print("\nSaved model_results_v2.json")


if __name__ == "__main__":
    main()
