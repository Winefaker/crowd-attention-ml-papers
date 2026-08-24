"""
05_models.py
------------
The statistical analysis from the proposal:

 (A) Count model for scholarly impact:
       negative-binomial regression of citation_count on the attention signal
       (log upvotes) plus controls (author-prestige proxy, n_authors, paper age,
       subfield fixed effects). The headline quantity is beta1 on log_upvotes:
       does community attention still predict citations after controls?

 (B) Hierarchical version: subfield random intercept (Poisson mixed GLM), to
       respect the "papers grouped within subfields" structure.

 (C) Selection adjustment: treat "high attention" as a pseudo-treatment, build a
       propensity score from prestige + subfield + age, nearest-neighbour match,
       and compare the naive attention-citation gap with the matched (ATT) gap.

 (D) Bootstrap confidence interval for beta1.

 (E) Overrated / underrated: residual-from-attention labels + a logistic model
       that predicts which papers are over/underrated from title/author features.

Writes: data/processed/model_results.json, data/processed/papers_scored.csv
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
    df = pd.read_csv(os.path.join(PROC, "papers_analysis.csv"))
    # analysis sample: must have citations and a sensible age (>=5 months old)
    df = df[df["citation_count"].notna()].copy()
    df = df[df["age_months"] >= 5].copy()
    df = df[df["upvotes"].notna()].copy()
    # cap absurd ages (bad timestamps)
    df = df[df["age_months"] <= 40].copy()
    df["log_age"] = np.log(df["age_months"].clip(lower=1))
    df["n_authors"] = df["n_authors"].fillna(df["n_authors"].median())
    df["log_n_authors"] = np.log1p(df["n_authors"])
    # keep subfields with at least 30 papers; others -> "Other"
    vc = df["subfield"].value_counts()
    keep = vc[vc >= 30].index
    df["subfield_grp"] = np.where(df["subfield"].isin(keep), df["subfield"], "Other")
    return df.reset_index(drop=True)


def count_models(df):
    print("\n=== (A) Count models: citations ~ attention + controls ===")
    base_terms = "log_upvotes"
    ctrl_terms = ("log_upvotes + log_author_max_appear + log_n_authors + log_age "
                  "+ has_github + C(subfield_grp)")
    out = {}

    # bivariate NB
    try:
        m0 = smf.negativebinomial(f"citation_count ~ {base_terms}", data=df).fit(disp=0)
        out["nb_bivariate_beta_logupvotes"] = float(m0.params["log_upvotes"])
        out["nb_bivariate_p"] = float(m0.pvalues["log_upvotes"])
    except Exception as e:
        print("  bivariate NB failed:", e)

    # full NB with controls
    try:
        m1 = smf.negativebinomial(f"citation_count ~ {ctrl_terms}", data=df).fit(disp=0)
        ci = m1.conf_int().loc["log_upvotes"].tolist()
        out["nb_full_beta_logupvotes"] = float(m1.params["log_upvotes"])
        out["nb_full_ci"] = [float(ci[0]), float(ci[1])]
        out["nb_full_p"] = float(m1.pvalues["log_upvotes"])
        out["nb_full_irr"] = float(np.exp(m1.params["log_upvotes"]))  # incidence-rate ratio
        out["nb_full_prestige_beta"] = float(m1.params["log_author_max_appear"])
        out["nb_full_age_beta"] = float(m1.params["log_age"])
        out["nb_full_nobs"] = int(m1.nobs)
        out["nb_full_summary"] = m1.summary().as_text()
        print(f"  NB bivariate  beta(log_upvotes) = {out.get('nb_bivariate_beta_logupvotes'):.3f}")
        print(f"  NB +controls  beta(log_upvotes) = {out['nb_full_beta_logupvotes']:.3f} "
              f"(IRR={out['nb_full_irr']:.2f}, p={out['nb_full_p']:.1e}, n={out['nb_full_nobs']})")
        print(f"    -> doubling upvotes multiplies expected citations by "
              f"{2**out['nb_full_beta_logupvotes']:.2f}x")
    except Exception as e:
        print("  full NB failed:", e)

    # OLS on log citations (interpretable elasticity, used for bootstrap & residuals)
    m2 = smf.ols(f"log_citations ~ {ctrl_terms}", data=df).fit()
    out["ols_log_beta_logupvotes"] = float(m2.params["log_upvotes"])
    out["ols_log_r2"] = float(m2.rsquared)
    # model WITHOUT upvotes (expected impact from prestige/subfield/age) for residuals
    m_noatt = smf.ols("log_citations ~ log_author_max_appear + log_n_authors + log_age "
                      "+ has_github + C(subfield_grp)", data=df).fit()
    df["pred_log_cit_no_attention"] = m_noatt.fittedvalues
    out["ols_noatt_r2"] = float(m_noatt.rsquared)
    print(f"  OLS log-citations elasticity wrt log_upvotes = {out['ols_log_beta_logupvotes']:.3f} "
          f"(R2={out['ols_log_r2']:.3f}); without upvotes R2={out['ols_noatt_r2']:.3f}")

    RES["count_models"] = out
    return df, m2


def hierarchical(df):
    print("\n=== (B) Hierarchical Poisson (subfield random intercept) ===")
    try:
        # Poisson mixed GLM with random intercept per subfield
        md = sm.PoissonBayesMixedGLM.from_formula(
            "citation_count ~ log_upvotes + log_author_max_appear + log_n_authors + log_age + has_github",
            {"subfield_grp": "0 + C(subfield_grp)"}, data=df)
        mres = md.fit_vb()
        # locate the fixed effect for log_upvotes
        names = list(mres.model.exog_names)
        idx = names.index("log_upvotes")
        beta = float(mres.fe_mean[idx])
        sd = float(mres.fe_sd[idx])
        RES["hierarchical"] = {
            "beta_logupvotes": beta, "sd": sd,
            "ci": [beta - 1.96 * sd, beta + 1.96 * sd],
            "note": "Poisson mixed GLM (variational Bayes), subfield random intercept",
        }
        print(f"  mixed-model beta(log_upvotes) = {beta:.3f} +/- {sd:.3f}")
    except Exception as e:
        print("  hierarchical model failed (reporting subfield fixed-effects instead):", e)
        RES["hierarchical"] = {"error": str(e)}


def matching_selection(df):
    print("\n=== (C) Selection adjustment via propensity matching ===")
    d = df.copy()
    # pseudo-treatment: top-tertile attention vs bottom-tertile attention
    hi = d["upvotes"].quantile(2 / 3)
    lo = d["upvotes"].quantile(1 / 3)
    d = d[(d["upvotes"] >= hi) | (d["upvotes"] <= lo)].copy()
    d["treat"] = (d["upvotes"] >= hi).astype(int)

    naive = d.loc[d.treat == 1, "log_citations"].mean() - d.loc[d.treat == 0, "log_citations"].mean()

    # propensity score from prestige + subfield + age + n_authors (NOT upvotes)
    sub_d = pd.get_dummies(d["subfield_grp"], prefix="sf", drop_first=True)
    X = pd.concat([d[["log_author_max_appear", "log_n_authors", "log_age", "has_github"]].reset_index(drop=True),
                   sub_d.reset_index(drop=True)], axis=1).astype(float)
    y = d["treat"].values
    Xs = StandardScaler().fit_transform(X)
    ps = LogisticRegression(max_iter=1000).fit(Xs, y).predict_proba(Xs)[:, 1]
    d = d.reset_index(drop=True)
    d["ps"] = ps

    # 1-NN match each treated unit to a control on propensity score
    treated = d[d.treat == 1].reset_index(drop=True)
    control = d[d.treat == 0].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]].values)
    dist, idx = nn.kneighbors(treated[["ps"]].values)
    matched_control = control.iloc[idx.flatten()].reset_index(drop=True)
    att = (treated["log_citations"].values - matched_control["log_citations"].values).mean()

    RES["selection"] = {
        "naive_gap_log_citations": float(naive),
        "matched_att_log_citations": float(att),
        "n_treated": int(treated.shape[0]),
        "interpretation": "gap in log-citations between high- and low-attention papers, "
                          "before vs after matching on prestige/subfield/age",
    }
    print(f"  naive high-vs-low attention gap (log cit) = {naive:.3f}")
    print(f"  after matching on prestige/subfield/age   = {att:.3f}")
    if naive:
        print(f"  -> confounders explain {100*(1-att/naive):.0f}% of the raw gap; "
              f"{100*att/naive:.0f}% remains after matching")


def bootstrap_beta(df, n_boot=600):
    print(f"\n=== (D) Bootstrap CI for beta1 ({n_boot} resamples) ===")
    formula = ("log_citations ~ log_upvotes + log_author_max_appear + log_n_authors "
               "+ log_age + has_github + C(subfield_grp)")
    betas = []
    rng = np.random.default_rng(42)
    n = len(df)
    for b in range(n_boot):
        samp = df.iloc[rng.integers(0, n, n)]
        try:
            m = smf.ols(formula, data=samp).fit()
            betas.append(m.params["log_upvotes"])
        except Exception:
            continue
    betas = np.array(betas)
    ci = np.percentile(betas, [2.5, 97.5]).tolist()
    RES["bootstrap"] = {
        "beta_mean": float(betas.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "n_boot": len(betas),
        "note": "OLS log-citations elasticity wrt log_upvotes, paper-level resampling",
    }
    print(f"  beta1 = {betas.mean():.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")


def over_under_rated(df):
    print("\n=== (E) Overrated / Underrated analysis ===")
    d = df.copy()
    # residual of actual citations vs what ATTENTION (+age/subfield) predicts
    pred = smf.ols("log_citations ~ log_upvotes + log_age + C(subfield_grp)", data=d).fit()
    d["impact_residual"] = d["log_citations"] - pred.fittedvalues

    # AGE-ADJUSTED labels: a young paper with high upvotes and few citations is
    # not necessarily "overrated" -- it just has not had time. So we residualise
    # BOTH attention and impact on paper age + subfield, then rank the residuals.
    # This compares each paper to others of similar age and field.
    att_m = smf.ols("log_upvotes ~ log_age + C(subfield_grp)", data=d).fit()
    imp_m = smf.ols("log_citations ~ log_age + C(subfield_grp)", data=d).fit()
    d["attention_adj"] = d["log_upvotes"] - att_m.fittedvalues
    d["impact_adj"] = d["log_citations"] - imp_m.fittedvalues
    d["attention_pct"] = d["attention_adj"].rank(pct=True)
    d["impact_pct"] = d["impact_adj"].rank(pct=True)
    # raw (un-adjusted) ranks kept for comparison / the scatter axes
    d["attention_pct_raw"] = d["upvotes"].rank(pct=True)
    d["impact_pct_raw"] = d["citation_count"].rank(pct=True)
    d["overrated"] = ((d["attention_pct"] >= 0.66) & (d["impact_pct"] <= 0.34)).astype(int)
    d["underrated"] = ((d["attention_pct"] <= 0.34) & (d["impact_pct"] >= 0.66)).astype(int)

    RES["over_under"] = {
        "n_overrated": int(d["overrated"].sum()),
        "n_underrated": int(d["underrated"].sum()),
        "pct_overrated": float(d["overrated"].mean()),
        "pct_underrated": float(d["underrated"].mean()),
    }
    print(f"  overrated (high attn, low cites): {d['overrated'].sum()} papers "
          f"| underrated (low attn, high cites): {d['underrated'].sum()} papers")

    # logistic models: which features predict over/under-rated?
    feat_cols = ["title_n_words", "title_has_colon", "log_n_authors",
                 "log_author_max_appear", "has_github", "abstract_n_chars",
                 "num_comments"] + [c for c in d.columns if c.startswith("kw_")]
    feat_cols = [c for c in feat_cols if c in d.columns]
    Xf = d[feat_cols].fillna(0).astype(float)
    Xs = StandardScaler().fit_transform(Xf)

    for label in ["overrated", "underrated"]:
        if d[label].sum() < 15:
            continue
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, d[label])
        coefs = sorted(zip(feat_cols, lr.coef_[0]), key=lambda x: -abs(x[1]))
        RES.setdefault("logit_drivers", {})[label] = [
            {"feature": f, "coef": float(c)} for f, c in coefs[:8]]
        top = ", ".join(f"{f}({c:+.2f})" for f, c in coefs[:5])
        print(f"  {label} top drivers: {top}")

    # save scored papers for the dashboard / inspection
    keep_cols = ["arxiv_id_clean", "title", "subfield_grp", "upvotes", "citation_count",
                 "influential_citations", "age_months", "log_author_max_appear",
                 "has_github", "num_comments", "attention_pct", "impact_pct",
                 "attention_pct_raw", "impact_pct_raw",
                 "impact_residual", "overrated", "underrated", "release_month",
                 "pred_log_cit_no_attention"]
    keep_cols = [c for c in keep_cols if c in d.columns]
    d[keep_cols].to_csv(os.path.join(PROC, "papers_scored.csv"), index=False)
    print("  saved data/processed/papers_scored.csv")


def descriptives(df):
    RES["descriptives"] = {
        "n_papers": int(len(df)),
        "date_range": [str(df["release_month"].min()), str(df["release_month"].max())],
        "median_upvotes": float(df["upvotes"].median()),
        "median_citations": float(df["citation_count"].median()),
        "mean_citations": float(df["citation_count"].mean()),
        "spearman_upvotes_citations": float(df[["upvotes", "citation_count"]].corr(method="spearman").iloc[0, 1]),
        "pearson_logupvotes_logcit": float(df[["log_upvotes", "log_citations"]].corr().iloc[0, 1]),
        "n_subfields": int(df["subfield_grp"].nunique()),
        "pct_with_github": float(df["has_github"].mean()),
    }
    print("\n=== Descriptives ===")
    for k, v in RES["descriptives"].items():
        print(f"  {k}: {v}")


def main():
    df = load()
    print(f"Analysis sample: {len(df)} papers")
    descriptives(df)
    df, _ = count_models(df)
    hierarchical(df)
    matching_selection(df)
    bootstrap_beta(df)
    over_under_rated(df)
    with open(os.path.join(PROC, "model_results.json"), "w") as f:
        json.dump(RES, f, indent=2)
    print("\nSaved data/processed/model_results.json")


if __name__ == "__main__":
    main()
