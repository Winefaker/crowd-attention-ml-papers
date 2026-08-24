"""
16_figures_v2.py
----------------
Paper-grade figures for the v2 analysis. All read from the saved result JSONs and
processed CSVs (no model re-fitting), so they are cheap to regenerate.

  fig_v2_1_spec_ladder.png   coefficient ladder + placebo (the central evidence)
  fig_v2_2_scatter_cohorts.png  attention vs impact with per-cohort binned means
  fig_v2_3_trending_vs_background.png  citation CCDFs, trending vs control
  fig_v2_4_random_slopes.png  attention slope by subfield (forest)
  fig_v2_5_prediction_lift.png  out-of-sample lift from attention
  fig_v2_6_text_themes.png   discriminative terms, overrated vs underrated
"""
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def jload(name):
    with open(os.path.join(PROC, name)) as f:
        return json.load(f)


def fig1_spec_ladder(res):
    sl = res["spec_ladder"]
    labels = {
        "M0_bivariate": "M0: bivariate",
        "M1_age_field": "M1: + age, subfield FE",
        "M2_v1_proxy": "M2: + recurrence prestige (v1)",
        "M3_hindex": "M3: + author h-index (main)",
        "M4_monthFE": "M4: month FE (robustness)",
    }
    rows = []
    for k, lab in labels.items():
        if "beta" in sl.get(k, {}):
            rows.append((lab, sl[k]["beta"], sl[k]["ci"][0], sl[k]["ci"][1], "#2c3e50"))
    p = res.get("placebo", {})
    if "beta" in p:
        rows.append(("Placebo: reference count", p["beta"],
                     p["beta"] - 1.96 * p["se"], p["beta"] + 1.96 * p["se"], "#C44E52"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (lab, b, lo, hi, col) in enumerate(rows[::-1]):
        ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="o", color=col, capsize=4, ms=7)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlabel(r"coefficient on log(1+upvotes)  [NB for citations; OLS for placebo]")
    ax.set_title("Attention's association with citations survives every control set;\n"
                 "the placebo outcome shows ~nothing", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_1_spec_ladder.png"), dpi=160)
    plt.close(fig)


def fig2_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.scatter(df["log_upvotes"], df["log_citations"], s=6, alpha=0.18, color="#777")
    pal = {2023: "#C44E52", 2024: "#4C72B0", 2025: "#55A868"}
    for yr, g in df.groupby("release_year"):
        if yr not in pal or len(g) < 200:
            continue
        g = g.copy()
        g["bin"] = pd.qcut(g["log_upvotes"], 10, duplicates="drop")
        m = g.groupby("bin", observed=True).agg(x=("log_upvotes", "mean"),
                                                y=("log_citations", "mean"))
        ax.plot(m["x"], m["y"], color=pal[yr], lw=2.2, marker="o", ms=4, label=f"{int(yr)} cohort")
    ax.set_xlabel("log(1+upvotes)")
    ax.set_ylabel("log(1+citations), snapshot 2026-06-11")
    ax.legend(title="binned means")
    ax.set_title("The attention-impact gradient is stable across release cohorts")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_2_scatter_cohorts.png"), dpi=160)
    plt.close(fig)


def fig3_control(ctrl):
    meta = pd.read_csv(os.path.join(RAW, "arxiv_control.csv"), dtype={"arxiv_id_clean": str})
    s2 = pd.read_csv(os.path.join(RAW, "arxiv_control_s2.csv"), dtype={"arxiv_id_clean": str})
    c = meta.merge(s2, on="arxiv_id_clean")
    c = c[c["ss_found"] == 1]
    t = pd.read_csv(os.path.join(PROC, "papers_v2.csv"), dtype={"arxiv_id_clean": str})
    t = t[t["citation_count"].notna()]
    sc = pd.read_csv(os.path.join(PROC, "papers_scored_v2.csv"), dtype={"arxiv_id_clean": str})
    und = sc[sc["underrated"] == 1]["citation_count"].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    for series, lab, col in [(c["citation_count"].dropna(), "background arXiv (control)", "#999999"),
                             (t["citation_count"].dropna(), "trending on HF", "#4C72B0"),
                             (und, "trending + 'underrated'", "#C44E52")]:
        x = np.sort(series.values)
        ccdf = 1 - np.arange(1, len(x) + 1) / len(x)
        ax.plot(x + 1, ccdf, lw=2, label=f"{lab} (n={len(x):,})", color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("1 + citations"); ax.set_ylabel("P(X > x)  (CCDF)")
    ax.set_title("Trending papers dominate the citation distribution of the background;\n"
                 "even 'underrated' trending papers beat most of it", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_3_trending_vs_background.png"), dpi=160)
    plt.close(fig)


def fig4_slopes(res):
    rs = res.get("random_slopes", {})
    slopes = rs.get("subfield_slopes")
    if not slopes:
        return
    items = sorted(slopes.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4C72B0")
    ax.axvline(rs["fixed_beta"], color="#C44E52", ls="--", lw=1.5,
               label=f"pooled slope {rs['fixed_beta']:.2f}")
    ax.set_xlabel("attention slope on log-citations (random-slope BLUP)")
    ax.set_title("Where the crowd is most informative, by subfield")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_4_random_slopes.png"), dpi=160)
    plt.close(fig)


def fig5_prediction(pred):
    rows = []
    for m in ["ridge", "gbm"]:
        for fs, lab in [("controls_only", "controls"), ("controls_plus_attention", "+ attention")]:
            r = pred[f"{m}__{fs}"]
            rows.append((m, lab, r["auc_top_decile"], r["precision_at_100"]))
    dfp = pd.DataFrame(rows, columns=["model", "features", "AUC", "P@100"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ["AUC", "P@100"]):
        sns.barplot(data=dfp, x="model", y=metric, hue="features",
                    palette=["#999999", "#4C72B0"], ax=ax)
        ax.set_title(f"2025 out-of-sample: {metric}")
        ax.set_ylim(0, 1)
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.2f", fontsize=9)
    axes[0].set_ylabel("AUC, top-decile-by-citations")
    axes[1].set_ylabel("precision in top-100 picks")
    fig.suptitle("Day-of-release attention adds real predictive lift", y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_5_prediction_lift.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig6_themes(themes):
    ov = [(t["term"], t["coef"]) for t in themes["overrated_terms"][:12]]
    un = [(t["term"], abs(t["coef"])) for t in themes["underrated_terms"][:12]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=False)
    for ax, data, col, lab in [(axes[0], ov, "#C44E52", "predicts OVERRATED"),
                               (axes[1], un, "#4C72B0", "predicts UNDERRATED")]:
        data = data[::-1]
        ax.barh([t for t, _ in data], [c for _, c in data], color=col)
        ax.set_title(lab)
        ax.set_xlabel("|L1-logistic coefficient|")
    fig.suptitle("What the community over- and under-values (TF-IDF terms)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_6_text_themes.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    res = jload("model_results_v2.json")
    df = pd.read_csv(os.path.join(PROC, "papers_v2.csv"), dtype={"arxiv_id_clean": str})
    df = df[df["citation_count"].notna() & (df["age_months"] >= 5) & (df["age_months"] <= 40)]
    fig1_spec_ladder(res)
    fig2_scatter(df)
    fig4_slopes(res)
    try:
        fig3_control(jload("control_results.json"))
    except FileNotFoundError:
        print("control results not ready; skipping fig3")
    try:
        fig5_prediction(jload("prediction_results.json"))
    except FileNotFoundError:
        print("prediction results not ready; skipping fig5")
    try:
        fig6_themes(jload("text_themes.json"))
    except FileNotFoundError:
        print("text themes not ready; skipping fig6")
    print("figures written to figures/")


if __name__ == "__main__":
    main()
