"""
06_figures.py
-------------
Static figures (PNG) for the report + a self-contained interactive
"overrated vs. underrated" explorer (HTML) that stands in for the Shiny
dashboard milestone. No server required: open the HTML in any browser.
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(PROC, "papers_analysis.csv"))
    df = df[df["citation_count"].notna() & (df["age_months"] >= 5) & (df["age_months"] <= 40)]
    scored = pd.read_csv(os.path.join(PROC, "papers_scored.csv"))
    with open(os.path.join(PROC, "model_results.json")) as f:
        res = json.load(f)

    # --- Fig 1: distribution of upvotes and citations -------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(np.log1p(df["upvotes"]), bins=40, color="#4C72B0")
    ax[0].set_title("Community attention"); ax[0].set_xlabel("log(1+upvotes)")
    ax[1].hist(np.log1p(df["citation_count"]), bins=40, color="#C44E52")
    ax[1].set_title("Scholarly impact"); ax[1].set_xlabel("log(1+citations)")
    fig.suptitle("Both signals are heavy-tailed", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_distributions.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: upvotes vs citations scatter (the core relationship) ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(np.log1p(df["upvotes"]), np.log1p(df["citation_count"]),
                    c=df["age_months"], cmap="viridis", s=14, alpha=0.5)
    ax.set_xlabel("log(1+upvotes)  (community attention)")
    ax.set_ylabel("log(1+citations)  (scholarly impact)")
    r = res["descriptives"]["spearman_upvotes_citations"]
    ax.set_title(f"Attention vs impact (Spearman rho = {r:.2f})")
    plt.colorbar(sc, label="age (months)")
    # add lowess-ish binned mean
    tmp = df.copy(); tmp["bin"] = pd.qcut(tmp["log_upvotes"], 12, duplicates="drop")
    g = tmp.groupby("bin", observed=True).agg(x=("log_upvotes", "mean"), y=("log_citations", "mean"))
    ax.plot(g["x"], g["y"], color="red", lw=2.5, label="binned mean")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_attention_vs_impact.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: beta1 across model specifications -----------------------
    cm = res["count_models"]
    labels, vals, errs = [], [], []
    if "nb_bivariate_beta_logupvotes" in cm:
        labels.append("NB bivariate"); vals.append(cm["nb_bivariate_beta_logupvotes"]); errs.append(0)
    if "nb_full_beta_logupvotes" in cm:
        lo, hi = cm["nb_full_ci"]; v = cm["nb_full_beta_logupvotes"]
        labels.append("NB + controls"); vals.append(v); errs.append((hi - lo) / 2)
    if "bootstrap" in res:
        b = res["bootstrap"]; lo, hi = b["ci95"]
        labels.append("OLS (bootstrap)"); vals.append(b["beta_mean"]); errs.append((hi - lo) / 2)
    if "hierarchical" in res and "beta_logupvotes" in res["hierarchical"]:
        h = res["hierarchical"]
        labels.append("Mixed (subfield RE)"); vals.append(h["beta_logupvotes"]); errs.append(1.96 * h["sd"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(vals, range(len(labels)), xerr=errs, fmt="o", color="#2c3e50", capsize=5, ms=9)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("coefficient on log(upvotes)")
    ax.set_title("Does attention predict citations? beta1 across specifications")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_beta_specs.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 4: citations by subfield -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("subfield")["citation_count"].median().sort_values(ascending=False).index[:12]
    sub = df[df["subfield"].isin(order)]
    sns.boxplot(data=sub, y="subfield", x="log_citations", order=order, ax=ax, color="#55A868")
    ax.set_xlabel("log(1+citations)"); ax.set_ylabel("")
    ax.set_title("Scholarly impact by subfield")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_subfield.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 5: the overrated/underrated quadrants ----------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(scored["attention_pct"], scored["impact_pct"], s=12, alpha=0.35, color="gray")
    over = scored[scored["overrated"] == 1]; under = scored[scored["underrated"] == 1]
    ax.scatter(over["attention_pct"], over["impact_pct"], s=22, color="#C44E52", label=f"overrated (n={len(over)})")
    ax.scatter(under["attention_pct"], under["impact_pct"], s=22, color="#4C72B0", label=f"underrated (n={len(under)})")
    ax.axhline(0.5, color="k", lw=0.6); ax.axvline(0.5, color="k", lw=0.6)
    ax.set_xlabel("attention percentile (upvotes)"); ax.set_ylabel("impact percentile (citations)")
    ax.set_title("Overrated vs underrated papers")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_quadrants.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("Saved 5 figures to figures/")

    # --- Interactive explorer (plotly, self-contained HTML) -------------
    build_dashboard(scored)


def build_dashboard(scored):
    import plotly.express as px
    d = scored.copy()
    d["category"] = np.where(d["overrated"] == 1, "Overrated",
                     np.where(d["underrated"] == 1, "Underrated", "Typical"))
    d["short_title"] = d["title"].str.slice(0, 80)
    d["citations"] = d["citation_count"]
    fig = px.scatter(
        d, x="upvotes", y="citation_count", color="category",
        color_discrete_map={"Overrated": "#C44E52", "Underrated": "#4C72B0", "Typical": "#BBBBBB"},
        hover_data={"short_title": True, "subfield_grp": True, "upvotes": True,
                    "citation_count": True, "age_months": ":.0f",
                    "attention_pct": ":.2f", "impact_pct": ":.2f"},
        log_x=True, log_y=True, opacity=0.7,
        title="Overrated vs. Underrated arXiv/ML Papers — Community Attention vs. Citations",
        labels={"upvotes": "HF Daily-Papers upvotes (log)", "citation_count": "Semantic Scholar citations (log)"},
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=720, legend_title="", template="plotly_white")
    out = os.path.join(FIG, "overrated_explorer.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved interactive explorer -> {out}")


if __name__ == "__main__":
    main()
