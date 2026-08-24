#!/usr/bin/env python
"""
make_figures_v3.py -- the analysis charts.

Inputs (read-only): results/prediction_v3.json, results/prediction_v3_scores.csv,
results/crowding_iv_v3.json, results/association_v3.json,
data/processed/analysis_final.csv, data/processed/overunder_v3.csv and the
never-trending control files under Project/data/raw (joined exactly as in
27_association_v3.py).

Outputs: figures/<name>.pdf + <name>.png (report, Latin Modern Roman,
figsize = printed width, 8-9 pt) and slide_<name>.png (16:9 deck boxes at 200 dpi,
14-16 pt) and FIGURES_INDEX.md.

Every number annotated in a figure is read from the JSON files (single source of
truth = results/D5_results_gate_v3.md ledger, which is generated from the same JSON).
Run:  ../.venv/bin/python scripts/make_figures_v3.py
"""
from __future__ import annotations

import glob
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

warnings.filterwarnings("ignore", message="Glyph")

# --------------------------------------------------------------------------- paths
BASE = Path(__file__).resolve().parents[1]
AUG = BASE
ROOT = AUG
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

PRED = json.load(open(RES / "prediction_v3.json"))
IV = json.load(open(RES / "crowding_iv_v3.json"))
ASSOC = json.load(open(RES / "association_v3.json"))
SCORES = pd.read_csv(RES / "prediction_v3_scores.csv", dtype={"arxiv_id_clean": str})
DF = pd.read_csv(ROOT / "data/processed/analysis_final.csv", dtype={"arxiv_id_clean": str})
OU = pd.read_csv(ROOT / "data/processed/overunder_v3.csv", dtype={"arxiv_id_clean": str})
CTRL_META = BASE / "data/raw/arxiv_control.csv"     # read-only
CTRL_S2 = BASE / "data/raw/arxiv_control_s2.csv"    # read-only
SNAPSHOT = pd.Timestamp("2026-06-11")                        # as in 27_association_v3.py

# --------------------------------------------------------------------------- palette
# Okabe-Ito, validated with the dataviz skill validator (all-pairs, light surface):
# blue = attention / +attention, vermillion = controls only / baseline / background,
# green = third estimator, grey = neutral (upvotes only, reference), ink = text.
PAL = {
    "blue": "#0072B2",       # attention, +attention, underrated, main estimate
    "orange": "#D55E00",     # controls only, baseline, never trending, overrated
    "green": "#009E73",      # third series (OLS ratio)
    "grey": "#7A7A7A",       # neutral series (upvotes only), reference lines
    "lightgrey": "#BDBDBD",  # flagged / mechanical rows
    "band": "#E8E8E8",       # shaded reference bands
    "bandblue": "#DCEAF5",   # highlight band for the main spec
    "ink": "#111111",
    "muted": "#666666",
    "grid": "#E4E4E4",
    "axis": "#9A9A9A",
}
SLIDE_PAL = dict(PAL)  # one place to re-map hues if the deck adopts other tokens

# --------------------------------------------------------------------------- fonts
LM_DIR = Path.home() / "Library/TinyTeX/texmf-dist/fonts/opentype/public/lm"
for f in glob.glob(str(LM_DIR / "lmroman10-*.otf")):
    fm.fontManager.addfont(f)
PPT_FONTS = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
for f in ["Calibri.ttf", "Calibrib.ttf", "Calibrii.ttf", "Calibriz.ttf"]:
    if (PPT_FONTS / f).exists():
        fm.fontManager.addfont(str(PPT_FONTS / f))
HAVE_LM = any(f.name == "Latin Modern Roman" for f in fm.fontManager.ttflist)
HAVE_CALIBRI = any(f.name == "Calibri" for f in fm.fontManager.ttflist)

TEXTWIDTH = 6.3   # in, print text width
BOX = {"wide": (8.0, 5.1), "stat": (7.6, 3.55), "full": (12.13, 4.6)}  # slide boxes (in)


def style(ctx: str) -> None:
    """One rcParams block per context. ctx = 'report' | 'slide'."""
    plt.rcdefaults()
    if ctx == "report":
        fam = ["Latin Modern Roman", "DejaVu Serif"] if HAVE_LM else ["DejaVu Serif"]
        base, tick, leg = 8.5, 8, 8
        mfs = "cm"
        lw, ms, alw = 1.2, 4.5, 0.6
    else:
        fam = ["Calibri", "Arial", "DejaVu Sans"] if HAVE_CALIBRI else ["Arial", "DejaVu Sans"]
        base, tick, leg = 16, 14, 14
        mfs = "stixsans"
        lw, ms, alw = 2.2, 8, 1.0
    plt.rcParams.update({
        "font.family": fam, "font.size": base, "mathtext.fontset": mfs,
        "axes.labelsize": base, "axes.titlesize": base, "xtick.labelsize": tick,
        "ytick.labelsize": tick, "legend.fontsize": leg, "legend.title_fontsize": leg,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": PAL["axis"], "axes.linewidth": alw,
        "xtick.color": PAL["axis"], "ytick.color": PAL["axis"],
        "xtick.labelcolor": PAL["ink"], "ytick.labelcolor": PAL["ink"],
        "xtick.major.width": alw, "ytick.major.width": alw,
        "xtick.major.size": 3 if ctx == "report" else 5,
        "ytick.major.size": 3 if ctx == "report" else 5,
        "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"],
        "axes.grid": False, "grid.color": PAL["grid"], "grid.linewidth": alw, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": lw, "lines.markersize": ms, "lines.markeredgewidth": 0,
        "errorbar.capsize": 0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200, "savefig.bbox": None, "pdf.fonttype": 3,
        "axes.unicode_minus": True,
    })


def ygrid(ax):
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)


def save(fig, name, ctx):
    if ctx == "report":
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=200)
    else:
        fig.savefig(OUT / f"slide_{name}.png", dpi=200)
    plt.close(fig)


def sz(ctx, report_size, slide_box):
    return report_size if ctx == "report" else BOX[slide_box]


def small(ctx):
    return 7.5 if ctx == "report" else 13


def hl(ci):
    """error bar half-lengths from [lo, hi] around a point (returned as 2xN)."""
    return ci


# =========================================================================== fig 1
def fig_dose_response(ctx):
    rows = ASSOC["dose_response_deciles"]["rows"]
    d = pd.DataFrame(rows)
    x = d.decile.values.astype(float)
    raw = np.exp(d.mean_log1p_cites.values)
    raw_lo = np.exp(np.array([c[0] for c in d.ci95]))
    raw_hi = np.exp(np.array([c[1] for c in d.ci95]))
    adj = np.exp(d.mean_log1p_cites_adjusted.values)
    adj_lo = np.exp(np.array([c[0] for c in d.ci95_adjusted]))
    adj_hi = np.exp(np.array([c[1] for c in d.ci95_adjusted]))
    share = d.share_top_decile_cites_in_quarter.values * 100
    rho = ASSOC["dose_response_deciles"]["spearman_upvotes_cites"]
    n = ASSOC["measurement"]["descriptives"]["n"]

    fig = plt.figure(figsize=sz(ctx, (TEXTWIDTH, 3.6), "wide"))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.0], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)
    off = 0.13
    ax.errorbar(x - off, raw, yerr=[raw - raw_lo, raw_hi - raw], fmt="o", color=PAL["orange"],
                ecolor=PAL["orange"], elinewidth=plt.rcParams["lines.linewidth"], label="raw")
    ax.plot(x - off, raw, color=PAL["orange"], lw=plt.rcParams["lines.linewidth"] * 0.8, alpha=0.9)
    ax.errorbar(x + off, adj, yerr=[adj - adj_lo, adj_hi - adj], fmt="s", color=PAL["blue"],
                ecolor=PAL["blue"], elinewidth=plt.rcParams["lines.linewidth"],
                label="adjusted for age, subfield and release month")
    ax.plot(x + off, adj, color=PAL["blue"], lw=plt.rcParams["lines.linewidth"] * 0.8, alpha=0.9)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50])
    ax.set_yticklabels(["5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(5.5, 70)
    ax.set_ylabel("geometric mean of 1 + citations\n(mean log(1 + citations), 95% CI)")
    ygrid(ax)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", title=None)
    ax.text(0.99, 0.04, f"Spearman rank correlation {rho:.3f}, n {n}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=small(ctx), color=PAL["muted"])
    # bottom strip: share of papers in the top citation decile (within release quarter)
    axb.bar(x, share, width=0.6, color=PAL["blue"], alpha=0.55, linewidth=0)
    axb.set_ylim(0, 44)
    axb.set_yticks([0, 20, 40])
    axb.set_yticklabels(["0", "20", "40%"])
    axb.set_ylabel("share in top\ncitation decile")
    ygrid(axb)
    for i in [0, len(x) - 1]:
        axb.text(x[i], share[i] + 2, f"{share[i]:.1f}%", ha="center", va="bottom", fontsize=small(ctx))
    axb.set_xticks(x)
    axb.set_xticklabels([f"{int(a)}\n{int(lo)} to {int(hi)}" for a, lo, hi in zip(x, d.upvotes_min, d.upvotes_max)])
    axb.set_xlabel("upvote decile (upvote range)")
    axb.set_xlim(0.35, 10.65)
    return fig


# =========================================================================== fig 2
def fig_ladder(ctx):
    order = ["M0_raw", "M1_age", "M2_subfield", "M3_month_dow", "M4_tierB_prestige",
             "M5_plus_hindex_LEAKY", "M6_plus_text"]
    labels = ["raw", "+ age", "+ subfield", "+ month and weekday", "+ prior output (leakage free)",
              "+ h-index (measured today)", "+ text and format"]
    L = ASSOC["ladder"]
    series = [
        ("Poisson QMLE", "poisson_qmle", "irr_per_doubling", "irr_per_doubling_ci95", PAL["blue"], "o"),
        ("NB2", "nb2", "irr_per_doubling", "irr_per_doubling_ci95", PAL["orange"], "s"),
        ("OLS on log(1 + citations)", "ols_log1p", "ratio_1p_cites_per_doubling",
         "ratio_1p_cites_per_doubling_ci95", PAL["green"], "D"),
    ]
    plac = ASSOC["placebo_permutation_M4"]["poisson"]["placebo_irr2x_range"]
    fig, ax = plt.subplots(figsize=sz(ctx, (TEXTWIDTH, 3.4), "wide"))
    ys = np.arange(len(order))[::-1].astype(float)
    xmin, xmax = 0.85, 2.6
    # background bands: main spec (blue), leaky specs (grey hatched), placebo range (grey)
    ax.axhspan(ys[4] - 0.5, ys[4] + 0.5, color=PAL["bandblue"], lw=0, zorder=0)
    ax.axhspan(ys[6] - 0.5, ys[5] + 0.5, facecolor="white", edgecolor=PAL["lightgrey"], hatch="////", lw=0,
               zorder=0, alpha=0.6)
    ax.axvspan(plac[0], plac[1], color=PAL["band"], lw=0, zorder=0)
    ax.axvline(1.0, ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4, zorder=1)
    offs = [0.24, 0.0, -0.24]
    for (name, key, pk, ck, col, mk), o in zip(series, offs):
        x = np.array([L[k][key][pk] for k in order])
        lo = np.array([L[k][key][ck][0] for k in order])
        hi = np.array([L[k][key][ck][1] for k in order])
        ax.errorbar(x, ys + o, xerr=[x - lo, hi - x], fmt=mk, color=col, ecolor=col,
                    elinewidth=plt.rcParams["lines.linewidth"], label=name, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[4].set_fontweight("bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlabel("citation ratio per doubling of upvotes (1 = no association)\n95% CI, clustered by release month")
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fs = small(ctx)
    ax.text(xmax - 0.02, ys[4], "main", ha="right", va="center", fontsize=fs, color=PAL["ink"])
    handles, labs = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=PAL["band"], edgecolor="none", label="placebo range (permuted upvotes)"))
    handles.append(Patch(facecolor="white", edgecolor=PAL["lightgrey"], hatch="////",
                         label="control measured after the outcome"))
    labs += [h.get_label() for h in handles[-2:]]
    fig.legend(handles, labs, loc="outside upper left", ncol=2, columnspacing=1.5, handletextpad=0.5)
    return fig


# =========================================================================== fig 3
def fig_roc_pr(ctx):
    f = SCORES[SCORES.split == "forward_test_2025"]
    y = f.y_q.values
    H = PRED["headline"]["primary"]
    cols = [
        ("controls only", "forward_yq__logistic__P_tierB__controls_only", PAL["orange"], "-",
         H["controls_only_auc"], H["controls_only_pr_auc"]),
        ("upvotes only", "forward_yq__logistic__P_tierB__upvotes_only", PAL["grey"], "--",
         H["upvotes_only_auc"], H["upvotes_only_pr_auc"]),
        ("controls + attention", "forward_yq__logistic__P_tierB__+attention", PAL["blue"], "-",
         H["attention_auc"], H["attention_pr_auc"]),
    ]
    base = PRED["experiments"]["forward_yq"]["base_rate_test"]
    fig, (a, b) = plt.subplots(1, 2, figsize=sz(ctx, (TEXTWIDTH, 3.35), "full"))
    a.plot([0, 1], [0, 1], ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4)  # chance diagonal
    b.axhline(base, ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4, label=f"base rate {base:.3f}")
    for name, c, col, ls, auc, prauc in cols:
        s = f[c].values
        fpr, tpr, _ = roc_curve(y, s)
        a.plot(fpr, tpr, color=col, ls=ls, label=f"{name} (AUC {auc:.3f})")
        pr, rc, _ = precision_recall_curve(y, s)
        b.plot(rc, pr, color=col, ls=ls, label=f"{name} (PR AUC {prauc:.3f})")
    a.set_xlabel("false positive rate")
    a.set_ylabel("true positive rate")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_aspect("equal")
    a.legend(loc="lower right")
    b.set_xlabel("recall")
    b.set_ylabel("precision")
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.set_aspect("equal")
    b.legend(loc="upper right")
    n = PRED["experiments"]["forward_yq"]["n_test"]
    fs = small(ctx)
    if ctx == "report":  # on slides the kicker/title carries this
        fig.supxlabel(f"logistic model, test cohort release year 2025 (n {n}), label = top citation decile within release quarter",
                      fontsize=fs, color=PAL["muted"], x=0.5, ha="center")
    a.text(-0.12, 1.02, "(a)", transform=a.transAxes, ha="left", va="bottom", fontweight="bold")
    b.text(-0.12, 1.02, "(b)", transform=b.transAxes, ha="left", va="bottom", fontweight="bold")
    return fig


# =========================================================================== fig 4
def _delta(exp, model, sub, branch="P_tierB"):
    r = PRED["experiments"][exp]["models"][model][sub]["rows"][f"{branch}|+attention"]["delta_vs_controls_only"]
    return r["d_auc"], r["ci"]["month_cluster"]["auc"]


def fig_delta_forest(ctx):
    spec = [  # (label, exp, sub, branch, group)
        ("logistic (prespecified headline)", "forward_yq", "main", "P_tierB", "head"),
        ("gradient boosting", "forward_yq", "main", "P_tierB", "head"),
        ("prestige control: career count (anachronistic)", "forward_yq", "main", "P_interim", "rob"),
        ("no prestige control", "forward_yq", "main", "P_none", "rob"),
        ("mature subset, at least 12 months old", "forward_yq", "mature_k12", "P_tierB", "rob"),
        ("label ranked within month", "forward_ym", "main", "P_tierB", "rob"),
        ("label: influential citations", "forward_yinf", "main", "P_tierB", "rob"),
        ("backward test, release year 2023 cohort", "backward_yq", "main", "P_tierB", "rob"),
        ("age dropped from the controls", "drop_age", "main", "P_tierB", "rob"),
        ("training excludes launch era months", "no_launch_months", "main", "P_tierB", "rob"),
        ("legacy subfield taxonomy (flagged, leaky)", "legacy_subfield", "main", "P_tierB", "flag"),
    ]
    rows = []
    for lab, exp, sub, br, grp in spec:
        if grp == "head":
            model = "logistic" if lab.startswith("logistic") else "hgb"
            d, ci = _delta(exp, model, sub, br)
            rows.append((lab, grp, {model: (d, ci)}))
        else:
            rows.append((lab, grp, {m: _delta(exp, m, sub, br) for m in ["logistic", "hgb"]}))
    n_test = {r[0]: PRED["experiments"][r[1]]["models"]["logistic"][r[2]]["n"] for r in spec}

    fig, ax = plt.subplots(figsize=sz(ctx, (TEXTWIDTH, 3.7), "wide"))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[2:] -= 0.6  # gap between headline block and robustness block
    ax.axvline(0, ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4, zorder=1)
    ax.axhspan(ys[1] - 0.5, ys[0] + 0.5, color=PAL["bandblue"], lw=0, zorder=0)
    mk = {"logistic": ("o", PAL["blue"], "logistic"), "hgb": ("s", PAL["grey"], "gradient boosting")}
    off = {"logistic": 0.16, "hgb": -0.16}
    fs = small(ctx)
    for (lab, grp, vals), yy in zip(rows, ys):
        for m, (d, ci) in vals.items():
            m_, col, _ = mk[m]
            o = 0.0 if grp == "head" else off[m]
            face = "white" if grp == "flag" else col
            ax.errorbar(d, yy + o, xerr=[[d - ci[0]], [ci[1] - d]], fmt=m_, color=col, ecolor=col,
                        mfc=face, mec=col, mew=1.0 if grp == "flag" else 0,
                        elinewidth=plt.rcParams["lines.linewidth"], zorder=3)
        if grp == "head":
            m = list(vals)[0]
            d, ci = vals[m]
            ax.text(ci[1] + 0.004, yy, f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", va="center", ha="left",
                    fontsize=fs, color=PAL["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    for t, (lab, grp, _) in zip(ax.get_yticklabels(), rows):
        if grp == "head":
            t.set_fontweight("bold")
        if grp == "flag":
            t.set_color(PAL["muted"])
    ax.set_xlim(-0.005, 0.19 if ctx == "report" else 0.22)
    ax.set_xlabel("gain in test AUC from adding attention to the controls\n(95% CI, month clustered paired bootstrap)" if ctx == "report"
                  else "gain in test AUC from adding attention\n(95% CI, month clustered paired bootstrap)")
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", color=PAL["blue"], ls="", label="logistic"),
               Line2D([], [], marker="s", color=PAL["grey"], ls="", label="gradient boosting")]
    fig.legend(handles=handles, loc="outside upper right", ncol=2)
    ax.text(0.0, ys[0] + 0.55, "headline, test cohort release year 2025", ha="left", va="bottom",
            fontsize=fs, color=PAL["muted"], transform=ax.get_yaxis_transform())
    ax.text(0.0, ys[2] + 0.55, "robustness rows", ha="left", va="bottom", fontsize=fs, color=PAL["muted"],
            transform=ax.get_yaxis_transform())
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.3)
    return fig, n_test


# =========================================================================== fig 5
def fig_misjudged(ctx):
    pa = ASSOC["over_under"]["prestige_asymmetry"]
    cnt = ASSOC["over_under"]["counts"]
    groups = [("overrated", "overrated", PAL["orange"]), ("neutral", "neutral", PAL["grey"]),
              ("underrated", "underrated", PAL["blue"])]
    drivers = ASSOC["over_under"]["logit_drivers"]["overrated_vs_underrated_NOLEAK"]
    keep = {
        "tierB_prior (per SD)": "prior papers (per SD)",
        "tierB_yrs (per SD)": "years active (per SD)",
        "log_abs_chars (per SD)": "abstract length (per SD)",
        "title_n_words (per SD)": "title length (per SD)",
        "log_n_authors (per SD)": "number of authors (per SD)",
        "has_github": "Github link",
        "kw_survey": "survey keyword",
        "kw_agent": "agent keyword",
        "kw_benchmark": "benchmark keyword",
        "kw_efficient": "efficiency keyword",
        "kw_reasoning": "reasoning keyword",
        "kw_multimodal": "multimodal keyword",
        "kw_rl": "RL keyword",
    }
    dr = [(keep[r["feature"]], r["or"], r["or_ci95"], r["p"]) for r in drivers["rows"] if r["feature"] in keep]
    dr.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=sz(ctx, (TEXTWIDTH, 3.9), "full"))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.9], wspace=0.06, hspace=0.08)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[:, 1])
    fs = small(ctx)

    def medpanel(ax, key, ylab, flagged, show_x):
        m = pa[key]
        for i, (g, lab, col) in enumerate(groups):
            med = m[g]["median"]
            lo, hi = m[g]["ci95"]
            ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", color=col, ecolor=col,
                        mfc="white" if flagged else col, mec=col, mew=1.0 if flagged else 0,
                        elinewidth=plt.rcParams["lines.linewidth"], zorder=3)
        ax.set_xticks(range(3))
        if show_x:
            ax.set_xticklabels([f"{lab}\nn {cnt[g]}" for g, lab, _ in groups], fontsize=fs)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(ylab)
        ygrid(ax)

    medpanel(a1, "max_prior_papers_true_w99", "median prior papers\n(leakage free)", False, False)
    medpanel(a2, "max_hindex", "median h-index\n(measured today, flagged)", True, True)
    a1.set_ylim(9, 18)
    a2.set_ylim(12, 23)
    a1.text(0.98, 0.04, "bootstrap 95% CI", transform=a1.transAxes, ha="right", va="bottom", fontsize=fs, color=PAL["muted"])
    for ax in (a1, a2):
        ax.yaxis.label.set_fontsize(plt.rcParams["axes.labelsize"] if ctx == "report" else fs)
    # driver forest
    yy = np.arange(len(dr))
    for i, (lab, o, ci, p) in enumerate(dr):
        sig = (ci[0] > 1) or (ci[1] < 1)
        a3.errorbar(o, i, xerr=[[o - ci[0]], [ci[1] - o]], fmt="o", color=PAL["ink"], ecolor=PAL["ink"],
                    mfc=PAL["ink"] if sig else "white", mec=PAL["ink"], mew=1.0,
                    elinewidth=plt.rcParams["lines.linewidth"], zorder=3)
    a3.set_xscale("log")
    a3.axvline(1, ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4)
    a3.set_yticks(yy)
    a3.set_yticklabels([t[0] for t in dr])
    a3.set_xlim(0.04, 4)
    a3.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
    a3.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
    a3.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a3.set_xlabel(f"odds ratio, overrated versus underrated (log scale)\nleakage free logit, n {drivers['n']}, 95% CI")
    xgrid(a3)
    a3.tick_params(axis="y", length=0)
    a3.spines["left"].set_visible(False)
    a3.set_ylim(-1.4, len(dr) + 0.7)
    a3.text(0.9, len(dr) + 0.1, "more often underrated", ha="right", va="bottom", fontsize=fs, color=PAL["muted"])
    a3.text(1.12, len(dr) + 0.1, "more often overrated", ha="left", va="bottom", fontsize=fs, color=PAL["muted"])
    a3.text(0.045, -1.2, "filled marker: 95% CI excludes 1", ha="left", va="bottom", fontsize=fs, color=PAL["muted"])
    a1.text(0.02, 0.97, "(a)", transform=a1.transAxes, ha="left", va="top", fontweight="bold")
    a2.text(0.02, 0.97, "(b)", transform=a2.transAxes, ha="left", va="top", fontweight="bold")
    a3.text(-0.02, 1.0, "(c)", transform=a3.transAxes, ha="left", va="bottom", fontweight="bold")
    return fig


# =========================================================================== fig 6
def _control_join():
    """Trending vs never-trending frame, exactly as 27_association_v3.control_comparison."""
    meta = pd.read_csv(CTRL_META, dtype={"arxiv_id_clean": str})
    s2 = pd.read_csv(CTRL_S2, dtype={"arxiv_id_clean": str})
    c = meta.merge(s2, on="arxiv_id_clean", how="inner", validate="1:1")
    c = c[c.ss_found == 1].copy()
    rel = pd.to_datetime(c.published_v1, errors="coerce", utc=True).dt.tz_localize(None)
    c["age_months"] = (SNAPSHOT - rel).dt.days / 30.44
    c["release_month"] = rel.dt.to_period("M").astype(str)
    c = c[(c.age_months >= 5) & (c.age_months <= 40)].copy()
    c["citation_count"] = pd.to_numeric(c.citation_count, errors="coerce")
    c = c[c.citation_count.notna()].copy()
    t = DF[["arxiv_id_clean", "citation_count", "release_month", "max_hindex"]].merge(
        OU[["arxiv_id_clean", "attention_resid_pct", "label"]], on="arxiv_id_clean")
    both = pd.concat([t.assign(trend=1),
                      c[["arxiv_id_clean", "citation_count", "release_month", "max_hindex"]].assign(trend=0)],
                     ignore_index=True)
    both = both[both.max_hindex.notna()].copy()
    return both


def ccdf(v):
    v = np.sort(np.asarray(v))
    n = len(v)
    xs = np.unique(v)
    # P(X >= x)
    ys = 1.0 - np.searchsorted(v, xs, side="left") / n
    return xs, ys


def fig_control_ccdf(ctx):
    both = _control_join()
    tr = both[both.trend == 1]
    co = both[both.trend == 0]
    low = tr[tr.attention_resid_pct <= 1 / 3]
    prem = ASSOC["control_comparison"]["trending_premium"]
    cem = prem["cem_month_subfield13_hbin"]
    lowj = ASSOC["control_comparison"]["low_attention_vs_background"]["bottom_tertile_attention_residual"]
    pct = lowj["percentile_in_background_same_month"]
    fig, ax = plt.subplots(figsize=sz(ctx, (3.6, 3.0), "wide"))
    for v, col, ls, lab in [
        (co.citation_count, PAL["orange"], "-", f"never trending background, n {len(co)}"),
        (tr.citation_count, PAL["blue"], "-", f"trending papers, n {len(tr)}"),
        (low.citation_count, PAL["blue"], "--", f"trending, bottom attention tertile, n {len(low)}"),
    ]:
        xs, ys = ccdf(v)
        xs = np.where(xs == 0, 0.5, xs)  # log axis: put 0 citations at 0.5
        ax.step(xs, ys, where="post", color=col, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 3000)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xticks([0.5, 1, 10, 100, 1000])
    ax.set_xticklabels(["0", "1", "10", "100", "1000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("citations c (log scale)")
    ax.set_ylabel("share of papers with at least c citations")
    ygrid(ax)
    ax.legend(loc="lower left", fontsize=small(ctx))
    fs = small(ctx)
    txt = (f"matched premium ×{cem['ratio']:.2f} [{cem['ratio_ci95'][0]:.2f}, {cem['ratio_ci95'][1]:.2f}]\n"
           f"(month, subfield, h-index quintile)\n"
           f"naive ratio ×{prem['naive']['ratio']:.2f}\n"
           f"bottom attention tertile:\n{pct['mean_percentile']*100:.0f}th percentile of the same\n"
           f"month background [{pct['ci95'][0]:.3f}, {pct['ci95'][1]:.3f}]")
    ax.text(0.56, 0.035, txt, ha="left", va="top", fontsize=fs, color=PAL["ink"], linespacing=1.25)
    return fig, dict(n_trend=len(tr), n_ctrl=len(co), n_low=len(low))


# =========================================================================== fig 7
def fig_iv(ctx):
    ols = IV["ols"]["honest_FE_tierB"]
    ph = IV["primary_honest"]
    cnt = IV["honest_count_instrument"]
    loo = IV["honest_own_subfield_loo"]
    dfe = IV["dayfe_Z1p_kw"]
    sing = IV["dayfe_singleton_cells"]
    nons = IV["dayfe_nonsingleton_cells"]
    z = 1.96
    rows = [  # label, beta, lo, hi, group, note
        ("OLS, month + weekday + subfield FE", ols["beta"], ols["beta"] - z * ols["se_cluster_day"],
         ols["beta"] + z * ols["se_cluster_day"], "ols", f"n {ols['N']}"),
        ("2SLS, other subfield upvotes, same FE\n(Anderson Rubin 95% CI)", ph["iv_2sls"]["beta"],
         ph["ar_ci95"]["lower"], ph["ar_ci95"]["upper"], "honest",
         f"first stage t {ph['first_stage']['t']:.1f}, F {ph['first_stage']['kp_f']:.1f}"),
        ("2SLS, other subfield paper count, same FE\n(Anderson Rubin 95% CI)", cnt["iv_2sls"]["beta"],
         cnt["ar_ci95"]["lower"], cnt["ar_ci95"]["upper"], "honest",
         f"first stage t {cnt['first_stage']['t']:.1f}, F {cnt['first_stage']['kp_f']:.1f}"),
        ("2SLS, own subfield leave one out peers, same FE", loo["iv_2sls"]["beta"], None, None, "honest",
         f"first stage t {loo['first_stage']['t']:.1f}, F {loo['first_stage']['kp_f']:.1f}"),
        ("2SLS with day FE, all cells\n(Anderson Rubin 95% CI)", dfe["iv_2sls"]["beta"],
         dfe["ar_ci95"]["lower"], dfe["ar_ci95"]["upper"], "mech",
         f"first stage t {dfe['first_stage']['t']:.1f}, F {dfe['first_stage']['kp_f']:.0f}"),
        ("day FE, singleton day by subfield cells (Wald CI)", sing["iv_2sls"]["beta"],
         sing["iv_2sls"]["wald_ci95"][0], sing["iv_2sls"]["wald_ci95"][1], "mech",
         f"first stage t {sing['first_stage']['t']:.1f}, F {sing['first_stage']['kp_f']:.0f}"),
        ("day FE, non singleton cells (Wald CI)", nons["iv_2sls"]["beta"],
         nons["iv_2sls"]["wald_ci95"][0], nons["iv_2sls"]["wald_ci95"][1], "mech",
         f"first stage t {nons['first_stage']['t']:.1f}, F {nons['first_stage']['kp_f']:.0f}"),
    ]
    fig, ax = plt.subplots(figsize=sz(ctx, (TEXTWIDTH, 3.5), "full"))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[4:] -= 0.5
    xmin, xmax = -0.15, 1.25
    ax.axvline(0, ls=":", color=PAL["ink"], lw=plt.rcParams["axes.linewidth"] * 1.4, zorder=1)
    ax.axvline(ols["beta"], ls="-", color=PAL["bandblue"], lw=plt.rcParams["lines.linewidth"] * 2.2, zorder=0)
    ax.axhspan(ys[-1] - 0.5, ys[4] + 0.5, facecolor="white", edgecolor=PAL["lightgrey"], hatch="////",
               lw=0, zorder=0, alpha=0.6)
    fs = small(ctx)
    for (lab, b, lo, hi, grp, note), yy in zip(rows, ys):
        col = {"ols": PAL["ink"], "honest": PAL["blue"], "mech": PAL["grey"]}[grp]
        mk = {"ols": "o", "honest": "D", "mech": "s"}[grp]
        if lo is None:  # unbounded AR interval: dotted line across the axis, hollow marker at beta if visible
            ax.plot([xmin, xmax], [yy, yy], ls=":", color=col, lw=plt.rcParams["lines.linewidth"], zorder=2)
            ax.text(xmax - 0.01, yy + 0.1, r"$\beta$ " + f"{b:.2f}, Anderson Rubin interval unbounded",
                    ha="right", va="bottom", fontsize=fs, color=PAL["muted"])
        else:
            ax.errorbar(b, yy, xerr=[[b - lo], [hi - b]], fmt=mk, color=col, ecolor=col,
                        mfc="white" if grp == "mech" else col, mec=col, mew=1.0 if grp == "mech" else 0,
                        elinewidth=plt.rcParams["lines.linewidth"], zorder=3)
            ax.text(b, yy + 0.1, f"{b:.3f}", ha="center", va="bottom", fontsize=fs, color=PAL["ink"])
        ax.text(1.005, yy, note.replace("-", "\u2212"), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=fs, color=PAL["muted"], clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
    ax.set_xlabel("coefficient on log upvotes in a log citations equation")
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.text(xmin + 0.01, ys[4] + 0.55, "day fixed effects: first stage is the within day adding up identity (mechanical)",
            ha="left", va="bottom", fontsize=fs, color=PAL["muted"])
    ax.text(ols["beta"] + 0.012, ys[0] + 0.62, "OLS", ha="left", va="center", fontsize=fs, color=PAL["muted"])
    ax.text(1.005, ys[0] + 0.62, "first stage strength", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=fs, color=PAL["muted"], clip_on=False)
    return fig


# =========================================================================== fig 8
def fig_hier_slopes(ctx):
    H = ASSOC["hierarchical_mixedlm"]
    sl = H["subfield_slopes"]
    items = sorted(sl.items(), key=lambda kv: kv[1]["slope"])
    fig, ax = plt.subplots(figsize=sz(ctx, (3.7, 3.1), "wide"))
    ys = np.arange(len(items))
    lo, hi = H["fixed_slope_ci95"]
    # the band and the fixed slope line stop below the annotation (top 14% of the axes) so the text never overlaps them
    ax.axvspan(lo, hi, ymin=0, ymax=0.86, color=PAL["bandblue"], lw=0, zorder=0)
    ax.axvline(H["fixed_slope_log_upvotes"], ymin=0, ymax=0.86, color=PAL["blue"], lw=plt.rcParams["lines.linewidth"], zorder=1)
    for i, (name, v) in enumerate(items):
        ax.errorbar(v["slope"], i, xerr=[[v["slope"] - v["ci95"][0]], [v["ci95"][1] - v["slope"]]], fmt="o",
                    color=PAL["ink"], ecolor=PAL["ink"], elinewidth=plt.rcParams["lines.linewidth"], zorder=3)
    ax.set_yticks(ys)
    NAMES = {"Vision/Image-Gen": "vision and image generation", "Multimodal": "multimodal",
             "Reasoning/RL": "reasoning and RL", "LLM-core": "LLM core",
             "Efficiency/Systems": "efficiency and systems", "Agents": "agents",
             "Benchmark/Eval": "benchmarks and evaluation", "Vision-Perception": "vision perception",
             "RAG/Retrieval": "retrieval", "Speech/Audio": "speech and audio", "Code/Math": "code and math",
             "Robotics/Embodied": "robotics and embodied", "Other": "other"}
    ax.set_yticklabels([f"{NAMES.get(n, n)} (n {v['n']})" for n, v in items])
    ax.set_xlim(0.52, 0.76)
    ax.set_xlabel("slope of log(1 + citations)\non log upvotes, 95% CI")
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fs = small(ctx)
    ax.text(0.02, 0.99,
            f"fixed slope {H['fixed_slope_log_upvotes']:.3f} [{lo:.3f}, {hi:.3f}]\nbetween subfield SD {H['variance_components']['slope_sd']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs, color=PAL["ink"], zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_ylim(-0.7, len(items) + 2.2)
    return fig


# =========================================================================== fig 9
def fig_measurement(ctx):
    d = DF.copy()
    d["rm"] = pd.PeriodIndex(d.release_month, freq="M")
    q = d.groupby("rm").upvotes.quantile([0.25, 0.5, 0.75]).unstack()
    q = q[q.index >= pd.Period("2023-05", "M")]  # two papers each in 2023-02 and 2023-04 (n=2) omitted from the panel
    ncount = d.groupby("rm").size().reindex(q.index)
    xs = q.index.to_timestamp()
    d["rq"] = pd.PeriodIndex(d.release_month, freq="M").asfreq("Q")
    z = d.groupby("rq").apply(lambda g: (g.citation_count == 0).mean())
    z = z[z.index >= pd.Period("2023Q2", "Q")]
    zn = d.groupby("rq").size().reindex(z.index)
    sp = ASSOC["measurement"]["spearman_upvotes_age_by_year"]
    zy = ASSOC["measurement"]["share_zero_cites_by_year"]
    rng = ASSOC["measurement"]["median_upvotes_range_2023H2_on"]
    fig, (a, b) = plt.subplots(1, 2, figsize=sz(ctx, (TEXTWIDTH, 2.5), "full"), gridspec_kw={"width_ratios": [1.5, 1]})
    fs = small(ctx)
    a.fill_between(xs, q[0.25], q[0.75], color=PAL["blue"], alpha=0.15, lw=0, label="interquartile range")
    a.plot(xs, q[0.5], color=PAL["blue"], marker="o", ms=plt.rcParams["lines.markersize"] * 0.6, label="monthly median")
    launch_end = pd.Period("2023-06", "M").to_timestamp(how="end")
    a.axvspan(xs[0] - pd.Timedelta(days=15), launch_end, color=PAL["band"], lw=0, zorder=0)
    a.text(xs[0] - pd.Timedelta(days=10), 61, "launch\nera", ha="left", va="top", fontsize=fs, color=PAL["muted"])
    a.set_ylabel("upvotes per paper")
    a.set_ylim(0, 63)
    a.set_xlabel("release month")
    a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ygrid(a)
    a.legend(loc="upper right", ncol=2)
    a.text(0.13, 0.76, f"medians after the launch era: {rng[0]:.0f} to {rng[1]:.0f}",
           transform=a.transAxes, ha="left", va="top", fontsize=fs, color=PAL["muted"])
    xq = np.arange(len(z))
    b.bar(xq, z.values * 100, color=PAL["orange"], alpha=0.8, linewidth=0, width=0.7)
    b.set_xticks(xq)
    b.set_xticklabels([f"Q{p.quarter}" for p in z.index], fontsize=fs)
    for yr in sorted({p.year for p in z.index}):
        idx = [i for i, p in enumerate(z.index) if p.year == yr]
        b.text(np.mean(idx), -0.16, str(yr), transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=fs)
    b.set_ylabel("share with zero citations, %")
    b.set_xlabel("release quarter", labelpad=14 if ctx == "report" else 40)
    ygrid(b)
    b.set_ylim(0, max(z.values * 100) * 1.55)
    txt = "Spearman(upvotes, age) within year\n" + "\n".join(
        f"{y}: {sp[y]['spearman_upvotes_age']:+.3f} (n {sp[y]['n']})".replace("-", "\u2212") for y in ["2023", "2024", "2025"])
    b.text(0.03, 0.97, txt, transform=b.transAxes, ha="left", va="top", fontsize=fs, color=PAL["ink"], linespacing=1.3)
    a.text(-0.1, 1.02, "(a)", transform=a.transAxes, ha="left", va="bottom", fontweight="bold")
    b.text(-0.15, 1.02, "(b)", transform=b.transAxes, ha="left", va="bottom", fontweight="bold")
    return fig, dict(quarters=[str(p) for p in z.index], share_zero=[float(v) for v in z.values],
                     n=[int(v) for v in zn.values], zy=zy)


# =========================================================================== table
# =========================================================================== index
def write_index(meta):
    H = PRED["headline"]["primary"]
    hg = PRED["experiments"]["forward_yq"]["models"]["hgb"]["main"]["rows"]["P_tierB|+attention"]["delta_vs_controls_only"]
    L = ASSOC["ladder"]["M4_tierB_prestige"]
    dose = ASSOC["dose_response_deciles"]["rows"]
    cem = ASSOC["control_comparison"]["trending_premium"]["cem_month_subfield13_hbin"]
    naive = ASSOC["control_comparison"]["trending_premium"]["naive"]
    lowp = ASSOC["control_comparison"]["low_attention_vs_background"]["bottom_tertile_attention_residual"]["percentile_in_background_same_month"]
    ph = IV["primary_honest"]
    dfe = IV["dayfe_Z1p_kw"]
    Hm = ASSOC["hierarchical_mixedlm"]
    pa = ASSOC["over_under"]["prestige_asymmetry"]
    dr = {r["feature"]: r for r in ASSOC["over_under"]["logit_drivers"]["overrated_vs_underrated_NOLEAK"]["rows"]}
    sp = ASSOC["measurement"]["spearman_upvotes_age_by_year"]
    rng = ASSOC["measurement"]["median_upvotes_range_2023H2_on"]
    zy = ASSOC["measurement"]["share_zero_cites_by_year"]
    ols = IV["ols"]["honest_FE_tierB"]
    m4p = L["poisson_qmle"]
    m4n = L["nb2"]
    m4o = L["ols_log1p"]
    plac = ASSOC["placebo_permutation_M4"]["poisson"]["placebo_irr2x_range"]

    def files(name):
        return f"`{name}.pdf`, `{name}.png`, `slide_{name}.png`"

    txt = f"""# figures_v3 index

Generated by `scripts/make_figures_v3.py`. Every number below is read from the JSON results at run time and
matches `results/D5_results_gate_v3.md` (ledger row numbers given as L<n>). Report files: PDF (vector, Latin Modern
Roman, 8 to 9 pt, sized for a 6.3 in text width or a 3.6 in half width) plus a 200 dpi PNG. Slide files:
`slide_<name>.png` at 200 dpi in the deck boxes (wide 8.0 x 5.1 in, full 12.13 x 4.6 in), 13 to 16 pt Calibri.
Colour semantics (Okabe Ito, CVD safe, validated): blue = attention or the +attention model or underrated,
vermillion = controls only, baseline, never trending background or overrated, grey = neutral (upvotes only,
gradient boosting), green = third estimator, hatched grey = mechanical or flagged rows.
Include in LaTeX with `\\includegraphics[width=\\linewidth]{{fig_v3_dose_response}}` (no extension, pdflatex takes the PDF).
CI style: error bars in figures, square brackets in annotations and captions. No in-figure titles.
Notes for writers: (i) the slide variant of the ROC/PR figure carries no footnote (put the test cohort and label in the
slide title or kicker), (ii) the CCDF trending count ({meta['ccdf']['n_trend']}) is the analysis sample with a recorded
author h-index, as in the matching join, (iii) all hues live in `PAL` / `SLIDE_PAL` at the top of the script, so a deck
that adopts other tokens can re-map them in one place, (iv) the money figures are fig_v3_roc_pr (prediction headline)
and fig_v3_dose_response (association).

| # | Figure | Files | Money figure? | Suggested width |
|---|---|---|---|---|
| 1 | dose response | {files('fig_v3_dose_response')} | yes, association | `\\linewidth` |
| 2 | specification ladder | {files('fig_v3_ladder')} | | `\\linewidth` |
| 3 | ROC and PR | {files('fig_v3_roc_pr')} | yes, prediction (headline) | `\\linewidth` |
| 4 | delta AUC forest | {files('fig_v3_delta_forest')} | | `\\linewidth` |
| 5 | misjudged papers | {files('fig_v3_misjudged')} | | `\\linewidth` |
| 6 | control CCDF | {files('fig_v3_control_ccdf')} | | `0.57\\linewidth` |
| 7 | crowding IV | {files('fig_v3_iv')} | | `\\linewidth` |
| 8 | hierarchical slopes | {files('fig_v3_hier_slopes')} | | `0.57\\linewidth` |
| 9 | measurement | {files('fig_v3_measurement')} | | `\\linewidth` |

## Captions (plain, style compliant, at most three sentences)

### 1. fig_v3_dose_response
Geometric mean of 1 + citations (equivalently the mean of log(1 + citations)) with 95% CI by upvote decile, raw and
after adjusting for age, subfield and release month, for {ASSOC['measurement']['descriptives']['n']} papers. The
bottom strip is the share of each decile in the top citation decile of its release quarter, from
{dose[0]['share_top_decile_cites_in_quarter']*100:.1f}% in the lowest upvote decile to
{dose[-1]['share_top_decile_cites_in_quarter']*100:.1f}% in the highest. Descriptive, association not causation.
Source: `association_v3.json: dose_response_deciles.rows` (mean_log1p_cites, ci95, mean_log1p_cites_adjusted,
ci95_adjusted, share_top_decile_cites_in_quarter, geo_mean_1p_cites), `dose_response_deciles.spearman_upvotes_cites`
= {ASSOC['dose_response_deciles']['spearman_upvotes_cites']} (association tables T3, ledger L73 for n).

### 2. fig_v3_ladder
Citation ratio per doubling of upvotes across specifications that add controls one at a time, for Poisson QMLE, NB2 and
the OLS ratio on log(1 + citations), with 95% CIs clustered by release month (n {m4p['nobs']}). In the main
specification (subfield, month and weekday fixed effects and leakage free Tier B prestige) the ratios are
{m4p['irr_per_doubling']:.2f} [{m4p['irr_per_doubling_ci95'][0]:.2f}, {m4p['irr_per_doubling_ci95'][1]:.2f}] (Poisson),
{m4n['irr_per_doubling']:.2f} [{m4n['irr_per_doubling_ci95'][0]:.2f}, {m4n['irr_per_doubling_ci95'][1]:.2f}] (NB2) and
{m4o['ratio_1p_cites_per_doubling']:.2f} (OLS, elasticity {m4o['beta']:.3f}). The grey band is the range of the
Poisson ratio when upvotes are permuted within month and subfield ({plac[0]:.2f} to {plac[1]:.2f}), and the hatched
specifications add a control measured after the outcome.
Source: `association_v3.json: ladder.M0_raw ... M6_plus_text.{{poisson_qmle,nb2,ols_log1p}}`,
`placebo_permutation_M4.poisson.placebo_irr2x_range` (ledger L63, L64, L65).

### 3. fig_v3_roc_pr
ROC (a) and precision recall (b) curves of the prespecified logistic model on the release year 2025 test cohort
(n {PRED['experiments']['forward_yq']['n_test']}, label = top citation decile within release quarter, base rate
{PRED['experiments']['forward_yq']['base_rate_test']:.3f}). Controls only reaches AUC {H['controls_only_auc']:.3f} and PR AUC
{H['controls_only_pr_auc']:.3f}, upvotes alone AUC {H['upvotes_only_auc']:.3f} and PR AUC {H['upvotes_only_pr_auc']:.3f},
controls plus attention AUC {H['attention_auc']:.3f} and PR AUC {H['attention_pr_auc']:.3f}. Predictive, not causal, and
upvotes are cumulative counts at collection.
Source: curves from `prediction_v3_scores.csv` (split forward_test_2025, columns
forward_yq__logistic__P_tierB__{{controls_only,upvotes_only,+attention}}), AUCs from `prediction_v3.json:
headline.primary.*` (ledger L1, L2, L4, L6, L24, the upvotes only PR AUC {H['upvotes_only_pr_auc']:.3f} is in the JSON
key `headline.primary.upvotes_only_pr_auc` and not in a ledger row).

### 4. fig_v3_delta_forest
Gain in test AUC from adding attention to the controls, with month clustered paired bootstrap 95% CIs. The headline
logistic gain is {H['delta_auc']:+.3f} [{H['delta_auc_ci_month'][0]:+.3f}, {H['delta_auc_ci_month'][1]:+.3f}] and the gradient
boosting gain {hg['d_auc']:+.3f} [{hg['ci']['month_cluster']['auc'][0]:+.3f}, {hg['ci']['month_cluster']['auc'][1]:+.3f}], and the
robustness rows change the prestige control, the label, the test cohort or the training window (n per row in the
ledger). The legacy subfield row is flagged because its taxonomy leaked upvote information into the baseline.
Source: `prediction_v3.json: experiments.<exp>.models.<model>.<main|mature_k12>.rows['<branch>|+attention'].delta_vs_controls_only`
(ledger L3, L14, L26 to L36). Test n: {', '.join(f"{k.split(' (')[0]} {v}" for k, v in meta['forest_n'].items())}.

### 5. fig_v3_misjudged
Overrated papers (top attention tertile, bottom impact tertile after adjusting for age, subfield and month, n
{ASSOC['over_under']['counts']['overrated']}) versus underrated papers (the reverse, n {ASSOC['over_under']['counts']['underrated']}) and neutral papers
(n {ASSOC['over_under']['counts']['neutral']}). (a) Median leakage free prior papers of the first or last author:
{pa['max_prior_papers_true_w99']['overrated']['median']:.0f} versus {pa['max_prior_papers_true_w99']['underrated']['median']:.0f}
(neutral {pa['max_prior_papers_true_w99']['neutral']['median']:.0f}), (b) median h-index measured today (flagged, downstream of the outcome):
{pa['max_hindex']['overrated']['median']:.0f} versus {pa['max_hindex']['underrated']['median']:.0f} (neutral {pa['max_hindex']['neutral']['median']:.0f}),
bootstrap 95% CIs. (c) Odds ratios from the leakage free head to head logit (overrated versus underrated, n
{ASSOC['over_under']['logit_drivers']['overrated_vs_underrated_NOLEAK']['n']}, cluster robust by month): Tier B prior papers
{dr['tierB_prior (per SD)']['or']:.2f} [{dr['tierB_prior (per SD)']['or_ci95'][0]:.2f}, {dr['tierB_prior (per SD)']['or_ci95'][1]:.2f}] per SD,
survey keyword {dr['kw_survey']['or']:.2f}, agent keyword {dr['kw_agent']['or']:.2f}, abstract length
{dr['log_abs_chars (per SD)']['or']:.2f} per SD, Github link {dr['has_github']['or']:.2f}.
Source: `association_v3.json: over_under.prestige_asymmetry.{{max_prior_papers_true_w99,max_hindex}}`,
`over_under.logit_drivers.overrated_vs_underrated_NOLEAK.rows`, `over_under.counts` (ledger L71, association tables T6).

### 6. fig_v3_control_ccdf
Share of papers with at least c citations (log log) for trending papers (n {meta['ccdf']['n_trend']}), the never trending
same month background sample (n {meta['ccdf']['n_ctrl']}) and trending papers in the bottom attention tertile (n
{meta['ccdf']['n_low']}). Exact matching on release month, subfield and h-index quintile gives a trending premium of
×{cem['ratio']:.2f} [{cem['ratio_ci95'][0]:.2f}, {cem['ratio_ci95'][1]:.2f}] (naive ×{naive['ratio']:.2f}). Even bottom
tertile attention trending papers sit on average at the {lowp['mean_percentile']*100:.0f}th percentile of the same month
background [{lowp['ci95'][0]:.3f}, {lowp['ci95'][1]:.3f}].
Source: curves recomputed from `analysis_final.csv`, `overunder_v3.csv` and the control files joined as in
`27_association_v3.py` (max h-index present, ages 5 to 40 months), numbers from `association_v3.json:
control_comparison.trending_premium.{{naive,cem_month_subfield13_hbin}}` and
`control_comparison.low_attention_vs_background.bottom_tertile_attention_residual.percentile_in_background_same_month`
(ledger L68, L69).

### 7. fig_v3_iv
Coefficient on log upvotes in a log citations equation. OLS with month, weekday and subfield fixed effects gives
{ols['beta']:.3f} (n {ols['N']}, SE clustered by day {ols['se_cluster_day']:.4f}), the honest 2SLS with the other subfield
upvote instrument gives {ph['iv_2sls']['beta']:.3f} with Anderson Rubin 95% CI ({ph['ar_ci95']['lower']:.3f}, {ph['ar_ci95']['upper']:.3f})
and first stage t {ph['first_stage']['t']:.1f} (F {ph['first_stage']['kp_f']:.1f}), and the own subfield leave one out instrument
has no first stage (t {IV['honest_own_subfield_loo']['first_stage']['t']:.1f}, unbounded interval). The hatched rows use day fixed
effects, where the first stage (F {dfe['first_stage']['kp_f']:.0f}) is the within day adding up identity and the coefficient is
{IV['dayfe_singleton_cells']['iv_2sls']['beta']:.3f} in singleton cells versus {IV['dayfe_nonsingleton_cells']['iv_2sls']['beta']:.3f} in
non singleton cells, so the IV is an attempted design and not causal evidence.
Source: `crowding_iv_v3.json: ols.honest_FE_tierB, primary_honest, honest_count_instrument, honest_own_subfield_loo,
dayfe_Z1p_kw, dayfe_singleton_cells, dayfe_nonsingleton_cells` (ledger L46, L47, L50, L51, L52, L53, L54).

### 8. fig_v3_hier_slopes
Random slopes of log(1 + citations) on log upvotes by subfield from a linear mixed model with month and weekday fixed
effects, log age and Tier B prestige (n {Hm['n']}, {Hm['n_groups']} subfields), with approximate 95% CIs. The fixed slope is
{Hm['fixed_slope_log_upvotes']:.3f} [{Hm['fixed_slope_ci95'][0]:.3f}, {Hm['fixed_slope_ci95'][1]:.3f}] and the between subfield
slope SD is {Hm['variance_components']['slope_sd']:.3f} (slopes {Hm['slope_range'][0]:.3f} to {Hm['slope_range'][1]:.3f}), so the
association is close to uniform across subfields.
Source: `association_v3.json: hierarchical_mixedlm.{{fixed_slope_log_upvotes,fixed_slope_ci95,variance_components.slope_sd,subfield_slopes}}` (ledger L72).

### 9. fig_v3_measurement
(a) Monthly median upvotes with the interquartile band by release month (release months with two papers omitted). Medians after the
launch era range from {rng[0]:.0f} to {rng[1]:.0f}. (b) Share of papers with zero recorded citations by release quarter (by year:
{zy['2023']:.3f}, {zy['2024']:.3f}, {zy['2025']:.3f}), with the within year Spearman correlation of upvotes and age
({sp['2023']['spearman_upvotes_age']:+.3f}, {sp['2024']['spearman_upvotes_age']:+.3f}, {sp['2025']['spearman_upvotes_age']:+.3f}), which shows that late
accrual of upvotes is small relative to cross paper variation outside the launch era.
Source: panel (a) recomputed from `analysis_final.csv` (upvotes by release_month), range from
`association_v3.json: measurement.median_upvotes_range_2023H2_on` (ledger L42), panel (b) recomputed from
`analysis_final.csv` (citation_count == 0 by quarter), text from `measurement.spearman_upvotes_age_by_year` and
`measurement.share_zero_cites_by_year` (ledger L41, association tables T7).


"""
    (OUT / "FIGURES_INDEX.md").write_text(txt)


# =========================================================================== main
def main():
    meta = {}
    for ctx in ["report", "slide"]:
        style(ctx)
        save(fig_dose_response(ctx), "fig_v3_dose_response", ctx)
        save(fig_ladder(ctx), "fig_v3_ladder", ctx)
        save(fig_roc_pr(ctx), "fig_v3_roc_pr", ctx)
        fig, n_test = fig_delta_forest(ctx)
        meta["forest_n"] = n_test
        save(fig, "fig_v3_delta_forest", ctx)
        save(fig_misjudged(ctx), "fig_v3_misjudged", ctx)
        fig, m = fig_control_ccdf(ctx)
        meta["ccdf"] = m
        save(fig, "fig_v3_control_ccdf", ctx)
        save(fig_iv(ctx), "fig_v3_iv", ctx)
        save(fig_hier_slopes(ctx), "fig_v3_hier_slopes", ctx)
        fig, m = fig_measurement(ctx)
        meta["measurement"] = m
        save(fig, "fig_v3_measurement", ctx)
    write_index(meta)
    print(f"wrote figures to {OUT}")
    print(json.dumps(meta, indent=1, default=str)[:1500])


if __name__ == "__main__":
    main()
