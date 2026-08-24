#!/usr/bin/env python3
"""
Figure generation: fig1_prediction_delta_auc.png and fig2_iv_triangulation.png
Fixed version: additive FE residualization for binscatter; no-overlap sub-labels for fig1.

Usage:
  python scripts/make_figures.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyfixest as pf
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = Path(__file__).resolve().parents[2]
DATA_PATH  = BASE / "data/processed/analysis_final.csv"
IV_JSON    = BASE / "results/crowding_iv.json"
PRED_JSON  = BASE / "results/prediction.json"
FIG_DIR    = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─── Load JSONs ───────────────────────────────────────────────────────────────
with open(IV_JSON)   as f: iv   = json.load(f)
with open(PRED_JSON) as f: pred = json.load(f)

# ─── Load data ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH, dtype={"arxiv_id_clean": str})

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — IV TRIANGULATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BUILDING FIG2: IV triangulation")
print("="*70)

# ── Estimation sample (same as primary-P in 25_crowding_iv.py) ───────────────
PRESTIGE = ["max_hindex", "max_papercount_cur2026_w99"]
X0 = [
    "age_months", "n_authors", "has_github", "title_n_words", "title_has_colon",
    "abstract_n_chars", "kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning",
    "kw_benchmark", "kw_survey", "kw_efficient", "kw_multimodal", "kw_rl", "kw_scaling",
]
X_P = X0 + PRESTIGE

need_cols = ["log_citations", "log_upvotes", "Z1p_othersub",
             "cohort_day", "subfield"] + X_P
df_est = (df_raw
          .loc[df_raw["citation_count"].notna() &
               (df_raw["age_months"] >= 5) &
               (df_raw["age_months"] <= 40)]
          .dropna(subset=need_cols)
          .copy()
          .reset_index(drop=True))
print(f"Estimation sample (before pyfixest singleton drop): N={len(df_est)}")

# ── Residualize on ADDITIVE FE: C(cohort_day) + C(subfield) ──────────────────
# This is the FE used in the model (additive main effects, NOT interaction).
# Z1p is a cell-level (cohort_day × subfield) aggregate, so under the INTERACTION
# FE it would be fully absorbed (machine-zero residual). Under the ADDITIVE FE
# the interaction-component variation is retained.
print("\nResidualizing log_upvotes and Z1p_othersub on C(cohort_day)+C(subfield) [ADDITIVE]...")

fit_z = pf.feols("Z1p_othersub ~ 1 | cohort_day + subfield", data=df_est, vcov="iid")
fit_d = pf.feols("log_upvotes  ~ 1 | cohort_day + subfield", data=df_est, vcov="iid")

resid_z = np.array(fit_z.resid())
resid_d = np.array(fit_d.resid())

sd_resid_z = float(np.std(resid_z, ddof=1))
print(f"\n*** SD of residualized Z1p_othersub (ADDITIVE FE): {sd_resid_z:.6f} ***")
if sd_resid_z < 1e-10:
    raise RuntimeError(
        f"SD of residualized Z1p is ~machine-zero ({sd_resid_z:.2e}). "
        "The instrument has no additive-FE residual variation — "
        "DO NOT draw a binscatter. Aborting."
    )
print(f"SANITY CHECK PASSED: SD={sd_resid_z:.4f} is clearly > 0")

# Note: pyfixest may drop singletons; align arrays
n_used = len(resid_z)
print(f"N after pyfixest singleton drop: {n_used}")

# ── Binscatter: 20 quantile bins of residualized Z1p ────────────────────────
n_bins = 20
bin_labels = pd.qcut(resid_z, q=n_bins, labels=False)

bin_means_z = np.array([resid_z[bin_labels == b].mean() for b in range(n_bins)])
bin_means_d = np.array([resid_d[bin_labels == b].mean() for b in range(n_bins)])

# Fitted slope (OLS of binned means, weighted by bin size)
bin_sizes = np.array([(bin_labels == b).sum() for b in range(n_bins)])
coef_bin  = np.polyfit(bin_means_z, bin_means_d, deg=1, w=np.sqrt(bin_sizes))
slope_bin = coef_bin[0]
x_line    = np.linspace(bin_means_z.min(), bin_means_z.max(), 200)
y_line    = np.polyval(coef_bin, x_line)

print(f"Binscatter fitted slope: {slope_bin:.4f}")
print(f"Residualized Z1p range: [{bin_means_z.min():.4f}, {bin_means_z.max():.4f}]")

# ── Numbers from JSON ─────────────────────────────────────────────────────────
prim_p      = iv["primary-P"]
ols_res     = iv["OLS-primaryFE"]
iv_beta     = prim_p["beta"]          # 0.757911
iv_se       = prim_p["se_cluster"]    # 0.055049
ols_beta    = ols_res["beta"]         # 0.596571
ols_se      = ols_res["se_cluster"]   # 0.018257
pi1         = prim_p["first_stage_pi1"]   # -1.292471
kp_f        = prim_p["first_stage_kp_f"]  # 354.6538
ar_lo       = prim_p["ar_ci_95"]["lower"] # 0.6490
ar_hi       = prim_p["ar_ci_95"]["upper"] # 0.8659
n_obs       = prim_p["N"]             # 11236
sing_pct    = iv["diagnostics"]["cell_size_structure"]["singleton_cell_share"]
papers_sing = iv["diagnostics"]["cell_size_structure"]["paper_singleton_share"]

# ── Draw fig2 ────────────────────────────────────────────────────────────────
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    "Fig. 2 — IV Triangulation: Crowding → Attention → Citations",
    fontsize=13, fontweight="bold", y=1.01
)

# Panel (a): binscatter
ax_a.scatter(bin_means_z, bin_means_d, s=60, color="#2166ac", zorder=5,
             alpha=0.85, label="Bin mean (20 quantile bins)")
ax_a.plot(x_line, y_line, color="#d6604d", lw=2, zorder=4, label=f"Fitted slope: {slope_bin:.3f}")
ax_a.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
ax_a.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.6)

ax_a.set_xlabel("Residualized Z1p (additive cohort-day + subfield FE)", fontsize=10)
ax_a.set_ylabel("Residualized log(upvotes) (same FE)", fontsize=10)
ax_a.set_title(
    f"(a) First stage binscatter\n"
    r"$\pi_1=$" + f"{pi1:.4f},  KP-F={kp_f:.2f}",
    fontsize=11
)

# Caveat annotation
caveat_txt = (
    f"Partly mechanical: {sing_pct:.1%} singleton cells\n"
    f"({papers_sing:.1%} of papers)"
)
ax_a.text(0.03, 0.03, caveat_txt, transform=ax_a.transAxes,
          fontsize=8, color="#666666", va="bottom",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                    edgecolor="#cccc99", alpha=0.85))
ax_a.legend(fontsize=8, loc="upper right")

# Panel (b): forest plot — OLS vs IV
ax_b.set_title("(b) OLS vs 2SLS estimate\n(log citations ~ log upvotes)", fontsize=11)

y_ols = 1.0
y_iv  = 0.0

# OLS point + 1.96*SE CI
ax_b.scatter([ols_beta], [y_ols], s=90, color="#1a9641", zorder=5, label="OLS")
ax_b.errorbar([ols_beta], [y_ols],
              xerr=[[1.96*ols_se], [1.96*ols_se]],
              fmt="none", color="#1a9641", capsize=5, lw=1.8)

# IV point + AR CI
ax_b.scatter([iv_beta], [y_iv], s=90, marker="D", color="#d73027", zorder=5, label="2SLS (IV)")
ax_b.errorbar([iv_beta], [y_iv],
              xerr=[[iv_beta - ar_lo], [ar_hi - iv_beta]],
              fmt="none", color="#d73027", capsize=5, lw=1.8,
              label=f"AR 95% CI [{ar_lo:.3f}, {ar_hi:.3f}]")

ax_b.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)

# Per-doubling axis label
def _per_double(beta):
    return f"×{2**beta:.2f}"

ax_b.set_yticks([y_iv, y_ols])
ax_b.set_yticklabels([
    f"2SLS  β={iv_beta:.3f} ({_per_double(iv_beta)}/doubling)\n"
    f"AR CI [{ar_lo:.3f}, {ar_hi:.3f}]",
    f"OLS  β={ols_beta:.3f} ({_per_double(ols_beta)}/doubling)\n"
    f"N={n_obs:,}, cluster SE={ols_se:.3f}",
], fontsize=9)
ax_b.set_xlabel("β on log(upvotes) → log(citations)", fontsize=10)
ax_b.set_ylim(-0.6, 1.6)

# Annotation box
ax_b.text(0.97, 0.05,
          f"Instrument: Z1p_othersub\ncohort_day + subfield FE\nN={n_obs:,}",
          transform=ax_b.transAxes, fontsize=8, ha="right", va="bottom",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa"))

fig2.tight_layout()
out2 = FIG_DIR / "fig2_iv_triangulation.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"\nFig2 saved → {out2}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — PREDICTION ΔAUC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BUILDING FIG1: prediction ΔAUC")
print("="*70)

# Pull numbers from prediction.json
results_list = pred["results"]

def get_branch(branch_name):
    for r in results_list:
        if r["branch"] == branch_name and not r.get("leaky", True):
            return r
    return None

p_int = get_branch("P_interim")
p_non = get_branch("P_none")
n_test_full   = 6325
n_test_mature = 2365

# Rows to plot (one row per configuration)
# Format: (label, delta_auc, ci_lo, ci_hi, subset, marker, color)
# We'll show HGB and Logistic for both branches, Full and Mature test sets.
# For mature, no bootstrap CI stored in JSON, so show point only.

rows = []
for res, branch_label in [(p_int, "P_interim\n(w/ productivity)"),
                           (p_non, "P_none\n(no prestige)")]:
    if res is None:
        continue
    for model_key, model_label in [("hgb_clf", "HGB"), ("logistic", "Logistic")]:
        # Full test
        delta_full = res["delta"][model_key]["delta_auc"]
        ci_lo      = res["delta"][model_key]["bootstrap_ci"]["ci_lo"]
        ci_hi      = res["delta"][model_key]["bootstrap_ci"]["ci_hi"]
        rows.append({
            "label":        f"{model_label}, {branch_label}",
            "delta":        delta_full,
            "ci_lo":        ci_lo,
            "ci_hi":        ci_hi,
            "subset":       "full",
        })
        # Mature test (AUC diff only, no separate CI)
        ctrl_mat  = res["controls_only"][model_key]["mature_auc"]
        attn_mat  = res["+attention"][model_key]["mature_auc"]
        delta_mat = attn_mat - ctrl_mat
        rows.append({
            "label":        f"{model_label}, {branch_label}",
            "delta":        delta_mat,
            "ci_lo":        None,
            "ci_hi":        None,
            "subset":       "mature",
        })

# ── Sort: group by model+branch, then full above mature ─────────────────────
# We'll interleave: for each (model, branch) pair: full row first, then mature
# And separate the two subsets visually using different markers/colors

full_rows   = [r for r in rows if r["subset"] == "full"]
mature_rows = [r for r in rows if r["subset"] == "mature"]

# Build y-positions with a gap between full and mature groups
n_full   = len(full_rows)
n_mature = len(mature_rows)
gap      = 1.2  # extra space between groups

y_full   = list(range(n_full - 1, -1, -1))
y_mature = [y - n_full - gap for y in range(n_mature - 1, -1, -1)]

fig1, ax1 = plt.subplots(figsize=(9, 6))

COLOR_FULL   = "#2166ac"
COLOR_MATURE = "#762a83"

# Full test rows
for i, (row, y) in enumerate(zip(full_rows, y_full)):
    delta = row["delta"]
    ax1.scatter([delta], [y], s=70, color=COLOR_FULL, zorder=5,
                marker="o", label="_nolegend_")
    if row["ci_lo"] is not None:
        ax1.errorbar([delta], [y],
                     xerr=[[delta - row["ci_lo"]], [row["ci_hi"] - delta]],
                     fmt="none", color=COLOR_FULL, capsize=4, lw=1.6)
    ax1.text(delta + 0.001, y, f" {delta:+.3f}", va="center", fontsize=8.5,
             color=COLOR_FULL)

# Mature test rows
for i, (row, y) in enumerate(zip(mature_rows, y_mature)):
    delta = row["delta"]
    ax1.scatter([delta], [y], s=70, color=COLOR_MATURE, zorder=5,
                marker="D", label="_nolegend_")
    ax1.text(delta + 0.001, y, f" {delta:+.3f}", va="center", fontsize=8.5,
             color=COLOR_MATURE)

# Y-axis labels
y_ticks  = list(y_full) + list(y_mature)
y_labels = [r["label"] for r in full_rows] + [r["label"] for r in mature_rows]
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels, fontsize=9)

# Section labels as ax.text() in left margin — placed above each group
# Calculate group top positions
full_top   = max(y_full) + 0.6
mature_top = max(y_mature) + 0.6

ax_xmin = ax1.get_xlim()[0] if ax1.get_xlim()[0] != 0.0 else -0.005
# Determine a fixed x position for section labels (far left)
x_section = -0.015  # will be adjusted by xlim below

# Draw section bracket lines
full_yspan   = (min(y_full) - 0.4, max(y_full) + 0.4)
mature_yspan = (min(y_mature) - 0.4, max(y_mature) + 0.4)

ax1.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

# Reference line at x=0
ax1.axvline(0, color="#333333", lw=1.0, ls=":", alpha=0.6)

# Group separator
mid_y = (min(y_full) + max(y_mature)) / 2
ax1.axhline(mid_y, color="#cccccc", lw=1, ls="-")

ax1.set_xlabel("ΔAUC (+attention vs controls-only), out-of-sample 2025 test", fontsize=10)
ax1.set_xlim(-0.01, max([r["delta"] for r in rows]) + 0.025)

# Legend patches
patch_full   = mpatches.Patch(color=COLOR_FULL,   label=f"Full test set (n={n_test_full:,})")
patch_mature = mpatches.Patch(color=COLOR_MATURE, label=f"Mature K≥12 (n={n_test_mature:,})")
ax1.legend(handles=[patch_full, patch_mature], loc="lower right", fontsize=9,
           framealpha=0.9, edgecolor="#cccccc")

ax1.set_title(
    "Fig. 1 — Incremental ΔAUC of HF Attention Signal\n"
    "Forward-in-time test (train ≤ 2024 → test 2025);  "
    "bootstrap 95% CI shown for full-test rows",
    fontsize=11, fontweight="bold"
)

fig1.tight_layout()
out1 = FIG_DIR / "fig1_prediction_delta_auc.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Fig1 saved → {out1}")

print("\n" + "="*70)
print("DONE")
print(f"  SD resid Z1p (additive FE): {sd_resid_z:.6f}  (> 0: real variation confirmed)")
print(f"  Binscatter x-range: [{bin_means_z.min():.4f}, {bin_means_z.max():.4f}]")
print(f"  Fig1 → {out1}")
print(f"  Fig2 → {out2}")
