#!/usr/bin/env python3
"""
D1 v3 — Crowding IV, honest re-specification (audit response; diagnostics/02_stats_methods_critique.md §B)
=========================================================================================================
Supersedes scripts/25_crowding_iv.py (kept for reference).  Purpose: report the same-day crowding
instrument as an ATTEMPTED design and show, with the numbers, why it is not informative causal evidence.

Changes vs v1
  * Uniform taxonomy: instruments rebuilt on `subfield_kw` (scripts/23_crowding_cohort_v3.py ->
    data/processed/crowding_v3.csv, merged here; analysis_final.csv is not modified).
  * Prestige control = leakage-free Tier B (log1p prior papers + years active) instead of today-measured
    max_hindex / 2026 counts (legacy prestige kept only in flagged replication rows).  n_authors enters as log1p.
  * PRIMARY (honest) 2SLS = Z1'_kw under release_month + dow + subfield_kw FE (no day FE) with
    first-stage t / KP-F, cluster-robust Anderson-Rubin CI (proper AR-regression residuals, pyfixest ssc),
    reduced form; OLS with the same FE clustered by cohort_day AND by release_month.
  * The v1 day-FE specification is reported as the "mechanical / reflection" comparison, with the
    singleton vs non-singleton split, the own-subfield leave-one-out peer sum (null first stage), the
    other-subfield paper COUNT instrument, and within-day reflection correlations.
  * Legacy replication rows reproduce the v1 headline (Z1p_othersub, legacy subfield, legacy prestige).
  * All prose in the JSON/NOTE is generated from the computed numbers.

Outputs: results/crowding_iv_v3.json, results/crowding_iv_v3_NOTE.md
Usage:   python scripts/25_crowding_iv_v3.py
"""
from pathlib import Path
import json
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
T0 = time.time()

BASE = str(Path(__file__).resolve().parents[1])
DATA_PATH = f"{BASE}/data/processed/analysis_final.csv"
CROWD_V3 = f"{BASE}/data/processed/crowding_v3.csv"
OUT_JSON = f"{BASE}/results/crowding_iv_v3.json"
OUT_NOTE = f"{BASE}/results/crowding_iv_v3_NOTE.md"

# ─── Load, merge, filter ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, dtype={"arxiv_id_clean": str})
cw = pd.read_csv(CROWD_V3, dtype={"arxiv_id_clean": str})
assert cw["arxiv_id_clean"].is_unique and df["arxiv_id_clean"].is_unique
n0 = len(df)
df = df.merge(cw.drop(columns=["cohort_day"]), on="arxiv_id_clean", how="left", validate="one_to_one")
assert len(df) == n0 and df["Z1p_kw"].notna().all(), "crowding_v3 merge incomplete"
df = df[df["citation_count"].notna() & (df["age_months"] >= 5) & (df["age_months"] <= 40)].copy().reset_index(drop=True)
df["log_n_authors"] = np.log1p(df["n_authors"])
df["release_month"] = df["release_month"].astype(str)
N_base = len(df)
print(f"Base sample N={N_base:,}")

X0 = ["age_months", "log_n_authors", "has_github", "title_n_words", "title_has_colon", "abstract_n_chars",
      "kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning", "kw_benchmark", "kw_survey", "kw_efficient",
      "kw_multimodal", "kw_rl", "kw_scaling"]
X0_LEGACY = [c if c != "log_n_authors" else "n_authors" for c in X0]          # v1 used raw n_authors
P_TIERB = ["log1p_max_prior_papers_true", "max_years_active"]
P_LEGACY = ["max_hindex", "max_papercount_cur2026_w99"]
X_T = X0 + P_TIERB            # primary controls (leakage-free prestige)
X_NP = X0                     # no prestige
X_LEG = X0_LEGACY + P_LEGACY  # v1 replication controls

FE_HONEST = "release_month + dow + subfield_kw"
FE_DAY = "cohort_day + subfield_kw"
FE_LEGACY_DAY = "cohort_day + subfield"
FE_LEGACY_HONEST = "release_month + dow + subfield"
Y, D = "log_citations", "log_upvotes"


def sample(cols):
    return df.dropna(subset=cols).copy().reset_index(drop=True)


def _fit_stats(fit, name):
    t = fit.tidy()
    return {"beta": float(t.loc[name, "Estimate"]), "se": float(t.loc[name, "Std. Error"]),
            "t": float(t.loc[name, "t value"]), "N": int(fit._N)}


def ols(fe, X, data, cluster="cohort_day"):
    f = pf.feols(f"{Y} ~ {D} + {' + '.join(X)} | {fe}", data=data, vcov={"CRV1": cluster})
    return _fit_stats(f, D), f


def first_stage(fe, X, Z, data, cluster="cohort_day"):
    f = pf.feols(f"{D} ~ {Z} + {' + '.join(X)} | {fe}", data=data, vcov={"CRV1": cluster})
    s = _fit_stats(f, Z)
    s["kp_f"] = s["t"] ** 2         # single instrument: KP rk Wald F == cluster-robust t^2
    return s, f


def reduced_form(fe, X, Z, data, cluster="cohort_day"):
    f = pf.feols(f"{Y} ~ {Z} + {' + '.join(X)} | {fe}", data=data, vcov={"CRV1": cluster})
    return _fit_stats(f, Z), f


def two_sls(fe, X, Z, data, cluster="cohort_day"):
    f = pf.feols(f"{Y} ~ {' + '.join(X)} | {fe} | {D} ~ {Z}", data=data, vcov={"CRV1": cluster})
    return _fit_stats(f, D), f


def ar_ci(fe, X, Z, data, cluster="cohort_day", half_width=6.0, n_grid=6001):
    """Cluster-robust Anderson-Rubin 95% CI by test inversion.
    AR(b0) = t^2 of Z in the regression (Y - b0*D) ~ Z + X | FE with CRV1(cluster), computed in closed
    form via FWL residuals (exact) with pyfixest's own CRV1 small-sample factor, verified against pyfixest at
    a check point (abs diff stored).  Grid: b0 in [b2sls - half_width, b2sls + half_width] with brentq refinement."""
    xs = " + ".join(X)
    fy = pf.feols(f"{Y} ~ {xs} | {fe}", data=data, vcov={"CRV1": cluster})
    fd = pf.feols(f"{D} ~ {xs} | {fe}", data=data, vcov={"CRV1": cluster})
    fz = pf.feols(f"{Z} ~ {xs} | {fe}", data=data, vcov={"CRV1": cluster})
    ey, ed, ez = np.asarray(fy.resid()), np.asarray(fd.resid()), np.asarray(fz.resid())
    cl = np.asarray(fy._data[cluster])
    ug, g_idx = np.unique(cl, return_inverse=True)
    G, N = len(ug), len(ey)
    # ssc factors as in the pyfixest AR regression (r ~ Z + X | FE)
    ref = pf.feols(f"{Y} ~ {Z} + {xs} | {fe}", data=data, vcov={"CRV1": cluster})
    k = int(ref._k)
    ssc = float(np.asarray(ref._ssc).ravel()[0])      # pyfixest's own CRV1 small-sample factor for this design
    zz = float(ez @ ez)
    Szy = np.bincount(g_idx, weights=ez * ey, minlength=G)
    Szd = np.bincount(g_idx, weights=ez * ed, minlength=G)
    Szz = np.bincount(g_idx, weights=ez * ez, minlength=G)
    a, b = float(ez @ ey), float(ez @ ed)
    beta_2sls = a / b

    def ar_stat(b0):
        gam = (a - b0 * b) / zz
        # cluster score sums of ez*u with u = (ey - b0 ed) - gam ez
        s = Szy - b0 * Szd - gam * Szz
        V = ssc * float(s @ s) / zz**2
        return gam**2 / V

    crit = float(stats.chi2.ppf(0.95, 1))
    # verification vs pyfixest at b0 = beta_2sls + 1 SE-ish offset (any point works)
    b_chk = beta_2sls + 0.5
    data_chk = data.copy()
    data_chk["_r"] = data_chk[Y] - b_chk * data_chk[D]
    fchk = pf.feols(f"_r ~ {Z} + {xs} | {fe}", data=data_chk, vcov={"CRV1": cluster})
    ar_pf = float(fchk.tstat()[Z]) ** 2
    ar_cf = ar_stat(b_chk)
    grid = np.linspace(beta_2sls - half_width, beta_2sls + half_width, n_grid)
    vals = np.array([ar_stat(x) for x in grid])
    inside = vals <= crit
    out = {"crit": crit, "G": int(G), "N": int(N), "k_slopes": k, "ssc_factor": ssc, "beta_2sls_fwl": float(beta_2sls),
           "closed_form_vs_pyfixest_abs_diff": abs(ar_pf - ar_cf), "grid_half_width": half_width}
    if not inside.any():
        out.update({"lower": None, "upper": None, "empty": True, "unbounded": False,
                    "string": "empty (AR rejects every b0 in grid)"})
        return out
    lo_open, hi_open = bool(inside[0]), bool(inside[-1])
    idx_in = np.where(inside)[0]
    f = lambda x: ar_stat(x) - crit
    lower = -np.inf if lo_open else brentq(f, grid[idx_in[0] - 1], grid[idx_in[0]], xtol=1e-9)
    upper = np.inf if hi_open else brentq(f, grid[idx_in[-1]], grid[idx_in[-1] + 1], xtol=1e-9)
    # detect disjoint set
    n_segments = int(np.sum(np.diff(inside.astype(int)) == 1) + (1 if inside[0] else 0))
    out.update({"lower": None if np.isinf(lower) else float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "unbounded": lo_open or hi_open, "empty": False, "n_segments": n_segments,
                "string": f"({'-inf' if np.isinf(lower) else f'{lower:.3f}'}, {'+inf' if np.isinf(upper) else f'{upper:.3f}'})"
                          + ("" if n_segments == 1 else f" [{n_segments} disjoint segments]")})
    return out


def iv_block(label, fe, X, Z, data, cluster="cohort_day", with_ar=True, note=""):
    d = sample(X + [Z, Y, D]) if data is None else data
    fs, _ = first_stage(fe, X, Z, d, cluster)
    rf, _ = reduced_form(fe, X, Z, d, cluster)
    iv, fit_iv = two_sls(fe, X, Z, d, cluster)
    o, _ = ols(fe, X, d, cluster)
    blk = {"label": label, "FE": fe, "instrument": Z, "controls": X, "cluster": cluster,
           "N": iv["N"], "N_dropped_singleton_fe": int(len(d) - iv["N"]),
           "first_stage": {"pi": fs["beta"], "se": fs["se"], "t": fs["t"], "kp_f": fs["kp_f"]},
           "reduced_form": {"beta": rf["beta"], "se": rf["se"], "t": rf["t"]},
           "iv_2sls": {"beta": iv["beta"], "se": iv["se"], "wald_ci95": [iv["beta"] - 1.96 * iv["se"], iv["beta"] + 1.96 * iv["se"]]},
           "ols_same_sample": {"beta": o["beta"], "se": o["se"]}, "note": note}
    if with_ar:
        used = fit_iv._data.copy()
        blk["ar_ci95"] = ar_ci(fe, X, Z, used, cluster)
    print(f"  {label:58s} FS pi={fs['beta']:+.3f} (t={fs['t']:+.1f}, F={fs['kp_f']:.1f})  "
          f"IV b={iv['beta']:.3f} (se {iv['se']:.3f})  OLS={o['beta']:.3f}  N={iv['N']}"
          + (f"  AR {blk['ar_ci95']['string']}" if with_ar else ""))
    return blk


R = {}
d_T = sample(X_T + ["Z1p_kw", "Zc_kw", "Zown_loo_kw", Y, D])
d_NP = sample(X_NP + ["Z1p_kw", Y, D])
d_LEG = sample(X_LEG + ["Z1p_othersub", Y, D])
print(f"samples: tierB-controls N={len(d_T):,}; no-prestige N={len(d_NP):,}; legacy N={len(d_LEG):,}")

# ─── (i) OLS under honest FE, two clusterings; and under day FE ──────────────
print("\n── (i) OLS ──")
o_day, _ = ols(FE_HONEST, X_T, d_T, "cohort_day")
o_mon, _ = ols(FE_HONEST, X_T, d_T, "release_month")
o_dayfe, _ = ols(FE_DAY, X_T, d_T, "cohort_day")
o_np, _ = ols(FE_HONEST, X_NP, d_NP, "cohort_day")
o_leg, _ = ols(FE_LEGACY_DAY, X_LEG, d_LEG, "cohort_day")
R["ols"] = {
    "honest_FE_tierB": {"FE": FE_HONEST, "controls": X_T, "beta": o_day["beta"], "N": o_day["N"],
                        "se_cluster_day": o_day["se"], "se_cluster_month": o_mon["se"],
                        "G_day": int(d_T["cohort_day"].nunique()), "G_month": int(d_T["release_month"].nunique())},
    "honest_FE_noprestige": {"FE": FE_HONEST, "controls": X_NP, "beta": o_np["beta"], "se_cluster_day": o_np["se"], "N": o_np["N"]},
    "day_FE_tierB": {"FE": FE_DAY, "controls": X_T, "beta": o_dayfe["beta"], "se_cluster_day": o_dayfe["se"], "N": o_dayfe["N"]},
    "legacy_v1_replication": {"FE": FE_LEGACY_DAY, "controls": X_LEG, "beta": o_leg["beta"], "se_cluster_day": o_leg["se"], "N": o_leg["N"],
                              "note": "v1 OLS-primaryFE replication (legacy subfield, today-measured prestige, raw n_authors)"},
}
for k, v in R["ols"].items():
    print(f"  OLS {k:24s} beta={v['beta']:.4f} se_day={v.get('se_cluster_day', float('nan')):.4f}"
          + (f" se_month={v['se_cluster_month']:.4f}" if "se_cluster_month" in v else "") + f" N={v['N']}")

# ─── (ii) PRIMARY honest 2SLS: Z1'_kw, month + dow + subfield_kw FE ─────────
print("\n── (ii) honest 2SLS (no day FE) ──")
R["primary_honest"] = iv_block("PRIMARY honest: Z1p_kw | month+dow+subfield_kw, tierB controls", FE_HONEST, X_T, "Z1p_kw", d_T,
                               note="No day FE: instrument variation is between-day crowding, not the within-day adding-up identity.")
R["primary_honest_month_cluster"] = iv_block("  same, clustered by release_month", FE_HONEST, X_T, "Z1p_kw", d_T,
                                             cluster="release_month", with_ar=True)
R["primary_honest_noprestige"] = iv_block("  honest, no prestige controls", FE_HONEST, X_NP, "Z1p_kw", d_NP)
R["honest_count_instrument"] = iv_block("  honest FE, other-subfield paper COUNT (Zc_kw)", FE_HONEST, X_T, "Zc_kw", d_T, with_ar=True,
                                        note="Count-based crowding: no upvote content in the instrument.")
d_T_multi = d_T[d_T["kw_singleton"] == 0].reset_index(drop=True)
R["honest_own_subfield_loo"] = iv_block("  honest FE, own-subfield leave-one-out peer sum (Zown_loo_kw)", FE_HONEST, X_T, "Zown_loo_kw",
                                        d_T_multi, with_ar=True,
                                        note="Same-topic crowding with no adding-up link to own upvotes (no day FE); non-singleton cells only.")

# ─── (iii) day-FE (mechanical / reflection) comparison ──────────────────────
print("\n── (iii) day-FE comparison (reflection) ──")
R["dayfe_Z1p_kw"] = iv_block("day+subfield_kw FE: Z1p_kw (v1 design, uniform taxonomy)", FE_DAY, X_T, "Z1p_kw", d_T,
                             note="Under day FE, day_total is absorbed and Z1p_kw = f(-own cell sum), which contains the paper's own upvotes.")
d_single = d_T[d_T["kw_singleton"] == 1].reset_index(drop=True)
d_multi = d_T[d_T["kw_singleton"] == 0].reset_index(drop=True)
R["dayfe_singleton_cells"] = iv_block("  singleton (day x subfield_kw) cells only", FE_DAY, X_T, "Z1p_kw", d_single, with_ar=False,
                                      note="In singleton cells Z1p_kw | day = log1p(day_total - own upvotes) exactly: pure reflection.")
R["dayfe_nonsingleton_cells"] = iv_block("  non-singleton cells only", FE_DAY, X_T, "Z1p_kw", d_multi, with_ar=False)
R["dayfe_own_subfield_loo"] = iv_block("  own-subfield leave-one-out peer sum (Zown_loo_kw)", FE_DAY, X_T, "Zown_loo_kw", d_multi, with_ar=False,
                                       note="Genuine same-topic crowding with no mechanical link to own upvotes (non-singleton cells; singleton cells have no peers).")
R["dayfe_count_instrument"] = iv_block("  other-subfield paper COUNT (Zc_kw), day FE", FE_DAY, X_T, "Zc_kw", d_T, with_ar=False)

# ─── legacy replication rows (v1 numbers) ───────────────────────────────────
print("\n── legacy replication ──")
R["legacy_v1_primary_P"] = iv_block("LEGACY v1 primary-P: Z1p_othersub | day+subfield(legacy), legacy prestige", FE_LEGACY_DAY, X_LEG,
                                    "Z1p_othersub", d_LEG, note="Replicates results/crowding_iv.json primary-P (v1 headline).")
R["legacy_v1_honest_FE"] = iv_block("LEGACY: Z1p_othersub | month+dow+subfield(legacy), legacy prestige", FE_LEGACY_HONEST, X_LEG,
                                    "Z1p_othersub", d_LEG)

# ─── diagnostics: reflection structure + balance ────────────────────────────
print("\n── diagnostics ──")
cells = d_T.groupby(["cohort_day", "subfield_kw"]).size()
cell_mean = d_T.groupby(["cohort_day", "subfield_kw"])["Z1p_kw"].transform("mean")
within_share = float(((d_T["Z1p_kw"] - cell_mean) ** 2).sum() / ((d_T["Z1p_kw"] - d_T["Z1p_kw"].mean()) ** 2).sum())
# reflection: within-day correlation of Z1p_kw with own log upvotes, singleton vs not
def within_day_corr(dd, a, b):
    ra = dd[a] - dd.groupby("cohort_day")[a].transform("mean")
    rb = dd[b] - dd.groupby("cohort_day")[b].transform("mean")
    return float(ra.corr(rb))
refl = {"corr_Z1p_kw_log_upvotes_within_day_all": within_day_corr(d_T, "Z1p_kw", D),
        "corr_Z1p_kw_log_upvotes_within_day_singleton": within_day_corr(d_single, "Z1p_kw", D),
        "corr_Z1p_kw_log_upvotes_within_day_nonsingleton": within_day_corr(d_multi, "Z1p_kw", D),
        "corr_Z1p_kw_log_upvotes_within_month": float((d_T["Z1p_kw"] - d_T.groupby("release_month")["Z1p_kw"].transform("mean")).corr(
            d_T[D] - d_T.groupby("release_month")[D].transform("mean")))}
fs_single = pf.feols(f"Z1p_kw ~ {D} | cohort_day", data=d_single, vcov={"CRV1": "cohort_day"}).tidy().loc[D]
fs_multi = pf.feols(f"Z1p_kw ~ {D} | cohort_day", data=d_multi, vcov={"CRV1": "cohort_day"}).tidy().loc[D]
R["diagnostics"] = {
    "taxonomy": "subfield_kw (13 levels, one keyword rule set for all papers)",
    "cells_day_x_subfield_kw": int(len(cells)), "singleton_cell_share": float((cells == 1).mean()),
    "paper_singleton_share": float(d_T["kw_singleton"].mean()), "median_cell_size": float(cells.median()),
    "Z1p_kw_within_cell_variance_share": within_share,
    "reflection_within_day": {**refl,
                              "slope_Z1p_on_logup_singleton_dayFE": {"coef": float(fs_single["Estimate"]), "t": float(fs_single["t value"])},
                              "slope_Z1p_on_logup_nonsingleton_dayFE": {"coef": float(fs_multi["Estimate"]), "t": float(fs_multi["t value"])}},
}
bal = {}
for cov in ["age_months", "log_n_authors", "has_github", "log1p_max_prior_papers_true", "max_years_active"]:
    f = pf.feols(f"{cov} ~ Z1p_kw | {FE_HONEST}", data=d_T, vcov={"CRV1": "cohort_day"}).tidy().loc["Z1p_kw"]
    bal[cov] = {"coef": float(f["Estimate"]), "se": float(f["Std. Error"]), "t": float(f["t value"]),
                "imbalance_flag": bool(abs(f["t value"]) > 2)}
R["diagnostics"]["balance_on_Z1p_kw_honest_FE"] = bal
print(f"  cells={len(cells):,} singleton cell share={R['diagnostics']['singleton_cell_share']:.3f} "
      f"paper share={R['diagnostics']['paper_singleton_share']:.3f}; within-cell var share={within_share:.4f}")
print(f"  reflection corr within day: all {refl['corr_Z1p_kw_log_upvotes_within_day_all']:+.3f}, singleton "
      f"{refl['corr_Z1p_kw_log_upvotes_within_day_singleton']:+.3f}, non-singleton {refl['corr_Z1p_kw_log_upvotes_within_day_nonsingleton']:+.3f}")
print("  balance (honest FE): " + ", ".join(f"{k} t={v['t']:+.2f}" for k, v in bal.items()))

# ─── (iv) verdict generated from numbers ────────────────────────────────────
ph = R["primary_honest"]
ar = ph["ar_ci95"]
ols_b = R["ols"]["honest_FE_tierB"]["beta"]
iv_b, iv_se = ph["iv_2sls"]["beta"], ph["iv_2sls"]["se"]
ar_lo, ar_hi = ar["lower"], ar["upper"]
lo_txt = "-inf" if ar_lo is None else f"{ar_lo:.3f}"
hi_txt = "+inf" if ar_hi is None else f"{ar_hi:.3f}"
includes_zero = (ar_lo is None or ar_lo <= 0) and (ar_hi is None or ar_hi >= 0)
includes_ols = (ar_lo is None or ar_lo <= ols_b) and (ar_hi is None or ar_hi >= ols_b)
width = None if (ar_lo is None or ar_hi is None) else ar_hi - ar_lo
fs_t = ph["first_stage"]["t"]
day = R["dayfe_Z1p_kw"]
sing_b, multi_b = R["dayfe_singleton_cells"]["iv_2sls"]["beta"], R["dayfe_nonsingleton_cells"]["iv_2sls"]["beta"]
loo_t = R["dayfe_own_subfield_loo"]["first_stage"]["t"]
cnt_day_pi = R["dayfe_count_instrument"]["first_stage"]["pi"]

facts = []
facts.append(f"Honest spec (month+dow+subfield_kw FE, leakage-free prestige, N={ph['N']:,}): first-stage t={fs_t:+.2f} "
             f"(KP-F={ph['first_stage']['kp_f']:.1f}), 2SLS beta={iv_b:.3f} (SE {iv_se:.3f}), AR 95% CI ({lo_txt}, {hi_txt})"
             + (f", width {width:.2f} log-points" if width is not None else "") + ".")
facts.append(("The AR interval includes zero" if includes_zero else "The AR interval excludes zero")
             + (" and includes the OLS coefficient" if includes_ols else " and excludes the OLS coefficient")
             + f" ({ols_b:.3f}); it is " + ("too wide to be informative about the size of the effect."
                                            if (width is None or width > 0.5) else "moderately informative."))
facts.append(f"Under day FE (v1 design) the first stage looks strong (t={day['first_stage']['t']:+.1f}, KP-F={day['first_stage']['kp_f']:.0f}) "
             f"but this is the within-day adding-up identity: {R['diagnostics']['paper_singleton_share']:.0%} of papers sit in singleton "
             f"(day x subfield) cells where Z1' is exactly log1p(day total - own upvotes); the 2SLS coefficient is {sing_b:.3f} in singleton cells vs "
             f"{multi_b:.3f} in non-singleton cells (all-sample {day['iv_2sls']['beta']:.3f}).")
facts.append(f"A crowding instrument with no mechanical link to own upvotes (own-subfield leave-one-out peer sum) has "
             + ("no usable first stage" if abs(loo_t) < 1.96 else "a first stage") + f" (t={loo_t:+.2f}); the other-subfield paper COUNT "
             f"under day FE has first-stage sign {'positive' if cnt_day_pi > 0 else 'negative'} ({cnt_day_pi:+.3f}), "
             + ("the wrong sign for a crowding story." if cnt_day_pi > 0 else "consistent with crowding."))
loo_h = R["honest_own_subfield_loo"]
loo_h_t = loo_h["first_stage"]["t"]
loo_sentence = (f"a same-topic crowding instrument with no adding-up link to own upvotes (own-subfield leave-one-out peer sum, no day FE) "
                f"has no usable first stage (t={loo_h_t:+.2f})" if abs(loo_h_t) < 1.96 else
                f"the same-topic leave-one-out peer sum (no day FE) has a first stage (t={loo_h_t:+.2f}, F={loo_h['first_stage']['kp_f']:.1f}) "
                f"but its 2SLS estimate is imprecise (beta={loo_h['iv_2sls']['beta']:.3f}, SE {loo_h['iv_2sls']['se']:.3f}, "
                f"AR {loo_h['ar_ci95']['string']})")
verdict = (f"VERDICT: the crowding IV is an attempted design, not informative causal evidence. Under the honest specification "
           f"(release_month + dow + subfield_kw FE, no day FE) the first stage is modest (t={fs_t:+.1f}, KP-F={ph['first_stage']['kp_f']:.0f}) "
           f"and the weak-IV-robust AR 95% CI ({lo_txt}, {hi_txt}) is "
           + (f"{width:.2f} log-points wide, " if width is not None else "unbounded, ")
           + ("includes zero" if includes_zero else "barely excludes zero" if (ar_lo is not None and ar_lo < 0.1) else "excludes zero")
           + (" and includes" if includes_ols else " and excludes") + f" the OLS estimate ({ols_b:.3f}). "
           f"The day-FE first stage (KP-F={day['first_stage']['kp_f']:.0f}) is the within-day adding-up identity, not exogenous variation "
           f"(2SLS beta {sing_b:.3f} in singleton cells vs {multi_b:.3f} in non-singleton cells); {loo_sentence}. "
           "Report as 'consistent with OLS but uninformative'; do not describe as suggestive causal support.")
R["verdict"] = {"string": verdict, "facts": facts,
                "flags": {"ar_includes_zero": includes_zero, "ar_includes_ols": includes_ols, "ar_width": width,
                          "own_loo_dayFE_first_stage_null": bool(abs(loo_t) < 1.96),
                          "own_loo_honestFE_first_stage_null": bool(abs(loo_h_t) < 1.96),
                          "count_instrument_wrong_sign_dayFE": bool(cnt_day_pi > 0)}}
R["headline"] = {"spec": "primary_honest", "beta_2sls": iv_b, "se_cluster_day": iv_se, "AR_CI_95": [ar_lo, ar_hi],
                 "AR_CI_string": f"({lo_txt}, {hi_txt})", "first_stage_t": fs_t, "first_stage_kp_f": ph["first_stage"]["kp_f"],
                 "reduced_form_beta": ph["reduced_form"]["beta"], "reduced_form_t": ph["reduced_form"]["t"],
                 "OLS_beta_same_FE": ols_b, "OLS_se_day": R["ols"]["honest_FE_tierB"]["se_cluster_day"],
                 "OLS_se_month": R["ols"]["honest_FE_tierB"]["se_cluster_month"], "N": ph["N"],
                 "day_FE_beta_for_comparison": day["iv_2sls"]["beta"], "day_FE_kp_f": day["first_stage"]["kp_f"],
                 "legacy_v1_beta": R["legacy_v1_primary_P"]["iv_2sls"]["beta"], "verdict": verdict}
R["meta"] = {"generated": datetime.now().isoformat(timespec="seconds"), "script": "scripts/25_crowding_iv_v3.py",
             "python_version": sys.version.split()[0], "pyfixest": pf.__version__, "N_base": N_base,
             "inputs": ["data/processed/analysis_final.csv", "data/processed/crowding_v3.csv"],
             "runtime_seconds": round(time.time() - T0, 1)}
print("\n" + verdict)


def js(o):
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(OUT_JSON, "w") as f:
    json.dump(js(R), f, indent=1)
print(f"JSON -> {OUT_JSON}")


# ─── NOTE ───────────────────────────────────────────────────────────────────
def row(name, b):
    fs, iv = b["first_stage"], b["iv_2sls"]
    arx = b.get("ar_ci95", {}).get("string", "—")
    return (f"| {name} | {b['FE']} | {b['instrument']} | {fs['pi']:+.3f} ({fs['t']:+.1f}) | {fs['kp_f']:.1f} | "
            f"{iv['beta']:.3f} ({iv['se']:.3f}) | {arx} | {b['ols_same_sample']['beta']:.3f} | {b['N']:,} |")


rows = [("PRIMARY honest (tierB controls)", "primary_honest"), ("honest, cluster=month", "primary_honest_month_cluster"),
        ("honest, no prestige", "primary_honest_noprestige"), ("honest, COUNT instrument", "honest_count_instrument"),
        ("honest, own-subfield LOO peers", "honest_own_subfield_loo"),
        ("day FE (v1 design, kw taxonomy)", "dayfe_Z1p_kw"), ("day FE, singleton cells", "dayfe_singleton_cells"),
        ("day FE, non-singleton cells", "dayfe_nonsingleton_cells"), ("day FE, own-subfield LOO peers", "dayfe_own_subfield_loo"),
        ("day FE, COUNT instrument", "dayfe_count_instrument"), ("LEGACY v1 primary-P replication", "legacy_v1_primary_P"),
        ("LEGACY, honest FE", "legacy_v1_honest_FE")]
o = R["ols"]["honest_FE_tierB"]
note = f"""# D1 v3 — Crowding IV: attempted design, honest re-specification

Generated {R['meta']['generated']} | script scripts/25_crowding_iv_v3.py | pyfixest {pf.__version__} | N_base={N_base:,}

## Verdict
{verdict}

""" + "\n".join(f"- {x}" for x in facts) + f"""

## (i) OLS, release_month + dow + subfield_kw FE (leakage-free prestige, log n_authors)
beta = {o['beta']:.4f}; SE clustered by cohort_day {o['se_cluster_day']:.4f} (G={o['G_day']}), by release_month {o['se_cluster_month']:.4f} (G={o['G_month']}); N={o['N']:,}.
No-prestige: {R['ols']['honest_FE_noprestige']['beta']:.4f} ({R['ols']['honest_FE_noprestige']['se_cluster_day']:.4f}); day+subfield_kw FE: {R['ols']['day_FE_tierB']['beta']:.4f} ({R['ols']['day_FE_tierB']['se_cluster_day']:.4f}); legacy v1 replication: {R['ols']['legacy_v1_replication']['beta']:.4f}.

## (ii)–(iii) IV ladder (cluster = cohort_day unless stated; SE in parentheses; AR = cluster-robust Anderson–Rubin 95% CI)
| Spec | FE | Z | first-stage π (t) | KP-F | 2SLS β (SE) | AR 95% CI | OLS same sample | N |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(row(n, R[k]) for n, k in rows) + f"""

AR CI implementation: test inversion of the cluster-robust t² of Z in (Y − β0·D) ~ Z + X | FE, closed-form via FWL residuals with
pyfixest small-sample factors; closed form vs pyfixest at a check point differ by {ar['closed_form_vs_pyfixest_abs_diff']:.2e}.

## Reflection diagnostics (uniform taxonomy)
- (day × subfield_kw) cells: {R['diagnostics']['cells_day_x_subfield_kw']:,}; singleton cells {R['diagnostics']['singleton_cell_share']:.1%} of cells, {R['diagnostics']['paper_singleton_share']:.1%} of papers; within-cell variance share of Z1'_kw = {R['diagnostics']['Z1p_kw_within_cell_variance_share']:.4f}.
- Within-day corr(Z1'_kw, log upvotes): all {refl['corr_Z1p_kw_log_upvotes_within_day_all']:+.3f}; singleton {refl['corr_Z1p_kw_log_upvotes_within_day_singleton']:+.3f}; non-singleton {refl['corr_Z1p_kw_log_upvotes_within_day_nonsingleton']:+.3f}. Within-month corr: {refl['corr_Z1p_kw_log_upvotes_within_month']:+.3f}.
- Balance of Z1'_kw under honest FE: """ + "; ".join(f"{k} t={v['t']:+.2f}{' (imbalance)' if v['imbalance_flag'] else ''}" for k, v in bal.items()) + f"""

## What may be said
- OLS association (β≈{o['beta']:.2f} log-points per log-upvote, month+dow+subfield_kw FE) is precise and robust to clustering level.
- The IV is an attempted design: honest first stage t={fs_t:+.1f}; AR interval ({lo_txt}, {hi_txt}) — consistent with OLS but uninformative; the day-FE F≈{day['first_stage']['kp_f']:.0f} is mechanical (reflection); own-subfield LOO peer sum first stage t={loo_t:+.2f} (day FE) / {loo_h_t:+.2f} (honest FE), both with wide 2SLS intervals.
- Not to be described as causal support, suggestive or otherwise.
"""
with open(OUT_NOTE, "w") as f:
    f.write(note)
print(f"NOTE -> {OUT_NOTE}\nDone in {time.time() - T0:.0f}s.")
