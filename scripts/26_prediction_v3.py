"""
D3 v3 — Leakage-audited incremental prediction (PROJECT HEADLINE), audit-response rewrite
=========================================================================================
Supersedes scripts/26_prediction.py (kept untouched for reference). Implements the
recommendations of diagnostics/01_reproducibility.md and diagnostics/02_stats_methods_critique.md
(§A, §F rows 1-5).

Design (unchanged): forward-in-time split, train = release_year <= 2024, test = release_year 2025.
Label (primary): top decile of citation_count rank within release QUARTER (y_q).
Robustness labels: within release MONTH (y_m); top-decile INFLUENTIAL citations within quarter (y_inf).

What changed vs v1
  * subfield control = `subfield_kw` (one keyword taxonomy for every paper); the legacy mixed-taxonomy
    `subfield` appears only in an explicitly-flagged "legacy_subfield" robustness row.
  * P_tierB (leakage-free prior-paper count + years active) is the DEFINITIVE baseline;
    P_interim (2026 productivity) and P_none are brackets.  tierB_resolved / first_year / raw
    unwinsorised counts are NOT features.
  * Pre-specified headline model = L2 logistic regression (standardised, C tuned by GroupKFold on
    train months).  HGB reported with equal prominence, tuned with early stopping.
  * Rows: controls-only, +attention, upvotes-only, +attention+comments (ablation).
  * Metrics: AUC, PR-AUC, precision@top-decile, P@100 (noisy), Brier; paired bootstrap CIs
    (2000 reps, seed 1626) both i.i.d. and release-month-clustered (month-clustered = primary).
  * Robustness: within-month label; influential-citation label; backward test (train 2024-25 ->
    test 2023); mature K=12 subset; drop age_months; legacy subfield; no launch-era months in
    train; v2 leaky replication; per-quarter upvotes-only AUC.
  * Audit: corrected subfield provenance; label-provenance-uniform check; upvotes are cumulative
    at collection (evidence rows: monthly median upvotes, Spearman(upvotes, age) by year).
  * No hard-coded sample sizes / dates in prose; everything computed and written to JSON.

Outputs
  results/prediction_v3.json, results/prediction_v3_NOTE.md, results/prediction_v3_tables.md,
  results/prediction_v3_scores.csv (per test row: ids, labels, predicted probabilities per model/branch/row)

Usage
  python scripts/26_prediction_v3.py
"""

import json
import sys
import time
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

T0 = time.time()
SEED = 1626
N_BOOT = 2000
N_JOBS = -1
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/analysis_final.csv"
OUT_JSON = ROOT / "results/prediction_v3.json"
OUT_NOTE = ROOT / "results/prediction_v3_NOTE.md"
OUT_TABLES = ROOT / "results/prediction_v3_tables.md"
OUT_SCORES = ROOT / "results/prediction_v3_scores.csv"

# Launch-era months of HF Daily Papers (platform ramp; audit §A.6): everything up to and
# including this month is excluded from train in the "no_launch_months" robustness row.
LAUNCH_ERA_END = "2023-06"
K12_MONTHS = 12.0          # mature-subset exposure threshold (months)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD + DERIVED COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH, dtype={"arxiv_id_clean": str})
assert df["arxiv_id_clean"].is_unique
n_all = len(df)
print(f"  n={n_all:,}  cols={df.shape[1]}")

df["release_quarter"] = df["release_month"].str[:4] + "Q" + (
    (df["release_month"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)

# labels (citations are label-only; never features)
df["q_i"] = df.groupby("release_quarter")["citation_count"].rank(pct=True)
df["y_q"] = (df["q_i"] >= 0.90).astype(int)
df["y_m"] = (df.groupby("release_month")["citation_count"].rank(pct=True) >= 0.90).astype(int)
df["y_inf"] = (df.groupby("release_quarter")["influential_citations"].rank(pct=True) >= 0.90).astype(int)

# attention + transformed controls
df["upvote_rank_within_month"] = df.groupby("release_month")["upvotes"].rank(pct=True)
df["log_n_authors"] = np.log1p(df["n_authors"])
df["log1p_num_comments"] = np.log1p(df["num_comments"])
df["arxiv_fetched_flag"] = df["primary_category"].notna().astype(int)   # legacy provenance flag (audit only)
df["mature_k12"] = (df["age_months"] >= K12_MONTHS).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE DISCIPLINE
# ─────────────────────────────────────────────────────────────────────────────
FORBIDDEN = {
    # outcomes / outcome-adjacent
    "citation_count", "log_citations", "influential_citations", "reference_count",
    "q_i", "y_q", "y_m", "y_inf",
    # post-hoc / today-measured
    "n_trend_days", "github_stars", "last_author_hindex", "author_max_appear",   # max_hindex: FLAG_ONLY
    "max_papercount_cur2026", "first_author_papercount_cur2026", "last_author_papercount_cur2026",
    "first_author_papercount_cur2026_w99", "last_author_papercount_cur2026_w99",
    # Tier-B admin / raw columns that are not predictors
    "tierB_resolved", "first_author_prior_papers_true", "last_author_prior_papers_true",
    "max_prior_papers_true", "first_author_prior_papers_true_w99", "last_author_prior_papers_true_w99",
    "max_prior_papers_true_w99",
    # D1 instruments
    "cohort_day", "cohort_size", "Z1_logcompet", "Z2_count", "Z3_blockbuster", "Z1p_othersub", "ego_upvotes",
    # provenance / admin / split keys
    "gap_days_v1", "gap_days_rawpub", "published_v1", "v1_source", "anchor_date_used",
    "prestige_resolved", "primary_category", "arxiv_fetched_flag", "subfield_kw_source",
    "release_month", "release_year", "release_quarter", "first_trend_date", "mature_k12",
    "upvotes", "num_comments", "n_authors", "arxiv_id_clean",
}
# columns allowed ONLY in explicitly flagged rows
FLAG_ONLY = {"max_hindex": "v2_leaky_replication", "subfield": "legacy_subfield"}

CONTROL_NUM = [
    "age_months", "title_n_words", "title_has_colon", "abstract_n_chars",
    "kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning", "kw_benchmark",
    "kw_survey", "kw_efficient", "kw_multimodal", "kw_rl", "kw_scaling",
    "log_n_authors", "has_github",
]
CAT_COLS = ["subfield_kw", "dow"]
ATTENTION = ["log_upvotes", "upvote_rank_within_month"]
COMMENTS = ["log1p_num_comments"]
PRESTIGE = {
    "P_tierB":   ["log1p_max_prior_papers_true", "max_years_active"],   # leakage-free (DEFINITIVE)
    "P_interim": ["max_papercount_cur2026_w99"],                       # anachronistic 2026 count
    "P_none":    [],
}
BRANCH_TAGS = {
    "P_tierB":   "DEFINITIVE — leakage-free prior-paper count (log1p, strictly before anchor date) + years active",
    "P_interim": "ROBUSTNESS — anachronistic 2026 career paper count (winsorised)",
    "P_none":    "ROBUSTNESS — no prestige control",
}
ROWS = {                                # extra columns added on top of controls (None = upvotes only)
    "controls_only":       [],
    "+attention":          ATTENTION,
    "+attention+comments": ATTENTION + COMMENTS,
    "upvotes_only":        None,
}
MODELS = ["logistic", "hgb"]
HEADLINE_MODEL = "logistic"             # pre-specified (v3 spec); HGB reported with equal prominence
HEADLINE_BRANCH = "P_tierB"
HEADLINE_ROW = "+attention"
HEADLINE_CI = "month_cluster"


def check_features(cols, allow_flag=None):
    bad = set(cols) & FORBIDDEN
    for c, tag in FLAG_ONLY.items():
        if c in cols and allow_flag != tag:
            bad.add(c)
    if bad:
        raise ValueError(f"FORBIDDEN columns in feature matrix: {sorted(bad)}")


def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("scl", StandardScaler())]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                             sparse_output=False))]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


# C listed in DESCENDING order so that exact CV ties (e.g. the one-feature upvotes-only model, whose AUC is
# invariant to C) resolve to the least-regularised, best-calibrated fit.
LOGIT_GRID = {"mdl__C": [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]}
HGB_GRID = {"mdl__max_depth": [2, 3, 4], "mdl__l2_regularization": [0.0, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
            "mdl__min_samples_leaf": [20, 50]}


def tune_and_fit(model_name, X, y, groups, num_cols, cat_cols):
    """GroupKFold(5) by release_month on TRAIN only; refit best on full train."""
    pre = make_preprocessor(num_cols, cat_cols)
    if model_name == "logistic":
        est = Pipeline([("pre", pre), ("mdl", LogisticRegression(penalty="l2", solver="lbfgs",
                                                                 max_iter=5000, random_state=SEED))])
        grid = LOGIT_GRID
    elif model_name == "hgb":
        est = Pipeline([("pre", pre), ("mdl", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=1000, early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=30, random_state=SEED))])
        grid = HGB_GRID
    else:
        raise ValueError(model_name)
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=5), scoring="roc_auc",
                      n_jobs=N_JOBS, refit=True)
    gs.fit(X, y, groups=groups)
    info = {"best_params": {k.replace("mdl__", ""): v for k, v in gs.best_params_.items()},
            "cv_auc_mean": float(gs.best_score_)}
    if model_name == "hgb":
        info["n_iter"] = int(gs.best_estimator_.named_steps["mdl"].n_iter_)
    return gs.best_estimator_, info


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS + PAIRED BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def p_at_k(y, s, k):
    order = np.argsort(-s, kind="stable")[:k]
    return float(y[order].mean())


def point_metrics(y, s, k):
    return {
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "p_at_k": p_at_k(y, s, k),
        "p_at_100": p_at_k(y, s, min(100, len(y))),
        "brier": float(brier_score_loss(y, s)),
        "log_loss": float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6))),
    }


BOOT_METRICS = ["auc", "pr_auc", "p_at_k", "p_at_100"]


def fast_auc(y, s):
    """AUC via average ranks (ties count 1/2) — identical to sklearn.roc_auc_score."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fast_ap(y, s):
    """Average precision with sklearn's step-wise definition (thresholds at distinct scores)."""
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    s_s = s[order]
    distinct = np.where(np.diff(s_s) != 0)[0]
    thr_idx = np.r_[distinct, len(y_s) - 1]
    tps = np.cumsum(y_s)[thr_idx]
    fps = 1 + thr_idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _boot_chunk(y, S, grp_idx, reps, seed):
    """Return array (reps, m, 4) of [auc, pr_auc, p@10%, p@100] for m score vectors under a
    common resample (paired).  grp_idx=None -> i.i.d. rows; else list of index arrays per cluster."""
    rng = np.random.default_rng(seed)
    n, m = S.shape
    out = np.full((reps, m, 4), np.nan)
    for r in range(reps):
        if grp_idx is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(grp_idx), len(grp_idx))
            idx = np.concatenate([grp_idx[g] for g in pick])
        yb = y[idx]
        npos = yb.sum()
        if npos == 0 or npos == len(yb):
            continue
        kb = max(1, int(round(0.1 * len(yb))))
        k100 = min(100, len(yb))
        for j in range(m):
            sb = S[idx, j]
            out[r, j, 0] = fast_auc(yb, sb)
            out[r, j, 1] = fast_ap(yb, sb)
            out[r, j, 2] = p_at_k(yb, sb, kb)
            out[r, j, 3] = p_at_k(yb, sb, k100)
    return out


def paired_bootstrap(y, score_dict, groups, n_boot=N_BOOT, seed=SEED, n_chunks=10):
    """Paired bootstrap for all score vectors at once. Returns {scheme: {name: (n_boot,4) array}}.
    Identical score vectors (e.g. the branch-free upvotes-only row) are computed once."""
    names = list(score_dict.keys())
    uniq, alias = [], {}
    for k in names:
        v = np.asarray(score_dict[k], dtype=float)
        hit = next((u for u in uniq if np.array_equal(score_dict[u], v)), None)
        if hit is None:
            uniq.append(k)
            alias[k] = k
        else:
            alias[k] = hit
    S = np.column_stack([np.asarray(score_dict[k], dtype=float) for k in uniq])
    y = np.asarray(y, dtype=int)
    g = np.asarray(groups)
    ug = np.unique(g)
    grp_idx = [np.where(g == u)[0] for u in ug]
    reps = [n_boot // n_chunks] * n_chunks
    reps[-1] += n_boot - sum(reps)
    out = {}
    for scheme, gi in [("iid", None), ("month_cluster", grp_idx)]:
        chunks = Parallel(n_jobs=N_JOBS)(
            delayed(_boot_chunk)(y, S, gi, reps[c], seed * 1000 + c + (0 if scheme == "iid" else 500))
            for c in range(n_chunks))
        arr = np.concatenate(chunks, axis=0)
        col = {nm: j for j, nm in enumerate(uniq)}
        out[scheme] = {nm: arr[:, col[alias[nm]], :] for nm in names}
    return out


def ci_from(arr_col):
    a = arr_col[~np.isnan(arr_col)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def level_cis(boot, name):
    return {scheme: {m: ci_from(boot[scheme][name][:, i]) for i, m in enumerate(BOOT_METRICS)}
            for scheme in boot}


def delta_cis(boot, name_b, name_a):
    return {scheme: {m: ci_from(boot[scheme][name_b][:, i] - boot[scheme][name_a][:, i])
                     for i, m in enumerate(BOOT_METRICS)} for scheme in boot}


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
FIT_CACHE = {}
TUNING_LOG = {}
SCORE_STORE = {}     # (experiment, split_name) -> DataFrame of scores (for the CSV)


def run_experiment(name, train_df, test_df, label, branches, rows, models=MODELS,
                   control_num=CONTROL_NUM, cat_cols=CAT_COLS, allow_flag=None,
                   extra_num=None, eval_sets=None, description=""):
    """Fit models for every branch x row; evaluate on test_df[label]; bootstrap CIs.
    eval_sets: optional {name: (mask over test_df, label_col)} for extra evaluations
    (e.g. mature K=12 with its own re-ranked label)."""
    print(f"\n=== Experiment: {name}  (label={label}; n_train={len(train_df):,}; n_test={len(test_df):,}) ===")
    y_train = train_df[label].values
    groups_train = train_df["release_month"].values
    y_test = test_df[label].values
    n_test = len(test_df)
    k_dec = int(round(0.1 * n_test))
    res = {"description": description, "label": label,
           "train_years": sorted(train_df["release_year"].unique().tolist()),
           "test_years": sorted(test_df["release_year"].unique().tolist()),
           "n_train": int(len(train_df)), "n_test": int(n_test),
           "n_pos_train": int(y_train.sum()), "n_pos_test": int(y_test.sum()),
           "base_rate_train": float(y_train.mean()), "base_rate_test": float(y_test.mean()),
           "k_top_decile": k_dec, "cat_cols": list(cat_cols), "models": {}}
    scores = {m: {} for m in models}
    feats = {}
    for m in models:
        for b, pcols in branches.items():
            for r, extra in rows.items():
                if extra is None:                                   # upvotes-only (branch-free)
                    num, cat = ["log_upvotes"], []
                else:
                    num = list(control_num) + list(pcols) + list(extra_num or []) + list(extra)
                    cat = list(cat_cols)
                check_features(num + cat, allow_flag=allow_flag)
                key = (m, label, tuple(num), tuple(cat), name.split("::")[0])
                if key not in FIT_CACHE:
                    t1 = time.time()
                    model, info = tune_and_fit(m, train_df[num + cat], y_train, groups_train, num, cat)
                    info["fit_seconds"] = round(time.time() - t1, 1)
                    FIT_CACHE[key] = (model, info)
                    print(f"  fit {m:8s} {b:9s} {r:20s} cv_auc={info['cv_auc_mean']:.4f} "
                          f"params={info['best_params']} ({info['fit_seconds']}s)")
                model, info = FIT_CACHE[key]
                probs = model.predict_proba(test_df[num + cat])[:, 1]
                scores[m][(b, r)] = probs
                feats[(b, r)] = {"num": num, "cat": cat}
                TUNING_LOG[f"{name}|{m}|{b}|{r}"] = info

    # ── metrics + bootstrap on the main test set ────────────────────────────
    eval_specs = {"main": (np.ones(n_test, dtype=bool), label)}
    if eval_sets:
        eval_specs.update(eval_sets)
    for m in models:
        res["models"][m] = {}
        for ev_name, (mask, ev_label) in eval_specs.items():
            y_ev = test_df.loc[mask, ev_label].values
            g_ev = test_df.loc[mask, "release_month"].values
            k_ev = int(round(0.1 * mask.sum()))
            sd = {f"{b}|{r}": scores[m][(b, r)][mask] for (b, r) in scores[m]}
            boot = paired_bootstrap(y_ev, sd, g_ev)
            block = {"n": int(mask.sum()), "n_pos": int(y_ev.sum()), "k_top_decile": k_ev,
                     "label": ev_label, "rows": {}}
            for (b, r), s in scores[m].items():
                nm = f"{b}|{r}"
                ent = point_metrics(y_ev, s[mask], k_ev)
                assert abs(ent["auc"] - fast_auc(y_ev, s[mask])) < 1e-9
                assert abs(ent["pr_auc"] - fast_ap(y_ev, s[mask])) < 1e-9
                ent["ci"] = level_cis(boot, nm)
                ent["features"] = feats[(b, r)]
                if r != "controls_only" and (b, "controls_only") in scores[m]:
                    base = f"{b}|controls_only"
                    pm_base = point_metrics(y_ev, scores[m][(b, "controls_only")][mask], k_ev)
                    ent["delta_vs_controls_only"] = {
                        **{f"d_{k}": ent[k] - pm_base[k] for k in BOOT_METRICS + ["brier"]},
                        "ci": delta_cis(boot, nm, base)}
                if r == "+attention" and (b, "upvotes_only") in scores[m]:
                    up = f"{b}|upvotes_only"
                    pm_up = point_metrics(y_ev, scores[m][(b, "upvotes_only")][mask], k_ev)
                    ent["delta_vs_upvotes_only"] = {
                        **{f"d_{k}": ent[k] - pm_up[k] for k in BOOT_METRICS},
                        "ci": delta_cis(boot, nm, up)}
                block["rows"][nm] = ent
                if ev_name == "main":
                    print(f"  {m:8s} {nm:32s} AUC={ent['auc']:.4f} PR={ent['pr_auc']:.4f} "
                          f"P@{k_ev}={ent['p_at_k']:.3f} Brier={ent['brier']:.4f}"
                          + (f"  dAUC={ent['delta_vs_controls_only']['d_auc']:+.4f} "
                             f"mCI={np.round(ent['delta_vs_controls_only']['ci']['month_cluster']['auc'], 4).tolist()}"
                             if "delta_vs_controls_only" in ent else ""))
            res["models"][m][ev_name] = block

    # ── store scores for the CSV ─────────────────────────────────────────────
    sc = pd.DataFrame({"arxiv_id_clean": test_df["arxiv_id_clean"].values})
    for m in models:
        for (b, r), s in scores[m].items():
            sc[f"{name}__{m}__{b}__{r}"] = s
    SCORE_STORE[name] = sc
    print(f"  [{name}] done in {time.time() - T0:.0f}s cumulative")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPLITS
# ─────────────────────────────────────────────────────────────────────────────
train_df = df[df["release_year"] <= 2024].copy().reset_index(drop=True)
test_df = df[df["release_year"] == 2025].copy().reset_index(drop=True)
assert set(train_df["release_quarter"]).isdisjoint(set(test_df["release_quarter"]))
assert train_df["release_year"].max() < test_df["release_year"].min()

# mature K=12 evaluation set within test: re-rank labels within quarter on the subset
mature_mask = (test_df["mature_k12"] == 1).values
test_df["y_q_mature"] = np.nan
test_df.loc[mature_mask, "y_q_mature"] = (
    test_df.loc[mature_mask].groupby("release_quarter")["citation_count"].rank(pct=True) >= 0.90).astype(int)
test_df["y_q_mature"] = test_df["y_q_mature"].fillna(0).astype(int)   # only used under mature_mask
n_mature = int(mature_mask.sum())

# backward split
train_bw = df[df["release_year"] >= 2024].copy().reset_index(drop=True)
test_bw = df[df["release_year"] == 2023].copy().reset_index(drop=True)

# launch-era exclusion for train
train_nl = train_df[train_df["release_month"] > LAUNCH_ERA_END].copy().reset_index(drop=True)

experiments = {}

# ── E1: forward, y_q, all branches, all rows (HEADLINE) ─────────────────────
experiments["forward_yq"] = run_experiment(
    "forward_yq", train_df, test_df, "y_q", PRESTIGE, ROWS,
    eval_sets={"mature_k12": (mature_mask, "y_q_mature")},
    description="Forward-in-time (train<=2024, test 2025), within-quarter top-decile label; "
                "3 prestige branches x 4 feature rows; mature K=12 evaluation uses the same fitted "
                "models on test rows with age_months>=12 and labels re-ranked within quarter on that subset.")

# ── E2: forward, y_m (within-month label) — P_tierB ─────────────────────────
ROWS_CORE = {k: ROWS[k] for k in ["controls_only", "+attention", "upvotes_only"]}
experiments["forward_ym"] = run_experiment(
    "forward_ym", train_df, test_df, "y_m", {"P_tierB": PRESTIGE["P_tierB"]}, ROWS_CORE,
    description="Robustness: label = top decile of citation_count within release MONTH.")

# ── E3: forward, y_inf (influential citations) — P_tierB ────────────────────
experiments["forward_yinf"] = run_experiment(
    "forward_yinf", train_df, test_df, "y_inf", {"P_tierB": PRESTIGE["P_tierB"]}, ROWS_CORE,
    description="Robustness: label = top decile of INFLUENTIAL citations within release quarter.")

# ── E4/E5: backward test — train 2024-25, test 2023 (>=29 months exposure) ─
experiments["backward_yq"] = run_experiment(
    "backward_yq", train_bw, test_bw, "y_q", {"P_tierB": PRESTIGE["P_tierB"]}, ROWS_CORE,
    description="Backward test: train on 2024-2025, test on 2023 (mature cohort, 29-40 months exposure); "
                "within-quarter label.")
experiments["backward_ym"] = run_experiment(
    "backward_ym", train_bw, test_bw, "y_m", {"P_tierB": PRESTIGE["P_tierB"]},
    {k: ROWS[k] for k in ["controls_only", "+attention"]},
    description="Backward test with within-month label.")

# ── E6: drop age_months (out-of-support across the split) ───────────────────
experiments["drop_age"] = run_experiment(
    "drop_age", train_df, test_df, "y_q", {"P_tierB": PRESTIGE["P_tierB"]},
    {k: ROWS[k] for k in ["controls_only", "+attention"]},
    control_num=[c for c in CONTROL_NUM if c != "age_months"],
    description="Robustness: age_months removed from controls (train and test age ranges do not overlap).")

# ── E7: legacy mixed-taxonomy subfield (what the v1 baseline did) ───────────
experiments["legacy_subfield"] = run_experiment(
    "legacy_subfield", train_df, test_df, "y_q", {"P_tierB": PRESTIGE["P_tierB"]},
    {k: ROWS[k] for k in ["controls_only", "+attention"]},
    cat_cols=["subfield", "dow"], allow_flag="legacy_subfield",
    description="FLAGGED robustness: legacy `subfield` (two taxonomies keyed on upvote-ordered arXiv fetch) "
                "instead of subfield_kw — shows the attention proxy that leaked into the v1 controls-only baseline.")

# ── E8: exclude launch-era months from train ────────────────────────────────
experiments["no_launch_months"] = run_experiment(
    "no_launch_months", train_nl, test_df, "y_q", {"P_tierB": PRESTIGE["P_tierB"]},
    {k: ROWS[k] for k in ["controls_only", "+attention"]},
    description=f"Robustness: train excludes launch-era months (release_month <= {LAUNCH_ERA_END}).")

# ── E9: v2 leaky replication (max_hindex in baseline; P_interim) ────────────
experiments["v2_leaky_replication"] = run_experiment(
    "v2_leaky_replication", train_df, test_df, "y_q", {"P_interim": PRESTIGE["P_interim"]},
    {k: ROWS[k] for k in ["controls_only", "+attention"]},
    extra_num=["max_hindex"], allow_flag="v2_leaky_replication",
    description="FLAGGED context row: today-measured max_hindex added to the P_interim baseline (v2-style leak).")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PER-QUARTER UPVOTES-ONLY AUC (2025 test) + descriptive checks
# ─────────────────────────────────────────────────────────────────────────────
per_quarter = []
for q, g in test_df.groupby("release_quarter"):
    per_quarter.append({"quarter": q, "n": int(len(g)), "n_pos": int(g["y_q"].sum()),
                        "median_age_months": float(g["age_months"].median()),
                        "median_citations": float(g["citation_count"].median()),
                        "p90_citations": float(g["citation_count"].quantile(0.9)),
                        "auc_log_upvotes": float(roc_auc_score(g["y_q"], g["log_upvotes"]))})
raw_auc = {"test_2025_log_upvotes": float(roc_auc_score(test_df["y_q"], test_df["log_upvotes"])),
           "test_2025_upvote_rank_within_month": float(roc_auc_score(test_df["y_q"], test_df["upvote_rank_within_month"])),
           "train_le2024_log_upvotes": float(roc_auc_score(train_df["y_q"], train_df["log_upvotes"])),
           "test_2023_backward_log_upvotes": float(roc_auc_score(test_bw["y_q"], test_bw["log_upvotes"])),
           "mature_k12_log_upvotes": float(roc_auc_score(test_df.loc[mature_mask, "y_q_mature"],
                                                         test_df.loc[mature_mask, "log_upvotes"]))}

# ─────────────────────────────────────────────────────────────────────────────
# 7. LEAKAGE AUDIT + EVIDENCE ROWS
# ─────────────────────────────────────────────────────────────────────────────
monthly = (df.groupby("release_month").agg(n=("upvotes", "size"), median_upvotes=("upvotes", "median"),
                                           p75_upvotes=("upvotes", lambda s: float(s.quantile(0.75))))
           .reset_index())
monthly["launch_era"] = monthly["release_month"] <= LAUNCH_ERA_END
sp_by_year = {}
for yr, g in df.groupby("release_year"):
    rho = spearmanr(g["upvotes"], g["age_months"]).statistic
    sp_by_year[str(int(yr))] = {"spearman_upvotes_age_months": float(rho), "n": int(len(g))}

# label-provenance checks (test set)
te = test_df
src_flag = (te["subfield_kw_source"] == "ai_keywords").astype(int)
prov = {
    "subfield_kw": {
        "taxonomy": "single ordered keyword rule set (v2 KEYWORD_RULES) applied to HF ai_keywords, "
                    "fallback title+summary, else Other — identical rules for every paper",
        "n_levels": int(df["subfield_kw"].nunique()),
        "source_shares_all": df["subfield_kw_source"].value_counts(normalize=True).round(4).to_dict(),
        "test_auc_y_given_source_is_ai_keywords": float(roc_auc_score(te["y_q"], src_flag)),
        "test_median_upvotes_by_source": te.groupby("subfield_kw_source")["upvotes"].median().to_dict(),
        "test_pos_rate_by_source": te.groupby("subfield_kw_source")["y_q"].mean().round(4).to_dict(),
    },
    "legacy_subfield": {
        "taxonomy": "MIXED: arXiv-category map for papers whose arXiv metadata was fetched (high-upvote-first "
                    "fetch, ~27% of papers) and keyword rules for the rest",
        "n_levels": int(df["subfield"].nunique()),
        "share_arxiv_fetched_all": float(df["arxiv_fetched_flag"].mean()),
        "test_auc_y_given_arxiv_fetched_flag": float(roc_auc_score(te["y_q"], te["arxiv_fetched_flag"])),
        "test_median_upvotes_by_arxiv_fetched": te.groupby("arxiv_fetched_flag")["upvotes"].median().to_dict(),
        "test_pos_rate_by_arxiv_fetched": te.groupby("arxiv_fetched_flag")["y_q"].mean().round(4).to_dict(),
    },
}
prov_uniform_pass = prov["subfield_kw"]["test_auc_y_given_source_is_ai_keywords"] < 0.55 and \
    prov["subfield_kw"]["source_shares_all"].get("ai_keywords", 0) > 0.85

age_support = {"train_min": float(train_df["age_months"].min()), "train_max": float(train_df["age_months"].max()),
               "test_min": float(test_df["age_months"].min()), "test_max": float(test_df["age_months"].max())}
age_support["overlap"] = bool(age_support["test_max"] > age_support["train_min"])
n_tierb_missing_train = int(train_df["log1p_max_prior_papers_true"].isna().sum())
n_tierb_missing_test = int(test_df["log1p_max_prior_papers_true"].isna().sum())

audit = {
    "split_temporal": {"status": "PASS", "detail": f"train=release_year<={int(train_df['release_year'].max())} "
                       f"(n={len(train_df):,}), test=release_year={int(test_df['release_year'].min())} (n={len(test_df):,}); "
                       "years and quarters disjoint (asserted)."},
    "labels_same_snapshot_caveat": {"status": "DISCLOSED", "detail": "All citation counts are one 2026-06 snapshot; train "
                                    "labels use longer exposure windows than test labels. Within-cohort ranking, the K=12 "
                                    "mature row and the backward-2023 test address this; a true forward simulation would need "
                                    "historical snapshots."},
    "scalers_fit_on_train_only": {"status": "PASS", "detail": "SimpleImputer/StandardScaler/OneHotEncoder inside a Pipeline "
                                  "fitted on train only; test transformed with frozen objects."},
    "hyperparameters_tuned_on_train_only": {"status": "PASS", "detail": "GridSearchCV with GroupKFold(5) grouped by "
                                            "release_month on train; one evaluation on test."},
    "headline_model_prespecified": {"status": "PASS", "detail": f"headline model={HEADLINE_MODEL}, branch={HEADLINE_BRANCH}, "
                                    f"row={HEADLINE_ROW}, CI={HEADLINE_CI}; no test-set-based selection of model or branch."},
    "target_is_label_only": {"status": "PASS", "detail": "citation_count/influential_citations used only to build labels; "
                             "FORBIDDEN set asserted for every feature matrix."},
    "within_group_ranks": {"status": "PASS", "detail": "upvote_rank_within_month and all labels are within-group ranks; "
                           "release_month/quarter never enter as features."},
    "forbidden_columns_absent": {"status": "PASS", "detail": "check_features() raised on any forbidden column; max_hindex only "
                                 "in v2_leaky_replication, legacy `subfield` only in legacy_subfield."},
    "tierB_columns_whitelisted": {"status": "PASS", "detail": "P_tierB uses exactly log1p_max_prior_papers_true + max_years_active; "
                                  "tierB_resolved, *_first_year and raw/winsorised counts excluded; "
                                  f"missing Tier-B values: train {n_tierb_missing_train}, test {n_tierb_missing_test} (median-imputed)."},
    "label_provenance_uniform": {"status": "PASS" if prov_uniform_pass else "FLAG",
                                 "detail": "subfield_kw comes from one rule set for every paper; source flag "
                                 f"(ai_keywords vs fallback) predicts the label with AUC "
                                 f"{prov['subfield_kw']['test_auc_y_given_source_is_ai_keywords']:.3f} in test. The legacy "
                                 f"`subfield` provenance flag (arXiv fetched) had AUC "
                                 f"{prov['legacy_subfield']['test_auc_y_given_arxiv_fetched_flag']:.3f} — that leak is now "
                                 "confined to the flagged legacy_subfield row.",
                                 "evidence": prov},
    "attention_is_cumulative_at_collection": {
        "status": "DISCLOSED",
        "detail": "upvotes are the cumulative HF count at collection (June 2026), not a day-one measurement; "
                  "n_trend_days is 1 for essentially all papers so 'peak' == snapshot. Evidence that late accrual is "
                  "small relative to cross-paper variation: flat monthly medians after the launch era and near-zero "
                  "Spearman(upvotes, age_months) within 2024 and 2025.",
        "n_trend_days_share_eq1": float((df["n_trend_days"] == 1).mean()),
        "spearman_upvotes_age_by_year": sp_by_year,
        "monthly_median_upvotes": monthly.to_dict(orient="records"),
    },
    "age_months_support": {"status": "DISCLOSED", "detail": "age_months is a deterministic function of release date and does not "
                           "overlap across the split; the drop_age robustness row shows the sensitivity.", **age_support},
    "features_available_at_day_one": {
        "age_months": {"available": "yes", "note": "planned exposure to snapshot; deterministic in release date (see age_months_support)"},
        "title_n_words / title_has_colon / abstract_n_chars": {"available": "yes", "note": "from title/abstract at submission"},
        "kw_* flags": {"available": "yes", "note": "keyword flags from abstract"},
        "log_n_authors": {"available": "yes", "note": "log1p(n_authors) from submission metadata"},
        "has_github": {"available": "flag", "note": "GitHub link on HF page at scrape time; may be added later"},
        "subfield_kw": {"available": "yes", "note": "keyword rules on HF ai_keywords/title/summary; uniform taxonomy for all papers"},
        "dow": {"available": "yes", "note": "day-of-week of HF posting"},
        "log1p_max_prior_papers_true / max_years_active": {"available": "yes", "note": "Tier B: papers strictly before anchor date; leakage-free"},
        "max_papercount_cur2026_w99": {"available": "flag", "note": "ANACHRONISTIC 2026 career count; P_interim only"},
        "log_upvotes": {"available": "flag", "note": "cumulative upvotes at collection (2026-06); not day-one"},
        "upvote_rank_within_month": {"available": "flag", "note": "needs the whole month cohort; retrospective, transductive"},
        "log1p_num_comments": {"available": "flag", "note": "cumulative HF comments at collection; ablation only"},
        "max_hindex": {"available": "no", "note": "today-measured; v2_leaky_replication row only"},
        "subfield (legacy)": {"available": "no", "note": "label source keyed on upvote-ordered fetch; legacy_subfield row only"},
    },
    "prestige_branch_tags": BRANCH_TAGS,
    "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "schemes": ["iid", "month_cluster"], "primary": HEADLINE_CI,
                  "note": "paired over test rows (or test release-months), conditional on the trained models; "
                          "ignores training-set variability and model selection."},
}
audit_status_pass = all(v.get("status") in ("PASS", "DISCLOSED") for v in audit.values()
                        if isinstance(v, dict) and "status" in v)

# ─────────────────────────────────────────────────────────────────────────────
# 8. HEADLINE + JSON
# ─────────────────────────────────────────────────────────────────────────────
def row_of(exp, model, branch, row, ev="main"):
    return experiments[exp]["models"][model][ev]["rows"][f"{branch}|{row}"]


def headline_block(model):
    ctrl = row_of("forward_yq", model, HEADLINE_BRANCH, "controls_only")
    att = row_of("forward_yq", model, HEADLINE_BRANCH, "+attention")
    up = row_of("forward_yq", model, HEADLINE_BRANCH, "upvotes_only")
    d = att["delta_vs_controls_only"]
    return {
        "model": model, "branch": HEADLINE_BRANCH, "label": "y_q",
        "controls_only_auc": ctrl["auc"], "controls_only_auc_ci_month": ctrl["ci"]["month_cluster"]["auc"],
        "attention_auc": att["auc"], "attention_auc_ci_month": att["ci"]["month_cluster"]["auc"],
        "upvotes_only_auc": up["auc"], "upvotes_only_auc_ci_month": up["ci"]["month_cluster"]["auc"],
        "delta_auc": d["d_auc"], "delta_auc_ci_month": d["ci"]["month_cluster"]["auc"],
        "delta_auc_ci_iid": d["ci"]["iid"]["auc"],
        "controls_only_pr_auc": ctrl["pr_auc"], "attention_pr_auc": att["pr_auc"], "upvotes_only_pr_auc": up["pr_auc"],
        "delta_pr_auc": d["d_pr_auc"], "delta_pr_auc_ci_month": d["ci"]["month_cluster"]["pr_auc"],
        "controls_only_p_at_k": ctrl["p_at_k"], "attention_p_at_k": att["p_at_k"], "upvotes_only_p_at_k": up["p_at_k"],
        "delta_p_at_k": d["d_p_at_k"], "delta_p_at_k_ci_month": d["ci"]["month_cluster"]["p_at_k"],
        "delta_p_at_100": d["d_p_at_100"], "delta_p_at_100_ci_month": d["ci"]["month_cluster"]["p_at_100"],
        "controls_only_brier": ctrl["brier"], "attention_brier": att["brier"],
        "attention_minus_upvotes_only_auc": att["delta_vs_upvotes_only"]["d_auc"],
        "attention_minus_upvotes_only_auc_ci_month": att["delta_vs_upvotes_only"]["ci"]["month_cluster"]["auc"],
        "k_top_decile": experiments["forward_yq"]["k_top_decile"],
    }


headline = {m: headline_block(m) for m in MODELS}
hl = headline[HEADLINE_MODEL]
hgb_fits = [v for k, v in TUNING_LOG.items() if "|hgb|" in k]
hgb_l2_at_max_share = float(np.mean([v["best_params"].get("l2_regularization") == max(HGB_GRID["mdl__l2_regularization"])
                                     for v in hgb_fits])) if hgb_fits else None
interp = (
    f"Pre-specified headline ({HEADLINE_MODEL}, {HEADLINE_BRANCH} leakage-free baseline, within-quarter top-decile label): "
    f"adding cumulative HF attention raises test AUC from {hl['controls_only_auc']:.3f} to {hl['attention_auc']:.3f} "
    f"(dAUC {hl['delta_auc']:+.3f}, month-clustered 95% CI [{hl['delta_auc_ci_month'][0]:+.3f}, {hl['delta_auc_ci_month'][1]:+.3f}]); "
    f"PR-AUC {hl['controls_only_pr_auc']:.3f} -> {hl['attention_pr_auc']:.3f}; precision@top-decile "
    f"{hl['controls_only_p_at_k']:.3f} -> {hl['attention_p_at_k']:.3f}. log_upvotes alone gives AUC {hl['upvotes_only_auc']:.3f}; "
    f"controls add {hl['attention_minus_upvotes_only_auc']:+.3f} AUC on top of upvotes "
    f"(CI [{hl['attention_minus_upvotes_only_auc_ci_month'][0]:+.3f}, {hl['attention_minus_upvotes_only_auc_ci_month'][1]:+.3f}]). "
    f"HGB: dAUC {headline['hgb']['delta_auc']:+.3f} [{headline['hgb']['delta_auc_ci_month'][0]:+.3f}, "
    f"{headline['hgb']['delta_auc_ci_month'][1]:+.3f}]. Predictive, not causal; upvotes are cumulative at collection."
)

output = {
    "meta": {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/26_prediction_v3.py", "spec": "pre-specified: logistic model, leakage free prestige baseline, top decile within release quarter label",
        "python_version": platform.python_version(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "seed": SEED, "n_boot": N_BOOT, "n_all": n_all,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)), "n_mature_k12": n_mature,
        "n_train_backward": int(len(train_bw)), "n_test_backward": int(len(test_bw)),
        "n_train_no_launch": int(len(train_nl)), "launch_era_end": LAUNCH_ERA_END,
        "k12_months": K12_MONTHS, "runtime_seconds": None,
        "headline_model": HEADLINE_MODEL, "headline_branch": HEADLINE_BRANCH, "headline_row": HEADLINE_ROW,
        "headline_ci": HEADLINE_CI,
        "hgb_grid": HGB_GRID, "logistic_grid": LOGIT_GRID, "hgb_share_of_fits_selecting_max_l2": hgb_l2_at_max_share,
        "hgb_fixed": {"learning_rate": 0.05, "max_iter": 1000, "early_stopping": True, "validation_fraction": 0.2,
                      "n_iter_no_change": 30},
    },
    "headline": {"primary": hl, "by_model": headline, "interpretation": interp,
                 "audit_verdict": "PASS/DISCLOSED — no outcome leakage in headline rows" if audit_status_pass else "REVIEW"},
    "experiments": experiments,
    "raw_attention_auc": raw_auc,
    "per_quarter_upvotes_only_2025": per_quarter,
    "audit": audit,
    "tuning": TUNING_LOG,
}


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


output["meta"]["runtime_seconds"] = round(time.time() - T0, 1)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(json_safe(output), f, indent=1)
print(f"\nWrote {OUT_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SCORES CSV
# ─────────────────────────────────────────────────────────────────────────────
base_cols = ["arxiv_id_clean", "release_month", "release_quarter", "release_year", "y_q", "y_m", "y_inf",
             "mature_k12", "log_upvotes", "upvote_rank_within_month", "age_months"]
fw = test_df[base_cols + ["y_q_mature"]].copy()
fw.loc[fw["mature_k12"] == 0, "y_q_mature"] = np.nan
fw["split"] = "forward_test_2025"
for nm in ["forward_yq", "forward_ym", "forward_yinf", "drop_age", "legacy_subfield", "no_launch_months",
           "v2_leaky_replication"]:
    fw = fw.merge(SCORE_STORE[nm], on="arxiv_id_clean", how="left")
bw = test_bw[base_cols].copy()
bw["y_q_mature"] = np.nan
bw["split"] = "backward_test_2023"
for nm in ["backward_yq", "backward_ym"]:
    bw = bw.merge(SCORE_STORE[nm], on="arxiv_id_clean", how="left")
scores_out = pd.concat([fw, bw], ignore_index=True, sort=False)
scores_out.to_csv(OUT_SCORES, index=False)
print(f"Wrote {OUT_SCORES}  ({scores_out.shape[0]:,} rows x {scores_out.shape[1]} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TABLES (markdown, paste-ready) + NOTE
# ─────────────────────────────────────────────────────────────────────────────
def fci(ci, nd=3, sign=False):
    fmt = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


MODEL_LABEL = {"logistic": "Logistic (headline)", "hgb": "HGB (early-stopped)"}
ROW_LABEL = {"controls_only": "controls only", "+attention": "+ attention (log upvotes, month rank)",
             "upvotes_only": "log upvotes only", "+attention+comments": "+ attention + comments"}

lines = []
lines.append("# D3 v3 — paste-ready tables (generated by scripts/26_prediction_v3.py)\n")
lines.append(f"Generated {output['meta']['generated']}; n_train={len(train_df):,}, n_test={len(test_df):,} "
             f"(top-decile k={experiments['forward_yq']['k_top_decile']}); bootstrap {N_BOOT} reps, seed {SEED}; "
             f"CIs in brackets are release-month-clustered unless stated.\n")

# Table 1: headline ladder P_tierB
lines.append("## Table 1 — Headline ladder (P_tierB leakage-free baseline; label = top decile within release quarter; test 2025)\n")
lines.append("| Model | Row | AUC [95% CI] | PR-AUC [95% CI] | P@top-decile [95% CI] | Brier | ΔAUC vs controls [95% CI] | ΔPR-AUC [95% CI] | ΔP@top-decile [95% CI] | ΔP@100 (noisy) |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for r in ["controls_only", "+attention", "upvotes_only", "+attention+comments"]:
        e = row_of("forward_yq", m, "P_tierB", r)
        d = e.get("delta_vs_controls_only")
        lines.append(
            f"| {MODEL_LABEL[m]} | {ROW_LABEL[r]} | {e['auc']:.3f} {fci(e['ci']['month_cluster']['auc'])} | "
            f"{e['pr_auc']:.3f} {fci(e['ci']['month_cluster']['pr_auc'])} | {e['p_at_k']:.3f} {fci(e['ci']['month_cluster']['p_at_k'])} | "
            f"{e['brier']:.4f} | "
            + (f"{d['d_auc']:+.3f} {fci(d['ci']['month_cluster']['auc'], sign=True)} | {d['d_pr_auc']:+.3f} "
               f"{fci(d['ci']['month_cluster']['pr_auc'], sign=True)} | {d['d_p_at_k']:+.3f} "
               f"{fci(d['ci']['month_cluster']['p_at_k'], sign=True)} | {d['d_p_at_100']:+.2f} {fci(d['ci']['month_cluster']['p_at_100'], 2, sign=True)} |"
               if d else "— | — | — | — |"))
lines.append("")
for m in MODELS:
    e = row_of("forward_yq", m, "P_tierB", "+attention")["delta_vs_upvotes_only"]
    lines.append(f"- {MODEL_LABEL[m]}: (+attention) − (upvotes only) ΔAUC = {e['d_auc']:+.3f} "
                 f"{fci(e['ci']['month_cluster']['auc'], sign=True)}; ΔPR-AUC = {e['d_pr_auc']:+.3f} "
                 f"{fci(e['ci']['month_cluster']['pr_auc'], sign=True)} — what the controls add given upvotes.")
lines.append("")

# Table 2: prestige-branch bracket
lines.append("## Table 2 — Prestige-branch bracket (ΔAUC of +attention over controls-only; test 2025; y_q)\n")
lines.append("| Branch | Model | Controls-only AUC | +Attention AUC | ΔAUC | 95% CI (month-cluster) | 95% CI (i.i.d.) | ΔPR-AUC | ΔP@top-decile |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for b in PRESTIGE:
    for m in MODELS:
        c = row_of("forward_yq", m, b, "controls_only"); a = row_of("forward_yq", m, b, "+attention")
        d = a["delta_vs_controls_only"]
        lines.append(f"| {b} | {MODEL_LABEL[m]} | {c['auc']:.3f} | {a['auc']:.3f} | {d['d_auc']:+.3f} | "
                     f"{fci(d['ci']['month_cluster']['auc'], sign=True)} | {fci(d['ci']['iid']['auc'], sign=True)} | "
                     f"{d['d_pr_auc']:+.3f} | {d['d_p_at_k']:+.3f} |")
lines.append("")

# Table 3: robustness rows
lines.append("## Table 3 — Robustness rows (P_tierB unless stated; ΔAUC = +attention − controls-only)\n")
lines.append("| Row | n_test | Model | Controls-only AUC | +Attention AUC | ΔAUC [95% CI month-cluster] | Upvotes-only AUC | ΔPR-AUC |")
lines.append("|---|---|---|---|---|---|---|---|")
rob_rows = [
    ("forward_yq", "main", "Primary: within-quarter label, test 2025"),
    ("forward_yq", "mature_k12", f"Mature K=12 subset (age ≥ {K12_MONTHS:.0f} mo; label re-ranked within quarter)"),
    ("forward_ym", "main", "Within-month label"),
    ("forward_yinf", "main", "Top-decile influential citations (within quarter)"),
    ("backward_yq", "main", "Backward: train 2024–25 → test 2023 (within-quarter label)"),
    ("backward_ym", "main", "Backward: train 2024–25 → test 2023 (within-month label)"),
    ("drop_age", "main", "Drop age_months from controls"),
    ("no_launch_months", "main", f"Train excludes launch-era months (≤ {LAUNCH_ERA_END})"),
    ("legacy_subfield", "main", "FLAGGED: legacy mixed-taxonomy `subfield` instead of subfield_kw"),
    ("v2_leaky_replication", "main", "FLAGGED: v2 leaky (max_hindex + P_interim baseline)"),
]
for exp, ev, lab in rob_rows:
    branch = "P_interim" if exp == "v2_leaky_replication" else "P_tierB"
    for m in MODELS:
        blk = experiments[exp]["models"][m][ev]
        c = blk["rows"][f"{branch}|controls_only"]; a = blk["rows"][f"{branch}|+attention"]
        d = a["delta_vs_controls_only"]
        up = blk["rows"].get(f"{branch}|upvotes_only")
        lines.append(f"| {lab} | {blk['n']:,} | {MODEL_LABEL[m]} | {c['auc']:.3f} | {a['auc']:.3f} | "
                     f"{d['d_auc']:+.3f} {fci(d['ci']['month_cluster']['auc'], sign=True)} | "
                     f"{(f'{up['auc']:.3f}' if up else '—')} | {d['d_pr_auc']:+.3f} |")
lines.append("")

# Table 4: per-quarter upvotes-only AUC
lines.append("## Table 4 — Per-quarter AUC of log upvotes alone (test 2025; within-quarter label)\n")
lines.append("| Quarter | n | positives | median age (mo) | median citations | p90 citations | AUC(log upvotes) |")
lines.append("|---|---|---|---|---|---|---|")
for q in per_quarter:
    lines.append(f"| {q['quarter']} | {q['n']:,} | {q['n_pos']} | {q['median_age_months']:.1f} | {q['median_citations']:.0f} | "
                 f"{q['p90_citations']:.0f} | {q['auc_log_upvotes']:.3f} |")
lines.append("")

# Table 5: attention-accrual evidence
lines.append("## Table 5 — Are upvotes 'early'? (cumulative at collection; evidence of low late accrual)\n")
lines.append("| Year | n | Spearman(upvotes, age_months) |")
lines.append("|---|---|---|")
for yr, v in sp_by_year.items():
    lines.append(f"| {yr} | {v['n']:,} | {v['spearman_upvotes_age_months']:+.3f} |")
lines.append("")
lines.append("| Release month | n | median upvotes | launch era |")
lines.append("|---|---|---|---|")
for rec in monthly.to_dict(orient="records"):
    lines.append(f"| {rec['release_month']} | {rec['n']} | {rec['median_upvotes']:.0f} | {'yes' if rec['launch_era'] else ''} |")
lines.append("")

# Table 6: provenance check
lines.append("## Table 6 — Label-provenance check (test 2025)\n")
lines.append("| Taxonomy | provenance flag | AUC(y | flag) | median upvotes (flag=1 / 0) | P(y=1) (flag=1 / 0) |")
lines.append("|---|---|---|---|---|")
pk = prov["subfield_kw"]; pl = prov["legacy_subfield"]
lines.append(f"| subfield_kw (uniform) | source = ai_keywords | {pk['test_auc_y_given_source_is_ai_keywords']:.3f} | "
             f"{pk['test_median_upvotes_by_source'].get('ai_keywords', float('nan')):.0f} / "
             f"{pk['test_median_upvotes_by_source'].get('title_summary', float('nan')):.0f} (title_summary) | "
             f"{pk['test_pos_rate_by_source'].get('ai_keywords', float('nan')):.3f} / {pk['test_pos_rate_by_source'].get('title_summary', float('nan')):.3f} |")
lines.append(f"| legacy subfield (mixed) | arXiv metadata fetched | {pl['test_auc_y_given_arxiv_fetched_flag']:.3f} | "
             f"{pl['test_median_upvotes_by_arxiv_fetched'].get(1, float('nan')):.0f} / {pl['test_median_upvotes_by_arxiv_fetched'].get(0, float('nan')):.0f} | "
             f"{pl['test_pos_rate_by_arxiv_fetched'].get(1, float('nan')):.3f} / {pl['test_pos_rate_by_arxiv_fetched'].get(0, float('nan')):.3f} |")
lines.append("")

with open(OUT_TABLES, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_TABLES}")

# NOTE
hh = headline["hgb"]
note = f"""# D3 v3 — Audited incremental prediction (PROJECT HEADLINE), audit-response run

Generated: {output['meta']['generated']} | script: scripts/26_prediction_v3.py | seed={SEED} | runtime {output['meta']['runtime_seconds']:.0f}s
Python {output['meta']['python_version']}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}

## Headline (pre-specified: logistic, P_tierB leakage-free baseline, within-quarter top-decile label, month-clustered CI)

- Controls-only AUC **{hl['controls_only_auc']:.4f}** {fci(hl['controls_only_auc_ci_month'], 4)} → +attention AUC **{hl['attention_auc']:.4f}** {fci(hl['attention_auc_ci_month'], 4)}
- **ΔAUC = {hl['delta_auc']:+.4f}**, 95% CI month-clustered {fci(hl['delta_auc_ci_month'], 4, sign=True)} (i.i.d. {fci(hl['delta_auc_ci_iid'], 4, sign=True)})
- PR-AUC {hl['controls_only_pr_auc']:.4f} → {hl['attention_pr_auc']:.4f} (**ΔPR-AUC {hl['delta_pr_auc']:+.4f}** {fci(hl['delta_pr_auc_ci_month'], 4, sign=True)})
- Precision@top-decile (k={hl['k_top_decile']}) {hl['controls_only_p_at_k']:.4f} → {hl['attention_p_at_k']:.4f} (**Δ {hl['delta_p_at_k']:+.4f}** {fci(hl['delta_p_at_k_ci_month'], 4, sign=True)}); ΔP@100 {hl['delta_p_at_100']:+.2f} {fci(hl['delta_p_at_100_ci_month'], 2, sign=True)} (noisy: 100-draw binomial)
- Brier {hl['controls_only_brier']:.4f} → {hl['attention_brier']:.4f}
- **log upvotes alone: AUC {hl['upvotes_only_auc']:.4f}** {fci(hl['upvotes_only_auc_ci_month'], 4)}, PR-AUC {hl['upvotes_only_pr_auc']:.4f}; controls add {hl['attention_minus_upvotes_only_auc']:+.4f} AUC on top of upvotes {fci(hl['attention_minus_upvotes_only_auc_ci_month'], 4, sign=True)}
- HGB (equal prominence): controls-only {hh['controls_only_auc']:.4f} → +attention {hh['attention_auc']:.4f}, ΔAUC {hh['delta_auc']:+.4f} {fci(hh['delta_auc_ci_month'], 4, sign=True)}; ΔPR-AUC {hh['delta_pr_auc']:+.4f}; ΔP@top-decile {hh['delta_p_at_k']:+.4f}; upvotes-only {hh['upvotes_only_auc']:.4f}
- n_train={len(train_df):,} (release_year ≤ {int(train_df['release_year'].max())}), n_test={len(test_df):,} (2025), base rate test {experiments['forward_yq']['base_rate_test']:.3f}

{interp}

## What changed vs v1 (scripts/26_prediction.py) and why
1. `subfield` → `subfield_kw` (uniform keyword taxonomy). Legacy label source was keyed on an upvote-ordered arXiv fetch and predicted the label with AUC {pl['test_auc_y_given_arxiv_fetched_flag']:.3f} on its own (Table 6); it inflated the controls-only baseline. Legacy kept only as a flagged robustness row.
2. P_tierB (leakage-free prior-paper count + years active) is the definitive baseline; P_interim/P_none are brackets. Bool/first-year/raw Tier-B columns are not features (v1 hook would have ingested them).
3. Headline model pre-specified as logistic; HGB re-tuned with early stopping (v1 HGB was under-tuned: +attention AUC below upvotes-alone).
4. Added: upvotes-only row, PR-AUC, precision@top-decile, Brier, month-clustered CIs, comments ablation, within-month + influential labels, backward-2023 test, drop-age, no-launch-months, per-quarter AUC, provenance and accrual evidence.
5. No hard-coded n / dates; regressors (ridge/HGB-reg on q_i) dropped in v3 (not part of the audited claim).

Model notes: the HGB "log upvotes only" row (AUC {hh['upvotes_only_auc']:.3f}) sits below the model-free AUC of log upvotes
({raw_auc['test_2025_log_upvotes']:.3f}; = the logistic upvotes-only row) because a boosted one-feature model is not monotone in its input;
the model-free number is the one to quote for "upvotes alone". HGB tuning selected the largest l2_regularization in the grid
({max(HGB_GRID['mdl__l2_regularization']):.0f}) in {hgb_l2_at_max_share:.0%} of fits — CV-AUC differences across the upper l2 values are small
and the early-stopped HGB does not beat the logistic here; both are reported.

## Robustness (see prediction_v3_tables.md Table 3)
"""
for exp, ev, lab in rob_rows:
    branch = "P_interim" if exp == "v2_leaky_replication" else "P_tierB"
    parts = []
    for m in MODELS:
        blk = experiments[exp]["models"][m][ev]
        d = blk["rows"][f"{branch}|+attention"]["delta_vs_controls_only"]
        parts.append(f"{m} ΔAUC {d['d_auc']:+.3f} {fci(d['ci']['month_cluster']['auc'], 3, sign=True)}")
    n_ev = experiments[exp]["models"][MODELS[0]][ev]["n"]
    note += f"- {lab} (n={n_ev:,}): " + "; ".join(parts) + "\n"

note += f"""
Per-quarter AUC of log upvotes alone in 2025: """ + ", ".join(f"{q['quarter']} {q['auc_log_upvotes']:.3f}" for q in per_quarter) + f""".
Raw AUC of log upvotes: test-2025 {raw_auc['test_2025_log_upvotes']:.3f}; backward test-2023 {raw_auc['test_2023_backward_log_upvotes']:.3f}; mature K=12 {raw_auc['mature_k12_log_upvotes']:.3f}.

## Leakage audit (verdict: {output['headline']['audit_verdict']})
""" + "\n".join(f"- {k}: **{v['status']}** — {v['detail']}" for k, v in audit.items() if isinstance(v, dict) and "status" in v) + f"""

Attention accrual evidence: Spearman(upvotes, age_months) within year — """ + ", ".join(
    f"{yr}: {v['spearman_upvotes_age_months']:+.3f}" for yr, v in sp_by_year.items()) + f""" (2023 reflects the launch-era ramp; months ≤ {LAUNCH_ERA_END} have medians """ + ", ".join(
    f"{r['release_month']}={r['median_upvotes']:.0f}" for r in monthly[monthly.launch_era].to_dict(orient="records") if r["n"] >= 50) + f"""). Monthly medians afterwards range {monthly.loc[~monthly.launch_era, 'median_upvotes'].min():.0f}–{monthly.loc[~monthly.launch_era, 'median_upvotes'].max():.0f}.

## CAN / CANNOT
**CAN:** say that cumulative HF attention (upvotes at collection) adds honest out-of-sample incremental predictive value for within-cohort citation rank, forward in time and backward onto a mature cohort, over a leakage-checked controls-only baseline — and that a single column (log upvotes) carries essentially all of it.
**CANNOT:** claim causation; claim day-one/early attention (no vote timestamps; cumulative counts); generalise beyond HF Daily Papers; treat the paired bootstrap as covering training-set variability.

## Files
- results/prediction_v3.json — all metrics, CIs, deltas, tuning, audit
- results/prediction_v3_tables.md — paste-ready tables
- results/prediction_v3_scores.csv — per-test-row labels + predicted probabilities (columns `<experiment>__<model>__<branch>__<row>`; forward-2025 and backward-2023 rows stacked, `split` column)
"""
with open(OUT_NOTE, "w") as f:
    f.write(note)
print(f"Wrote {OUT_NOTE}")
print(f"\nTotal runtime {time.time() - T0:.0f}s. Done.")
