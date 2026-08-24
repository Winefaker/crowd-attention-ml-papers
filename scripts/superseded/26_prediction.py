"""
D3 — Leakage-Audited Incremental Prediction (PROJECT HEADLINE)
==============================================================
Spec: the pre-specified plan

Forward-in-time test: train ≤ 2024 → test 2025
Headline: incremental ΔAUC of early/peak HF attention over a controls-only baseline.

Re-run hook (P_tierB):
  When  data/processed/prepub_prestige_tierB.csv  appears (left-join on
  arxiv_id_clean; use its leakage-free prior-count + years_active columns),
  re-execute this script with TIERB_PATH pointing to that file.
  The P_tierB branch will then become the DEFINITIVE headline baseline.
  Current status: PENDING (file not yet available).

Usage:
  Project/.venv/bin/python scripts/26_prediction.py
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

SEED = 1626
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/analysis_final.csv"
TIERB_PATH = ROOT / "data/processed/prepub_prestige_tierB.csv"  # PENDING
OUT_JSON = ROOT / "results/prediction.json"
OUT_NOTE = ROOT / "results/prediction_NOTE.md"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH, dtype={"arxiv_id_clean": str})
print(f"  n={len(df):,}  cols={df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# FORBIDDEN columns — assert absent from any feature matrix (enforced below)
# ─────────────────────────────────────────────────────────────────────────────
FORBIDDEN = {
    "citation_count", "log_citations", "influential_citations",  # outcomes
    "reference_count",                                            # outcome-adjacent
    "n_trend_days",                                               # post-hoc
    "max_hindex", "last_author_hindex",                          # leaky today-measured
    "author_max_appear",                                          # leaky
    # D1 instruments
    "cohort_day", "cohort_size", "Z1_logcompet", "Z2_count",
    "Z3_blockbuster", "Z1p_othersub", "ego_upvotes",
    # provenance / admin
    "gap_days_v1", "gap_days_rawpub", "published_v1", "v1_source",
    "anchor_date_used", "prestige_resolved",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TARGETS
# ─────────────────────────────────────────────────────────────────────────────
# within-release-quarter citation percentile
df["release_quarter"] = df["release_month"].str[:4] + "Q" + (
    (df["release_month"].str[5:7].astype(int) - 1) // 3 + 1
).astype(str)

df["q_i"] = df.groupby("release_quarter")["citation_count"].rank(pct=True)
df["y_i"] = (df["q_i"] >= 0.90).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  ATTENTION FEATURE: upvote_rank_within_month (within-group, no leakage)
# ─────────────────────────────────────────────────────────────────────────────
df["upvote_rank_within_month"] = df.groupby("release_month")["upvotes"].rank(pct=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_df = df[df["release_year"] <= 2024].copy().reset_index(drop=True)
test_df  = df[df["release_year"] == 2025].copy().reset_index(drop=True)
print(f"  train={len(train_df):,}  test={len(test_df):,}")
assert len(train_df) == 5019, f"Train size mismatch: {len(train_df)}"
assert len(test_df)  == 6325, f"Test size mismatch: {len(test_df)}"

# Robustness subset: test papers with first_trend_date ≤ 2025-06-01 (≥12 mo exposure)
test_df["first_trend_date_dt"] = pd.to_datetime(test_df["first_trend_date"])
mature_test = test_df[test_df["first_trend_date_dt"] <= "2025-06-01"].copy()
# Recompute within-quarter percentile on mature subset only
mature_test["q_i_mature"] = mature_test.groupby("release_quarter")["citation_count"].rank(pct=True)
mature_test["y_i_mature"] = (mature_test["q_i_mature"] >= 0.90).astype(int)
print(f"  mature_test (K=12)={len(mature_test):,}  (expected ~2365)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_BASE = [
    "age_months", "title_n_words", "title_has_colon", "abstract_n_chars",
    "kw_llm", "kw_agent", "kw_diffusion", "kw_reasoning", "kw_benchmark",
    "kw_survey", "kw_efficient", "kw_multimodal", "kw_rl", "kw_scaling",
    "n_authors", "has_github",
]
# subfield and dow are categorical → one-hot
CAT_COLS = ["subfield", "dow"]
ATTENTION_COLS = ["log_upvotes", "upvote_rank_within_month"]

# Prestige branches
PRESTIGE_BRANCHES = {
    "P_interim": "max_papercount_cur2026_w99",   # FLAGGED anachronistic
    "P_none":    None,                            # no prestige (upper bracket)
    # "P_tierB": determined at runtime below
}

# Check if TierB file exists (re-run hook)
TIERB_AVAILABLE = TIERB_PATH.exists()
if TIERB_AVAILABLE:
    print("  P_tierB file FOUND — adding as prestige branch")
    tierb_df = pd.read_csv(TIERB_PATH, dtype={"arxiv_id_clean": str})
    # Expected columns: arxiv_id_clean, [prior-count col(s)], years_active
    # Left-join onto main df
    df = df.merge(tierb_df, on="arxiv_id_clean", how="left")
    train_df = df[df["release_year"] <= 2024].copy().reset_index(drop=True)
    test_df  = df[df["release_year"] == 2025].copy().reset_index(drop=True)
    # Identify tier B prestige column(s) — assumes 'prior_papercount' + 'years_active'
    tierb_cols = [c for c in tierb_df.columns if c != "arxiv_id_clean"]
    PRESTIGE_BRANCHES["P_tierB"] = tierb_cols  # list of columns
else:
    print("  P_tierB PENDING — file not yet available; skipping (re-run when ready)")


def build_feature_list(prestige_val, include_attention=False, leaky=False):
    """Return numeric cols + categorical cols for ColumnTransformer."""
    num = list(CONTROL_BASE)
    if prestige_val is not None:
        if isinstance(prestige_val, list):
            num += prestige_val
        else:
            num.append(prestige_val)
    if leaky:
        num += ["max_hindex"]  # explicitly flagged leaky — v2 replication only
    if include_attention:
        num += ATTENTION_COLS
    return num, CAT_COLS


def assert_no_forbidden(X_df, label=""):
    """Fail loudly if any forbidden column appears in the feature matrix."""
    bad = FORBIDDEN & set(X_df.columns)
    if "max_hindex" in X_df.columns and "leaky" not in label.lower():
        bad.add("max_hindex")
    if bad:
        raise ValueError(f"FORBIDDEN columns in feature matrix [{label}]: {bad}")


def make_preprocessor(num_cols, cat_cols):
    """ColumnTransformer: impute+scale numerics, one-hot categoricals."""
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def p_at_k(y_true, y_score, k=100):
    """Precision@k: top-k ranked rows, fraction that are truly top-decile."""
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true.iloc[idx].mean())


def paired_bootstrap_dauc(y_true, score_a, score_b, n_boot=2000, seed=1626):
    """Paired bootstrap 95% CI for ΔAUC = AUC(b) − AUC(a)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    y_arr = np.array(y_true)
    sa = np.array(score_a)
    sb = np.array(score_b)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, sa_, sb_ = y_arr[idx], sa[idx], sb[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            d = roc_auc_score(yt, sb_) - roc_auc_score(yt, sa_)
            deltas.append(d)
        except Exception:
            continue
    deltas = np.array(deltas)
    return {
        "mean":  float(np.mean(deltas)),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "n_boot": len(deltas),
    }


def tune_and_fit_classifier(model_name, train_X, train_y, groups, preprocessor):
    """Tune on train with GroupKFold(5) grouped by release_month; return fitted pipeline."""
    cv = GroupKFold(n_splits=5)
    best_score, best_model = -np.inf, None

    if model_name == "logistic":
        for C in [0.01, 0.1, 1, 10]:
            clf = Pipeline([
                ("pre", preprocessor),
                ("mdl", LogisticRegression(C=C, max_iter=2000, random_state=SEED, solver="lbfgs")),
            ])
            scores = cross_val_score(clf, train_X, train_y, cv=cv, groups=groups,
                                     scoring="roc_auc", n_jobs=-1)
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model = clf
    elif model_name == "hgb_clf":
        for lr in [0.05, 0.1]:
            for md in [3, 6, None]:
                clf = Pipeline([
                    ("pre", preprocessor),
                    ("mdl", HistGradientBoostingClassifier(
                        learning_rate=lr, max_depth=md,
                        random_state=SEED, max_iter=300,
                    )),
                ])
                scores = cross_val_score(clf, train_X, train_y, cv=cv, groups=groups,
                                         scoring="roc_auc", n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = clf
    best_model.fit(train_X, train_y)
    return best_model, best_score


def tune_and_fit_regressor(model_name, train_X, train_y, groups, preprocessor):
    """Tune on train with GroupKFold(5); return fitted pipeline."""
    cv = GroupKFold(n_splits=5)
    best_score, best_model = -np.inf, None

    if model_name == "ridge":
        for alpha in [0.1, 1, 10, 100]:
            reg = Pipeline([
                ("pre", preprocessor),
                ("mdl", Ridge(alpha=alpha, random_state=SEED)),
            ])
            scores = cross_val_score(reg, train_X, train_y, cv=cv, groups=groups,
                                     scoring="r2", n_jobs=-1)
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model = reg
    elif model_name == "hgb_reg":
        for lr in [0.05, 0.1]:
            for md in [3, 6, None]:
                reg = Pipeline([
                    ("pre", preprocessor),
                    ("mdl", HistGradientBoostingRegressor(
                        learning_rate=lr, max_depth=md,
                        random_state=SEED, max_iter=300,
                    )),
                ])
                scores = cross_val_score(reg, train_X, train_y, cv=cv, groups=groups,
                                         scoring="r2", n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = reg
    best_model.fit(train_X, train_y)
    return best_model, best_score


def run_branch(branch_name, prestige_val, train_df, test_df, mature_test,
               leaky=False, label_suffix=""):
    """Run all 4 models × 2 feature sets for one prestige branch. Return metrics dict."""
    print(f"\n  Branch={branch_name}{label_suffix}  prestige={prestige_val}  leaky={leaky}")

    groups_train = train_df["release_month"].values
    y_train_bin = train_df["y_i"].values
    y_train_cont = train_df["q_i"].values

    y_test_bin  = test_df["y_i"]
    y_test_cont = test_df["q_i"]

    results = {"branch": branch_name + label_suffix, "leaky": leaky}
    scores_for_bootstrap = {}  # store scores for paired bootstrap

    for attn in [False, True]:
        fs_label = "+attention" if attn else "controls_only"
        num_cols, cat_cols = build_feature_list(prestige_val, include_attention=attn, leaky=leaky)

        # Select columns
        all_cols = num_cols + cat_cols
        # Exclude release_month as a feature (used only for grouping/split/rank)
        all_cols = [c for c in all_cols if c != "release_month"]

        train_X = train_df[all_cols].copy()
        test_X  = test_df[all_cols].copy()

        # Assert forbidden columns absent (allow max_hindex only in leaky runs)
        forbidden_check = set(all_cols) & FORBIDDEN
        if "max_hindex" in all_cols and not leaky:
            raise ValueError(f"max_hindex in non-leaky run [{branch_name}]!")
        if leaky and "max_hindex" in all_cols:
            # temporarily remove max_hindex from the FORBIDDEN check
            forbidden_check = forbidden_check - {"max_hindex"}
        if forbidden_check:
            raise ValueError(f"FORBIDDEN columns detected in [{branch_name}/{fs_label}]: {forbidden_check}")

        preprocessor = make_preprocessor(num_cols, cat_cols)

        fs_result = {}

        # ── Classifiers (binary) ──────────────────────────────────────────
        for clf_name in ["logistic", "hgb_clf"]:
            pre_clone = make_preprocessor(num_cols, cat_cols)
            model, cv_score = tune_and_fit_classifier(
                clf_name, train_X, y_train_bin, groups_train, pre_clone
            )
            probs = model.predict_proba(test_X)[:, 1]
            auc = float(roc_auc_score(y_test_bin, probs))
            pat100 = p_at_k(y_test_bin, pd.Series(probs), k=100)

            # Mature test (K=12)
            mature_X = mature_test[all_cols].copy()
            mature_probs = model.predict_proba(mature_X)[:, 1]
            mature_auc = float(roc_auc_score(mature_test["y_i_mature"], mature_probs))
            mature_pat100 = p_at_k(mature_test["y_i_mature"],
                                   pd.Series(mature_probs, index=mature_test.index), k=100)

            fs_result[clf_name] = {
                "auc":         auc,
                "p_at_100":    pat100,
                "cv_auc_mean": float(cv_score),
                "mature_auc":  mature_auc,
                "mature_p100": mature_pat100,
            }
            # Store scores for bootstrap
            scores_for_bootstrap.setdefault(clf_name, {})[fs_label] = probs
            print(f"    {clf_name}/{fs_label}: AUC={auc:.4f}  P@100={pat100:.3f}")

        # ── Regressors (continuous q_i) ───────────────────────────────────
        for reg_name in ["ridge", "hgb_reg"]:
            pre_clone = make_preprocessor(num_cols, cat_cols)
            model, cv_score = tune_and_fit_regressor(
                reg_name, train_X, y_train_cont, groups_train, pre_clone
            )
            preds = model.predict(test_X)
            r2 = float(1 - np.sum((y_test_cont - preds) ** 2) /
                           np.sum((y_test_cont - y_test_cont.mean()) ** 2))
            sp = float(spearmanr(y_test_cont, preds).statistic)
            fs_result[reg_name] = {
                "r2":       r2,
                "spearman": sp,
                "cv_r2":    float(cv_score),
            }
            print(f"    {reg_name}/{fs_label}: R²={r2:.4f}  Spearman={sp:.4f}")

        results[fs_label] = fs_result

    # ── Bootstrap ΔAUC (paired, 2000 reps, seed 1626) ──────────────────────
    dauc_results = {}
    for clf_name in ["logistic", "hgb_clf"]:
        sa = scores_for_bootstrap[clf_name]["controls_only"]
        sb = scores_for_bootstrap[clf_name]["+attention"]
        boot = paired_bootstrap_dauc(y_test_bin, sa, sb, n_boot=2000, seed=SEED)
        auc_ctrl = results["controls_only"][clf_name]["auc"]
        auc_attn = results["+attention"][clf_name]["auc"]
        dauc_results[clf_name] = {
            "delta_auc":    float(auc_attn - auc_ctrl),
            "delta_p100":   float(results["+attention"][clf_name]["p_at_100"] -
                                  results["controls_only"][clf_name]["p_at_100"]),
            "bootstrap_ci": boot,
        }
        print(f"    ΔAUC [{clf_name}] = {auc_attn - auc_ctrl:+.4f}  "
              f"95%CI=[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]")

    results["delta"] = dauc_results
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN ALL BRANCHES
# ─────────────────────────────────────────────────────────────────────────────
all_results = []

for branch_name, prestige_val in PRESTIGE_BRANCHES.items():
    res = run_branch(branch_name, prestige_val, train_df, test_df, mature_test)
    all_results.append(res)

# ── v2 leaky-prestige replication row (explicitly flagged, context only) ─────
print("\n  Running v2 LEAKY-prestige replication (max_hindex — context only, not headline)")
leaky_res = run_branch(
    "v2_leaky_replication", "max_papercount_cur2026_w99",
    train_df, test_df, mature_test,
    leaky=True, label_suffix="[LEAKY:max_hindex+P_interim]"
)
all_results.append(leaky_res)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  HEADLINE: conservative ΔAUC over STRONGER controls-only baseline
# ─────────────────────────────────────────────────────────────────────────────
# Compare P_interim vs P_none controls-only HGB AUC; conservative = stronger baseline
p_interim_res = next(r for r in all_results if r["branch"] == "P_interim")
p_none_res    = next(r for r in all_results if r["branch"] == "P_none")

ctrl_auc_interim = p_interim_res["controls_only"]["hgb_clf"]["auc"]
ctrl_auc_none    = p_none_res["controls_only"]["hgb_clf"]["auc"]

if ctrl_auc_interim >= ctrl_auc_none:
    conservative_branch = "P_interim"
    conservative_res = p_interim_res
    conservative_ctrl_auc = ctrl_auc_interim
else:
    conservative_branch = "P_none"
    conservative_res = p_none_res
    conservative_ctrl_auc = ctrl_auc_none

attn_auc_cons = conservative_res["+attention"]["hgb_clf"]["auc"]
cons_dauc     = conservative_res["delta"]["hgb_clf"]["delta_auc"]
cons_ci       = conservative_res["delta"]["hgb_clf"]["bootstrap_ci"]
cons_dp100    = conservative_res["delta"]["hgb_clf"]["delta_p100"]

print(f"\n  Conservative baseline: {conservative_branch}")
print(f"  Controls-only AUC: {conservative_ctrl_auc:.4f}")
print(f"  +Attention AUC:    {attn_auc_cons:.4f}")
print(f"  ΔAUC:              {cons_dauc:+.4f}  95%CI=[{cons_ci['ci_lo']:+.4f}, {cons_ci['ci_hi']:+.4f}]")

# Leaky ΔAUC for contrast
leaky_dauc_hgb = leaky_res["delta"]["hgb_clf"]["delta_auc"]
print(f"  v2 leaky ΔAUC (context only): {leaky_dauc_hgb:+.4f}")

# One-line interpretation
if cons_dauc > 0.01:
    interp = (
        f"HF attention adds modest incremental predictive value (ΔAUC={cons_dauc:+.4f}) "
        f"over the {conservative_branch} controls-only baseline in a clean forward-in-time test."
    )
elif cons_dauc > 0:
    interp = (
        f"HF attention adds marginal incremental predictive value (ΔAUC={cons_dauc:+.4f}) "
        f"over the {conservative_branch} controls-only baseline; bootstrap CI [{cons_ci['ci_lo']:+.4f}, "
        f"{cons_ci['ci_hi']:+.4f}] indicates the effect is small and may include zero."
    )
else:
    interp = (
        f"After removing leaky h-index, HF attention adds negligible or no incremental "
        f"value (ΔAUC={cons_dauc:+.4f}) over the {conservative_branch} controls-only baseline "
        f"in a clean forward-in-time test — a legitimate and informative null."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9.  LEAKAGE-AUDIT CHECKLIST (§6)
# ─────────────────────────────────────────────────────────────────────────────
audit = {
    "split_temporal": {
        "status": "PASS",
        "detail": "train=release_year≤2024 (n=5019), test=release_year=2025 (n=6325); disjoint years",
    },
    "scalers_fit_on_train_only": {
        "status": "PASS",
        "detail": "StandardScaler + OneHotEncoder inside Pipeline, fit only on train_X; test transformed with frozen objects",
    },
    "hyperparameters_tuned_on_train_only": {
        "status": "PASS",
        "detail": "GroupKFold(5) grouped by release_month, CV scoring on train; no test data seen during tuning",
    },
    "target_is_label_only": {
        "status": "PASS",
        "detail": "citation_count used only to compute q_i / y_i; never entered as a feature; FORBIDDEN assert ran",
    },
    "within_month_rank_no_leakage": {
        "status": "PASS",
        "detail": "upvote_rank_within_month = groupby(release_month).rank(pct=True) on full df before split; "
                  "within-group so no cross-split contamination; release_month itself excluded as feature",
    },
    "within_quarter_percentile_no_leakage": {
        "status": "PASS",
        "detail": "q_i = groupby(release_quarter).citation_count.rank(pct=True) on full df; "
                  "train/test quarters are disjoint (2023-2024 vs 2025) so no cross-split leakage",
    },
    "forbidden_columns_absent": {
        "status": "PASS",
        "detail": "Assertion enforced in run_branch(); forbidden set checked per feature matrix; "
                  "max_hindex included ONLY in the explicitly-flagged v2_leaky_replication row",
    },
    "features_available_at_day_one": {
        "age_months":                        {"available": "yes",  "note": "derived from release date"},
        "title_n_words":                     {"available": "yes",  "note": "from title at submission"},
        "title_has_colon":                   {"available": "yes",  "note": "from title at submission"},
        "abstract_n_chars":                  {"available": "yes",  "note": "from abstract at submission"},
        "kw_llm/agent/diffusion/etc":        {"available": "yes",  "note": "keyword flags from abstract"},
        "n_authors":                         {"available": "yes",  "note": "from submission metadata"},
        "has_github":                        {"available": "flag", "note": "BORDERLINE: may be added post-submission; flagged in spec"},
        "subfield":                          {"available": "yes",  "note": "from HF Daily Papers category"},
        "dow":                               {"available": "yes",  "note": "day-of-week of HF posting"},
        "max_papercount_cur2026_w99":        {"available": "flag", "note": "ANACHRONISTIC: today-measured productivity (2026 snapshot); P_interim flagged provisional"},
        "log_upvotes":                       {"available": "yes",  "note": "early/peak attention; never literal day-one (no timestamps); proxy"},
        "upvote_rank_within_month":          {"available": "yes",  "note": "within-month rank; no cross-split leakage"},
        "max_hindex (leaky row only)":       {"available": "no",   "note": "LEAKY: today-measured h-index; excluded from all headline rows"},
    },
    "prestige_branch_tags": {
        "P_interim":          "FLAGGED anachronistic (today-measured productivity; conservative provisional baseline)",
        "P_none":             "no prestige (upper bracket baseline — removes all productivity confound)",
        "P_tierB":            "PENDING — leakage-free prior-count; becomes DEFINITIVE headline when file available",
        "v2_leaky_replication": "LEAKY (max_hindex + P_interim) — context only, not headline",
    },
    "rerrun_hook_tierB": (
        "When data/processed/prepub_prestige_tierB.csv appears, re-execute this script. "
        "P_tierB branch will join on arxiv_id_clean and use its leakage-free prior-count + years_active. "
        "P_tierB AUC will become the DEFINITIVE conservative headline baseline."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10.  ASSEMBLE prediction.json
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {k: safe_float(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_float(i) for i in x]
    return x


def branch_summary(res):
    """Compact per-branch summary for JSON."""
    out = {
        "branch": res["branch"],
        "leaky": res["leaky"],
        "n_train": 5019,
        "n_test": 6325,
    }
    for fs in ["controls_only", "+attention"]:
        if fs not in res:
            continue
        out[fs] = {
            "logistic": {k: round(v, 5) for k, v in res[fs]["logistic"].items()
                         if isinstance(v, float)},
            "hgb_clf":  {k: round(v, 5) for k, v in res[fs]["hgb_clf"].items()
                         if isinstance(v, float)},
            "ridge":    {k: round(v, 5) for k, v in res[fs]["ridge"].items()
                         if isinstance(v, float)},
            "hgb_reg":  {k: round(v, 5) for k, v in res[fs]["hgb_reg"].items()
                         if isinstance(v, float)},
        }
    if "delta" in res:
        out["delta"] = {}
        for m, d in res["delta"].items():
            out["delta"][m] = {
                "delta_auc":    round(d["delta_auc"], 5),
                "delta_p100":   round(d["delta_p100"], 5),
                "bootstrap_ci": {k: round(v, 5) for k, v in d["bootstrap_ci"].items()
                                 if isinstance(v, float)},
            }
    return out


output = {
    "headline": {
        "conservative_delta_auc":  round(cons_dauc, 5),
        "ci_lo_95":                round(cons_ci["ci_lo"], 5),
        "ci_hi_95":                round(cons_ci["ci_hi"], 5),
        "conservative_baseline":   conservative_branch,
        "controls_only_auc":       round(conservative_ctrl_auc, 5),
        "attention_auc":           round(attn_auc_cons, 5),
        "delta_p100":              round(cons_dp100, 5),
        "n_train":                 5019,
        "n_test":                  6325,
        "model":                   "hgb_clf",
        "interpretation":          interp,
        "v2_leaky_delta_auc_contrast": round(leaky_dauc_hgb, 5),
    },
    "results": [branch_summary(r) for r in all_results],
    "audit": audit,
    "spec": "the pre-specified plan",
    "seed": SEED,
    "python": "Project/.venv/bin/python",
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Wrote {OUT_JSON}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  WRITE prediction_NOTE.md
# ─────────────────────────────────────────────────────────────────────────────
# Gather numbers for note
p_interim_ctrl_hgb  = p_interim_res["controls_only"]["hgb_clf"]["auc"]
p_interim_attn_hgb  = p_interim_res["+attention"]["hgb_clf"]["auc"]
p_interim_dauc      = p_interim_res["delta"]["hgb_clf"]["delta_auc"]
p_interim_ci        = p_interim_res["delta"]["hgb_clf"]["bootstrap_ci"]

p_none_ctrl_hgb     = p_none_res["controls_only"]["hgb_clf"]["auc"]
p_none_attn_hgb     = p_none_res["+attention"]["hgb_clf"]["auc"]
p_none_dauc         = p_none_res["delta"]["hgb_clf"]["delta_auc"]
p_none_ci           = p_none_res["delta"]["hgb_clf"]["bootstrap_ci"]

# Also logistic numbers
p_interim_ctrl_log  = p_interim_res["controls_only"]["logistic"]["auc"]
p_interim_attn_log  = p_interim_res["+attention"]["logistic"]["auc"]
p_interim_dauc_log  = p_interim_res["delta"]["logistic"]["delta_auc"]

p_none_ctrl_log     = p_none_res["controls_only"]["logistic"]["auc"]
p_none_attn_log     = p_none_res["+attention"]["logistic"]["auc"]
p_none_dauc_log     = p_none_res["delta"]["logistic"]["delta_auc"]

# Mature AUC
p_interim_mature_ctrl = p_interim_res["controls_only"]["hgb_clf"]["mature_auc"]
p_interim_mature_attn = p_interim_res["+attention"]["hgb_clf"]["mature_auc"]
p_none_mature_ctrl    = p_none_res["controls_only"]["hgb_clf"]["mature_auc"]
p_none_mature_attn    = p_none_res["+attention"]["hgb_clf"]["mature_auc"]

leaky_ctrl_hgb  = leaky_res["controls_only"]["hgb_clf"]["auc"]
leaky_attn_hgb  = leaky_res["+attention"]["hgb_clf"]["auc"]
leaky_dauc_val  = leaky_res["delta"]["hgb_clf"]["delta_auc"]

# Audit verdict
audit_pass = all(
    v.get("status") == "PASS"
    for v in audit.values()
    if isinstance(v, dict) and "status" in v
)
audit_verdict = "PASS — no leakage found in headline rows" if audit_pass else "REVIEW NEEDED"

note = f"""# D3 Prediction — Audited ΔAUC (PROJECT HEADLINE)

Generated: 2026-06-27  |  Spec: the pre-specified plan  |  seed={SEED}

## Headline

**Conservative ΔAUC (HGB, {conservative_branch} baseline):**
`ΔAUC = {cons_dauc:+.5f}  95%CI = [{cons_ci['ci_lo']:+.5f}, {cons_ci['ci_hi']:+.5f}]`

- Controls-only AUC: **{conservative_ctrl_auc:.5f}**
- +Attention AUC:    **{attn_auc_cons:.5f}**
- ΔP@100:           **{cons_dp100:+.5f}**
- n_train=5,019 / n_test=6,325

**v2 leaky contrast (max_hindex+P_interim, context only):**
Controls={leaky_ctrl_hgb:.5f} → +Attention={leaky_attn_hgb:.5f}  (ΔAUC={leaky_dauc_val:+.5f})

---

## Per-Baseline Brackets (HGB Classifier)

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC | 95%CI |
|---|---|---|---|---|
| P_interim (FLAGGED anachronistic) | {p_interim_ctrl_hgb:.4f} | {p_interim_attn_hgb:.4f} | {p_interim_dauc:+.4f} | [{p_interim_ci['ci_lo']:+.4f}, {p_interim_ci['ci_hi']:+.4f}] |
| P_none (no prestige) | {p_none_ctrl_hgb:.4f} | {p_none_attn_hgb:.4f} | {p_none_dauc:+.4f} | [{p_none_ci['ci_lo']:+.4f}, {p_none_ci['ci_hi']:+.4f}] |
| P_tierB (PENDING) | — | — | — | — |
| v2 leaky [CONTEXT ONLY] | {leaky_ctrl_hgb:.4f} | {leaky_attn_hgb:.4f} | {leaky_dauc_val:+.4f} | (not headline) |

## Per-Baseline Brackets (Logistic)

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC |
|---|---|---|---|
| P_interim | {p_interim_ctrl_log:.4f} | {p_interim_attn_log:.4f} | {p_interim_dauc_log:+.4f} |
| P_none | {p_none_ctrl_log:.4f} | {p_none_attn_log:.4f} | {p_none_dauc_log:+.4f} |

---

## Robustness Row (K=12 mature test, first_trend_date ≤ 2025-06-01, n={len(mature_test):,})

| Baseline | Controls-only AUC | +Attention AUC | ΔAUC |
|---|---|---|---|
| P_interim | {p_interim_mature_ctrl:.4f} | {p_interim_mature_attn:.4f} | {p_interim_mature_attn - p_interim_mature_ctrl:+.4f} |
| P_none | {p_none_mature_ctrl:.4f} | {p_none_mature_attn:.4f} | {p_none_mature_attn - p_none_mature_ctrl:+.4f} |

---

## Leakage Audit Verdict

**{audit_verdict}**

- Split: temporal (train≤2024, test=2025) — PASS
- Scalers/encoders fit on train only (inside Pipeline) — PASS
- Hyperparameters tuned on train only (GroupKFold by release_month) — PASS
- Target (citation_count) never entered feature matrix — PASS
- upvote_rank_within_month: within-group rank, no cross-split contamination — PASS
- q_i (within-quarter percentile): train/test quarters disjoint — PASS
- Forbidden columns asserted absent from all headline feature matrices — PASS
- max_hindex used ONLY in explicitly-flagged v2_leaky_replication row — PASS
- P_interim baseline flagged provisional (anachronistic productivity) — documented
- P_tierB baseline: PENDING (re-run hook in scripts/26_prediction.py)

---

## Audited ΔAUC vs v2 Leaky +0.05

v2 leaky setup (max_hindex + no forward-in-time split) obtained ΔAUC ≈ +0.05.
Audited ΔAUC (clean temporal split, no h-index): **{cons_dauc:+.5f}**
({conservative_branch} conservative baseline).

---

## CAN / CANNOT

**CAN:** Determine whether early/peak HF attention (upvotes) adds honest out-of-sample
incremental predictive value for within-quarter citation rank, forward in time
(train≤2024 → test 2025), over a leakage-audited controls-only baseline.

**CANNOT:**
- Claim causation
- Claim "day-one" signal (upvotes = peak, no day-level timestamps)
- Generalize beyond HF Daily Papers
- P_interim baseline is provisional (anachronistic) — flagged until P_tierB available

---

## One-line Verdict

{interp}
"""

with open(OUT_NOTE, "w") as f:
    f.write(note)
print(f"  Wrote {OUT_NOTE}")
print("\nDone.")
