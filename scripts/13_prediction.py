"""
13_prediction.py
----------------
v2 (D7.3): out-of-sample test — can day-of-release attention help PREDICT which
papers become high-impact, beyond what controls (prestige, field, age) already say?

Design: train on papers released 2023-05 .. 2024-12, test on 2025-01 .. 2025-12.
Strictly forward in time; nothing from the test year touches training.

Target: the paper's CITATION PERCENTILE WITHIN ITS RELEASE QUARTER. Predicting raw
log-citations fails across cohorts because the test year is younger (level shift
that tree models cannot extrapolate); the within-quarter percentile is level- and
age-free, which is also the quantity a reader actually wants ("will this paper be
a top paper of its cohort?").

Two model families (linear ridge + gradient boosting), each fit twice:
   controls-only  vs  controls + attention (log upvotes, comments, trend days)
Metrics on the 2025 test set:
   - R^2 / Spearman for the percentile target
   - AUC + precision@100 for "top decile of its quarter"

Output: data/processed/prediction_results.json
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

CONTROLS = ["log_age", "log_max_hindex", "log_last_hindex", "log_n_authors",
            "has_github", "title_n_words", "title_has_colon", "abstract_n_chars"]
ATTENTION = ["log_upvotes", "num_comments", "n_trend_days"]


def load():
    df = pd.read_csv(os.path.join(PROC, "papers_v2.csv"), dtype={"arxiv_id_clean": str})
    df = df[df["citation_count"].notna() & df["upvotes"].notna()]
    df = df[(df["age_months"] >= 5) & (df["age_months"] <= 40)].copy()
    df["log_age"] = np.log(df["age_months"].clip(lower=1))
    for c in CONTROLS + ATTENTION:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    sf = pd.get_dummies(df["subfield"], prefix="sf")
    df = pd.concat([df, sf], axis=1)
    # level-free target: citation percentile within release quarter
    df["quarter"] = pd.PeriodIndex(df["release_month"], freq="M").asfreq("Q").astype(str)
    df["cit_pct"] = df.groupby("quarter")["citation_count"].rank(pct=True)
    df["top_decile"] = (df["cit_pct"] >= 0.9).astype(int)
    return df, [c for c in sf.columns]


def evaluate(y_true, y_pred, top_mask):
    auc = roc_auc_score(top_mask, y_pred)
    order = np.argsort(-y_pred)[:100]
    p100 = float(np.mean(top_mask.values[order]))
    return {"r2": float(r2_score(y_true, y_pred)),
            "spearman": float(spearmanr(y_true, y_pred).statistic),
            "auc_top_decile": float(auc), "precision_at_100": p100}


def main():
    df, sf_cols = load()
    train = df[df["release_month"] <= "2024-12"]
    test = df[(df["release_month"] >= "2025-01") & (df["release_month"] <= "2025-12")]
    print(f"train n={len(train)} (2023-05..2024-12), test n={len(test)} (2025)")

    y_tr, y_te = train["cit_pct"], test["cit_pct"]
    out = {"n_train": int(len(train)), "n_test": int(len(test))}
    feature_sets = {
        "controls_only": CONTROLS + sf_cols,
        "controls_plus_attention": CONTROLS + sf_cols + ATTENTION,
    }
    models = {
        "ridge": lambda: Ridge(alpha=1.0),
        "gbm": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                 learning_rate=0.05, subsample=0.8,
                                                 random_state=0),
    }
    for mname, mk in models.items():
        for fname, feats in feature_sets.items():
            m = mk().fit(train[feats], y_tr)
            pred = m.predict(test[feats])
            out[f"{mname}__{fname}"] = evaluate(y_te, pred, test["top_decile"])
            r = out[f"{mname}__{fname}"]
            print(f"  {mname:5s} {fname:24s} R2={r['r2']:.3f} rho={r['spearman']:.3f} "
                  f"AUC(top10%)={r['auc_top_decile']:.3f} P@100={r['precision_at_100']:.2f}")

    for mname in models:
        a = out[f"{mname}__controls_plus_attention"]; b = out[f"{mname}__controls_only"]
        out[f"{mname}__lift"] = {k: a[k] - b[k] for k in a}
        print(f"  {mname} LIFT from attention: dR2={out[f'{mname}__lift']['r2']:+.3f} "
              f"dAUC={out[f'{mname}__lift']['auc_top_decile']:+.3f} "
              f"dP@100={out[f'{mname}__lift']['precision_at_100']:+.2f}")

    with open(os.path.join(PROC, "prediction_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved prediction_results.json")


if __name__ == "__main__":
    main()
