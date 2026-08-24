"""
14_text_themes.py
-----------------
v2 (D7.4): what CONTENT separates overrated from underrated papers?

TF-IDF (1-2 grams) over title + HF summary + ai_keywords, then:
  - L1-regularized logistic (overrated vs underrated) -> discriminative terms
  - top distinctive terms per group by mean TF-IDF ratio

Output: data/processed/text_themes.json
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    d = pd.read_csv(os.path.join(PROC, "papers_scored_v2.csv"), dtype={"arxiv_id_clean": str})
    d = d[(d["overrated"] == 1) | (d["underrated"] == 1)].copy()
    d["text"] = (d["title"].fillna("") + ". " + d.get("ai_keywords", "").fillna("")
                 + ". " + d.get("hf_summary", "").fillna("")).str.lower()
    y = d["overrated"].values  # 1 = overrated, 0 = underrated
    print(f"texts: {len(d)} ({y.sum()} overrated / {(1-y).sum():.0f} underrated)")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=8, max_df=0.5,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(d["text"])
    terms = np.array(vec.get_feature_names_out())
    print(f"vocab: {len(terms)}")

    lr = LogisticRegression(solver="saga", l1_ratio=1.0, C=3.0,
                            class_weight="balanced", max_iter=5000).fit(X, y)
    coef = lr.coef_[0]
    idx = np.argsort(coef)
    res = {
        "n_overrated": int(y.sum()), "n_underrated": int(len(y) - y.sum()),
        "cv_note": "L1 logistic, overrated(1) vs underrated(0)",
        "overrated_terms": [{"term": terms[i], "coef": float(coef[i])}
                            for i in idx[::-1][:25] if coef[i] > 0],
        "underrated_terms": [{"term": terms[i], "coef": float(coef[i])}
                             for i in idx[:25] if coef[i] < 0],
    }

    # distinctive-term ratio as a second, model-free view
    Xa = X.toarray()
    mo = Xa[y == 1].mean(axis=0) + 1e-6
    mu = Xa[y == 0].mean(axis=0) + 1e-6
    ratio = np.log(mo / mu)
    ridx = np.argsort(ratio)
    res["overrated_distinctive"] = [terms[i] for i in ridx[::-1][:20]]
    res["underrated_distinctive"] = [terms[i] for i in ridx[:20]]

    print("\nOVERRATED terms:", ", ".join(t["term"] for t in res["overrated_terms"][:12]))
    print("UNDERRATED terms:", ", ".join(t["term"] for t in res["underrated_terms"][:12]))

    with open(os.path.join(PROC, "text_themes.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("Saved text_themes.json")


if __name__ == "__main__":
    main()
