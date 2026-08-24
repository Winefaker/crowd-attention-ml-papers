"""
04_merge_and_features.py
------------------------
Join the three raw sources on the arXiv id, deduplicate to one row per paper,
and engineer the analysis variables described in the proposal:

  outcome  : citation_count (and log1p)          [Semantic Scholar]
  signal   : upvotes (and log1p)                 [Hugging Face]
  controls : author-prestige proxy, n_authors,
             subfield, release month, paper age  [arXiv + derived]
  features : title length, has-colon title,
             abstract length, num_comments,
             github stars, reference count, OA    [derived]

Output: data/processed/papers_analysis.csv  (one row per paper)
"""
import pandas as pd
import numpy as np
import os
import re

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC, exist_ok=True)

TODAY = pd.Timestamp("2026-06-05")


def clean_arxiv_id(x):
    x = str(x).strip()
    if "v" in x.split(".")[-1]:
        return x.split("v")[0]
    return x


ARXIV_CAT_MAP = {
    "cs.CL": "NLP/LLM", "cs.CV": "Vision", "cs.LG": "ML/Theory",
    "cs.AI": "AI/Agents", "cs.RO": "Robotics", "cs.IR": "IR/RecSys",
    "cs.SD": "Audio/Speech", "eess.AS": "Audio/Speech", "eess.IV": "Vision",
    "cs.MA": "AI/Agents", "cs.HC": "HCI", "stat.ML": "ML/Theory",
    "cs.CR": "Security", "cs.SE": "Software", "cs.GR": "Graphics",
    "cs.NE": "ML/Theory", "cs.DC": "Systems",
}

# Ordered keyword rules: HF ai_keywords -> coarse subfield. First match wins, so
# more specific buckets (multimodal, RAG, agents) are checked before generic LLM.
KEYWORD_RULES = [
    ("Multimodal", r"multimodal|vision-language|vision language|mllm|vlm|image-text|video-language"),
    ("Vision/Image-Gen", r"diffusion|image generation|text-to-image|video generation|gaussian splatting|nerf|3d generation|super-resolution"),
    ("Agents", r"\bagent|agentic|tool use|tool-use|llm agent|multi-agent|autonomous"),
    ("RAG/Retrieval", r"retrieval-augmented|\brag\b|retrieval|dense retrieval|reranking"),
    ("Reasoning/RL", r"reasoning|chain-of-thought|chain of thought|reinforcement learning|\brlhf\b|reward model|preference optimization|\bgrpo\b|\bppo\b"),
    ("Efficiency/Systems", r"quantization|efficient|kv cache|inference|distillation|pruning|moe|mixture-of-experts|long context|flash"),
    ("Vision-Perception", r"object detection|segmentation|depth|pose|tracking|optical flow|point cloud"),
    ("Speech/Audio", r"speech|audio|asr|text-to-speech|\btts\b|music|voice"),
    ("Robotics/Embodied", r"robot|embodied|manipulation|navigation|locomotion|sim-to-real"),
    ("Benchmark/Eval", r"benchmark|evaluation|dataset|leaderboard"),
    ("Code/Math", r"\bcode\b|program synthesis|coding|software|theorem|mathematical reasoning"),
    ("LLM-core", r"large language model|\bllm\b|language model|pretraining|fine-tuning|instruction tuning|transformer|attention|scaling"),
]


def coarse_subfield(primary, fields_of_study, ai_keywords):
    """Subfield label. Prefer the arXiv primary category; else keyword rules on
    the HF ai_keywords; else the Semantic Scholar field of study."""
    p = primary if isinstance(primary, str) else ""
    if p in ARXIV_CAT_MAP:
        return ARXIV_CAT_MAP[p]
    if p.startswith(("cs.", "eess.", "stat.")):
        return "Other-CS"
    import re as _re
    kw = (ai_keywords if isinstance(ai_keywords, str) else "").lower()
    if kw:
        for label, pat in KEYWORD_RULES:
            if _re.search(pat, kw):
                return label
    fos = fields_of_study if isinstance(fields_of_study, str) else ""
    if fos and "Computer Science" not in fos:
        return fos.split(";")[0].strip()[:18] or "Unknown"
    if fos:
        return "Other-CS"
    return "Unknown"


def main():
    hf = pd.read_csv(os.path.join(RAW, "hf_daily_papers.csv"), dtype={"arxiv_id": str})
    hf["arxiv_id_clean"] = hf["arxiv_id"].map(clean_arxiv_id)

    # ---- collapse HF to one row per paper -------------------------------
    # A paper can trend on multiple days; keep its peak upvotes and first trend.
    hf_sorted = hf.sort_values("trend_date")
    agg = hf_sorted.groupby("arxiv_id_clean").agg(
        upvotes=("upvotes", "max"),
        num_comments=("num_comments", "max"),
        github_stars=("github_stars", "max"),
        n_trend_days=("trend_date", "nunique"),
        first_trend_date=("trend_date", "first"),
        hf_title=("hf_title", "first"),
        hf_summary=("hf_summary", "first"),
        published_at=("published_at", "first"),
        n_authors_hf=("n_authors_hf", "first"),
        submitted_by=("submitted_by", "first"),
        is_author_participating=("is_author_participating", "first"),
        ai_keywords=("ai_keywords", "first"),
        github_repo=("github_repo", "first"),
    ).reset_index()

    df = agg

    # ---- Semantic Scholar (outcome) ------------------------------------
    ss_path = os.path.join(RAW, "semantic_scholar.csv")
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path, dtype={"arxiv_id_clean": str})
        ss = ss.drop_duplicates("arxiv_id_clean")
        df = df.merge(ss, on="arxiv_id_clean", how="left")
    else:
        print("WARNING: no semantic_scholar.csv yet")

    # ---- arXiv (subfield + affiliations) -------------------------------
    ax_path = os.path.join(RAW, "arxiv_meta.csv")
    if os.path.exists(ax_path) and os.path.getsize(ax_path) > 0:
        ax = pd.read_csv(ax_path, dtype={"arxiv_id_clean": str})
        ax = ax.drop_duplicates("arxiv_id_clean")
        n_cat = ax["primary_category"].notna().sum() if "primary_category" in ax.columns else 0
        print(f"arXiv metadata available for {n_cat} papers")
        df = df.merge(ax, on="arxiv_id_clean", how="left")
    else:
        print("WARNING: no arxiv_meta.csv yet (subfield falls back to HF keywords / S2 fields)")
    # ensure expected arXiv columns exist even if collection was partial
    for c in ["primary_category", "n_authors_arxiv", "n_affiliations", "arxiv_title",
              "affiliations", "published_v1", "latest_version", "all_categories"]:
        if c not in df.columns:
            df[c] = np.nan

    # ---- dates & age ---------------------------------------------------
    df["release_dt"] = pd.to_datetime(
        df["published_v1"].fillna(df["published_at"]), errors="coerce", utc=True
    ).dt.tz_localize(None)
    df["release_dt"] = df["release_dt"].fillna(
        pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    )
    df["release_month"] = df["release_dt"].dt.to_period("M").astype(str)
    df["age_days"] = (TODAY - df["release_dt"]).dt.days
    df["age_months"] = df["age_days"] / 30.44

    # ---- outcome -------------------------------------------------------
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce")
    df["log_citations"] = np.log1p(df["citation_count"])
    df["citation_rate"] = df["citation_count"] / df["age_months"].clip(lower=1)

    # ---- attention signal ---------------------------------------------
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce")
    df["log_upvotes"] = np.log1p(df["upvotes"])
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0)
    df["github_stars"] = pd.to_numeric(df["github_stars"], errors="coerce").fillna(0)
    df["has_github"] = df["github_repo"].notna().astype(int)

    # ---- author count + prestige proxy --------------------------------
    df["n_authors"] = df["n_authors_arxiv"].fillna(df["n_authors_hf"])

    # Prestige proxy: how often this paper's authors recur across *other*
    # HF Daily Papers. Authors who repeatedly land featured papers are, within
    # this venue, prominent. We build an author -> #distinct-papers table from
    # the raw HF author lists, then for each paper take the max over its authors
    # (a "star author" signal) and the mean (overall team prominence).
    from collections import defaultdict
    author_papers = defaultdict(set)
    hf_authornames = hf.dropna(subset=["author_names"]) if "author_names" in hf.columns else hf.iloc[0:0]
    for _, r in hf_authornames.iterrows():
        names = [n.strip() for n in str(r["author_names"]).split(";") if n.strip()]
        for nm in names:
            author_papers[nm].add(r["arxiv_id_clean"])
    author_count = {nm: len(s) for nm, s in author_papers.items()}

    # map each paper to its author list (first trend appearance)
    paper_authors = (hf_authornames.sort_values("trend_date")
                     .groupby("arxiv_id_clean")["author_names"].first().to_dict())

    def prestige_stats(aid):
        names = [n.strip() for n in str(paper_authors.get(aid, "")).split(";") if n.strip()]
        if not names:
            return pd.Series({"author_max_appear": 1, "author_mean_appear": 1.0})
        # subtract this paper itself so we measure track record beyond this paper
        counts = [max(author_count.get(nm, 1) - 1, 0) for nm in names]
        return pd.Series({"author_max_appear": max(counts) if counts else 0,
                          "author_mean_appear": float(np.mean(counts)) if counts else 0.0})

    pres = df["arxiv_id_clean"].apply(prestige_stats)
    df = pd.concat([df, pres], axis=1)
    df["log_author_max_appear"] = np.log1p(df["author_max_appear"])
    df["log_author_mean_appear"] = np.log1p(df["author_mean_appear"])

    # keep submitter recurrence too (secondary visibility proxy)
    sub_counts = hf.groupby("submitted_by")["arxiv_id_clean"].nunique()
    df["submitter_n_papers"] = df["submitted_by"].map(sub_counts).fillna(1)
    df["log_submitter_papers"] = np.log1p(df["submitter_n_papers"])

    df["reference_count"] = pd.to_numeric(df.get("reference_count"), errors="coerce")
    df["influential_citations"] = pd.to_numeric(df.get("influential_citations"), errors="coerce")

    # ---- subfield ------------------------------------------------------
    df["subfield"] = df.apply(
        lambda r: coarse_subfield(r.get("primary_category"),
                                  r.get("fields_of_study"),
                                  r.get("ai_keywords")), axis=1)

    # ---- text-derived features ----------------------------------------
    df["title"] = df["arxiv_title"].fillna(df["hf_title"]).fillna("")
    df["title_n_chars"] = df["title"].str.len()
    df["title_n_words"] = df["title"].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    df["title_has_colon"] = df["title"].str.contains(":").astype(int)
    df["abstract_n_chars"] = df["hf_summary"].fillna("").str.len()
    df["title_lower"] = df["title"].str.lower()
    for kw in ["llm", "agent", "diffusion", "reasoning", "benchmark",
               "survey", "efficient", "multimodal", "rl", "scaling"]:
        df[f"kw_{kw}"] = df["title_lower"].str.contains(rf"\b{kw}", regex=True).astype(int)

    # ---- open access ---------------------------------------------------
    if "is_open_access" in df.columns:
        df["is_open_access"] = df["is_open_access"].fillna(False).astype(int)

    df.to_csv(os.path.join(PROC, "papers_analysis.csv"), index=False)
    print(f"Merged dataset: {df.shape[0]} papers x {df.shape[1]} columns")
    print(f"  with citations: {df['citation_count'].notna().sum()}")
    print(f"  with subfield != Unknown: {(df['subfield']!='Unknown').sum()}")
    print(f"  median upvotes={df['upvotes'].median()}, median citations={df['citation_count'].median()}")
    print(f"Saved -> data/processed/papers_analysis.csv")


if __name__ == "__main__":
    main()
