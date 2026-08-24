"""
30 — Uniform subfield taxonomy (one rule set for every paper).

Why: the v2 `subfield` column mixes two taxonomies. arXiv categories were fetched
high-upvote-first and merged partially, so ~27% of papers (the most upvoted) carry
arXiv-category labels (Vision, NLP/LLM, AI/Agents, ...) and the rest carry keyword
labels (Vision/Image-Gen, LLM-core, Agents, ...). Label *source* therefore proxies
attention and leaks into "controls-only" models. This script assigns every paper a
label from the SAME ordered keyword rule set (v2 KEYWORD_RULES, unchanged), applied to
HF ai_keywords, falling back to the HF title + summary text, else "Other".

Output: data/subfield_kw.csv  (arxiv_id_clean, subfield_kw, subfield_kw_source)
"""
import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PAPERS_V2 = BASE / "data/processed/papers_v2.csv"
OUT = BASE / "data/subfield_kw.csv"

# identical to Project/scripts/04_merge_and_features.py KEYWORD_RULES (first match wins)
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
RULES = [(lab, re.compile(pat)) for lab, pat in KEYWORD_RULES]

def label(text):
    t = (text if isinstance(text, str) else "").lower()
    if not t:
        return None
    for lab, rx in RULES:
        if rx.search(t):
            return lab
    return None

def main():
    p = pd.read_csv(PAPERS_V2, dtype={"arxiv_id_clean": str}, low_memory=False)
    out = []
    for r in p.itertuples(index=False):
        lab = label(getattr(r, "ai_keywords", None)); src = "ai_keywords"
        if lab is None:
            title = getattr(r, "hf_title", None) or getattr(r, "title", None)
            lab = label(f"{title or ''} {getattr(r, 'hf_summary', '') or ''}"); src = "title_summary"
        if lab is None:
            lab, src = "Other", "none"
        out.append((r.arxiv_id_clean, lab, src))
    df = pd.DataFrame(out, columns=["arxiv_id_clean", "subfield_kw", "subfield_kw_source"])
    assert df.arxiv_id_clean.is_unique
    df.to_csv(OUT, index=False)
    print(df.subfield_kw.value_counts().to_string())
    print(df.subfield_kw_source.value_counts().to_string())
    # diagnostic: does label source still proxy attention?
    m = p[["arxiv_id_clean", "upvotes", "subfield"]].merge(df, on="arxiv_id_clean")
    print("median upvotes by kw source:\n", m.groupby("subfield_kw_source")["upvotes"].median().to_string())
    print("wrote", OUT, len(df))

if __name__ == "__main__":
    main()
