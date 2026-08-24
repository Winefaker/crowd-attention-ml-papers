"""
10_collect_arxiv_control.py
---------------------------
v2 (D5): control sample of arXiv papers that did NOT trend on HF Daily Papers.

Sampling rule (popularity-blind, deterministic): for each month from 2023-05 to
2025-12 and each category in {cs.CL, cs.CV, cs.LG}, take the first N submissions of
the month ordered by submission date. arXiv's date ordering carries no attention
information, so this is a fair "background" sample of the same fields and period.

Then fetch Semantic Scholar citations for the control ids (batch endpoint).

Outputs:
  data/raw/arxiv_control.csv          (one row per control paper, arXiv metadata)
  data/raw/arxiv_control_s2.csv       (citations etc. for control papers)
"""
import requests
import pandas as pd
import time
import os
import xml.etree.ElementTree as ET

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_META = os.path.join(RAW, "arxiv_control.csv")
OUT_S2 = os.path.join(RAW, "arxiv_control_s2.csv")

URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}

CATS = ["cs.CL", "cs.CV", "cs.LG"]
PER_CELL = 40  # papers per (month, category)


def month_windows():
    import calendar
    wins = []
    for y in (2023, 2024, 2025):
        for m in range(1, 13):
            if y == 2023 and m < 5:
                continue
            last_day = calendar.monthrange(y, m)[1]  # arXiv rejects invalid dates like 0631
            start = f"{y}{m:02d}010000"
            endm = f"{y}{m:02d}{last_day}2359"
            wins.append((f"{y}-{m:02d}", start, endm))
    return wins


def parse_entry(e):
    def txt(path):
        node = e.find(path, NS)
        return node.text.strip() if node is not None and node.text else None
    raw_id = (txt("a:id") or "").split("/abs/")[-1]
    aid = raw_id.split("v")[0] if "v" in raw_id.split(".")[-1] else raw_id
    cats = [c.attrib.get("term") for c in e.findall("a:category", NS)]
    prim = e.find("arxiv:primary_category", NS)
    title = txt("a:title")
    summary = txt("a:summary")
    return {
        "arxiv_id_clean": aid,
        "title": title.replace("\n", " ") if title else None,
        "published_v1": txt("a:published"),
        "n_authors": len(e.findall("a:author", NS)),
        "primary_category": prim.attrib.get("term") if prim is not None else (cats[0] if cats else None),
        "all_categories": "; ".join([c for c in cats if c]),
        "abstract_n_chars": len(summary) if summary else 0,
    }


def fetch_cell(cat, start, end, session):
    q = f"cat:{cat} AND submittedDate:[{start} TO {end}]"
    params = {"search_query": q, "max_results": PER_CELL,
              "sortBy": "submittedDate", "sortOrder": "ascending"}
    for attempt in range(4):
        try:
            r = session.get(URL, params=params, timeout=60)
            if r.status_code == 200 and "<entry>" in r.text:
                root = ET.fromstring(r.text)
                return [parse_entry(e) for e in root.findall("a:entry", NS)]
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(4 * (attempt + 1))
        except Exception:
            time.sleep(4 * (attempt + 1))
    return []


def fetch_s2(ids):
    rows = []
    fields = ("title,citationCount,influentialCitationCount,referenceCount,year,"
              "publicationDate,isOpenAccess,authors.hIndex,authors.paperCount")
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        for attempt in range(5):
            try:
                r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                                  params={"fields": fields},
                                  json={"ids": [f"arXiv:{x}" for x in batch]},
                                  headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    for qid, p in zip(batch, r.json()):
                        if not p:
                            rows.append({"arxiv_id_clean": qid, "ss_found": 0})
                            continue
                        hs = [a.get("hIndex") for a in (p.get("authors") or []) if a and a.get("hIndex") is not None]
                        rows.append({
                            "arxiv_id_clean": qid, "ss_found": 1,
                            "citation_count": p.get("citationCount"),
                            "influential_citations": p.get("influentialCitationCount"),
                            "reference_count": p.get("referenceCount"),
                            "ss_pub_date": p.get("publicationDate"),
                            "max_hindex": max(hs) if hs else None,
                            "mean_hindex": sum(hs) / len(hs) if hs else None,
                            "last_author_hindex": hs[-1] if hs else None,
                        })
                    break
                time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(3 * (attempt + 1))
        if (i // 100) % 5 == 0:
            print(f"  s2 control {i + len(batch)}/{len(ids)}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_S2, index=False)
        time.sleep(1.1)
    pd.DataFrame(rows).to_csv(OUT_S2, index=False)
    print(f"S2 control DONE: {len(rows)} rows, found={sum(r0.get('ss_found',0) for r0 in rows)}")


def main():
    session = requests.Session()
    session.headers.update(HEADERS)
    wins = month_windows()
    rows = []
    total = len(wins) * len(CATS)
    k = 0
    for ym, start, end in wins:
        for cat in CATS:
            k += 1
            res = fetch_cell(cat, start, end, session)
            for r0 in res:
                r0["strat_month"] = ym
                r0["strat_cat"] = cat
            rows.extend(res)
            print(f"  [{k}/{total}] {ym} {cat}: +{len(res)} (total {len(rows)})", flush=True)
            pd.DataFrame(rows).to_csv(OUT_META, index=False)
            time.sleep(3.2)

    df = pd.DataFrame(rows).drop_duplicates("arxiv_id_clean")
    df.to_csv(OUT_META, index=False)
    print(f"LISTING DONE. {len(df)} unique control papers -> {OUT_META}")

    # exclude any that actually trended on HF
    hf_ids = set()
    for f in ["hf_daily_papers.csv", "hf_daily_papers_2023.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            h = pd.read_csv(p, dtype={"arxiv_id": str})
            hf_ids |= set(h["arxiv_id"].dropna().map(
                lambda x: x.split("v")[0] if "v" in str(x).split(".")[-1] else str(x)))
    df = df[~df["arxiv_id_clean"].isin(hf_ids)]
    print(f"After removing HF-trending overlap: {len(df)} control papers")
    df.to_csv(OUT_META, index=False)

    fetch_s2(df["arxiv_id_clean"].dropna().tolist())
    print("ALL DONE")


if __name__ == "__main__":
    main()
