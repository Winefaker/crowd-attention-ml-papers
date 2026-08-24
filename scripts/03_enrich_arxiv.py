"""
03_enrich_arxiv.py
------------------
arXiv metadata: primary + cross-listed categories (the "subfield"), author list,
affiliations when present, and the v1 submission timestamp + revision count.

arXiv's query endpoint accepts a comma-separated id_list, so we batch many ids
per request (with the requested ~3s politeness delay) instead of one-at-a-time.

Output: data/raw/arxiv_meta.csv  (one row per unique arxiv id)
"""
import requests
import pandas as pd
import time
import os
import xml.etree.ElementTree as ET

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HF = os.path.join(RAW, "hf_daily_papers.csv")
OUT = os.path.join(RAW, "arxiv_meta.csv")

URL = "https://export.arxiv.org/api/query"  # https avoids the http->https 301
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS = {"User-Agent": "hf-papers-study/1.0"
                         + (f" (mailto:{CONTACT})" if CONTACT else "")}


def clean_arxiv_id(x):
    x = str(x).strip()
    if "v" in x.split(".")[-1]:
        return x.split("v")[0]
    return x


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_entry(e):
    def txt(path):
        node = e.find(path, NS)
        return node.text.strip() if node is not None and node.text else None

    # arxiv id sits in the <id> url, e.g. http://arxiv.org/abs/2501.08313v1
    raw_id = txt("a:id") or ""
    aid = raw_id.split("/abs/")[-1]
    aid_clean = clean_arxiv_id(aid)
    version = None
    if "v" in aid.split("/")[-1]:
        try:
            version = int(aid.split("v")[-1])
        except ValueError:
            version = None

    authors = e.findall("a:author", NS)
    affs = []
    for a in authors:
        af = a.find("arxiv:affiliation", NS)
        if af is not None and af.text:
            affs.append(af.text.strip())

    cats = [c.attrib.get("term") for c in e.findall("a:category", NS)]
    prim = e.find("arxiv:primary_category", NS)
    primary = prim.attrib.get("term") if prim is not None else (cats[0] if cats else None)
    comment = txt("arxiv:comment")

    return {
        "arxiv_id_clean": aid_clean,
        "arxiv_found": 1,
        "arxiv_title": txt("a:title").replace("\n", " ") if txt("a:title") else None,
        "published_v1": txt("a:published"),   # v1 submission timestamp
        "updated_latest": txt("a:updated"),   # latest revision timestamp
        "latest_version": version,
        "n_authors_arxiv": len(authors),
        "n_affiliations": len(affs),
        "affiliations": " | ".join(affs[:10]) if affs else None,
        "primary_category": primary,
        "all_categories": "; ".join([c for c in cats if c]),
        "n_categories": len([c for c in cats if c]),
        "arxiv_comment": comment,
    }


def fetch_batch(ids, session):
    # The export endpoint is slow (~15-20s/req) and rate-limits bursts, so we use
    # modest batches, a generous timeout, and back off hard on 429.
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    for attempt in range(4):
        try:
            r = session.get(URL, params=params, timeout=45)
            if r.status_code == 200 and r.text.lstrip().startswith("<?xml"):
                root = ET.fromstring(r.text)
                entries = root.findall("a:entry", NS)
                return [parse_entry(e) for e in entries]
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
            else:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    return []


def main():
    hf = pd.read_csv(HF)
    hf["arxiv_id_clean"] = hf["arxiv_id"].map(clean_arxiv_id)
    # Prioritise the most-upvoted papers first, so if the slow endpoint forces an
    # early stop we still have categories/affiliations for the papers that matter
    # most to the analysis and the dashboard.
    order = (hf.groupby("arxiv_id_clean")["upvotes"].max()
             .sort_values(ascending=False).index.tolist())
    ids = order
    print(f"Fetching arXiv metadata for {len(ids)} unique ids (high-upvote first)")

    session = requests.Session()
    session.headers.update(HEADERS)
    rows = []
    BATCH = 25
    batches = list(chunks(ids, BATCH))
    for bi, batch in enumerate(batches):
        res = fetch_batch(batch, session)
        got = {r["arxiv_id_clean"] for r in res}
        rows.extend(res)
        for qid in batch:
            if qid not in got:
                rows.append({"arxiv_id_clean": qid, "arxiv_found": 0})
        if (bi + 1) % 5 == 0 or bi == len(batches) - 1:
            print(f"  batch {bi+1}/{len(batches)}: {len(rows)} rows", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        time.sleep(3.0)  # arXiv asks for ~1 request / 3 seconds

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    found = int(out["arxiv_found"].fillna(0).sum())
    print(f"DONE. {len(out)} ids, {found} found on arXiv -> {OUT}")


if __name__ == "__main__":
    main()
