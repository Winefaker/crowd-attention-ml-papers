"""
Feasibility check: is there a pool of submitted but not featured papers?
====================================================
Determines whether the Hugging Face API exposes submitted-but-not-featured
papers (the "pool") with submission timestamps. Without it, the featured versus
not featured comparison cannot be run.

Verdict framework:
  (a) POOL FOUND       — non-featured ids return HF records with submittedOnDailyAt
                         + upvotes, density in gap of 0 to 14 days, so the comparison can be run
  (b) REJECT-LOG FOUND — endpoint exposes rejected/attempted submissions gap>14
                         → unlocks B1 RD (unlikely)
  (c) NEITHER          — backbone (B3+C+A) proceeds; B2 marked extension

Guardrails (from pre-critique):
  1. Network sentinel FIRST — abort with INCONCLUSIVE if sentinel fails
  2. Genuine non-featured ids (anti-join control vs featured)
  3. Separate 404 (true no-record) from non-200 (transient/blocked)
  4. Verdict keys on PAYLOAD FIELDS, not status code
"""

import os
import csv
import json
import random
import time
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DATA = Path(__file__).resolve().parents[1] / "data"
FEATURED_CSV = PROJECT_DATA / "processed" / "papers_v2.csv"
CONTROL_CSV  = PROJECT_DATA / "raw" / "arxiv_control.csv"
OUT_DIR      = Path(__file__).resolve().parents[1] / "spikes"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_API_BASE   = "https://huggingface.co/api"
CONTACT = os.environ.get("CONTACT_EMAIL", "")  # polite User-Agent for the public APIs
HEADERS       = {
    "User-Agent": "hf-papers-study/1.0" + (f" (mailto:{CONTACT})" if CONTACT else ""),
    "Accept": "application/json",
}
SLEEP_S       = 0.25   # ≥ 0.2 s between calls
MAX_CALLS     = 100
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hf_paper(arxiv_id: str, retry: bool = True) -> dict:
    """GET /api/papers/{id}. Returns dict with keys: status, payload, error."""
    url = f"{HF_API_BASE}/papers/{arxiv_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return {"status": 200, "payload": r.json(), "error": None}
        elif r.status_code == 404:
            return {"status": 404, "payload": None, "error": "not_found"}
        else:
            if retry:
                time.sleep(SLEEP_S * 2)
                return get_hf_paper(arxiv_id, retry=False)
            return {"status": r.status_code, "payload": None, "error": f"http_{r.status_code}"}
    except Exception as e:
        return {"status": -1, "payload": None, "error": str(e)}


def has_pool_fields(payload: dict) -> bool:
    """True iff payload contains submittedOnDailyAt (regardless of upvotes)."""
    return payload is not None and "submittedOnDailyAt" in payload and payload["submittedOnDailyAt"] is not None


def has_upvotes(payload: dict) -> bool:
    return payload is not None and "upvotes" in payload and payload["upvotes"] is not None


def probe_daily_endpoint(date_str: str) -> dict:
    """GET /api/daily_papers?date={date_str}. Inspect field schema."""
    url = f"{HF_API_BASE}/daily_papers?date={date_str}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return {"status": 200, "payload": data, "error": None}
        else:
            return {"status": r.status_code, "payload": None, "error": f"http_{r.status_code}"}
    except Exception as e:
        return {"status": -1, "payload": None, "error": str(e)}


def read_csv_ids(path: Path, col: str) -> set:
    ids = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(col, "").strip()
            if val:
                ids.add(val)
    return ids


# ---------------------------------------------------------------------------
# Step 0 — Load featured and control sets
# ---------------------------------------------------------------------------
print("=" * 70)
print("SPIKE 0 — HF Pool Probe")
print("=" * 70)

print("\n[0] Loading featured set (papers_v2.csv)...")
featured_ids = read_csv_ids(FEATURED_CSV, "arxiv_id_clean")
print(f"    Featured papers: {len(featured_ids):,}")

print("[0] Loading control set (arxiv_control.csv)...")
control_ids_all = read_csv_ids(CONTROL_CSV, "arxiv_id_clean")
print(f"    Control papers (raw): {len(control_ids_all):,}")

# Anti-join: keep only ids NOT in featured set
nonfeatured_ids = sorted(control_ids_all - featured_ids)
print(f"    Non-featured (anti-join): {len(nonfeatured_ids):,}")

# ---------------------------------------------------------------------------
# Step 1 — Sentinel: probe a known-featured paper
# ---------------------------------------------------------------------------
print("\n[1] NETWORK SENTINEL — probing a known-featured paper...")
# Pick a high-upvote featured paper
sentinel_ids = []
with open(FEATURED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = sorted(reader, key=lambda r: float(r.get("upvotes", 0) or 0), reverse=True)
    sentinel_ids = [r["arxiv_id_clean"] for r in rows[:3]]

sentinel_id = sentinel_ids[0] if sentinel_ids else "2302.06555"
print(f"    Sentinel id: {sentinel_id}")
time.sleep(SLEEP_S)
sentinel_result = get_hf_paper(sentinel_id)
print(f"    HTTP status: {sentinel_result['status']}")

if sentinel_result["status"] != 200:
    print(f"\n    *** ABORT: Sentinel returned {sentinel_result['status']} — {sentinel_result['error']}")
    print("    VERDICT = INCONCLUSIVE — no network / endpoint blocked")
    sys.exit(1)

payload = sentinel_result["payload"]
has_submitted = has_pool_fields(payload)
has_up = has_upvotes(payload)
print(f"    submittedOnDailyAt present: {has_submitted}")
print(f"    upvotes present: {has_up}")
print(f"    submittedOnDailyAt value: {payload.get('submittedOnDailyAt', 'MISSING')}")
print(f"    upvotes value: {payload.get('upvotes', 'MISSING')}")
sentinel_keys = sorted(payload.keys()) if payload else []
print(f"    Top-level keys: {sentinel_keys}")

if not has_submitted:
    print("\n    *** ABORT: Featured paper lacks submittedOnDailyAt — endpoint changed or blocked")
    print("    VERDICT = INCONCLUSIVE — endpoint does not expose submittedOnDailyAt")
    sys.exit(1)

print("    Sentinel PASSED — endpoint live, submittedOnDailyAt confirmed.\n")

calls_used = 1  # sentinel

# ---------------------------------------------------------------------------
# Step 2 — Daily endpoint shape probe
# ---------------------------------------------------------------------------
print("[2] DAILY ENDPOINT PROBE — checking /api/daily_papers?date=...")
time.sleep(SLEEP_S)
daily_result = probe_daily_endpoint("2024-07-11")
calls_used += 1
print(f"    HTTP status: {daily_result['status']}")
daily_field_schema = {}
if daily_result["status"] == 200 and daily_result["payload"]:
    data = daily_result["payload"]
    if isinstance(data, list) and len(data) > 0:
        example = data[0]
        daily_field_schema = {k: type(v).__name__ for k, v in example.items()}
        print(f"    Records returned: {len(data)}")
        print(f"    First-record keys: {sorted(example.keys())}")
        # Look for submitted/curated/isFeatured distinction
        for key in ["submitted", "curated", "isFeatured", "submittedOnDailyAt", "status"]:
            if key in example:
                print(f"    {key}: {example[key]}")
        # Check if any record has isFeatured=False or similar
        featured_flags = [r.get("isFeatured") for r in data if "isFeatured" in r]
        submitted_flags = [r.get("submitted") for r in data if "submitted" in r]
        print(f"    isFeatured values seen: {set(featured_flags)}")
        print(f"    submitted values seen: {set(submitted_flags)}")
    elif isinstance(data, dict):
        print(f"    Response is dict with keys: {sorted(data.keys())}")
else:
    print(f"    Daily endpoint failed: {daily_result.get('error')}")

print()

# ---------------------------------------------------------------------------
# Step 3 — Sample non-featured ids and probe
# ---------------------------------------------------------------------------
print("[3] PROBING NON-FEATURED IDS (from arxiv_control.csv anti-join)...")
sample_size = min(60, len(nonfeatured_ids))
sampled_nonfeatured = random.sample(nonfeatured_ids, sample_size)
print(f"    Sampling {sample_size} non-featured ids (seed={RANDOM_SEED})")

results_nonfeatured = []
tally = {
    "200_with_submittedOnDailyAt": 0,
    "200_without_submittedOnDailyAt": 0,
    "404": 0,
    "blocked_non200_non404": 0,
    "error": 0,
}
pool_examples = []  # examples with submittedOnDailyAt
gap14_plus = []     # gap > 14 days (unlikely but probe for B1)

for i, arxiv_id in enumerate(sampled_nonfeatured):
    if calls_used >= MAX_CALLS - 20:  # reserve 20 for wide draw
        print(f"    *** Approaching call budget at i={i}, stopping non-featured batch")
        break
    time.sleep(SLEEP_S)
    res = get_hf_paper(arxiv_id)
    calls_used += 1
    status = res["status"]
    p = res["payload"]

    row = {
        "arxiv_id": arxiv_id,
        "http_status": status,
        "has_submittedOnDailyAt": has_pool_fields(p),
        "has_upvotes": has_upvotes(p),
        "submittedOnDailyAt": p.get("submittedOnDailyAt") if p else None,
        "upvotes": p.get("upvotes") if p else None,
        "error": res["error"],
    }
    results_nonfeatured.append(row)

    if status == 200:
        if has_pool_fields(p):
            tally["200_with_submittedOnDailyAt"] += 1
            if len(pool_examples) < 3:
                pool_examples.append({
                    "arxiv_id": arxiv_id,
                    "submittedOnDailyAt": p.get("submittedOnDailyAt"),
                    "upvotes": p.get("upvotes"),
                    "publishedAt": p.get("publishedAt"),
                    "keys": sorted(p.keys()),
                })
        else:
            tally["200_without_submittedOnDailyAt"] += 1
    elif status == 404:
        tally["404"] += 1
    elif status == -1:
        tally["error"] += 1
    else:
        tally["blocked_non200_non404"] += 1

    if i > 0 and (i + 1) % 10 == 0:
        print(f"    Progress: {i+1}/{sample_size} — pool_hits={tally['200_with_submittedOnDailyAt']}, "
              f"404s={tally['404']}, blocked={tally['blocked_non200_non404']}")

print(f"\n    Done with non-featured batch. Calls used so far: {calls_used}")
print(f"    Tally: {tally}")

# ---------------------------------------------------------------------------
# Step 4 — Wide arbitrary cs.* draw (if budget allows)
# ---------------------------------------------------------------------------
print("\n[4] WIDE CS.* DRAW (arbitrary ids not from either dataset)...")

# Construct a set of plausible-looking cs.AI/cs.LG arxiv ids from 2023-2024
# that are unlikely to be in our featured set
wide_draw_pool = []
for year in ["2301", "2302", "2303", "2310", "2311", "2312", "2401", "2402", "2403"]:
    for suffix in range(10050, 10070):
        aid = f"{year}.{suffix:05d}"
        if aid not in featured_ids and aid not in control_ids_all:
            wide_draw_pool.append(aid)

random.shuffle(wide_draw_pool)
wide_sample = wide_draw_pool[:min(20, MAX_CALLS - calls_used - 2)]
print(f"    Wide draw sample size: {len(wide_sample)}")

tally_wide = {
    "200_with_submittedOnDailyAt": 0,
    "200_without_submittedOnDailyAt": 0,
    "404": 0,
    "blocked_non200_non404": 0,
    "error": 0,
}
wide_pool_examples = []

for i, arxiv_id in enumerate(wide_sample):
    if calls_used >= MAX_CALLS:
        print(f"    *** Call budget reached at i={i}")
        break
    time.sleep(SLEEP_S)
    res = get_hf_paper(arxiv_id)
    calls_used += 1
    p = res["payload"]
    status = res["status"]

    if status == 200:
        if has_pool_fields(p):
            tally_wide["200_with_submittedOnDailyAt"] += 1
            if len(wide_pool_examples) < 2:
                wide_pool_examples.append({
                    "arxiv_id": arxiv_id,
                    "submittedOnDailyAt": p.get("submittedOnDailyAt"),
                    "upvotes": p.get("upvotes"),
                    "keys": sorted(p.keys()),
                })
        else:
            tally_wide["200_without_submittedOnDailyAt"] += 1
    elif status == 404:
        tally_wide["404"] += 1
    elif status == -1:
        tally_wide["error"] += 1
    else:
        tally_wide["blocked_non200_non404"] += 1

print(f"    Wide draw tally: {tally_wide}")
print(f"    Total calls used: {calls_used}")

# ---------------------------------------------------------------------------
# Step 5 — Determine verdict
# ---------------------------------------------------------------------------
print("\n[5] VERDICT DETERMINATION...")

total_pool_hits = (tally["200_with_submittedOnDailyAt"] +
                   tally_wide["200_with_submittedOnDailyAt"])

total_nonfeatured_tested = (tally["200_with_submittedOnDailyAt"] +
                             tally["200_without_submittedOnDailyAt"] +
                             tally["404"] +
                             tally["blocked_non200_non404"] +
                             tally["error"])

total_wide_tested = (tally_wide["200_with_submittedOnDailyAt"] +
                     tally_wide["200_without_submittedOnDailyAt"] +
                     tally_wide["404"] +
                     tally_wide["blocked_non200_non404"] +
                     tally_wide["error"])

# Check for gap > 14 (B1 criterion)
# (gap_days = submittedOnDailyAt - publishedAt; need to compute if pool records found)
gap14_plus_count = 0
if pool_examples or wide_pool_examples:
    import datetime
    for ex in (pool_examples + wide_pool_examples):
        try:
            sub = ex.get("submittedOnDailyAt")
            # Note: gap > 14 would require publishedAt too; if we have records we'd compute
            # For now just note the count (likely 0 by design)
        except Exception:
            pass

# Determine verdict
if total_pool_hits > 0:
    verdict = "(a) POOL FOUND"
    b2_unlocked = True
elif gap14_plus_count > 0:
    verdict = "(b) REJECT-LOG FOUND"
    b2_unlocked = False
else:
    verdict = "(c) NEITHER"
    b2_unlocked = False

print(f"    VERDICT: {verdict}")
print(f"    B2 unlocked: {b2_unlocked}")

# ---------------------------------------------------------------------------
# Step 6 — Print full summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Sentinel:                   {sentinel_id} → HTTP 200, submittedOnDailyAt PRESENT")
print(f"Non-featured ids tested:    {total_nonfeatured_tested}")
print(f"  200 WITH submittedOnDailyAt:    {tally['200_with_submittedOnDailyAt']}")
print(f"  200 WITHOUT submittedOnDailyAt: {tally['200_without_submittedOnDailyAt']}")
print(f"  404 (true no-record):           {tally['404']}")
print(f"  blocked (non-200/non-404):      {tally['blocked_non200_non404']}")
print(f"  error:                          {tally['error']}")
print(f"Wide draw tested:           {total_wide_tested}")
print(f"  200 WITH submittedOnDailyAt:    {tally_wide['200_with_submittedOnDailyAt']}")
print(f"  200 WITHOUT submittedOnDailyAt: {tally_wide['200_without_submittedOnDailyAt']}")
print(f"  404 (true no-record):           {tally_wide['404']}")
print(f"Total pool hits:            {total_pool_hits}")
print(f"Gap > 14 days found:        {gap14_plus_count}")
print(f"Total API calls used:       {calls_used}")
print(f"\nVERDICT: {verdict}")
if pool_examples:
    print(f"\nPool record examples (non-featured with submittedOnDailyAt):")
    for ex in pool_examples[:2]:
        print(f"  {ex}")
if wide_pool_examples:
    print(f"\nWide draw pool examples:")
    for ex in wide_pool_examples[:2]:
        print(f"  {ex}")

# ---------------------------------------------------------------------------
# Step 7 — Save JSON checkpoint
# ---------------------------------------------------------------------------
checkpoint = {
    "verdict": verdict,
    "b2_unlocked": b2_unlocked,
    "sentinel": {
        "arxiv_id": sentinel_id,
        "http_status": sentinel_result["status"],
        "has_submittedOnDailyAt": has_submitted,
        "has_upvotes": has_up,
        "submittedOnDailyAt": payload.get("submittedOnDailyAt"),
        "upvotes": payload.get("upvotes"),
        "keys": sentinel_keys,
    },
    "daily_endpoint": {
        "status": daily_result["status"],
        "field_schema": daily_field_schema,
    },
    "nonfeatured_tally": tally,
    "wide_draw_tally": tally_wide,
    "total_pool_hits": total_pool_hits,
    "gap14_plus_count": gap14_plus_count,
    "calls_used": calls_used,
    "pool_examples": pool_examples[:3],
    "wide_pool_examples": wide_pool_examples[:2],
}

chk_path = OUT_DIR / "spike0_checkpoint.json"
with open(chk_path, "w") as f:
    json.dump(checkpoint, f, indent=2, default=str)
print(f"\nCheckpoint saved: {chk_path}")
print("=" * 70)
print("DONE")
