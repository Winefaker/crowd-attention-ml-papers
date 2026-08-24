"""
17_dashboard_v2.py
------------------
Generate the v2 interactive dashboard (self-contained HTML, vanilla JS/canvas,
no server, no external libraries). Iterates on dashboard_v1_2026-06-10.html with
the v2 dataset and findings:

  - v2 stat cards (n=11,347; beta; E-value; trending premium; OOS lift)
  - scatter explorer with year filter + click-through to arXiv
  - trending vs background CCDF (canvas, log-log) -- the control-sample result
  - prestige asymmetry (median h-index by group)
  - spec ladder table + prediction lift bars
  - "where the crowd is most informative" subfield slopes
  - hidden-gems leaderboard (underrated, sorted by citations)
  - paper explorer table with h-index column and 4 sort keys

Output: figures/dashboard_v2_2026-06-11.html
"""
import pandas as pd
import numpy as np
import json
import math
import os

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "dashboard_v2_2026-06-11.html")


def papers_payload():
    d = pd.read_csv(os.path.join(PROC, "papers_scored_v2.csv"), dtype={"arxiv_id_clean": str})
    rows = []
    for _, r in d.iterrows():
        t = str(r.get("title", "") or "").strip().replace("\n", " ")
        rows.append({
            "id": r.arxiv_id_clean,
            "t": t[:95] + ("…" if len(t) > 95 else ""),
            "s": str(r.subfield_grp),
            "u": int(r.upvotes) if pd.notna(r.upvotes) else 0,
            "c": int(r.citation_count) if pd.notna(r.citation_count) else 0,
            "h": int(r.max_hindex) if pd.notna(r.max_hindex) else None,
            "a": round(float(r.age_months), 1),
            "y": int(r.release_year) if pd.notna(r.release_year) else None,
            "o": int(r.overrated), "n": int(r.underrated),
            "ap": round(float(r.attention_pct), 3), "ip": round(float(r.impact_pct), 3),
            "g": int(r.has_github) if pd.notna(r.has_github) else 0,
        })
    return rows, d


def ccdf_payload():
    """CCDF point lists for background / trending / underrated (downsampled)."""
    meta = pd.read_csv(os.path.join(RAW, "arxiv_control.csv"), dtype={"arxiv_id_clean": str})
    s2 = pd.read_csv(os.path.join(RAW, "arxiv_control_s2.csv"), dtype={"arxiv_id_clean": str})
    c = meta.merge(s2, on="arxiv_id_clean")
    c = c[c["ss_found"] == 1]["citation_count"].dropna()
    sc = pd.read_csv(os.path.join(PROC, "papers_scored_v2.csv"), dtype={"arxiv_id_clean": str})
    t = sc["citation_count"].dropna()
    u = sc[sc["underrated"] == 1]["citation_count"].dropna()

    def ccdf(series, npts=140):
        x = np.sort(series.values) + 1
        p = 1 - np.arange(1, len(x) + 1) / len(x)
        keep = p > 0
        x, p = x[keep], p[keep]
        idx = np.unique(np.geomspace(1, len(x), npts).astype(int) - 1)
        return [[float(x[i]), float(p[i])] for i in idx]

    return {"background": {"pts": ccdf(c), "n": int(len(c))},
            "trending": {"pts": ccdf(t), "n": int(len(t))},
            "underrated": {"pts": ccdf(u), "n": int(len(u))}}


def main():
    papers, d = papers_payload()
    res = json.load(open(os.path.join(PROC, "model_results_v2.json")))
    pred = json.load(open(os.path.join(PROC, "prediction_results.json")))
    ctrl = json.load(open(os.path.join(PROC, "control_results.json")))

    sl = res["spec_ladder"]
    ladder = [
        ("M0 bivariate", sl["M0_bivariate"]),
        ("M1 + age, subfield", sl["M1_age_field"]),
        ("M2 + recurrence prestige", sl["M2_v1_proxy"]),
        ("M3 + author h-index (main)", sl["M3_hindex"]),
        ("M4 month fixed effects", sl["M4_monthFE"]),
    ]
    ladder_js = json.dumps([
        {"name": nm, "beta": round(s["beta"], 3), "lo": round(s["ci"][0], 3),
         "hi": round(s["ci"][1], 3), "irr": round(s["irr_per_doubling"], 2)}
        for nm, s in ladder] + [
        {"name": "Placebo: reference count", "beta": round(res["placebo"]["beta"], 3),
         "lo": round(res["placebo"]["beta"] - 1.96 * res["placebo"]["se"], 3),
         "hi": round(res["placebo"]["beta"] + 1.96 * res["placebo"]["se"], 3), "irr": None}])

    slopes = res["random_slopes"]["subfield_slopes"]
    sub_stats = d.groupby("subfield_grp").agg(
        n=("upvotes", "size"), mc=("citation_count", "median")).to_dict("index")
    slopes_js = json.dumps([
        {"s": k, "v": round(v, 2), "n": int(sub_stats.get(k, {}).get("n", 0)),
         "mc": float(sub_stats.get(k, {}).get("mc", 0))}
        for k, v in slopes.items() if sub_stats.get(k, {}).get("n", 0) >= 80])

    hmed = {
        "over": float(d[d.overrated == 1]["max_hindex"].median()),
        "typ": float(d[(d.overrated == 0) & (d.underrated == 0)]["max_hindex"].median()),
        "under": float(d[d.underrated == 1]["max_hindex"].median()),
    }

    meta_js = json.dumps({
        "n": int(len(d)),
        "beta": round(sl["M3_hindex"]["beta"], 3),
        "irr": round(sl["M3_hindex"]["irr_per_doubling"], 2),
        "evalue": round(res["e_value"]["e_value_point"], 2),
        "placebo_pct": round(abs(res["placebo"]["beta"]) / res["ols_main"]["beta"] * 100),
        "premium": round(math.exp(ctrl["trending_premium"]["matched_log_gap"]), 1),
        "under_pctile": round(ctrl["underrated_vs_background"]
                              ["mean_percentile_of_underrated_in_background"] * 100),
        "auc0": round(pred["ridge__controls_only"]["auc_top_decile"], 3),
        "auc1": round(pred["ridge__controls_plus_attention"]["auc_top_decile"], 3),
        "p0": round(pred["ridge__controls_only"]["precision_at_100"], 2),
        "p1": round(pred["ridge__controls_plus_attention"]["precision_at_100"], 2),
        "nov": int(res["over_under_v2"]["n_overrated"]),
        "nun": int(res["over_under_v2"]["n_underrated"]),
        "hmed": hmed,
        "ctrl_med": ctrl["distributions"]["control"]["citation_quantiles"]["0.5"],
        "trend_med": ctrl["distributions"]["trending"]["citation_quantiles"]["0.5"],
        "spearman": round(res["descriptives_v2"]["spearman"], 2),
    })

    html_top = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Attention vs Impact</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f4f5f8; color: #18182b; }
  header { background: linear-gradient(110deg,#18182b 60%,#2d2d52); color: #fff; padding: 20px 30px; }
  header h1 { font-size: 1.3rem; font-weight: 650; }
  header p { font-size: 0.82rem; opacity: 0.75; margin-top: 4px; }
  .badge { display:inline-block; background:#4f46e5; color:#fff; font-size:0.7rem; font-weight:700;
           padding:2px 9px; border-radius:10px; margin-left:8px; vertical-align:middle; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap: 13px; padding: 18px 30px 0; }
  .card { background: #fff; border-radius: 9px; padding: 14px 17px; box-shadow: 0 1px 4px #0002; }
  .card .val { font-size: 1.45rem; font-weight: 750; line-height: 1.15; }
  .card .lbl { font-size: 0.72rem; color: #667; margin-top: 4px; line-height: 1.35; }
  .section { background: #fff; border-radius: 9px; margin: 15px 30px; box-shadow: 0 1px 4px #0002; }
  .section > h2 { font-size: 0.92rem; font-weight: 650; padding: 13px 18px 10px; border-bottom: 1px solid #eee; }
  .section .note { font-size: 0.76rem; color: #778; padding: 8px 18px 0; }
  .filters { padding: 10px 18px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
             border-bottom: 1px solid #f0f0f4; font-size: 0.82rem; }
  select, input[type=text] { border: 1px solid #ccd; border-radius: 5px; padding: 4px 8px; font-size: 0.82rem; background: #fff; }
  .btn { padding: 4px 12px; border-radius: 5px; border: 1px solid #ccd; cursor: pointer; font-size: 0.8rem; background: #fff; }
  .btn.active { background: #18182b; color: #fff; border-color: #18182b; }
  canvas { display: block; }
  #tooltip { position: fixed; background: rgba(24,24,43,0.94); color: #fff; padding: 8px 12px;
             border-radius: 7px; font-size: 0.78rem; pointer-events: none; display: none;
             max-width: 300px; line-height: 1.45; z-index: 99; }
  .legend { display: flex; gap: 18px; padding: 4px 18px 13px; font-size: 0.77rem; color:#445; flex-wrap: wrap; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  @media(max-width:900px){ .grid2{grid-template-columns:1fr;} }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { background: #f0f1f6; padding: 7px 10px; text-align: left; font-weight: 620; color: #445; cursor: pointer; user-select: none; }
  th.sortable:hover { background: #e4e5ef; }
  td { padding: 6px 10px; border-bottom: 1px solid #f2f2f6; }
  tr:hover td { background: #f8f8fc; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .tag { display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.69rem; font-weight:650; }
  .tag-ov { background:#fde5e5; color:#b91c1c; } .tag-un { background:#dbeafe; color:#1d4ed8; }
  .tag-ok { background:#eee; color:#667; }
  .pg { padding: 9px 18px 13px; font-size: 0.78rem; color: #667; display: flex; gap: 12px; align-items: center; }
  .pg button { padding: 3px 11px; font-size: 0.78rem; border: 1px solid #ccd; border-radius: 5px; cursor: pointer; background: #fff; }
  .pg button:disabled { opacity: 0.35; cursor: default; }
  .hbar-row { display: grid; grid-template-columns: 150px 1fr 56px; gap: 9px; align-items: center;
              padding: 3px 18px; font-size: 0.78rem; }
  .hbar-bg { background: #eceef4; border-radius: 4px; height: 13px; position: relative; }
  .hbar-fill { background: #4f46e5; border-radius: 4px; height: 13px; }
  .lad-row { display: grid; grid-template-columns: 215px 1fr 120px; gap: 10px; align-items: center;
             padding: 5px 18px; font-size: 0.79rem; }
  .lad-track { position: relative; height: 16px; background: #f0f1f6; border-radius: 4px; }
  .lad-ci { position: absolute; height: 8px; top: 4px; background: #b9bdf0; border-radius: 3px; }
  .lad-pt { position: absolute; width: 9px; height: 9px; top: 3.5px; border-radius: 50%; background: #2c3160; }
  .gems li { padding: 5px 0; border-bottom: 1px solid #f2f2f6; font-size: 0.81rem; line-height: 1.4; }
  .gems { list-style: none; padding: 8px 18px 14px; }
  .gems a { color: #3b4bd8; text-decoration: none; }
  footer { padding: 14px 30px 26px; font-size: 0.75rem; color: #889; }
</style>
</head>
<body>
<header>
  <h1>Can the Crowd Spot Important ML Papers on Day One? <span class="badge">v2 · 2026-06-11</span></h1>
  <p>11,347 HF Daily-Papers trending papers (2023-05 → 2025-12) + 3,280 never-trending arXiv control ·
     author h-index controls · uniform citation snapshot 2026-06-11 · full paper: reports/05_paper_v2_2026-06-11.md</p>
</header>

<div class="cards" id="cards"></div>

<div class="section">
  <h2>Explorer — every trending paper, age- &amp; field-adjusted percentiles (click a dot → arXiv)</h2>
  <div class="filters">
    <label>Subfield</label><select id="f-sub"><option value="">All</option></select>
    <label>Year</label><select id="f-year"><option value="">All</option>
      <option>2023</option><option>2024</option><option>2025</option></select>
    <label>Show</label>
    <button class="btn active" id="b-all">All</button>
    <button class="btn" id="b-ov">Overrated</button>
    <button class="btn" id="b-un">Underrated</button>
  </div>
  <div style="padding: 12px 18px;"><canvas id="scatter" width="900" height="460" style="border:1px solid #eee;border-radius:6px;cursor:pointer;max-width:100%;"></canvas></div>
  <div class="legend">
    <span><span class="dot" style="background:#c3c7cf"></span>Typical</span>
    <span><span class="dot" style="background:#ef4444"></span>Overrated (high buzz, low impact for its age+field)</span>
    <span><span class="dot" style="background:#3b82f6"></span>Underrated (low buzz, high impact)</span>
  </div>
</div>

<div class="section">
  <h2>Trending vs the never-trending background (CCDF, log–log)</h2>
  <div class="note" id="ccdf-note"></div>
  <div style="padding: 12px 18px;"><canvas id="ccdf" width="880" height="380" style="max-width:100%;"></canvas></div>
  <div class="legend">
    <span><span class="dot" style="background:#9aa0a6"></span>background arXiv (control)</span>
    <span><span class="dot" style="background:#3b82f6"></span>trending on HF</span>
    <span><span class="dot" style="background:#ef4444"></span>trending &amp; "underrated"</span>
  </div>
</div>

<div class="grid2">
<div class="section" style="margin-right:8px;">
  <h2>Does β survive controls? (NB spec ladder + placebo)</h2>
  <div class="note">coefficient on log(1+upvotes); bar = 95% CI. Placebo (reference count) should be ~0 — and is.</div>
  <div id="ladder" style="padding: 8px 0 14px;"></div>
</div>
<div class="section" style="margin-left:8px;">
  <h2>The prestige asymmetry &amp; out-of-sample lift</h2>
  <div class="note">median team max h-index by group — the crowd over-rewards newcomer flash, under-rewards establishment substance.</div>
  <div id="hbars" style="padding: 8px 0 4px;"></div>
  <div class="note" style="padding-top:10px;">2025 held-out prediction (ridge): adding day-one attention…</div>
  <div id="predbars" style="padding: 6px 0 14px;"></div>
</div>
</div>

<div class="grid2">
<div class="section" style="margin-right:8px;">
  <h2>Where the crowd's votes are most informative (random slopes)</h2>
  <div class="note">attention→citation slope by subfield (mixed model BLUPs); n ≥ 80 shown.</div>
  <div id="slopes" style="padding: 8px 0 14px;"></div>
</div>
<div class="section" style="margin-left:8px;">
  <h2>Hidden gems — most-cited "underrated" papers</h2>
  <div class="note">low day-one buzz relative to their age+field, top-tertile citations. The alpha is real: these sit at the 91st percentile of the background.</div>
  <ul class="gems" id="gems"></ul>
</div>
</div>

<div class="section">
  <h2>Paper table</h2>
  <div class="filters">
    <input type="text" id="q" placeholder="Search title…" style="width:230px">
    <label>Subfield</label><select id="t-sub"><option value="">All</option></select>
    <label>Type</label><select id="t-type"><option value="">All</option>
      <option value="ov">Overrated</option><option value="un">Underrated</option><option value="ok">Typical</option></select>
    <label>Year</label><select id="t-year"><option value="">All</option>
      <option>2023</option><option>2024</option><option>2025</option></select>
  </div>
  <div style="overflow-x:auto; padding: 0 18px 6px;">
  <table><thead><tr>
    <th class="num sortable" data-k="u">Upvotes ▾</th>
    <th class="num sortable" data-k="c">Citations</th>
    <th class="num sortable" data-k="h">max h</th>
    <th>Type</th><th>Subfield</th>
    <th class="num sortable" data-k="a">Age mo</th>
    <th>Title</th>
  </tr></thead><tbody id="tbody"></tbody></table></div>
  <div class="pg">
    <button id="prev">← Prev</button><span id="pginfo"></span><button id="next">Next →</button>
  </div>
</div>

<div id="tooltip"></div>
<footer>First pass dashboard · generated by scripts/17_dashboard_v2.py · data: HF Daily Papers, Semantic Scholar, arXiv ·
decisions log: reports/DECISIONS.md · previous iteration: dashboard_v1_2026-06-10.html</footer>

<script>
"""

    html_js = """
// ---------- payloads ----------
const META = __META__;
const PAPERS = __PAPERS__;
const CCDF = __CCDF__;
const LADDER = __LADDER__;
const SLOPES = __SLOPES__;

// ---------- stat cards ----------
document.getElementById("cards").innerHTML = [
  {v: META.n.toLocaleString(), l: "trending papers (+3,280 control)"},
  {v: "×" + META.irr, l: "citations per 2× upvotes, h-index controlled (β=" + META.beta + ")"},
  {v: "E=" + META.evalue, l: "E-value; placebo effect only " + META.placebo_pct + "% — confounding implausible"},
  {v: "×" + META.premium, l: "trending vs matched never-trending background (median " + META.trend_med + " vs " + META.ctrl_med + " cites)"},
  {v: META.auc0 + "→" + META.auc1, l: "2025 out-of-sample AUC, top-decile detection (P@100 " + META.p0 + "→" + META.p1 + ")"},
  {v: META.nov + " / " + META.nun, l: "overrated / underrated papers (h-index medians " + META.hmed.over + " vs " + META.hmed.under + ")"},
].map(c => `<div class="card"><div class="val">${c.v}</div><div class="lbl">${c.l}</div></div>`).join("");

// ---------- scatter ----------
const cv = document.getElementById("scatter"), cx = cv.getContext("2d");
const P = {l: 52, r: 18, t: 16, b: 44};
let mode = "all";
const subs = [...new Set(PAPERS.map(p => p.s))].sort();
["f-sub","t-sub"].forEach(id => { const el = document.getElementById(id);
  subs.forEach(s => el.add(new Option(s, s))); });

function filt() {
  const sf = document.getElementById("f-sub").value;
  const yr = document.getElementById("f-year").value;
  return PAPERS.filter(p => (!sf || p.s === sf) && (!yr || String(p.y) === yr));
}
function draw() {
  const W = cv.width - P.l - P.r, H = cv.height - P.t - P.b;
  cx.clearRect(0, 0, cv.width, cv.height);
  cx.fillStyle = "rgba(239,68,68,0.05)"; cx.fillRect(P.l + W*2/3, P.t + H*2/3, W/3, H/3);
  cx.fillStyle = "rgba(59,130,246,0.06)"; cx.fillRect(P.l, P.t, W/3, H/3);
  cx.strokeStyle = "#e8e9ee"; [1/3, 2/3].forEach(v => {
    cx.beginPath(); cx.moveTo(P.l + v*W, P.t); cx.lineTo(P.l + v*W, P.t+H); cx.stroke();
    cx.beginPath(); cx.moveTo(P.l, P.t + v*H); cx.lineTo(P.l+W, P.t + v*H); cx.stroke(); });
  cx.strokeStyle = "#ccd"; cx.strokeRect(P.l, P.t, W, H);
  cx.fillStyle = "#667"; cx.font = "12px system-ui";
  cx.fillText("attention percentile (adjusted for age + subfield) →", P.l + W/2 - 130, cv.height - 8);
  cx.save(); cx.translate(14, P.t + H/2 + 95); cx.rotate(-Math.PI/2);
  cx.fillText("impact percentile (citations, adjusted) →", 0, 0); cx.restore();
  cx.fillStyle = "#99a"; cx.font = "10.5px system-ui";
  [0,0.5,1].forEach(v => { cx.fillText(v.toFixed(1), P.l + v*W - 7, P.t + H + 15);
    cx.fillText(v.toFixed(1), P.l - 26, P.t + (1-v)*H + 4); });
  const data = filt();
  const dot = (p, col, r) => { cx.beginPath();
    cx.arc(P.l + p.ap*W, P.t + (1-p.ip)*H, r, 0, 7); cx.fillStyle = col; cx.fill(); };
  if (mode !== "ov" && mode !== "un")
    data.filter(p => !p.o && !p.n).forEach(p => dot(p, "rgba(165,170,180,0.4)", 2.3));
  if (mode !== "ov") data.filter(p => p.n).forEach(p => dot(p, "rgba(59,130,246,0.85)", 3.4));
  if (mode !== "un") data.filter(p => p.o).forEach(p => dot(p, "rgba(239,68,68,0.85)", 3.4));
}
["f-sub","f-year"].forEach(id => document.getElementById(id).onchange = draw);
[["b-all","all"],["b-ov","ov"],["b-un","un"]].forEach(([id,m]) => {
  document.getElementById(id).onclick = () => { mode = m;
    ["b-all","b-ov","b-un"].forEach(x => document.getElementById(x).classList.toggle("active", x===id));
    draw(); };
});
const tip = document.getElementById("tooltip");
let hoverP = null;
cv.addEventListener("mousemove", e => {
  const r = cv.getBoundingClientRect();
  const mx = (e.clientX - r.left) * cv.width / r.width, my = (e.clientY - r.top) * cv.height / r.height;
  const W = cv.width - P.l - P.r, H = cv.height - P.t - P.b;
  const ax = (mx - P.l)/W, ay = 1 - (my - P.t)/H;
  let best = null, bd = 1;
  filt().forEach(p => { const dx = p.ap - ax, dy = p.ip - ay, d2 = dx*dx + dy*dy;
    if (d2 < bd) { bd = d2; best = p; } });
  hoverP = (best && bd < 0.0006) ? best : null;
  if (!hoverP) { tip.style.display = "none"; return; }
  const tg = hoverP.o ? "🔴 OVERRATED" : hoverP.n ? "🔵 UNDERRATED" : "·";
  tip.innerHTML = `<b>${tg}</b> ${hoverP.t}<br><span style="opacity:.8">${hoverP.s} · ${hoverP.u} upvotes · ${hoverP.c} cites · h=${hoverP.h ?? "?"} · ${hoverP.a} mo</span>`;
  tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY - 8) + "px";
});
cv.addEventListener("mouseleave", () => { tip.style.display = "none"; hoverP = null; });
cv.addEventListener("click", () => { if (hoverP) window.open("https://arxiv.org/abs/" + hoverP.id, "_blank"); });
draw();

// ---------- CCDF (log-log canvas) ----------
const c2 = document.getElementById("ccdf"), g2 = c2.getContext("2d");
(function() {
  const Q = {l: 60, r: 16, t: 12, b: 42};
  const W = c2.width - Q.l - Q.r, H = c2.height - Q.t - Q.b;
  const xmax = Math.log10(20000), ymin = -4;
  const X = v => Q.l + Math.log10(Math.max(v,1)) / xmax * W;
  const Y = p => Q.t + (Math.log10(Math.max(p, 1e-4)) / ymin) * H;
  g2.strokeStyle = "#e8e9ee"; g2.fillStyle = "#99a"; g2.font = "10.5px system-ui";
  [1,10,100,1000,10000].forEach(v => { const x = X(v);
    g2.beginPath(); g2.moveTo(x, Q.t); g2.lineTo(x, Q.t+H); g2.stroke();
    g2.fillText(v >= 1000 ? (v/1000)+"k" : v, x - 8, Q.t + H + 15); });
  [1,0.1,0.01,0.001,0.0001].forEach(p => { const y = Y(p);
    g2.beginPath(); g2.moveTo(Q.l, y); g2.lineTo(Q.l+W, y); g2.stroke();
    g2.fillText(p >= 0.01 ? (p*100)+"%" : p, Q.l - 44, y + 4); });
  g2.strokeStyle = "#ccd"; g2.strokeRect(Q.l, Q.t, W, H);
  g2.fillStyle = "#667"; g2.font = "12px system-ui";
  g2.fillText("citations (1+x, log)", Q.l + W/2 - 50, c2.height - 6);
  g2.save(); g2.translate(13, Q.t + H/2 + 55); g2.rotate(-Math.PI/2);
  g2.fillText("share of papers exceeding x", 0, 0); g2.restore();
  const colors = {background: "#9aa0a6", trending: "#3b82f6", underrated: "#ef4444"};
  for (const [k, obj] of Object.entries(CCDF)) {
    g2.strokeStyle = colors[k]; g2.lineWidth = 2.2; g2.beginPath();
    obj.pts.forEach(([x, p], i) => { const px = X(x), py = Y(p);
      i ? g2.lineTo(px, py) : g2.moveTo(px, py); });
    g2.stroke();
  }
  document.getElementById("ccdf-note").textContent =
    `Even the trending papers the crowd under-voted (red) out-cite ~${META.under_pctile}% of the background. ` +
    `Medians: background ${META.ctrl_med}, trending ${META.trend_med} citations.`;
})();

// ---------- spec ladder ----------
(function() {
  const lo = Math.min(...LADDER.map(d => d.lo), 0) - 0.04, hi = Math.max(...LADDER.map(d => d.hi)) + 0.04;
  const pct = v => ((v - lo) / (hi - lo) * 100).toFixed(1) + "%";
  document.getElementById("ladder").innerHTML = LADDER.map(d => `
    <div class="lad-row">
      <div style="${d.irr === null ? 'color:#b91c1c;font-weight:600;' : ''}">${d.name}</div>
      <div class="lad-track">
        <div class="lad-ci" style="left:${pct(d.lo)}; width:calc(${pct(d.hi)} - ${pct(d.lo)}); ${d.irr === null ? 'background:#f3b9b9;' : ''}"></div>
        <div class="lad-pt" style="left:calc(${pct(d.beta)} - 4px); ${d.irr === null ? 'background:#b91c1c;' : ''}"></div>
        <div style="position:absolute;left:${pct(0)};top:0;width:1px;height:16px;background:#888;"></div>
      </div>
      <div class="num" style="font-variant-numeric:tabular-nums;">β=${d.beta}${d.irr ? " (×" + d.irr + ")" : ""}</div>
    </div>`).join("");
})();

// ---------- h-index asymmetry + prediction bars ----------
(function() {
  const H = META.hmed, mx = 24;
  const rows = [["Overrated teams", H.over, "#ef4444"], ["Typical", H.typ, "#9aa0a6"], ["Underrated teams", H.under, "#3b82f6"]];
  document.getElementById("hbars").innerHTML = rows.map(([l, v, c]) => `
    <div class="hbar-row"><div>${l}</div>
      <div class="hbar-bg"><div class="hbar-fill" style="width:${v/mx*100}%;background:${c}"></div></div>
      <div class="num">h = ${v}</div></div>`).join("");
  const pr = [["controls only", META.auc0, META.p0, "#9aa0a6"], ["+ day-one attention", META.auc1, META.p1, "#4f46e5"]];
  document.getElementById("predbars").innerHTML = pr.map(([l, a, p, c]) => `
    <div class="hbar-row"><div>${l}</div>
      <div class="hbar-bg"><div class="hbar-fill" style="width:${(a-0.5)/0.5*100}%;background:${c}"></div></div>
      <div class="num">AUC ${a}</div></div>
    <div class="hbar-row"><div style="color:#99a;font-size:0.72rem;">precision@100</div>
      <div class="hbar-bg"><div class="hbar-fill" style="width:${p*100}%;background:${c};opacity:0.55;"></div></div>
      <div class="num">${Math.round(p*100)}%</div></div>`).join("");
})();

// ---------- subfield slopes ----------
(function() {
  const s = SLOPES.sort((a, b) => b.v - a.v), mx = Math.max(...s.map(d => d.v));
  document.getElementById("slopes").innerHTML = s.map(d => `
    <div class="hbar-row"><div>${d.s}</div>
      <div class="hbar-bg"><div class="hbar-fill" style="width:${d.v/mx*100}%"></div></div>
      <div class="num">${d.v.toFixed(2)}</div></div>`).join("");
})();

// ---------- hidden gems ----------
(function() {
  const gems = PAPERS.filter(p => p.n).sort((a, b) => b.c - a.c).slice(0, 10);
  document.getElementById("gems").innerHTML = gems.map(p => `
    <li><a href="https://arxiv.org/abs/${p.id}" target="_blank">${p.t}</a><br>
      <span style="color:#778;">${p.u} upvotes → <b>${p.c.toLocaleString()} citations</b> · ${p.s} · team h=${p.h ?? "?"}${p.g ? " · code" : ""}</span></li>`).join("");
})();

// ---------- table ----------
let rows = [...PAPERS], page = 0, sortK = "u", sortD = -1;
const PAGE = 25;
function apply() {
  const q = document.getElementById("q").value.toLowerCase();
  const sf = document.getElementById("t-sub").value, ty = document.getElementById("t-type").value,
        yr = document.getElementById("t-year").value;
  rows = PAPERS.filter(p =>
    (!q || p.t.toLowerCase().includes(q)) && (!sf || p.s === sf) &&
    (!yr || String(p.y) === yr) &&
    (!ty || (ty === "ov" ? p.o : ty === "un" ? p.n : !p.o && !p.n)));
  rows.sort((a, b) => sortD * (((a[sortK] ?? -1)) - ((b[sortK] ?? -1))) * -1);
  page = 0; render();
}
function render() {
  const sl = rows.slice(page*PAGE, (page+1)*PAGE);
  document.getElementById("tbody").innerHTML = sl.map(p => `<tr>
    <td class="num">${p.u}</td><td class="num">${p.c.toLocaleString()}</td>
    <td class="num">${p.h ?? "–"}</td>
    <td>${p.o ? '<span class="tag tag-ov">overrated</span>' : p.n ? '<span class="tag tag-un">underrated</span>' : '<span class="tag tag-ok">typical</span>'}</td>
    <td>${p.s}</td><td class="num">${p.a}</td>
    <td><a href="https://arxiv.org/abs/${p.id}" target="_blank" style="color:#3b4bd8;text-decoration:none;">${p.t}</a></td></tr>`).join("");
  const tot = Math.max(1, Math.ceil(rows.length / PAGE));
  document.getElementById("pginfo").textContent = `Page ${page+1}/${tot} · ${rows.length.toLocaleString()} papers`;
  document.getElementById("prev").disabled = page === 0;
  document.getElementById("next").disabled = page >= tot - 1;
}
document.querySelectorAll("th.sortable").forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  if (sortK === k) sortD *= -1; else { sortK = k; sortD = -1; }
  document.querySelectorAll("th.sortable").forEach(t =>
    t.textContent = t.textContent.replace(/ [▾▴]/, "") + (t === th ? (sortD < 0 ? " ▾" : " ▴") : ""));
  apply();
});
["q","t-sub","t-type","t-year"].forEach(id => {
  const el = document.getElementById(id);
  el[el.tagName === "INPUT" ? "oninput" : "onchange"] = apply;
});
document.getElementById("prev").onclick = () => { page--; render(); };
document.getElementById("next").onclick = () => { page++; render(); };
apply();
</script>
</body>
</html>"""

    html = (html_top + html_js
            .replace("__META__", meta_js)
            .replace("__PAPERS__", json.dumps(papers, separators=(",", ":")))
            .replace("__CCDF__", json.dumps(ccdf_payload(), separators=(",", ":")))
            .replace("__LADDER__", ladder_js)
            .replace("__SLOPES__", slopes_js))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"written {OUT} ({len(html)//1024} KB, {len(papers)} papers)")


if __name__ == "__main__":
    main()
