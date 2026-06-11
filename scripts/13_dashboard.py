"""
Phase 3 Step 4 — Build the interactive HTML dashboard.

Self-contained, offline-capable (Plotly via CDN). Embeds the country-year
PRAS-scored panel as JSON inside the file. Filters by Species+Drug and
Country; shows %above-ECOFF, %above-BP, and PRAS trajectories.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_scored.parquet")
# Round numbers to keep file small
for c in ["pct_above_ecoff","pct_above_bp","pct_above_ecoff_lo","pct_above_ecoff_hi",
          "pct_above_bp_lo","pct_above_bp_hi","reservoir","vel_ecoff_3y","acc_ecoff","PRAS"]:
    if c in df: df[c] = df[c].round(3)

# Records
records = df[["Species","Drug","Country","Year","n","pct_above_ecoff","pct_above_ecoff_lo",
              "pct_above_ecoff_hi","pct_above_bp","pct_above_bp_lo","pct_above_bp_hi",
              "PRAS","ECOFF","R_breakpoint"]].to_dict(orient="records")
data_json = json.dumps(records, default=str)
print(f"Embedded {len(records):,} records")

# Top-alerts table: most recent year per country-pair with highest PRAS
latest = df.sort_values("Year").groupby(["Species","Drug","Country"], as_index=False).last()
latest = latest.sort_values("PRAS", ascending=False).head(50)
top_alerts = latest[["Species","Drug","Country","Year","n","pct_above_ecoff",
                     "pct_above_bp","PRAS"]].to_dict(orient="records")
top_json = json.dumps(top_alerts, default=str)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Before the Breakpoint — Pre-Resistance Alert Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
         margin: 0; padding: 16px 24px; background: #fafafa; color: #222; }
  h1 { color: #2c3e50; margin: 0 0 4px; font-size: 22px; }
  h1 small { color: #7a8a99; font-size: 13px; font-weight: 400; }
  h2 { color: #2c3e50; margin-top: 30px; font-size: 17px; border-bottom: 1px solid #d0d8e0; padding-bottom: 6px; }
  .controls { background: #fff; border: 1px solid #d0d8e0; border-radius: 6px;
              padding: 12px 16px; margin: 14px 0; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
  label { font-weight: 600; font-size: 13px; }
  select { padding: 5px 10px; font-size: 14px; border: 1px solid #c4ced8; border-radius: 4px;
           background: white; }
  #chart { width: 100%; height: 480px; background: white; border: 1px solid #d0d8e0;
           border-radius: 6px; padding: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; background: white;
          border: 1px solid #d0d8e0; border-radius: 6px; overflow: hidden; }
  th, td { padding: 7px 10px; border-bottom: 1px solid #eaeef2; text-align: left; }
  th { background: #2c3e50; color: white; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pras-high { background: #fdecea; color: #7a1a13; font-weight: 600; }
  .pras-med  { background: #fff4e0; color: #8a4d00; }
  .pras-low  { background: #eef7ee; color: #2a5b2c; }
  .note { font-size: 12px; color: #7a8a99; margin: 6px 0; }
  .legend { font-size: 12px; color: #555; margin-top: 4px; }
  .summary { background: #fff; border: 1px solid #d0d8e0; border-radius: 6px;
             padding: 10px 14px; margin-top: 12px; font-size: 13px; }
</style>
</head>
<body>

<h1>Before the Breakpoint <small>— Pre-Resistance Alert Dashboard</small></h1>
<div class="note">
  Data: Pfizer ATLAS (non-USA, 2004–2024). PRAS = predicted probability that
  the breakpoint-resistance fraction will exceed 10% within the next 5 years,
  trained on years 2007–2014 and validated on 2015–2024
  (out-of-time test AUC = 0.902, AUPRC = 0.848).
</div>

<div class="controls">
  <label>Pathogen × Drug:
    <select id="pair"></select>
  </label>
  <label>Country:
    <select id="country"></select>
  </label>
</div>

<div id="chart"></div>
<div class="legend">
  <strong>Orange = % above ECOFF</strong> (pre-resistance reservoir).
  <strong>Red = % above breakpoint</strong> (clinical resistance).
  <strong>Purple = PRAS</strong> (predicted probability of future BP crossing,
  right axis).
</div>

<div class="summary" id="summary"></div>

<h2>Top 50 active alerts (latest available year per country-pair, sorted by PRAS)</h2>
<table>
  <thead>
    <tr>
      <th>Species</th><th>Drug</th><th>Country</th><th>Year</th><th>n</th>
      <th>% above ECOFF</th><th>% above BP</th><th>PRAS</th>
    </tr>
  </thead>
  <tbody id="alertTable"></tbody>
</table>

<script>
const DATA = ''' + data_json + ''';
const ALERTS = ''' + top_json + ''';

// Build sorted unique lists
const pairs = [...new Set(DATA.map(d => d.Species + " \\u00D7 " + d.Drug))].sort();
const sel_pair = document.getElementById("pair");
pairs.forEach(p => {
  const o = document.createElement("option");
  o.value = p; o.textContent = p;
  sel_pair.appendChild(o);
});
sel_pair.value = "Klebsiella pneumoniae \\u00D7 Ceftazidime avibactam";

const sel_country = document.getElementById("country");

function refreshCountries() {
  const pair = sel_pair.value;
  const [sp, dr] = pair.split(" \\u00D7 ");
  const countries = [...new Set(DATA.filter(d => d.Species === sp && d.Drug === dr).map(d => d.Country))].sort();
  sel_country.innerHTML = "";
  countries.forEach(c => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    sel_country.appendChild(o);
  });
  // default to Greece if available
  if (countries.includes("Greece")) sel_country.value = "Greece";
  else if (countries.length) sel_country.value = countries[0];
}

function drawChart() {
  const pair = sel_pair.value;
  const [sp, dr] = pair.split(" \\u00D7 ");
  const country = sel_country.value;
  const series = DATA.filter(d => d.Species === sp && d.Drug === dr && d.Country === country)
                     .sort((a,b) => a.Year - b.Year);
  if (!series.length) {
    Plotly.purge("chart");
    document.getElementById("summary").innerHTML = "No data for this combination.";
    return;
  }
  const years = series.map(d => d.Year);
  const ecoff = series.map(d => d.pct_above_ecoff);
  const ecoff_lo = series.map(d => d.pct_above_ecoff_lo);
  const ecoff_hi = series.map(d => d.pct_above_ecoff_hi);
  const bp = series.map(d => d.pct_above_bp);
  const bp_lo = series.map(d => d.pct_above_bp_lo);
  const bp_hi = series.map(d => d.pct_above_bp_hi);
  const pras = series.map(d => d.PRAS);

  const traces = [
    // ECOFF CI band
    { x: [...years, ...years.slice().reverse()],
      y: [...ecoff_hi, ...ecoff_lo.slice().reverse()],
      fill: "toself", fillcolor: "rgba(255,140,0,0.15)", line: { color: "transparent" },
      hoverinfo: "skip", showlegend: false, name: "ECOFF CI" },
    // BP CI band
    { x: [...years, ...years.slice().reverse()],
      y: [...bp_hi, ...bp_lo.slice().reverse()],
      fill: "toself", fillcolor: "rgba(220,20,60,0.15)", line: { color: "transparent" },
      hoverinfo: "skip", showlegend: false, name: "BP CI" },
    // ECOFF line
    { x: years, y: ecoff, name: "% above ECOFF (non-WT)",
      mode: "lines+markers", line: { color: "#ff8c00", width: 2 }, marker: { size: 6 } },
    // BP line
    { x: years, y: bp, name: "% above breakpoint (R)",
      mode: "lines+markers", line: { color: "#dc143c", width: 2 }, marker: { size: 6 } },
    // PRAS line on secondary axis
    { x: years, y: pras, name: "PRAS",
      mode: "lines+markers", line: { color: "#6a3da0", width: 2, dash: "dot" },
      marker: { size: 6, color: "#6a3da0" }, yaxis: "y2" },
  ];

  const layout = {
    title: { text: country + " — " + sp + " \\u00D7 " + dr, font: { size: 14 } },
    xaxis: { title: "Year", dtick: 1, tickangle: -45 },
    yaxis: { title: "% of isolates", rangemode: "tozero" },
    yaxis2: { title: { text: "PRAS", font: { color: "#6a3da0" } },
              tickfont: { color: "#6a3da0" },
              overlaying: "y", side: "right", range: [0, 1] },
    legend: { orientation: "h", x: 0.02, y: 1.08 },
    margin: { l: 60, r: 70, t: 70, b: 50 },
    shapes: [
      { type: "line", x0: years[0], x1: years[years.length-1],
        y0: 10, y1: 10, line: { color: "#dc143c", width: 1, dash: "dot" } }
    ],
    annotations: [
      { x: years[0], y: 10, xref: "x", yref: "y", text: "BP-crossing (10%)",
        showarrow: false, font: { size: 10, color: "#dc143c" }, xanchor: "left", yanchor: "bottom" }
    ]
  };
  Plotly.newPlot("chart", traces, layout, { displaylogo: false, responsive: true });

  // Summary text
  const latest = series[series.length - 1];
  const summaryHtml = `
    <strong>Latest year: ${latest.Year}</strong> &nbsp;
    n=${latest.n} isolates &nbsp; | &nbsp;
    % above ECOFF: <strong>${latest.pct_above_ecoff.toFixed(1)}%</strong> &nbsp; | &nbsp;
    % above breakpoint: <strong>${latest.pct_above_bp.toFixed(1)}%</strong> &nbsp; | &nbsp;
    PRAS: <strong style="color:${latest.PRAS > 0.5 ? "#c0392b" : latest.PRAS > 0.25 ? "#d68910" : "#27ae60"}">${latest.PRAS.toFixed(3)}</strong>
    &nbsp;<em>(${latest.PRAS > 0.5 ? "HIGH" : latest.PRAS > 0.25 ? "MODERATE" : "LOW"} alert)</em>
    <br>
    Clinical breakpoint (CLSI): R ≥ ${latest.R_breakpoint} mg/L &nbsp; | &nbsp;
    ECOFF: ${latest.ECOFF} mg/L
  `;
  document.getElementById("summary").innerHTML = summaryHtml;
}

// Populate the alerts table
function populateAlerts() {
  const tbody = document.getElementById("alertTable");
  tbody.innerHTML = "";
  ALERTS.forEach(a => {
    const tr = document.createElement("tr");
    const cls = a.PRAS > 0.5 ? "pras-high" : a.PRAS > 0.25 ? "pras-med" : "pras-low";
    tr.innerHTML = `
      <td>${a.Species}</td>
      <td>${a.Drug}</td>
      <td>${a.Country}</td>
      <td class="num">${a.Year}</td>
      <td class="num">${a.n}</td>
      <td class="num">${a.pct_above_ecoff.toFixed(1)}%</td>
      <td class="num">${a.pct_above_bp.toFixed(1)}%</td>
      <td class="num ${cls}">${a.PRAS.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

sel_pair.addEventListener("change", () => { refreshCountries(); drawChart(); });
sel_country.addEventListener("change", drawChart);
refreshCountries();
drawChart();
populateAlerts();
</script>
</body>
</html>
'''

out = ROOT / "reports/dashboard.html"
out.write_text(HTML, encoding="utf-8")
size_kb = out.stat().st_size / 1024
print(f"Wrote {out}  ({size_kb:.0f} KB)")
