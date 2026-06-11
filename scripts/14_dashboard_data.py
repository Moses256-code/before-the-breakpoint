"""
Prepare all the data slices the polished dashboard needs.

Outputs a single JSON file with structured payloads:
  meta:        global stats (n isolates, n countries, etc., model performance)
  pairs:       list of all 17 pairs with summary stats
  countries:   list of countries with ISO codes, latest-year aggregates
  timeseries:  full (pair, country, year) PRAS-scored series
  alerts:      top alerts, sorted by PRAS, with sparkline series
  global_traj: per-pair global yearly trajectory
  bucket_real: PRAS bucket realization (the killer figure data)
  per_pair_metrics: test-set AUC/AUPRC per pair
  genotype_kp: K. pneumoniae genotype carriage by country-year
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/claude/atlas")
scored = pd.read_parquet(ROOT/"data/pras_scored.parquet")
panel  = pd.read_csv(ROOT/"tables/country_year_panel.csv")
glob_traj = pd.read_csv(ROOT/"tables/global_yearly_trajectory.csv")
bucket = pd.read_csv(ROOT/"tables/pras_bucket_realization.csv")
per_pair = pd.read_csv(ROOT/"tables/pras_per_pair_test_metrics.csv")
geno_kp = pd.read_csv(ROOT/"tables/genotype_country_year_kp.csv")
coeffs = pd.read_csv(ROOT/"tables/pras_coefficients.csv")
lead = pd.read_csv(ROOT/"tables/pras_leadtime_before_crossing.csv")

# ISO 3166-1 alpha-3 codes for choropleth map
COUNTRY_ISO = {
    "Argentina":"ARG","Australia":"AUS","Austria":"AUT","Belgium":"BEL","Brazil":"BRA",
    "Bulgaria":"BGR","Cameroon":"CMR","Canada":"CAN","Chile":"CHL","China":"CHN",
    "Colombia":"COL","Costa Rica":"CRI","Croatia":"HRV","Czech Republic":"CZE",
    "Denmark":"DNK","Dominican Republic":"DOM","Egypt":"EGY","El Salvador":"SLV",
    "Estonia":"EST","Finland":"FIN","France":"FRA","Germany":"DEU","Ghana":"GHA",
    "Greece":"GRC","Guatemala":"GTM","Honduras":"HND","Hong Kong":"HKG","Hungary":"HUN",
    "India":"IND","Indonesia":"IDN","Ireland":"IRL","Israel":"ISR","Italy":"ITA",
    "Ivory Coast":"CIV","Jamaica":"JAM","Japan":"JPN","Jordan":"JOR","Kazakhstan":"KAZ",
    "Kenya":"KEN","Korea, South":"KOR","Kuwait":"KWT","Latvia":"LVA","Lebanon":"LBN",
    "Lithuania":"LTU","Malawi":"MWI","Malaysia":"MYS","Mauritius":"MUS","Mexico":"MEX",
    "Morocco":"MAR","Namibia":"NAM","Netherlands":"NLD","New Zealand":"NZL","Nicaragua":"NIC",
    "Nigeria":"NGA","Norway":"NOR","Oman":"OMN","Pakistan":"PAK","Panama":"PAN",
    "Philippines":"PHL","Poland":"POL","Portugal":"PRT","Puerto Rico":"PRI","Qatar":"QAT",
    "Romania":"ROU","Russia":"RUS","Saudi Arabia":"SAU","Serbia":"SRB","Singapore":"SGP",
    "Slovak Republic":"SVK","Slovenia":"SVN","South Africa":"ZAF","Spain":"ESP",
    "Sweden":"SWE","Switzerland":"CHE","Taiwan":"TWN","Thailand":"THA","Tunisia":"TUN",
    "Turkey":"TUR","Uganda":"UGA","Ukraine":"UKR","United Arab Emirates":"ARE",
    "United Kingdom":"GBR","Venezuela":"VEN","Vietnam":"VNM",
}

def safe_round(x, n=3):
    if pd.isna(x): return None
    return round(float(x), n)

# ---- META ----
iso = pd.read_parquet(ROOT/"data/atlas_isolates.parquet")
meta = dict(
    title="Before the Breakpoint",
    subtitle="Pre-Resistance Alert Dashboard",
    data_source="Pfizer ATLAS, non-USA partition, 2004–2024",
    total_isolates=int(len(iso)),
    n_countries=int(iso["Country"].nunique()),
    n_pairs_modeled=17,
    n_framework_pairs=7,
    year_min=int(iso["Year"].min()),
    year_max=int(iso["Year"].max()),
    model_train_window="2007–2014",
    model_test_window="2015–2019 (outcomes through 2024)",
    test_auc=0.902,
    test_auprc=0.848,
    n_test_cells=1154,
    test_base_rate=0.338,
    risk_buckets=bucket.to_dict(orient="records"),
    per_pair_metrics=per_pair.fillna("").to_dict(orient="records"),
    coefficients=coeffs.to_dict(orient="records"),
)

# ---- PAIRS ----
pair_summary = []
ALL_PAIRS = [
    ("Klebsiella pneumoniae","Meropenem",                "carbapenem",   1, True),
    ("Klebsiella pneumoniae","Imipenem",                 "carbapenem",   1, True),
    ("Klebsiella pneumoniae","Ceftazidime avibactam",    "novel BLI",    3, True),
    ("Escherichia coli","Meropenem",                     "carbapenem",   1, True),
    ("Escherichia coli","Ceftazidime avibactam",         "novel BLI",    3, True),
    ("Enterobacter cloacae","Meropenem",                 "carbapenem",   1, True),
    ("Pseudomonas aeruginosa","Ceftazidime avibactam",   "novel BLI",    3, True),
    ("Pseudomonas aeruginosa","Meropenem",               "carbapenem",   1, False),
    ("Acinetobacter baumannii","Meropenem",              "carbapenem",   1, False),
    ("Escherichia coli","Ceftriaxone",                   "cephalosporin",2, False),
    ("Klebsiella pneumoniae","Ceftriaxone",              "cephalosporin",2, False),
    ("Escherichia coli","Cefepime",                      "cephalosporin",2, False),
    ("Klebsiella pneumoniae","Cefepime",                 "cephalosporin",2, False),
    ("Pseudomonas aeruginosa","Cefepime",                "cephalosporin",2, False),
    ("Klebsiella pneumoniae","Colistin",                 "last-resort",  3, False),
    ("Escherichia coli","Colistin",                      "last-resort",  3, False),
    ("Pseudomonas aeruginosa","Colistin",                "last-resort",  3, False),
]
for sp, dr, klass, tier, in_framework in ALL_PAIRS:
    sub = panel[(panel["Species"]==sp) & (panel["Drug"]==dr)]
    latest_y = sub["Year"].max() if len(sub) else None
    latest = sub[sub["Year"]==latest_y] if len(sub) else pd.DataFrame()
    scored_sub = scored[(scored["Species"]==sp) & (scored["Drug"]==dr)]
    latest_scored = scored_sub[scored_sub["Year"]==scored_sub["Year"].max()] if len(scored_sub) else pd.DataFrame()
    # Global trajectory: list of {year, ecoff, bp}
    gt = glob_traj[(glob_traj["Species"]==sp) & (glob_traj["Drug"]==dr)].sort_values("Year")
    sparkline = [safe_round(v,1) for v in gt["pct_above_bp"].values] if len(gt) else []
    pair_summary.append(dict(
        species=sp, drug=dr, class_=klass, tier=tier, in_framework=in_framework,
        n_total=int(sub["n"].sum()) if len(sub) else 0,
        n_countries=int(sub["Country"].nunique()) if len(sub) else 0,
        year_min=int(sub["Year"].min()) if len(sub) else None,
        year_max=int(latest_y) if pd.notna(latest_y) else None,
        latest_pct_ecoff=safe_round(latest["pct_above_ecoff"].mean(),1) if len(latest) else None,
        latest_pct_bp=safe_round(latest["pct_above_bp"].mean(),1) if len(latest) else None,
        latest_PRAS=safe_round(latest_scored["PRAS"].mean(),3) if len(latest_scored) else None,
        ecoff=safe_round(sub["ECOFF"].iloc[0],3) if len(sub) else None,
        bp=safe_round(sub["R_breakpoint"].iloc[0],3) if len(sub) else None,
        sparkline_bp=sparkline,
        sparkline_years=[int(y) for y in gt["Year"].values] if len(gt) else [],
    ))

# ---- COUNTRIES summary ----
ALL_COUNTRIES = sorted(panel["Country"].unique())
country_summary = []
for c in ALL_COUNTRIES:
    sub = panel[panel["Country"]==c]
    scored_sub = scored[scored["Country"]==c]
    latest_y = int(sub["Year"].max()) if len(sub) else None
    n_pairs = int(sub[["Species","Drug"]].drop_duplicates().shape[0])
    # Most recent average PRAS across this country's framework pairs
    if len(scored_sub):
        latest_scored_y = int(scored_sub["Year"].max())
        ls = scored_sub[scored_sub["Year"]==latest_scored_y]
        avg_pras = safe_round(ls["PRAS"].mean(),3)
        max_pras = safe_round(ls["PRAS"].max(),3)
        n_high_alerts = int((ls["PRAS"]>0.5).sum())
    else:
        avg_pras = max_pras = None
        n_high_alerts = 0
        latest_scored_y = None
    country_summary.append(dict(
        country=c, iso=COUNTRY_ISO.get(c, ""),
        n_pairs_with_data=n_pairs,
        n_total=int(sub["n"].sum()),
        year_min=int(sub["Year"].min()) if len(sub) else None,
        year_max=latest_y,
        latest_scored_year=latest_scored_y,
        avg_PRAS_latest=avg_pras,
        max_PRAS_latest=max_pras,
        n_high_alerts=n_high_alerts,
    ))

# ---- TIME SERIES (the heavy one) ----
# Keep only minimally rounded fields; this is the data that powers all charts
ts = scored[["Species","Drug","Country","Year","n","pct_above_ecoff",
             "pct_above_ecoff_lo","pct_above_ecoff_hi","pct_above_bp",
             "pct_above_bp_lo","pct_above_bp_hi","PRAS","ECOFF","R_breakpoint"]].copy()
# Also bring in pct_above_ecoff/bp for pairs OUTSIDE framework (no PRAS) so dashboard can show them
extras = panel.merge(scored[["Species","Drug","Country","Year","PRAS"]],
                     on=["Species","Drug","Country","Year"], how="left")
extras = extras[extras["PRAS"].isna()].copy()  # rows not in scored
ts_full = pd.concat([
    ts.assign(in_framework=True),
    extras[["Species","Drug","Country","Year","n","pct_above_ecoff",
             "pct_above_ecoff_lo","pct_above_ecoff_hi","pct_above_bp",
             "pct_above_bp_lo","pct_above_bp_hi","ECOFF","R_breakpoint"]].assign(
                 PRAS=np.nan, in_framework=False)
], ignore_index=True)

for c in ["pct_above_ecoff","pct_above_ecoff_lo","pct_above_ecoff_hi",
          "pct_above_bp","pct_above_bp_lo","pct_above_bp_hi","PRAS"]:
    ts_full[c] = ts_full[c].round(3)

ts_records = ts_full.to_dict(orient="records")
# Clean NaN -> null for JSON
import math
def clean(r):
    return {k:(None if (isinstance(v,float) and math.isnan(v)) else v) for k,v in r.items()}
ts_records = [clean(r) for r in ts_records]

# ---- TOP ALERTS ----
latest_scored = scored.sort_values("Year").groupby(
    ["Species","Drug","Country"], as_index=False).last()
latest_scored = latest_scored.sort_values("PRAS", ascending=False)
top_alerts = []
for _, r in latest_scored.head(40).iterrows():
    # 5-year sparkline of pct_above_bp for this country-pair
    series = scored[(scored["Species"]==r["Species"]) &
                    (scored["Drug"]==r["Drug"]) &
                    (scored["Country"]==r["Country"])].sort_values("Year").tail(8)
    spark = [safe_round(v,1) for v in series["pct_above_bp"].values]
    spark_pras = [safe_round(v,3) for v in series["PRAS"].values]
    spark_years = [int(y) for y in series["Year"].values]
    top_alerts.append(dict(
        species=r["Species"], drug=r["Drug"], country=r["Country"], iso=COUNTRY_ISO.get(r["Country"],""),
        year=int(r["Year"]), n=int(r["n"]),
        pct_ecoff=safe_round(r["pct_above_ecoff"],1),
        pct_bp=safe_round(r["pct_above_bp"],1),
        PRAS=safe_round(r["PRAS"],3),
        sparkline_bp=spark, sparkline_pras=spark_pras, sparkline_years=spark_years,
    ))

# ---- BUNDLE ----
payload = dict(
    meta=meta,
    pairs=pair_summary,
    countries=country_summary,
    timeseries=ts_records,
    alerts=top_alerts,
    bucket_realization=bucket.to_dict(orient="records"),
    lead_time=lead.to_dict(orient="records"),
    country_iso=COUNTRY_ISO,
)

OUT = ROOT/"data/dashboard_payload.json"
OUT.write_text(json.dumps(payload, default=str, separators=(",",":")), encoding="utf-8")
size_kb = OUT.stat().st_size/1024
print(f"Wrote {OUT}  ({size_kb:.0f} KB)")
print(f"  meta keys: {list(meta.keys())[:10]}")
print(f"  pairs: {len(pair_summary)}")
print(f"  countries: {len(country_summary)}")
print(f"  timeseries rows: {len(ts_records)}")
print(f"  top alerts: {len(top_alerts)}")
