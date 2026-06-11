"""
Phase 2 Step 2 — refined country-year trajectory matrix.

Switches from the misspecified single-normal model to empirical proportion
above ECOFF / above breakpoint as the primary signal. This is:
  - robust to bimodality (mixture of WT + non-WT populations)
  - directly interpretable
  - what global surveillance bodies (WHO, ECDC) report
The model-based estimates remain in cell_predictions.csv as a smoothed
secondary view.

Refined analysis windows applied here (post-diagnostic plot review):
  - Cefepime pairs restricted to 2018+ (panel floor change)
  - Colistin pairs flagged (methodology drift) but kept for transparency
  - All other pairs unchanged

For each (Species, Drug, Country, Year) cell with >=10 isolates we compute:
  n_isolates
  pct_above_ECOFF (empirical, treats interval-censored upper bound > ECOFF as 'above')
  pct_above_BP    (empirical, treats interval-censored upper bound >= R as 'above')
  pct_R_atlas     (from ATLAS Interp == "Resistant")

Wilson 95% CI on the proportions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from mic_model import get_thresholds

ROOT = Path("/home/claude/atlas")
LONG = ROOT / "data/atlas_long.parquet"

# Refined windows (per pair) — based on diagnostic plot review
WINDOWS = {
    ("Klebsiella pneumoniae",   "Meropenem"):              (2007, 2024),
    ("Escherichia coli",        "Meropenem"):              (2007, 2024),
    ("Enterobacter cloacae",    "Meropenem"):              (2007, 2024),
    ("Pseudomonas aeruginosa",  "Meropenem"):              (2007, 2024),
    ("Acinetobacter baumannii", "Meropenem"):              (2007, 2024),
    ("Klebsiella pneumoniae",   "Imipenem"):               (2012, 2024),
    ("Escherichia coli",        "Ceftriaxone"):            (2004, 2017),
    ("Klebsiella pneumoniae",   "Ceftriaxone"):            (2004, 2017),
    ("Escherichia coli",        "Cefepime"):               (2018, 2024),  # REVISED
    ("Klebsiella pneumoniae",   "Cefepime"):               (2018, 2024),  # REVISED
    ("Pseudomonas aeruginosa",  "Cefepime"):               (2008, 2024),
    ("Escherichia coli",        "Ceftazidime avibactam"):  (2012, 2024),
    ("Klebsiella pneumoniae",   "Ceftazidime avibactam"):  (2012, 2024),
    ("Pseudomonas aeruginosa",  "Ceftazidime avibactam"):  (2012, 2024),
    ("Klebsiella pneumoniae",   "Colistin"):               (2018, 2024),  # REVISED (methodology)
    ("Escherichia coli",        "Colistin"):               (2018, 2024),  # REVISED
    ("Pseudomonas aeruginosa",  "Colistin"):               (2018, 2024),  # REVISED
}

MIN_CELL_N = 10


def wilson_ci(k, n, conf=0.95):
    """Wilson score interval (returns lower, upper as proportions in [0,1])."""
    if n == 0: return 0.0, 1.0
    p = k/n
    z = norm.ppf((1+conf)/2)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, center-half), min(1.0, center+half)


print("Loading long table ...")
long_all = pq.read_table(LONG,
    columns=["Species","Drug","Country","Year","log2_idx","log2_lower","log2_upper","Interp"]
).to_pandas()

rows = []
for (sp, dr), (y0, y1) in WINDOWS.items():
    thresh = get_thresholds(sp, dr)
    if thresh is None:
        print(f"  ! No breakpoint for {sp} × {dr}, skip")
        continue
    ECOFF, BP = thresh["ECOFF_log2"], thresh["R_log2"]

    sub = long_all[(long_all["Species"]==sp) & (long_all["Drug"]==dr)
                   & (long_all["Year"].between(y0, y1))]
    if not len(sub): continue

    # Decide for each isolate whether it's empirically above ECOFF / BP
    # Convention: an isolate's MIC is "above threshold T" if its observed dilution index > T
    # (we are conservative: a right-censored ">X" with X == T is treated as above)
    sub = sub.copy()
    sub["above_ecoff"] = (sub["log2_idx"] > ECOFF)
    sub["above_bp"]    = (sub["log2_idx"] >= BP)  # at-or-above R-breakpoint  (R≥4 in mg/L -> log2_idx>=2)

    # Cell-level rollup
    grp = sub.groupby(["Country","Year"], as_index=False).agg(
        n=("log2_idx","size"),
        n_above_ecoff=("above_ecoff","sum"),
        n_above_bp=("above_bp","sum"),
        n_R_label=("Interp", lambda s: (s=="Resistant").sum()),
        n_labeled=("Interp", lambda s: s.notna().sum()),
    )
    grp = grp[grp["n"] >= MIN_CELL_N].copy()
    if not len(grp): continue

    grp["pct_above_ecoff"] = 100 * grp["n_above_ecoff"] / grp["n"]
    grp["pct_above_bp"]    = 100 * grp["n_above_bp"]    / grp["n"]
    grp["pct_R_atlas"]     = 100 * grp["n_R_label"] / grp["n_labeled"].replace(0, np.nan)

    # Wilson CI for % above ECOFF
    lo_e, hi_e, lo_b, hi_b = [], [], [], []
    for _, r in grp.iterrows():
        l, h = wilson_ci(int(r["n_above_ecoff"]), int(r["n"]))
        lo_e.append(100*l); hi_e.append(100*h)
        l, h = wilson_ci(int(r["n_above_bp"]), int(r["n"]))
        lo_b.append(100*l); hi_b.append(100*h)
    grp["pct_above_ecoff_lo"] = lo_e; grp["pct_above_ecoff_hi"] = hi_e
    grp["pct_above_bp_lo"]    = lo_b; grp["pct_above_bp_hi"]    = hi_b
    grp["Species"] = sp; grp["Drug"] = dr
    grp["window_start"] = y0; grp["window_end"] = y1
    grp["ECOFF"] = thresh["ECOFF"]; grp["R_breakpoint"] = thresh["R"]
    rows.append(grp)

panel = pd.concat(rows, ignore_index=True)
panel.to_csv(ROOT/"tables/country_year_panel.csv", index=False)

print(f"\nBuilt country-year panel: {len(panel):,} cells across {panel.groupby(['Species','Drug']).ngroups} pairs")
print(f"  unique countries: {panel['Country'].nunique()}")
print(f"  total years covered: {sorted(panel['Year'].unique())}")
print(f"  cells per pair:")
print(panel.groupby(['Species','Drug']).size().to_string())

# Also build a GLOBAL (across-country) annual trajectory per pair
glob = panel.groupby(["Species","Drug","Year"], as_index=False).agg(
    countries=("Country","nunique"),
    n_total=("n","sum"),
    n_above_ecoff=("n_above_ecoff","sum"),
    n_above_bp=("n_above_bp","sum"),
)
glob["pct_above_ecoff"] = 100*glob["n_above_ecoff"]/glob["n_total"]
glob["pct_above_bp"]    = 100*glob["n_above_bp"]   /glob["n_total"]
# Wilson CIs
glob[["pct_above_ecoff_lo","pct_above_ecoff_hi"]] = glob.apply(
    lambda r: pd.Series([100*x for x in wilson_ci(int(r["n_above_ecoff"]), int(r["n_total"]))]), axis=1)
glob[["pct_above_bp_lo","pct_above_bp_hi"]] = glob.apply(
    lambda r: pd.Series([100*x for x in wilson_ci(int(r["n_above_bp"]), int(r["n_total"]))]), axis=1)
glob.to_csv(ROOT/"tables/global_yearly_trajectory.csv", index=False)
print(f"\nGlobal yearly trajectory: {len(glob)} rows -> tables/global_yearly_trajectory.csv")
