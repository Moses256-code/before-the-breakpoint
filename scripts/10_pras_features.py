"""
Phase 3 Step 1 — Feature engineering for the Pre-Resistance Alert Score.

For each (Country, Species, Drug, Year) cell we want predictors of
"future breakpoint exceedance".

Features (computed at "observation year" y, using only data <= y):
  - ecoff_now:      empirical %above-ECOFF in year y
  - bp_now:         empirical %above-breakpoint in year y
  - reservoir:      ecoff_now - bp_now   (the "pre-resistance pool")
  - vel_ecoff_3y:   mean annual change in %above-ECOFF over years y-2..y
  - vel_bp_3y:      same for %above-BP
  - acc_ecoff:      (vel from y-2..y) - (vel from y-5..y-3)
  - n_recent:       average isolates per year (data weight)

Outcome (forward-looking):
  - bp_will_cross_<H>y_at_<T>: 1 if %above-BP exceeds threshold T at any
    year in (y+1 .. y+H), 0 otherwise.

We choose framework-applicable pairs (per Phase 2 lead-time analysis) and
restrict to (country, pair) histories with >=6 years of data.

Output: tables/pras_features.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/claude/atlas")
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")

# Restrict to pairs where the pre-resistance framework actually works (Phase 2 finding)
FRAMEWORK_PAIRS = [
    ("Klebsiella pneumoniae",  "Meropenem"),
    ("Klebsiella pneumoniae",  "Imipenem"),
    ("Klebsiella pneumoniae",  "Ceftazidime avibactam"),
    ("Escherichia coli",       "Meropenem"),
    ("Escherichia coli",       "Ceftazidime avibactam"),
    ("Enterobacter cloacae",   "Meropenem"),
    ("Pseudomonas aeruginosa", "Ceftazidime avibactam"),
]
panel = panel[panel.apply(lambda r: (r["Species"], r["Drug"]) in FRAMEWORK_PAIRS, axis=1)].copy()
panel = panel.sort_values(["Species","Drug","Country","Year"]).reset_index(drop=True)
print(f"Filtered to {len(panel):,} cells across {len(FRAMEWORK_PAIRS)} framework-applicable pairs")
print(f"Country-pair series: {panel.groupby(['Species','Drug','Country']).ngroups}")

# Compute features per (Species, Drug, Country)
def add_dynamics(g):
    g = g.sort_values("Year").copy()
    # Velocity over last 3 years (linear slope of ECOFF / BP fraction)
    def slope(arr):
        n = len(arr)
        if n < 2: return np.nan
        x = np.arange(n)
        return np.polyfit(x, arr, 1)[0]
    g["vel_ecoff_3y"] = g["pct_above_ecoff"].rolling(3, min_periods=2).apply(slope)
    g["vel_bp_3y"]    = g["pct_above_bp"]   .rolling(3, min_periods=2).apply(slope)
    g["vel_ecoff_5y"] = g["pct_above_ecoff"].rolling(5, min_periods=3).apply(slope)
    # Acceleration = ECOFF velocity now minus ECOFF velocity 3y ago
    g["acc_ecoff"]    = g["vel_ecoff_3y"] - g["vel_ecoff_3y"].shift(3)
    # Reservoir
    g["reservoir"]    = g["pct_above_ecoff"] - g["pct_above_bp"]
    # Smoothed level (mean of last 3 years)
    g["ecoff_smooth_3y"] = g["pct_above_ecoff"].rolling(3, min_periods=2).mean()
    g["bp_smooth_3y"]    = g["pct_above_bp"]   .rolling(3, min_periods=2).mean()
    return g

panel = panel.groupby(["Species","Drug","Country"], group_keys=False).apply(add_dynamics)
print(f"After feature engineering: {len(panel):,} rows")

# Forward-looking outcomes — for each row, look ahead H years
def add_outcomes(g, horizons=(3,5), thresholds=(5,10,20)):
    g = g.sort_values("Year").copy()
    for H in horizons:
        future_bp = g["pct_above_bp"].shift(-1).rolling(H, min_periods=1).max()
        # Maximum %above-BP in next H years
        # For each row, look at next H years (excluding current year)
        future_max = []
        for i in range(len(g)):
            window = g["pct_above_bp"].iloc[i+1 : i+1+H]
            future_max.append(window.max() if len(window) else np.nan)
        g[f"future_max_bp_{H}y"] = future_max
        for T in thresholds:
            g[f"bp_will_cross_{T}_{H}y"] = (np.array(future_max) >= T).astype(float)
            g.loc[np.isnan(future_max), f"bp_will_cross_{T}_{H}y"] = np.nan
    return g

panel = panel.groupby(["Species","Drug","Country"], group_keys=False).apply(add_outcomes)

# Save
out = ROOT/"data/pras_features.parquet"
panel.to_parquet(out, index=False)
print(f"\nWrote {out}  ({len(panel):,} rows)")

# Quick summary
print("\nFeature availability (non-null):")
for c in ["vel_ecoff_3y","vel_bp_3y","acc_ecoff","reservoir","ecoff_smooth_3y",
          "future_max_bp_3y","future_max_bp_5y",
          "bp_will_cross_10_3y","bp_will_cross_10_5y"]:
    print(f"  {c:<25} non-null: {panel[c].notna().sum():>5} / {len(panel)}")

# Outcome class balance for the main label
print("\nOutcome distribution (bp_will_cross_10_5y):")
print(panel["bp_will_cross_10_5y"].value_counts(dropna=False).to_string())
