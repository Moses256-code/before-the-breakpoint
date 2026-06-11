"""
Phase 1 viability table.

Decides, for each (Species, Drug) combination:
  - the recommended analysis window (start_year, end_year) of panel stability
  - whether the combination is VIABLE for longitudinal MIC-drift modeling
       (>=10 years of data in window AND >=30 isolates / year on average
        AND non-degenerate censoring profile)
  - sample size summary

Outputs:
  tables/viable_species_drug.csv     -- ALL combinations, with status flags
  tables/viable_focus.csv            -- subset for the priority pathogen list
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/claude/atlas")
LONG = ROOT / "data/atlas_long.parquet"

# Priority pathogens for the headline analysis
PRIORITY_BUGS = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",
    "Enterobacter cloacae",
    "Klebsiella oxytoca",
    "Serratia marcescens",
    "Staphylococcus aureus",
    "Streptococcus pneumoniae",
    "Enterococcus faecium",
    "Enterococcus faecalis",
    "Haemophilus influenzae",
]

# Recommended stable-panel windows derived from qc02_panel_ranges figure
# Format: drug -> (start, end) inclusive; None means "no usable contiguous window"
PANEL_WINDOWS = {
    "Meropenem":               (2007, 2024),  # floor stabilises -8 then -6 from 2015
    "Imipenem":                (2012, 2024),  # gap 2008-2011, then -5..3
    "Ertapenem":               (2023, 2024),  # only 2 years, not viable longitudinally
    "Doripenem":               (2012, 2017),  # stable single panel
    "Ceftriaxone":             (2004, 2017),  # then ceiling drops to 2, drug largely dropped
    "Ceftazidime":             (2012, 2024),  # pre-2012 erratic
    "Cefepime":                (2008, 2024),  # floor change in 2008
    "Ceftazidime avibactam":   (2012, 2024),
    "Ceftolozane tazobactam":  (2014, 2024),
    "Cefiderocol":             None,          # only 2024
    "Aztreonam":               (2012, 2024),
    "Aztreonam avibactam":     (2012, 2024),
    "Ciprofloxacin":           (2018, 2024),  # major change in 2018
    "Levofloxacin":            (2004, 2024),  # 1 minor change 2018
    "Colistin":                (2014, 2024),
    "Amikacin":                (2012, 2024),  # minor floor change 2012
    "Gentamicin":              (2014, 2024),
    "Piperacillin tazobactam": (2007, 2024),
    "Trimethoprim sulfa":      (2014, 2024),
    "Minocycline":             (2007, 2017),  # dropped 2018
    "Tigecycline":             (2007, 2024),
    "Amoxycillin clavulanate": (2007, 2024),
    "Ampicillin":              (2007, 2024),
    "Ampicillin sulbactam":    (2012, 2017),  # low volume after
    "Ceftaroline":             (2012, 2024),
    "Meropenem vaborbactam":   (2020, 2024),
    "Cefoperazone sulbactam":  (2018, 2024),  # only 7 yrs
    "Cefoxitin":               None,
    "Cefpodoxime":             None,
    "Tetracycline":            None,
}

print("Loading ...")
long = pq.read_table(LONG, columns=["Species","Drug","Year","Country","cens_type"]).to_pandas()

# Restrict to priority bugs for the focus table; keep all bugs for global table
def summarise(group):
    drug = group.name[1]  # group.name = (Species, Drug)
    win = PANEL_WINDOWS.get(drug)
    if win is None:
        return None
    y0, y1 = win
    in_win = group[(group["Year"]>=y0) & (group["Year"]<=y1)]
    n_total = len(in_win)
    yrs = sorted(in_win["Year"].unique())
    n_yrs = len(yrs)
    yr_min, yr_max = (min(yrs), max(yrs)) if yrs else (None, None)
    # Country-year cells with >=30
    cy = in_win.groupby(["Year","Country"]).size()
    cells_ge30 = int((cy>=30).sum())
    yearly_n = in_win.groupby("Year").size()
    median_yearly_n = int(yearly_n.median()) if len(yearly_n) else 0
    # Censoring profile
    cs = in_win["cens_type"].value_counts(normalize=True).to_dict()
    pct_left  = 100*cs.get("left", 0)
    pct_right = 100*cs.get("right", 0)
    pct_exact = 100*cs.get("exact", 0)
    # Viability criteria
    viable = (n_yrs >= 10) and (median_yearly_n >= 30) and (pct_left < 85) and (pct_right < 85)
    return pd.Series({
        "window_start": y0, "window_end": y1,
        "n_years": n_yrs, "yr_min": yr_min, "yr_max": yr_max,
        "n_total": n_total,
        "median_yearly_n": median_yearly_n,
        "country_year_cells_ge30": cells_ge30,
        "pct_left": round(pct_left,1),
        "pct_right": round(pct_right,1),
        "pct_exact": round(pct_exact,1),
        "VIABLE": viable,
    })

print("Building (Species,Drug) viability table ...")
focus = long[long["Species"].isin(PRIORITY_BUGS)]
viab = focus.groupby(["Species","Drug"]).apply(summarise, include_groups=False).dropna(how="all")
viab = viab.reset_index().sort_values(["Species","Drug"])
out = ROOT/"tables/viable_focus.csv"
viab.to_csv(out, index=False)
print(f"  wrote {out}  ({len(viab)} rows)")

# Print summary of viable combinations only
v_only = viab[viab["VIABLE"]==True].sort_values(["Species","n_total"], ascending=[True, False])
print(f"\nVIABLE combinations (n={len(v_only)}):\n")
cols = ["Species","Drug","window_start","window_end","n_years","n_total","median_yearly_n","country_year_cells_ge30","pct_left","pct_right","pct_exact"]
with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 30):
    print(v_only[cols].to_string(index=False))

# Quick aggregate counts
print(f"\nViable combinations per pathogen:")
print(v_only.groupby("Species").size().sort_values(ascending=False).to_string())
print(f"\nViable combinations per drug:")
print(v_only.groupby("Drug").size().sort_values(ascending=False).to_string())
