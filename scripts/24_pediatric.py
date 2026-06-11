"""
Phase 4l — Pediatric vs adult subgroup analysis.

ATLAS age groups: 0-17, 18-30, 31-60, 61+, Unknown.
Pediatric (0-17): 1,144,097 observations across all drugs.
Adult (18+):    10,092,902 observations.

Questions:
  (1) Are pediatric MIC distributions different from adult at the same time/place?
  (2) Do pediatric trends LEAD adult trends? (pediatric isolates often reflect
       community acquisition; adult often nosocomial)
  (3) For our headline pairs, what's the country-year-pair-age decomposition?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("/home/claude/atlas")

# Recompute the country-year panel BUT stratified by age
print("Loading long table ...")
long_all = pq.read_table(ROOT/"data/atlas_long.parquet",
    columns=["Species","Drug","Country","Year","Age Group","log2_idx","Interp"]).to_pandas()
print(f"  {len(long_all):,} obs")

# Map age to broader buckets
long_all["AgeBucket"] = long_all["Age Group"].map({
    "0 - 17": "Pediatric (0-17)",
    "18 - 30": "Adult",
    "31 - 60": "Adult",
    "61+": "Elderly (61+)",
    "Unknown": "Unknown",
})

# Headline pairs
PAIRS = [
    ("Klebsiella pneumoniae", "Meropenem"),
    ("Klebsiella pneumoniae", "Imipenem"),
    ("Klebsiella pneumoniae", "Ceftazidime avibactam"),
    ("Escherichia coli", "Meropenem"),
    ("Enterobacter cloacae", "Meropenem"),
]
WINDOWS = {
    ("Klebsiella pneumoniae", "Meropenem"):              (2007, 2024),
    ("Klebsiella pneumoniae", "Imipenem"):               (2012, 2024),
    ("Klebsiella pneumoniae", "Ceftazidime avibactam"):  (2012, 2024),
    ("Escherichia coli", "Meropenem"):                   (2007, 2024),
    ("Enterobacter cloacae", "Meropenem"):               (2007, 2024),
}

# CLSI thresholds (mg/L) -> log2_idx
THRESH = {
    ("Klebsiella pneumoniae", "Meropenem"):              dict(ECOFF=-3, BP=2),  # ECOFF=0.125→-3; R≥4→2
    ("Klebsiella pneumoniae", "Imipenem"):               dict(ECOFF=-1, BP=2),
    ("Klebsiella pneumoniae", "Ceftazidime avibactam"):  dict(ECOFF=-1, BP=4),  # ECOFF=0.5→-1; R≥16→4
    ("Escherichia coli", "Meropenem"):                   dict(ECOFF=-4, BP=2),
    ("Enterobacter cloacae", "Meropenem"):               dict(ECOFF=-3, BP=2),
}

# Compute yearly %above-ECOFF, %above-BP by age bucket, for each pair (global aggregate)
rows = []
for sp, dr in PAIRS:
    y0, y1 = WINDOWS[(sp, dr)]
    ECOFF, BP = THRESH[(sp, dr)]["ECOFF"], THRESH[(sp, dr)]["BP"]
    sub = long_all[(long_all["Species"]==sp) & (long_all["Drug"]==dr)
                    & (long_all["Year"].between(y0, y1))]
    if not len(sub): continue
    sub = sub.copy()
    sub["above_ecoff"] = sub["log2_idx"] > ECOFF
    sub["above_bp"]    = sub["log2_idx"] >= BP
    for (yr, age), g in sub.groupby(["Year","AgeBucket"]):
        if age == "Unknown" or len(g) < 30: continue
        rows.append(dict(
            Species=sp, Drug=dr, Year=int(yr), AgeBucket=age,
            n=len(g),
            pct_above_ecoff=100*g["above_ecoff"].sum()/len(g),
            pct_above_bp=100*g["above_bp"].sum()/len(g),
        ))
age_panel = pd.DataFrame(rows)
age_panel.to_csv(ROOT/"tables/age_stratified_yearly.csv", index=False)
print(f"\nAge-stratified yearly panel: {len(age_panel)} rows")

# Stat tests: pediatric vs adult differences in % above ECOFF
print("\n=== Mean %above-ECOFF by age, headline pairs (2018-2024 window for stability) ===")
recent = age_panel[age_panel["Year"]>=2018]
piv = recent.groupby(["Species","Drug","AgeBucket"])[["pct_above_ecoff","pct_above_bp","n"]].agg(
    pct_ecoff_mean=("pct_above_ecoff","mean"),
    pct_bp_mean=("pct_above_bp","mean"),
    n_total=("n","sum"),
).reset_index()
print(piv.round(2).to_string(index=False))

# Lead-time test: for each pair-country, does pediatric %above-ECOFF in year y predict adult %above-ECOFF in year y+2?
# This requires country-level stratification
print("\nComputing country-level age-stratified series for lead-lag test...")
country_age = []
for sp, dr in PAIRS:
    y0, y1 = WINDOWS[(sp, dr)]
    ECOFF = THRESH[(sp, dr)]["ECOFF"]
    sub = long_all[(long_all["Species"]==sp) & (long_all["Drug"]==dr)
                    & (long_all["Year"].between(y0, y1))]
    sub = sub.copy()
    sub["above_ecoff"] = sub["log2_idx"] > ECOFF
    for (cy, yr, age), g in sub.groupby(["Country","Year","AgeBucket"]):
        if age not in ["Pediatric (0-17)","Adult","Elderly (61+)"]: continue
        if len(g) < 15: continue  # need adequate cell size
        country_age.append(dict(
            Species=sp, Drug=dr, Country=cy, Year=int(yr), AgeBucket=age,
            n=len(g),
            pct_above_ecoff=100*g["above_ecoff"].sum()/len(g),
        ))
ca = pd.DataFrame(country_age)
print(f"  Country-year-age cells: {len(ca)}")

# For lead-lag: pivot pediatric vs adult time series per country-pair, cross-correlate
lead_lag = []
for (sp, dr, c), g in ca.groupby(["Species","Drug","Country"]):
    ped = g[g["AgeBucket"]=="Pediatric (0-17)"].sort_values("Year").set_index("Year")["pct_above_ecoff"]
    adt = g[g["AgeBucket"]=="Adult"].sort_values("Year").set_index("Year")["pct_above_ecoff"]
    eld = g[g["AgeBucket"]=="Elderly (61+)"].sort_values("Year").set_index("Year")["pct_above_ecoff"]
    # need overlap >=5 years for both
    overlap_pa = ped.index.intersection(adt.index)
    overlap_pe = ped.index.intersection(eld.index)
    if len(overlap_pa) < 5: continue
    # contemporaneous correlation
    r_pa, _ = stats.pearsonr(ped[overlap_pa], adt[overlap_pa])
    # lag-1 correlation: pediatric at year y predicts adult at year y+1
    if len(overlap_pa) >= 6:
        ped_lag = ped.shift(0).loc[overlap_pa[:-1]]
        adt_fut = adt.shift(0).loc[overlap_pa[1:]]
        if len(ped_lag) >= 4 and ped_lag.std() > 0 and adt_fut.std() > 0:
            r_lag, _ = stats.pearsonr(ped_lag.values, adt_fut.values)
        else: r_lag = None
    else: r_lag = None
    lead_lag.append(dict(Species=sp, Drug=dr, Country=c,
                          n_overlap=len(overlap_pa),
                          r_contemporaneous=r_pa,
                          r_pediatric_lead_1y=r_lag))
ll = pd.DataFrame(lead_lag)
print(f"\nCountry-pair series for lead-lag test: {len(ll)}")
print(f"Mean contemporaneous correlation (pediatric vs adult): {ll['r_contemporaneous'].mean():.3f}")
ll_clean = ll.dropna(subset=["r_pediatric_lead_1y"])
print(f"Mean lag-1 correlation (pediatric_y vs adult_y+1): {ll_clean['r_pediatric_lead_1y'].mean():.3f}  (n={len(ll_clean)})")
# Direct test: lead correlation > contemporaneous?
diff = (ll_clean["r_pediatric_lead_1y"] - ll_clean["r_contemporaneous"])
print(f"Mean (lag-1 corr − contemp corr): {diff.mean():+.3f} ± {diff.std():.3f}")
ll.to_csv(ROOT/"tables/pediatric_lead_test.csv", index=False)

# ============== PLOT ==============
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Panel 1-5: yearly %above-ECOFF by age for each headline pair
for i, (sp, dr) in enumerate(PAIRS):
    ax = axes.flat[i]
    sub = age_panel[(age_panel["Species"]==sp) & (age_panel["Drug"]==dr)]
    colors = {"Pediatric (0-17)":"#65d39a","Adult":"#7aa9ff","Elderly (61+)":"#e6677a"}
    for age, color in colors.items():
        g = sub[sub["AgeBucket"]==age].sort_values("Year")
        if not len(g): continue
        ax.plot(g["Year"], g["pct_above_ecoff"], "o-", color=color, lw=1.8, ms=4,
                label=f"{age} (n={g['n'].sum():,})")
    ax.set_title(f"{sp.split()[0][0]}. {sp.split()[1]} × {dr}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("% above ECOFF")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend(fontsize=8, loc="upper left")

# Panel 6: Cross-pair summary - pediatric vs adult contemporaneous correlation
ax = axes.flat[5]
ll_with_pair = ll.dropna(subset=["r_pediatric_lead_1y"])
ll_with_pair["pair_label"] = ll_with_pair.apply(lambda r: f"{r['Species'].split()[0][0]}.{r['Species'].split()[1][:3]} ×\n{r['Drug'][:8]}", axis=1)
pair_groups = ll_with_pair.groupby("pair_label")
positions = np.arange(len(pair_groups))
data_contemp = [g["r_contemporaneous"].values for _, g in pair_groups]
data_lag = [g["r_pediatric_lead_1y"].values for _, g in pair_groups]
bp1 = ax.boxplot(data_contemp, positions=positions-0.18, widths=0.32, patch_artist=True)
bp2 = ax.boxplot(data_lag,     positions=positions+0.18, widths=0.32, patch_artist=True)
for box in bp1["boxes"]: box.set(facecolor="lightsteelblue")
for box in bp2["boxes"]: box.set(facecolor="lightcoral")
ax.set_xticks(positions)
ax.set_xticklabels(pair_groups.groups.keys(), fontsize=8)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Correlation (pediatric vs adult ECOFF%)")
ax.set_title("Contemporaneous (blue) vs lag-1 (red)\nDoes pediatric lead adult?", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
# Manual legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="lightsteelblue", label="r(ped_y, adult_y)"),
                    Patch(facecolor="lightcoral", label="r(ped_y, adult_{y+1})")],
          fontsize=8, loc="lower right")

fig.suptitle("Pediatric vs adult subgroup analysis — headline pathogen-drug pairs",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_pediatric.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_pediatric.png")

# Print headline equity finding
print("\n=== KEY EQUITY FINDING ===")
print(f"Mean contemporaneous correlation (pediatric vs adult resistance levels): {ll['r_contemporaneous'].mean():.3f}")
print(f"  → pediatric and adult resistance track together at the country-year level")
print(f"Lag-1 correlation (pediatric year y → adult year y+1): {ll_clean['r_pediatric_lead_1y'].mean():.3f}")
print(f"Difference: {diff.mean():+.3f}  (positive means pediatric LEADS adult)")
