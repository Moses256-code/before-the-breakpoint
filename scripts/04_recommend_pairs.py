"""
Phase 1 closeout: produce the recommended pair list for Phase 2 modeling
and a summary figure of test-volume-over-time for those pairs.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")

# Tier 1: carbapenem story (the headline)
# Tier 2: ESBL/cephalosporin story  
# Tier 3: novel agents + last-resort
RECOMMENDED = [
    # Tier, Species, Drug, Rationale
    (1, "Klebsiella pneumoniae", "Meropenem",             "Headline. KPC/NDM/OXA-48 era; long window; rich genotype overlay"),
    (1, "Escherichia coli",      "Meropenem",             "Headline. CTX-M then carbapenemase spillover"),
    (1, "Enterobacter cloacae",  "Meropenem",             "AmpC overproducers; meropenem MIC drift is clinically actionable"),
    (1, "Pseudomonas aeruginosa","Meropenem",             "Intrinsic + acquired resistance; OprD loss + carbapenemases"),
    (1, "Acinetobacter baumannii","Meropenem",            "Mostly resistant; tracks how HIGH the upper tail has climbed"),
    (1, "Klebsiella pneumoniae", "Imipenem",              "Complementary carbapenem for triangulation"),
    (1, "Escherichia coli",      "Ertapenem",             "Most sensitive carbapenem for ESBL/AmpC – but only 2yrs, skip"),

    (2, "Escherichia coli",      "Ceftriaxone",           "ESBL marker; 2004-2017 = CTX-M-15 expansion era"),
    (2, "Klebsiella pneumoniae", "Ceftriaxone",           "ESBL marker; companion to E. coli story"),
    (2, "Escherichia coli",      "Cefepime",              "4th-gen cephalosporin; longer window than ceftriaxone"),
    (2, "Klebsiella pneumoniae", "Cefepime",              "AmpC + ESBL story"),
    (2, "Pseudomonas aeruginosa","Cefepime",              "Anti-pseudomonal; drift indicates Por loss/AmpC overprod."),

    (3, "Escherichia coli",      "Ceftazidime avibactam", "Novel BL/BLI; emerging resistance via KPC mutations"),
    (3, "Klebsiella pneumoniae", "Ceftazidime avibactam", "Where CAZ-AVI resistance is most reported"),
    (3, "Pseudomonas aeruginosa","Ceftazidime avibactam", "PsA-specific CAZ-AVI resistance story"),
    (3, "Klebsiella pneumoniae", "Colistin",              "Last-resort; mcr-1 emergence"),
    (3, "Escherichia coli",      "Colistin",              "mcr plasmid spread"),
    (3, "Pseudomonas aeruginosa","Colistin",              "Last-resort PsA; baseline higher than Entero"),
]

rec = pd.DataFrame(RECOMMENDED, columns=["Tier","Species","Drug","Rationale"])

# Join with viability for sample sizes
viab = pd.read_csv(ROOT/"tables/viable_focus.csv")
rec = rec.merge(viab[["Species","Drug","window_start","window_end","n_years","n_total",
                       "median_yearly_n","country_year_cells_ge30",
                       "pct_left","pct_right","pct_exact","VIABLE"]],
                 on=["Species","Drug"], how="left")

# Drop Ertapenem (only 2 yrs, fails viability)
rec_ok = rec[rec["VIABLE"]==True].copy()
rec_ok.to_csv(ROOT/"tables/recommended_pairs.csv", index=False)

print("=" * 100)
print("RECOMMENDED PAIRS FOR PHASE 2 MODELING")
print("=" * 100)
for tier in [1,2,3]:
    t = rec_ok[rec_ok["Tier"]==tier]
    print(f"\n--- Tier {tier} ({len(t)} pairs) ---")
    for _, r in t.iterrows():
        print(f"  {r['Species']:<26} × {r['Drug']:<24}  n={int(r['n_total']):>6,}  "
              f"[{int(r['window_start'])}–{int(r['window_end'])}]  "
              f"left={r['pct_left']:>4.1f}% right={r['pct_right']:>4.1f}%")
        print(f"      → {r['Rationale']}")
    
print(f"\nTotal pairs for modeling: {len(rec_ok)}")
print(f"Dropped (not viable): {rec[rec['VIABLE']!=True]['Species'].tolist()} × {rec[rec['VIABLE']!=True]['Drug'].tolist()}")

# Summary figure: yearly test volume for each recommended pair
print("\nBuilding summary figure ...")
long = pq.read_table(ROOT/"data/atlas_long.parquet",
                     columns=["Species","Drug","Year"]).to_pandas()
pairs = list(zip(rec_ok["Species"], rec_ok["Drug"]))
fig, axes = plt.subplots(len(pairs)//3 + (len(pairs)%3>0), 3, figsize=(15, 1.6*len(pairs)//3 + 3), sharex=True)
axes = axes.flat
yrs_all = sorted(long["Year"].dropna().unique())
for ax, (sp, dr) in zip(axes, pairs):
    sub = long[(long["Species"]==sp) & (long["Drug"]==dr)]
    yc = sub.groupby("Year").size().reindex(yrs_all, fill_value=0)
    # mark in/out of recommended window
    info = rec_ok[(rec_ok["Species"]==sp) & (rec_ok["Drug"]==dr)].iloc[0]
    y0, y1 = info["window_start"], info["window_end"]
    in_win = (yc.index >= y0) & (yc.index <= y1)
    ax.bar(yc.index, yc.values, color=["steelblue" if i else "lightgray" for i in in_win], width=0.85)
    ax.set_title(f"{sp.split()[0][0]}. {' '.join(sp.split()[1:])} × {dr}", fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(labelsize=7)
for ax in axes[len(pairs):]:
    ax.set_visible(False)
fig.suptitle("Recommended pairs — yearly isolates tested.  Blue = inside recommended analysis window; grey = outside (panel change era).", y=1.00, fontsize=11)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase1_recommended_pairs.png", dpi=130)
plt.close()
print(f"  wrote figures/phase1_recommended_pairs.png")
