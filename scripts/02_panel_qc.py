"""
Phase 1 QC: panel stability over time per drug.

For each (Drug, Year) cell, report:
  n           : isolates tested
  min_idx     : lowest log2 dilution observed (= panel floor)
  max_idx     : highest log2 dilution observed (= panel ceiling)
  pct_left    : % left-censored
  pct_right   : % right-censored
  pct_exact   : % on-step

A drug whose [min_idx, max_idx] range JUMPS between years is a red flag —
any drift signal in that window is partly artefactual.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
LONG = ROOT / "data/atlas_long.parquet"

print("Loading long table ...")
long = pq.read_table(LONG, columns=["Drug","Year","log2_idx","cens_type"]).to_pandas()
print(f"  {len(long):,} rows")

# Per drug-year summary
def cell_summary(g):
    n = len(g)
    return pd.Series({
        "n": n,
        "min_idx": g["log2_idx"].min(),
        "max_idx": g["log2_idx"].max(),
        "pct_left":  100 * (g["cens_type"]=="left").sum()/n,
        "pct_right": 100 * (g["cens_type"]=="right").sum()/n,
        "pct_exact": 100 * (g["cens_type"]=="exact").sum()/n,
    })

print("Computing per drug-year panel summary ...")
dy = long.groupby(["Drug","Year"]).apply(cell_summary, include_groups=False).reset_index()
dy.to_csv(ROOT/"tables/panel_stability_by_drug_year.csv", index=False)
print(f"  wrote tables/panel_stability_by_drug_year.csv ({len(dy)} rows)")

# Per-drug overall: number of distinct (min_idx, max_idx) panels across years
print("\nPanel range stability per drug:")
stab = dy.groupby("Drug").agg(
    yrs=("Year","nunique"),
    total_n=("n","sum"),
    distinct_panels=("min_idx", lambda x: len({(a,b) for a,b in zip(x, dy.loc[x.index,"max_idx"])})),
    min_idx_min=("min_idx","min"),
    min_idx_max=("min_idx","max"),
    max_idx_min=("max_idx","min"),
    max_idx_max=("max_idx","max"),
).reset_index().sort_values("total_n", ascending=False)
print(stab.to_string(index=False))
stab.to_csv(ROOT/"tables/panel_stability_per_drug.csv", index=False)

# Plot: heat-map of test volume per drug-year, with panel-shift markers
drugs_plot = stab["Drug"].head(20).tolist()
yrs = sorted(long["Year"].dropna().unique())
heat = dy.pivot(index="Drug", columns="Year", values="n").reindex(drugs_plot).reindex(columns=yrs)

fig, ax = plt.subplots(figsize=(14, 9))
data = np.log10(heat.values + 1)
im = ax.imshow(data, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(yrs))); ax.set_xticklabels([int(y) for y in yrs], rotation=45)
ax.set_yticks(range(len(drugs_plot))); ax.set_yticklabels(drugs_plot)
cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.set_label("log10(N isolates tested + 1)")
ax.set_title("ATLAS — test volume by drug × year (top 20 drugs)\nGaps reveal when drugs were added/dropped from the panel")
plt.tight_layout()
plt.savefig(ROOT/"figures/qc01_drug_year_volume.png", dpi=130)
plt.close()
print("  wrote figures/qc01_drug_year_volume.png")

# Plot: panel range trajectory (min_idx, max_idx) over years, for key drugs
key = ["Meropenem","Ertapenem","Imipenem","Ceftriaxone","Ceftazidime","Cefepime",
       "Ciprofloxacin","Colistin","Ceftazidime avibactam","Cefiderocol",
       "Piperacillin tazobactam","Levofloxacin","Amikacin"]
fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
for ax, drug in zip(axes.flat, key):
    sub = dy[dy["Drug"]==drug].sort_values("Year")
    if not len(sub):
        ax.set_visible(False); continue
    ax.fill_between(sub["Year"], sub["min_idx"], sub["max_idx"], alpha=0.3, color="steelblue", label="tested range (log2 µg/mL)")
    ax.plot(sub["Year"], sub["min_idx"], ".-", color="steelblue", lw=0.8)
    ax.plot(sub["Year"], sub["max_idx"], ".-", color="steelblue", lw=0.8)
    ax.set_title(drug, fontsize=10)
    ax.grid(alpha=0.3)
for ax in axes.flat[len(key):]:
    ax.set_visible(False)
fig.suptitle("ATLAS panel ranges by year (log2 µg/mL).  A flat band = stable panel;\njumps = panel reformulation, naive drift estimates will be biased over those boundaries.", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/qc02_panel_ranges.png", dpi=130)
plt.close()
print("  wrote figures/qc02_panel_ranges.png")
print("\nDone.")
