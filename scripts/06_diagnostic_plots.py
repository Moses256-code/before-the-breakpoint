"""
Diagnostic plot — for each headline pair show:
  (left)  empirical MIC distribution (heatmap of mic_label proportion per year)
  (right) % above ECOFF (empirical) + median (model-based) trajectory

This reveals bimodality, panel-edge effects, and methodology shifts.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from mic_model import get_thresholds

ROOT = Path("/home/claude/atlas")
LONG = ROOT / "data/atlas_long.parquet"
PAIRS = pd.read_csv(ROOT / "tables/recommended_pairs.csv")

long_all = pq.read_table(LONG,
    columns=["Species","Drug","Year","log2_idx","mic_label","cens_type","Interp"]
).to_pandas()

cells = pd.read_csv(ROOT / "tables/cell_predictions.csv")

# All ATLAS dilution labels for the y-axis
all_dils = sorted(long_all["log2_idx"].dropna().unique())
dil_labels = {-10:"≤0.001",-9:"0.002",-8:"0.004",-7:"0.008",-6:"0.015",-5:"0.03",
              -4:"0.06",-3:"0.12",-2:"0.25",-1:"0.5",0:"1",1:"2",2:"4",3:"8",4:"16",
              5:"32",6:"64",7:"128",8:"256"}

def plot_pair(ax_h, ax_t, sp, dr, y0, y1):
    sub = long_all[(long_all["Species"]==sp) & (long_all["Drug"]==dr)
                   & (long_all["Year"].between(y0, y1))]
    if not len(sub): return
    yrs = sorted(sub["Year"].unique())
    dils_used = sorted(sub["log2_idx"].unique())
    # heatmap matrix: % within year that fell on each dilution step (or were censored at boundary)
    mat = np.zeros((len(dils_used), len(yrs)))
    for j, y in enumerate(yrs):
        g = sub[sub["Year"]==y]
        for i, d in enumerate(dils_used):
            mat[i,j] = 100*(g["log2_idx"]==d).sum()/len(g)
    im = ax_h.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                     extent=[yrs[0]-0.5, yrs[-1]+0.5, dils_used[0]-0.5, dils_used[-1]+0.5])
    plt.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02, label="% of isolates")
    ax_h.set_yticks(dils_used)
    ax_h.set_yticklabels([dil_labels.get(d, str(d)) for d in dils_used], fontsize=7)
    ax_h.set_xlabel("Year"); ax_h.set_ylabel("MIC (mg/L)")
    # mark ECOFF and BP if available
    thresh = get_thresholds(sp, dr)
    if thresh:
        ax_h.axhline(thresh["ECOFF_log2"], color="white", lw=1.0, ls="--", alpha=0.8)
        ax_h.axhline(thresh["R_log2"],     color="red",   lw=1.0, ls="-",  alpha=0.7)
        ax_h.text(yrs[0]-0.4, thresh["ECOFF_log2"]+0.1, f"ECOFF={thresh['ECOFF']}",
                  color="white", fontsize=7, va="bottom")
        ax_h.text(yrs[0]-0.4, thresh["R_log2"]+0.1, f"R≥{thresh['R']}",
                  color="red", fontsize=7, va="bottom")
    ax_h.set_title(f"{sp} × {dr} — MIC distribution by year", fontsize=9)

    # right: %above ECOFF (empirical) + %resistant (empirical) over time
    ax_t.grid(alpha=0.3)
    # Empirical % above ECOFF (use upper bound > ECOFF as proxy: any isolate whose log2 upper > ECOFF_log2)
    if thresh:
        ecoff = thresh["ECOFF_log2"]
        bp    = thresh["R_log2"]
        by_yr = []
        for y in yrs:
            g = sub[sub["Year"]==y]
            # an isolate is "above ECOFF" iff its log2_lower >= ECOFF (strict — could underestimate
            # if interval straddles ECOFF; we use idx > ecoff)
            n = len(g)
            above_ecoff = (g["log2_idx"] > ecoff).sum()
            above_bp    = (g["log2_idx"] >= bp).sum()
            # % resistant from Interp
            interps = g["Interp"].dropna()
            pct_R = 100*(interps=="Resistant").sum()/max(len(interps),1)
            by_yr.append((y, 100*above_ecoff/n, 100*above_bp/n, pct_R, n))
        by = pd.DataFrame(by_yr, columns=["Year","pct_above_ecoff","pct_above_bp","pct_R_emp","n"])
        ax_t.plot(by["Year"], by["pct_above_ecoff"], "o-", color="darkorange",
                  label="% above ECOFF (non-WT)", lw=1.5, ms=3)
        ax_t.plot(by["Year"], by["pct_above_bp"],   "s-", color="crimson",
                  label="% above breakpoint", lw=1.5, ms=3)
        ax_t.plot(by["Year"], by["pct_R_emp"],      "x--", color="black",
                  label="% R (ATLAS label)",   lw=0.9, ms=4, alpha=0.7)
        ax_t.set_ylim(0, max(5, by[["pct_above_ecoff","pct_above_bp","pct_R_emp"]].max().max()*1.1))
        ax_t.set_xlabel("Year"); ax_t.set_ylabel("%")
        ax_t.legend(loc="upper left", fontsize=7)
        ax_t.set_title("% non-WT / R over time", fontsize=9)


# Plot Tier 1 pairs (6 pairs in 6 rows, 2 columns: heatmap + trajectory)
tier1 = PAIRS[PAIRS["Tier"]==1].reset_index(drop=True)
fig, axes = plt.subplots(6, 2, figsize=(15, 22))
for i, row in tier1.iterrows():
    plot_pair(axes[i,0], axes[i,1], row["Species"], row["Drug"],
              int(row["window_start"]), int(row["window_end"]))
plt.tight_layout()
plt.savefig(ROOT/"figures/phase2_tier1_diagnostics.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase2_tier1_diagnostics.png")

# Tier 2
tier2 = PAIRS[PAIRS["Tier"]==2].reset_index(drop=True)
fig, axes = plt.subplots(5, 2, figsize=(15, 18))
for i, row in tier2.iterrows():
    plot_pair(axes[i,0], axes[i,1], row["Species"], row["Drug"],
              int(row["window_start"]), int(row["window_end"]))
plt.tight_layout()
plt.savefig(ROOT/"figures/phase2_tier2_diagnostics.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase2_tier2_diagnostics.png")

# Tier 3
tier3 = PAIRS[PAIRS["Tier"]==3].reset_index(drop=True)
fig, axes = plt.subplots(6, 2, figsize=(15, 22))
for i, row in tier3.iterrows():
    plot_pair(axes[i,0], axes[i,1], row["Species"], row["Drug"],
              int(row["window_start"]), int(row["window_end"]))
plt.tight_layout()
plt.savefig(ROOT/"figures/phase2_tier3_diagnostics.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase2_tier3_diagnostics.png")
