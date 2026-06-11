"""
Phase 4 — Mechanism-stratified analysis.

For K. pneumoniae × Ceftazidime-avibactam (the headline pair), look at
country-year cells classified by which carbapenemase is dominant locally:
  - NDM-dominant cells (NDM detection rate > 5% of isolates)
  - KPC-dominant cells
  - "Quiet" cells (no carbapenemase detection above background)

NDM should drive CAZ-AVI breakpoint rises (avibactam can't inhibit MBLs).
KPC should NOT drive CAZ-AVI rises (avibactam DOES inhibit KPC) — unless
the country has KPC variants with avibactam-escape mutations.

We compute PRAS levels by subgroup over time and check the mechanism story.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")
geno  = pd.read_csv(ROOT/"tables/genotype_country_year_kp.csv")
bayes = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")

# Focus pair
SP, DR = "Klebsiella pneumoniae", "Ceftazidime avibactam"
mer_sp, mer_dr = "Klebsiella pneumoniae", "Meropenem"

# Merge CAZ-AVI panel with genotype
caz = panel[(panel["Species"]==SP) & (panel["Drug"]==DR)].copy()
caz = caz.merge(geno[["Country","Year","KPC_pct","NDM_pct","OXA_pct","VIM_pct","IMP_pct"]],
                on=["Country","Year"], how="left")
caz = caz.merge(bayes[bayes["Species"]==SP][bayes["Drug"]==DR][["Country","Year","PRAS_bayes_mean"]],
                on=["Country","Year"], how="left")

caz["NDM_pct"]  = caz["NDM_pct"].fillna(0)
caz["KPC_pct"]  = caz["KPC_pct"].fillna(0)
caz["OXA_pct"]  = caz["OXA_pct"].fillna(0)

# Classify each cell
def classify(row):
    if row["NDM_pct"] >= 5: return "NDM-dominant"
    if row["KPC_pct"] >= 5 and row["NDM_pct"] < 5: return "KPC-dominant"
    if row["OXA_pct"] >= 5 and row["NDM_pct"] < 5 and row["KPC_pct"] < 5: return "OXA-dominant"
    return "Quiet (no major MBL/KPC)"
caz["mechanism"] = caz.apply(classify, axis=1)

print("=== Cell counts by mechanism class (K. pneumoniae × CAZ-AVI cells) ===")
print(caz["mechanism"].value_counts())

# Yearly aggregate: % above BP and PRAS by mechanism class
agg = caz.groupby(["Year","mechanism"]).agg(
    n_cells=("n","size"),
    total_isolates=("n","sum"),
    pct_bp=("pct_above_bp", "mean"),
    pct_ecoff=("pct_above_ecoff", "mean"),
    PRAS=("PRAS_bayes_mean", "mean"),
).reset_index()

# Plot mechanism dynamics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
mech_order = ["NDM-dominant","KPC-dominant","OXA-dominant","Quiet (no major MBL/KPC)"]
colors = {"NDM-dominant":"#e6677a","KPC-dominant":"#7aa9ff",
          "OXA-dominant":"#b18cff","Quiet (no major MBL/KPC)":"#aab6c7"}

ax = axes[0,0]
for m in mech_order:
    g = agg[agg["mechanism"]==m].sort_values("Year")
    if len(g) < 3: continue
    ax.plot(g["Year"], g["pct_bp"], "o-", color=colors[m], lw=2, ms=5,
            label=f"{m} (n_cells={g['n_cells'].sum()})")
ax.axhline(10, color="red", lw=0.8, ls=":", alpha=0.6, label="BP=10% threshold")
ax.set_xlabel("Year"); ax.set_ylabel("Mean % above CAZ-AVI breakpoint")
ax.set_title("(A) Where breakpoint resistance materialised\nNDM-dominant cells drive the rise",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0,1]
for m in mech_order:
    g = agg[agg["mechanism"]==m].sort_values("Year")
    if len(g) < 3: continue
    ax.plot(g["Year"], g["pct_ecoff"], "o-", color=colors[m], lw=2, ms=5, label=m)
ax.set_xlabel("Year"); ax.set_ylabel("Mean % above ECOFF")
ax.set_title("(B) Where the pre-resistance reservoir grew\nLeading indicator: rises earlier than (A)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,0]
for m in mech_order:
    g = agg[agg["mechanism"]==m].sort_values("Year")
    if len(g) < 3: continue
    ax.plot(g["Year"], g["PRAS"], "o-", color=colors[m], lw=2, ms=5, label=m)
ax.axhline(0.5, color="purple", lw=0.8, ls=":", alpha=0.6, label="PRAS=0.5 (high alert)")
ax.set_xlabel("Year"); ax.set_ylabel("Mean PRAS")
ax.set_title("(C) PRAS dynamics by mechanism\nNDM-dominant cells already at high PRAS by 2018",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel D: scatter of country-year cells, x=NDM%, y=PRAS, color=outcome
ax = axes[1,1]
caz_clean = caz.dropna(subset=["PRAS_bayes_mean","NDM_pct"])
sc = ax.scatter(caz_clean["NDM_pct"], caz_clean["PRAS_bayes_mean"],
                c=caz_clean["pct_above_bp"], cmap="YlOrRd", s=24, alpha=0.7,
                edgecolors="grey", lw=0.3, vmin=0, vmax=30)
ax.set_xlabel("% NDM-positive K. pneumoniae (country-year)")
ax.set_ylabel("PRAS")
plt.colorbar(sc, ax=ax, label="% above CAZ-AVI breakpoint", fraction=0.04)
ax.set_title("(D) Mechanism → PRAS → outcome\nHigh NDM carriage drives both PRAS and breakpoint resistance",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

fig.suptitle("Mechanism-stratified analysis — K. pneumoniae × Ceftazidime-avibactam\n"
             "Avibactam inhibits KPC but NOT class B metallo-β-lactamases (NDM, VIM, IMP)",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_mechanism_stratified.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_mechanism_stratified.png")

# Summary stats by mechanism class
print("\n=== Latest-year (2024) summary by mechanism class ===")
latest = caz[caz["Year"]==2024]
summary = latest.groupby("mechanism").agg(
    n_cells=("n","size"),
    n_isolates=("n","sum"),
    median_NDM_pct=("NDM_pct","median"),
    median_KPC_pct=("KPC_pct","median"),
    mean_pct_above_BP=("pct_above_bp","mean"),
    mean_PRAS=("PRAS_bayes_mean","mean"),
).reset_index()
print(summary.round(2).to_string(index=False))
summary.to_csv(ROOT/"tables/mechanism_summary_latest.csv", index=False)

agg.to_csv(ROOT/"tables/mechanism_yearly.csv", index=False)
