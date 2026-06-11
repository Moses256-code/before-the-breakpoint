"""
Phase 2 Step 3 — genotype overlay.

For the K. pneumoniae × Meropenem and × Ceftazidime-avibactam pairs,
align observed MIC distribution shifts with β-lactamase gene detections.

Key question: in countries with rising MIC drift, do we see rising
KPC / NDM / OXA-48 carriage in the same years?

Outputs:
  tables/genotype_country_year_kp.csv
  figures/phase2_figF_genotype_kp_carbapenem.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
iso = pd.read_parquet(ROOT/"data/atlas_isolates.parquet")
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")

iso["Year"] = pd.to_numeric(iso["Year"], errors="coerce").astype("Int64")
kp = iso[iso["Species"]=="Klebsiella pneumoniae"].copy()

# Carbapenemase + ESBL genes of interest
GENES = ["KPC","NDM","OXA","VIM","IMP","GES","CTX-M-1","CTX-M-2","CTX-M-9","SHV","TEM"]
for g in GENES:
    kp[f"{g}_pos"] = kp[g].notna() & (kp[g].astype(str).str.upper() != "NEGATIVE")

print(f"K. pneumoniae isolates with genotype testing: {len(kp):,}")
print(f"Genotype-tested by year:")
print(kp.groupby("Year")[[f"{g}_pos" for g in GENES]].sum().to_string())

# By country-year: ATLAS only assays genes on screened isolates, so the
# meaningful surveillance metric is (gene-positive / TOTAL isolates of species),
# not (positive / tested). Below 'pct' = absolute detection rate in the country-year.
country_year_geno = []
for (cy, yr), g in kp.groupby(["Country","Year"]):
    if len(g) < 10: continue
    row = dict(Country=cy, Year=int(yr), n=len(g))
    for gene in GENES:
        pos = g[f"{gene}_pos"].sum()
        row[f"{gene}_pos"] = int(pos)
        row[f"{gene}_pct"] = 100*pos/len(g)  # of all isolates of this species
    country_year_geno.append(row)
geno_df = pd.DataFrame(country_year_geno)
geno_df.to_csv(ROOT/"tables/genotype_country_year_kp.csv", index=False)
print(f"\nWrote tables/genotype_country_year_kp.csv  ({len(geno_df)} country-year cells)")

# Global yearly fraction (out of ALL species isolates) - for K. pneumoniae
yearly_geno = (
    kp.groupby("Year")
      .agg(n=("Year","size"),
           KPC_pos=("KPC_pos","sum"),
           NDM_pos=("NDM_pos","sum"),
           OXA_pos=("OXA_pos","sum"),
           VIM_pos=("VIM_pos","sum"),
           IMP_pos=("IMP_pos","sum"),
           CTXM15_pos=("CTX-M-1_pos","sum"))
      .reset_index()
)
for g in ["KPC","NDM","OXA","VIM","IMP","CTXM15"]:
    yearly_geno[f"{g}_pct"] = 100*yearly_geno[f"{g}_pos"] / yearly_geno["n"]

# Plot: MIC drift overlay with genotype detection trends
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: K. pneumoniae × Meropenem global trajectory + carbapenemase genes
ax = axes[0,0]
mer_glob = panel[(panel["Species"]=="Klebsiella pneumoniae") & (panel["Drug"]=="Meropenem")] \
    .groupby("Year").apply(
        lambda g: pd.Series({
            "pct_above_ecoff": 100*g["n_above_ecoff"].sum()/g["n"].sum(),
            "pct_above_bp":    100*g["n_above_bp"].sum()/g["n"].sum(),
        }), include_groups=False).reset_index()
ax.plot(mer_glob["Year"], mer_glob["pct_above_ecoff"], "o-", color="darkorange",
        label="MIC: % above ECOFF", lw=1.7, ms=4)
ax.plot(mer_glob["Year"], mer_glob["pct_above_bp"],    "s-", color="crimson",
        label="MIC: % above breakpoint", lw=1.7, ms=4)
ax2 = ax.twinx()
yg = yearly_geno[yearly_geno["Year"].between(2007, 2024)]
ax2.plot(yg["Year"], yg["KPC_pct"], "^--", color="navy",      label="KPC% (of tested)", lw=1.2, ms=4)
ax2.plot(yg["Year"], yg["NDM_pct"], "v--", color="darkgreen", label="NDM% (of tested)", lw=1.2, ms=4)
ax2.plot(yg["Year"], yg["OXA_pct"], "D--", color="purple",    label="OXA% (of tested)", lw=1.2, ms=4)
ax2.plot(yg["Year"], yg["VIM_pct"], "p--", color="teal",      label="VIM% (of tested)", lw=1.2, ms=3.5)
ax.set_xlabel("Year"); ax.set_ylabel("% above MIC threshold", color="crimson")
ax2.set_ylabel("% of tested isolates carrying gene", color="navy")
ax.set_title("K. pneumoniae × Meropenem (global)\nMIC drift aligns with carbapenemase emergence",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: K. pneumoniae × CAZ-AVI global trajectory + genes
ax = axes[0,1]
cav_glob = panel[(panel["Species"]=="Klebsiella pneumoniae") & (panel["Drug"]=="Ceftazidime avibactam")] \
    .groupby("Year").apply(
        lambda g: pd.Series({
            "pct_above_ecoff": 100*g["n_above_ecoff"].sum()/g["n"].sum(),
            "pct_above_bp":    100*g["n_above_bp"].sum()/g["n"].sum(),
        }), include_groups=False).reset_index()
ax.plot(cav_glob["Year"], cav_glob["pct_above_ecoff"], "o-", color="darkorange",
        label="MIC: % above ECOFF", lw=1.7, ms=4)
ax.plot(cav_glob["Year"], cav_glob["pct_above_bp"],    "s-", color="crimson",
        label="MIC: % above breakpoint", lw=1.7, ms=4)
ax2 = ax.twinx()
yg = yearly_geno[yearly_geno["Year"].between(2012, 2024)]
ax2.plot(yg["Year"], yg["KPC_pct"], "^--", color="navy",      label="KPC%",   lw=1.2, ms=4)
ax2.plot(yg["Year"], yg["NDM_pct"], "v--", color="darkgreen", label="NDM%",   lw=1.2, ms=4)
ax2.plot(yg["Year"], yg["OXA_pct"], "D--", color="purple",    label="OXA%",   lw=1.2, ms=4)
ax.set_xlabel("Year"); ax.set_ylabel("% above MIC threshold", color="crimson")
ax2.set_ylabel("% of tested isolates carrying gene", color="navy")
ax.set_title("K. pneumoniae × Ceftazidime-avibactam (global)\nMIC drift aligns with metallo-β-lactamase rise",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: scatter — country-year cells: % KPC pos vs % above BP for meropenem
ax = axes[1,0]
mer = panel[(panel["Species"]=="Klebsiella pneumoniae") & (panel["Drug"]=="Meropenem")]
mer_geno = mer.merge(geno_df[["Country","Year","KPC_pct","NDM_pct","OXA_pct"]],
                     on=["Country","Year"], how="inner")
mer_geno = mer_geno.dropna(subset=["KPC_pct"])
mer_geno["any_carbapenemase_pct"] = mer_geno[["KPC_pct","NDM_pct","OXA_pct"]].fillna(0).sum(axis=1).clip(upper=100)
ax.scatter(mer_geno["any_carbapenemase_pct"], mer_geno["pct_above_bp"],
           s=np.sqrt(mer_geno["n"])*2.5, alpha=0.35,
           c=mer_geno["Year"], cmap="viridis", edgecolors="gray", lw=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax, fraction=0.04)
cb.set_label("Year", fontsize=8)
# 1:1 reference line
ax.plot([0,100],[0,100], "k--", alpha=0.4, lw=0.8)
ax.set_xlabel("% with KPC / NDM / OXA detected (any carbapenemase)")
ax.set_ylabel("% above meropenem breakpoint (R)")
ax.set_xlim(0, max(mer_geno["any_carbapenemase_pct"].max(), 5)+5)
ax.set_ylim(0, max(mer_geno["pct_above_bp"].max(), 5)+5)
ax.set_title("Country-year alignment:\ncarbapenemase carriage ≈ meropenem resistance",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 4: same for CAZ-AVI (where NDM is biggest mechanism)
ax = axes[1,1]
cav = panel[(panel["Species"]=="Klebsiella pneumoniae") & (panel["Drug"]=="Ceftazidime avibactam")]
cav_geno = cav.merge(geno_df[["Country","Year","KPC_pct","NDM_pct"]],
                     on=["Country","Year"], how="inner")
cav_geno = cav_geno.dropna(subset=["NDM_pct","KPC_pct"], how="all")
ax.scatter(cav_geno["NDM_pct"].fillna(0), cav_geno["pct_above_bp"],
           s=np.sqrt(cav_geno["n"])*2.5, alpha=0.4,
           c=cav_geno["Year"], cmap="viridis", edgecolors="gray", lw=0.3)
cb = plt.colorbar(ax.collections[0], ax=ax, fraction=0.04)
cb.set_label("Year", fontsize=8)
ax.set_xlabel("% with NDM detected")
ax.set_ylabel("% above CAZ-AVI breakpoint")
ax.set_title("NDM is the main driver of CAZ-AVI resistance\n(CAZ-AVI doesn't inhibit metallo-β-lactamases)",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT/"figures/phase2_figF_genotype_kp.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase2_figF_genotype_kp.png")
