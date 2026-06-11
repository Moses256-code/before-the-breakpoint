"""
Headline narrative figures from the refined country-year panel.

  fig_A:  Global %above-ECOFF vs %above-BP trajectories for all 17 pairs.
          Each subplot shows ECOFF-exceeding fraction (orange) leading the
          breakpoint-exceeding fraction (crimson) — the "before the breakpoint"
          claim in pictures.
  fig_B:  Lead-time scatter — for each pair, by how many years does ECOFF crossing
          precede breakpoint crossing? (Across countries.)
  fig_C:  K. pneumoniae × Ceftazidime-avibactam country heatmap of % above
          breakpoint — the "winning story" deep-dive.
  fig_D:  K. pneumoniae × Meropenem country heatmap — the carbapenem story.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path("/home/claude/atlas")
glob = pd.read_csv(ROOT/"tables/global_yearly_trajectory.csv")
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")

# Order pairs for consistent display
ORDER = [
    # Tier 1 carbapenem
    ("Klebsiella pneumoniae",   "Meropenem"),
    ("Klebsiella pneumoniae",   "Imipenem"),
    ("Escherichia coli",        "Meropenem"),
    ("Enterobacter cloacae",    "Meropenem"),
    ("Pseudomonas aeruginosa",  "Meropenem"),
    ("Acinetobacter baumannii", "Meropenem"),
    # Tier 2 ESBL/ceph
    ("Klebsiella pneumoniae",   "Ceftriaxone"),
    ("Escherichia coli",        "Ceftriaxone"),
    ("Klebsiella pneumoniae",   "Cefepime"),
    ("Escherichia coli",        "Cefepime"),
    ("Pseudomonas aeruginosa",  "Cefepime"),
    # Tier 3 novel
    ("Klebsiella pneumoniae",   "Ceftazidime avibactam"),
    ("Escherichia coli",        "Ceftazidime avibactam"),
    ("Pseudomonas aeruginosa",  "Ceftazidime avibactam"),
    # Tier 3 colistin (flagged)
    ("Klebsiella pneumoniae",   "Colistin"),
    ("Escherichia coli",        "Colistin"),
    ("Pseudomonas aeruginosa",  "Colistin"),
]

# ============================================================
# Fig A — global trajectories (ECOFF in orange, BP in crimson)
# ============================================================
fig, axes = plt.subplots(6, 3, figsize=(15, 18), sharex=False)
axes = axes.flat
for ax, (sp, dr) in zip(axes, ORDER):
    g = glob[(glob["Species"]==sp) & (glob["Drug"]==dr)].sort_values("Year")
    if not len(g):
        ax.set_visible(False); continue
    ax.fill_between(g["Year"], g["pct_above_ecoff_lo"], g["pct_above_ecoff_hi"],
                    alpha=0.18, color="darkorange")
    ax.plot(g["Year"], g["pct_above_ecoff"], "o-", color="darkorange", lw=1.6, ms=3.5,
            label="% above ECOFF\n(non-wild-type)")
    ax.fill_between(g["Year"], g["pct_above_bp_lo"], g["pct_above_bp_hi"],
                    alpha=0.18, color="crimson")
    ax.plot(g["Year"], g["pct_above_bp"], "s-", color="crimson", lw=1.6, ms=3.5,
            label="% above breakpoint\n(resistant)")
    ax.set_title(f"{sp} × {dr}\n(n={g['n_total'].sum():,}, {g['Year'].min()}–{g['Year'].max()})",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("% of isolates", fontsize=8)
    ax.tick_params(labelsize=8)
    # Annotate the "Colistin: methodology drift" warning
    if dr == "Colistin":
        ax.text(0.02, 0.97, "⚠ Colistin methodology\n changed ~2016", transform=ax.transAxes,
                fontsize=7, va="top", color="firebrick", fontweight="bold")
# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005),
           ncol=2, fontsize=10, frameon=True)
fig.suptitle("Global yearly trajectories — ECOFF-exceeding fraction (non-WT) "
             "leads breakpoint-exceeding fraction (R)",
             y=1.02, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT/"figures/phase2_figA_global_trajectories.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase2_figA_global_trajectories.png")

# ============================================================
# Fig B — Lead-time analysis: for each country-pair, find the first year
# where %above-ECOFF crosses 10% and the first year where %above-BP does.
# Lead time = year_BP_crossed - year_ECOFF_crossed.
# ============================================================
def first_cross(g, col, thr):
    g = g.sort_values("Year")
    above = g[g[col] >= thr]
    return int(above["Year"].min()) if len(above) else None

lead_rows = []
for (sp, dr), g in panel.groupby(["Species","Drug"]):
    if dr == "Colistin": continue  # methodology issue, skip
    # Compute lead times per country
    for country, gc in g.groupby("Country"):
        if len(gc) < 5: continue  # need years
        y_e = first_cross(gc, "pct_above_ecoff", 10)
        y_b = first_cross(gc, "pct_above_bp",    10)
        if y_e is not None and y_b is not None:
            lead_rows.append({"Species": sp, "Drug": dr, "Country": country,
                              "y_ecoff": y_e, "y_bp": y_b,
                              "lead_yrs": y_b - y_e})

lead = pd.DataFrame(lead_rows)
lead.to_csv(ROOT/"tables/lead_time_estimates.csv", index=False)

if len(lead):
    print(f"\nLead-time analysis (country-pair combos with both thresholds crossed): {len(lead)}")
    print(lead.groupby(["Species","Drug"])["lead_yrs"].agg(["count","median","mean","min","max"]).to_string())

    fig, ax = plt.subplots(figsize=(11, 7))
    # Build x ticks: pair labels
    pair_lbls = []
    pair_data = []
    for (sp, dr), gp in lead.groupby(["Species","Drug"]):
        if len(gp) < 5: continue  # need >=5 country-points
        lbl = f"{sp.split()[0][0]}. {' '.join(sp.split()[1:])} ×\n{dr}"
        pair_lbls.append(lbl)
        pair_data.append(gp["lead_yrs"].values)
    bp = ax.boxplot(pair_data, labels=pair_lbls, showfliers=True,
                    patch_artist=True, widths=0.6)
    for box in bp["boxes"]:
        box.set(facecolor="lightsteelblue", edgecolor="steelblue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Lead time (years) — ECOFF crossing → breakpoint crossing", fontsize=10)
    ax.set_title("Pre-resistance signal lead time, by country & pair\n"
                 "Positive = ECOFF-crossing preceded breakpoint-crossing (early warning works)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=35)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT/"figures/phase2_figB_lead_time.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Wrote figures/phase2_figB_lead_time.png")
else:
    print("\nNo lead-time data accumulated (need 10% thresholds crossed in both metrics)")

# ============================================================
# Fig C/D — country heatmaps for the two flagship pairs:
#   K. pneumoniae × Ceftazidime-avibactam   (modern emergence)
#   K. pneumoniae × Meropenem               (carbapenem era)
# ============================================================
def country_heatmap(sp, dr, metric, title, out, n_countries=25):
    g = panel[(panel["Species"]==sp) & (panel["Drug"]==dr)].copy()
    if not len(g):
        print(f"  no data for {sp} × {dr}"); return
    # Pick countries by total n in window
    top = g.groupby("Country")["n"].sum().sort_values(ascending=False).head(n_countries).index.tolist()
    g = g[g["Country"].isin(top)]
    pv = g.pivot_table(index="Country", columns="Year", values=metric, aggfunc="mean")
    # Order countries by mean of the metric (most-affected at top)
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    # Mask sparse cells (cells with n<10 already excluded; but cells absent are NaN)
    fig, ax = plt.subplots(figsize=(13, max(5, 0.32*len(pv))))
    cmap = plt.cm.YlOrRd
    cmap.set_bad("lightgrey")
    im = ax.imshow(pv.values, aspect="auto", cmap=cmap, vmin=0,
                   vmax=max(np.nanpercentile(pv.values, 98), 5))
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(metric.replace("_"," "), fontsize=9)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns.astype(int), rotation=45, fontsize=8)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=8)
    # Annotate values
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i,j]
            if not np.isnan(v):
                tcol = "white" if v > pv.values[~np.isnan(pv.values)].max()*0.5 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=tcol)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out.name}")

country_heatmap(
    "Klebsiella pneumoniae", "Ceftazidime avibactam",
    metric="pct_above_bp",
    title="Klebsiella pneumoniae × Ceftazidime-avibactam — % resistant by country × year\n"
          "Emergence of CAZ-AVI resistance, 2012–2024 (top 25 countries by isolate volume)",
    out=ROOT/"figures/phase2_figC_kp_cazavi_country.png", n_countries=25)

country_heatmap(
    "Klebsiella pneumoniae", "Meropenem",
    metric="pct_above_bp",
    title="Klebsiella pneumoniae × Meropenem — % resistant by country × year\n"
          "Carbapenem-resistant K. pneumoniae landscape, 2007–2024 (top 25 countries)",
    out=ROOT/"figures/phase2_figD_kp_meropenem_country.png", n_countries=25)

# Bonus: also show % above ECOFF for the same K. pneumoniae × CAZ-AVI pair
country_heatmap(
    "Klebsiella pneumoniae", "Ceftazidime avibactam",
    metric="pct_above_ecoff",
    title="Klebsiella pneumoniae × Ceftazidime-avibactam — % above ECOFF (non-wild-type) by country\n"
          "Pre-resistance signal: shows movement out of WT distribution BEFORE breakpoint resistance",
    out=ROOT/"figures/phase2_figE_kp_cazavi_ecoff.png", n_countries=25)

print("\nDone.")
