"""
Phase 3 Step 3 — worked example & lead-time analysis.

Story: For K. pneumoniae × Ceftazidime-avibactam, four countries
(India, Greece, Argentina, Turkey) crossed the 10% breakpoint-resistance
threshold AFTER 2018. The PRAS, computed using only data <= 2017,
should already have been elevated for them.

We show:
  (a) PRAS trajectories for those 4 countries vs control countries that
      never crossed (Germany, UK, France, Australia).
  (b) Lead-time figure: for every test-set cell that became a "BP crosser"
      (positive outcome), how high was the PRAS in each of the 5 years
      preceding the crossing?
  (c) Generalized worked-example: same story across the meropenem pairs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
scored = pd.read_parquet(ROOT/"data/pras_scored.parquet")
print(f"Loaded scored data: {len(scored):,} rows")

# ===== (a) CAZ-AVI worked example =====
HOTSPOTS_KP_CAZAVI = ["India", "Greece", "Argentina", "Turkey", "Brazil", "South Africa"]
CONTROLS_KP_CAZAVI = ["Germany", "United Kingdom", "France", "Australia", "Spain", "Canada"]

caz = scored[(scored["Species"]=="Klebsiella pneumoniae") &
             (scored["Drug"]=="Ceftazidime avibactam")].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel A: %above-BP trajectories
ax = axes[0,0]
for c in HOTSPOTS_KP_CAZAVI:
    g = caz[caz["Country"]==c].sort_values("Year")
    if len(g):
        ax.plot(g["Year"], g["pct_above_bp"], "o-", lw=2, ms=5, label=c, alpha=0.85)
for c in CONTROLS_KP_CAZAVI:
    g = caz[caz["Country"]==c].sort_values("Year")
    if len(g):
        ax.plot(g["Year"], g["pct_above_bp"], "--", lw=1, color="gray", alpha=0.4)
ax.axhline(10, color="red", lw=1, ls=":", alpha=0.6)
ax.text(2012, 11, "BP-crossing threshold (10%)", color="red", fontsize=8)
ax.axvspan(2007, 2014, color="lightblue", alpha=0.15, label="TRAIN window")
ax.axvspan(2015, 2019, color="lightyellow", alpha=0.25, label="TEST window")
ax.set_title("(A) Observed % above breakpoint over time\nK. pneumoniae × Ceftazidime-avibactam",
             fontsize=11, fontweight="bold")
ax.set_ylabel("% above breakpoint"); ax.set_xlabel("Year")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8, ncol=2)

# Panel B: PRAS trajectories (the prediction)
ax = axes[0,1]
for c in HOTSPOTS_KP_CAZAVI:
    g = caz[caz["Country"]==c].sort_values("Year")
    if len(g):
        ax.plot(g["Year"], g["PRAS"], "o-", lw=2, ms=5, label=c, alpha=0.85)
for c in CONTROLS_KP_CAZAVI:
    g = caz[caz["Country"]==c].sort_values("Year")
    if len(g):
        ax.plot(g["Year"], g["PRAS"], "--", lw=1, color="gray", alpha=0.4)
ax.axhline(0.5, color="red", lw=1, ls=":", alpha=0.6)
ax.text(2012, 0.52, "High-alert threshold (0.5)", color="red", fontsize=8)
ax.axvspan(2007, 2014, color="lightblue", alpha=0.15)
ax.axvspan(2015, 2019, color="lightyellow", alpha=0.25)
ax.set_title("(B) PRAS — Pre-Resistance Alert Score\nFires BEFORE %above-BP rises",
             fontsize=11, fontweight="bold")
ax.set_ylabel("PRAS  (probability of BP crossing within 5y)")
ax.set_xlabel("Year"); ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8, ncol=2)

# ===== (b) Lead-time on test set =====
test = scored[scored["Year"].between(2015, 2019)].copy()
test["crossed"] = test["bp_will_cross_10_5y"]==1
# For each (Species, Drug, Country) series that has at least one positive year,
# find the FIRST year %above-bp actually crossed 10%
def find_first_crossing(g):
    g = g.sort_values("Year")
    above = g[g["pct_above_bp"] >= 10]
    return int(above["Year"].iloc[0]) if len(above) else None

crossings = []
for (sp, dr, cy), g in scored.groupby(["Species","Drug","Country"]):
    fc = find_first_crossing(g)
    if fc is None: continue
    # need PRAS values for fc-5 ... fc-1
    history = g[g["Year"] < fc].sort_values("Year").tail(5)
    if len(history) < 3: continue
    for offset in [1,2,3,4,5]:
        yr = fc - offset
        row = g[g["Year"]==yr]
        if not len(row): continue
        crossings.append(dict(
            Species=sp, Drug=dr, Country=cy,
            first_cross_year=fc, years_before=offset,
            PRAS=row["PRAS"].iloc[0],
            pct_above_ecoff=row["pct_above_ecoff"].iloc[0],
            pct_above_bp=row["pct_above_bp"].iloc[0],
        ))
cd = pd.DataFrame(crossings)

ax = axes[1,0]
# Box plot of PRAS by years_before across all framework pairs that crossed
data = [cd[cd["years_before"]==o]["PRAS"].values for o in [5,4,3,2,1]]
labels_b = [f"{o} yrs\nbefore" for o in [5,4,3,2,1]]
bp = ax.boxplot(data, tick_labels=labels_b, patch_artist=True,
                showfliers=True, widths=0.65, medianprops=dict(color="red", lw=1.5))
for box in bp["boxes"]:
    box.set(facecolor="lightcoral", edgecolor="darkred", alpha=0.6)
ax.axhline(0.5, color="darkred", lw=1, ls=":", alpha=0.7)
ax.set_ylabel("PRAS"); ax.set_xlabel("Time prior to BP crossing")
ax.set_title("(C) PRAS rises in the years BEFORE breakpoint crossing\n"
             f"All (country, pair) cells that eventually crossed (n={cd['Country'].nunique()} countries)",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)
print(f"\nLead-time analysis: {len(cd)} pre-crossing observations across "
      f"{cd[['Species','Drug','Country']].drop_duplicates().shape[0]} country-pair series")
print(cd.groupby("years_before")["PRAS"].agg(["mean","median","std"]).round(3).to_string())

# Panel D: Top alerts in test set that actually crossed
# Find country-pair cells in test set where PRAS > 0.5 -> what fraction crossed?
ax = axes[1,1]
test_complete = test.dropna(subset=["bp_will_cross_10_5y"])
buckets = pd.cut(test_complete["PRAS"], bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
                  labels=["0.0–0.1","0.1–0.25","0.25–0.5","0.5–0.75","0.75–1.0"])
hit_rate = test_complete.groupby(buckets, observed=True).agg(
    n=("PRAS","size"),
    pct_crossed=("bp_will_cross_10_5y", lambda s: 100*s.mean()),
).reset_index()
hit_rate.columns = ["bucket","n","pct_crossed"]
print(f"\n=== Alert bucket realization (test set) ===")
print(hit_rate.to_string(index=False))

x = np.arange(len(hit_rate))
bars = ax.bar(x, hit_rate["pct_crossed"],
              color=["lightblue","steelblue","orange","tomato","crimson"],
              edgecolor="k", lw=0.6)
for i, (n, p) in enumerate(zip(hit_rate["n"], hit_rate["pct_crossed"])):
    ax.text(i, p+1.5, f"n={n}\n{p:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(hit_rate["bucket"], fontsize=9)
ax.set_ylabel("% that crossed BP within 5 yrs (observed)")
ax.set_xlabel("PRAS bucket")
ax.set_title("(D) PRAS bucket → observed crossing rate (test set)\nWell-stratified risk gradient",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, max(hit_rate["pct_crossed"])+15)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Pre-Resistance Alert Score: worked example & lead-time validation\n"
             "K. pneumoniae × Ceftazidime-avibactam as case study + cross-pair lead-time analysis",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase3_pras_worked_example.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase3_pras_worked_example.png")

# Save the lead-time data
cd.to_csv(ROOT/"tables/pras_leadtime_before_crossing.csv", index=False)
hit_rate.to_csv(ROOT/"tables/pras_bucket_realization.csv", index=False)

# Print the specific CAZ-AVI hotspot story
print("\n=== K. pneumoniae × Ceftazidime-avibactam — pre-2018 PRAS for future hotspots ===")
caz_pre = caz[caz["Year"].between(2014, 2017)].sort_values(["Country","Year"])
for c in HOTSPOTS_KP_CAZAVI:
    g = caz_pre[caz_pre["Country"]==c]
    if not len(g): continue
    p_mean = g["PRAS"].mean()
    bp_now = g["pct_above_bp"].mean()
    # When did this country actually cross BP=10?
    actual = caz[(caz["Country"]==c) & (caz["pct_above_bp"]>=10)]
    cross_yr = int(actual["Year"].min()) if len(actual) else None
    print(f"  {c:<18}  mean PRAS 2014–17={p_mean:.3f}  mean %above-BP 2014–17={bp_now:.1f}%  "
          f"first crossed BP={cross_yr}")
print("\nControls (countries that never crossed BP=10%):")
for c in CONTROLS_KP_CAZAVI:
    g = caz_pre[caz_pre["Country"]==c]
    if not len(g): continue
    p_mean = g["PRAS"].mean()
    bp_now = g["pct_above_bp"].mean()
    actual = caz[(caz["Country"]==c) & (caz["pct_above_bp"]>=10)]
    cross_yr = int(actual["Year"].min()) if len(actual) else "never"
    print(f"  {c:<18}  mean PRAS 2014–17={p_mean:.3f}  mean %above-BP 2014–17={bp_now:.1f}%  "
          f"first crossed BP={cross_yr}")
