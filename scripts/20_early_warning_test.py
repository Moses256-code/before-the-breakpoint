"""
Phase 4 — The TRUE early-warning test.

Restrict to test cells where %above-BP < 5% at the observation year:
these are countries with low current resistance, where naive "watch the
breakpoint number" surveillance gives NO warning. Among them, can PRAS
still identify which will cross BP=10% within the next 5 years?

This is the real test of the framework's contribution beyond the trivial
"the metric is already high" signal.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
df    = pd.read_parquet(ROOT/"data/pras_features.parquet")
freq  = pd.read_parquet(ROOT/"data/pras_scored.parquet")
bayes = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")

FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir", "vel_ecoff_3y", "acc_ecoff"]
OUTCOME = "bp_will_cross_10_5y"
df = df.dropna(subset=FEATURES + [OUTCOME]).reset_index(drop=True)

test = df[df["Year"].between(2015, 2019)].copy()
test = test.merge(freq[["Species","Drug","Country","Year","PRAS"]].rename(columns={"PRAS":"PRAS_freq"}),
                  on=["Species","Drug","Country","Year"], how="left")
test = test.merge(bayes[["Species","Drug","Country","Year","PRAS_bayes_mean","PRAS_bayes_lo","PRAS_bayes_hi"]],
                  on=["Species","Drug","Country","Year"], how="left")

# Restrict to "low baseline" cells
LOW_BP_THRESH = 5.0
low = test[test["pct_above_bp"] < LOW_BP_THRESH].copy()
print(f"Restricted test set: {len(test)} -> {len(low)} cells with %above-BP < {LOW_BP_THRESH}%")
print(f"Of these, future-crossers (BP>=10% within 5y): {int(low[OUTCOME].sum())} ({100*low[OUTCOME].mean():.1f}%)")

# AUC for each method on this restricted set
y = low[OUTCOME].values.astype(int)
methods = {
    "PRAS (Bayesian)":            low["PRAS_bayes_mean"].values,
    "PRAS (frequentist)":         low["PRAS_freq"].values,
    "%above-BP (current level)":  low["pct_above_bp"].values,
    "%above-ECOFF (level)":       low["pct_above_ecoff"].values,
    "ECOFF velocity (3y)":        low["vel_ecoff_3y"].values,
}
print(f"\n=== Performance on LOW-BASELINE subset (these are the genuinely hard cases) ===\n")
rows = []
for name, s in methods.items():
    mask = ~np.isnan(s)
    if mask.sum() < 10 or y[mask].sum() < 5: 
        print(f"  {name:<35}  insufficient data"); continue
    auc = roc_auc_score(y[mask], s[mask])
    auprc = average_precision_score(y[mask], s[mask])
    print(f"  {name:<35}  n={mask.sum():>4}  AUC={auc:.3f}  AUPRC={auprc:.3f}")
    rows.append(dict(method=name, n=mask.sum(), AUC=auc, AUPRC=auprc))
pd.DataFrame(rows).to_csv(ROOT/"tables/early_warning_low_baseline.csv", index=False)

# Plot: ROC on low-baseline test cells + side-by-side examples
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
colors = {"PRAS (Bayesian)":"#b18cff", "PRAS (frequentist)":"#7aa9ff",
          "%above-BP (current level)":"#aab6c7", "%above-ECOFF (level)":"#f4a747",
          "ECOFF velocity (3y)":"#65d39a"}
for name, s in methods.items():
    mask = ~np.isnan(s)
    if mask.sum() < 10 or y[mask].sum() < 5: continue
    fpr, tpr, _ = roc_curve(y[mask], s[mask])
    auc = roc_auc_score(y[mask], s[mask])
    lw = 2.4 if "PRAS" in name else 1.3
    ax.plot(fpr, tpr, lw=lw, color=colors[name], label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=0.6, alpha=0.5)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC — predicting BP=10% crossing among cells with %above-BP < {LOW_BP_THRESH}%\n"
             f"This is the actual early-warning challenge", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

# Right panel: example country trajectories showing PRAS picking up the signal early
ax = axes[1]
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")
panel = panel.merge(bayes[["Species","Drug","Country","Year","PRAS_bayes_mean"]],
                    on=["Species","Drug","Country","Year"], how="left")
# Show 4 worked examples: cells where PRAS was high but %above-BP was still <5%
examples = [
    ("Klebsiella pneumoniae","Ceftazidime avibactam","Argentina"),
    ("Klebsiella pneumoniae","Ceftazidime avibactam","South Africa"),
    ("Klebsiella pneumoniae","Ceftazidime avibactam","Brazil"),
    ("Pseudomonas aeruginosa","Ceftazidime avibactam","India"),
]
colors_ex = ["#b18cff","#4cc9c1","#f4a747","#e6677a"]
for (sp, dr, c), col in zip(examples, colors_ex):
    g = panel[(panel["Species"]==sp) & (panel["Drug"]==dr) & (panel["Country"]==c)].sort_values("Year")
    if not len(g): continue
    # primary axis: %above-BP
    ax.plot(g["Year"], g["pct_above_bp"], "s-", color=col, lw=1.2, ms=4, alpha=0.5)
    # also PRAS scaled by 50 to overlay
    ax.plot(g["Year"], g["PRAS_bayes_mean"]*50, "o-", color=col, lw=2, ms=5,
            label=f"{c}: {sp.split()[0][0]}.{sp.split()[1][:3]} × {dr.split()[0][:5]}")
ax.axhline(10, color="red", lw=0.8, ls=":", alpha=0.6)
ax.axhline(5*50/100, color="purple", lw=0.8, ls=":", alpha=0.6)
ax.text(2009, 11, "BP crossing (%above-BP=10%)", color="red", fontsize=8)
ax.text(2009, 26, "PRAS=0.5 (scaled)", color="purple", fontsize=8)
ax.set_ylabel("% above BP (squares) /  PRAS × 50 (circles)")
ax.set_xlabel("Year")
ax.set_title("Worked examples: PRAS rises while %above-BP is still low\n"
             "Solid circles = PRAS (scaled), open squares = %above-BP",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

fig.suptitle("Early-warning test — does PRAS work where the breakpoint metric is still asleep?",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_early_warning.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_early_warning.png")

# Concrete "PRAS picks the future crossers" worked examples
print("\n=== Cells where PRAS was elevated (>0.3) while %above-BP was still <5% ===")
hits = low[(low[OUTCOME]==1) & (low["PRAS_bayes_mean"]>0.3)].copy()
hits = hits[["Species","Drug","Country","Year","pct_above_ecoff","pct_above_bp",
             "PRAS_bayes_mean","PRAS_bayes_lo","PRAS_bayes_hi","future_max_bp_5y"]].sort_values("PRAS_bayes_mean", ascending=False)
print(f"\nTotal: {len(hits)} cells")
print(hits.head(20).to_string(index=False))
hits.to_csv(ROOT/"tables/early_warning_hits.csv", index=False)
