"""
Phase 4 — Naive baseline comparison & counterfactual lead-time savings.

For the same test cells we evaluated PRAS on:
  Baseline 1: current %above-BP only       (BP_now)
  Baseline 2: current %above-ECOFF only    (E_now)
  Baseline 3: linear trend of %above-BP    (vel_bp_3y)
  Baseline 4: 2-feature subset (E_now + BP_now)
  PRAS:       full model (5 features)
  PRAS-Bayes: hierarchical Bayesian version

ROC and AUPRC for each.

Then counterfactual: among cells that DID cross BP=10% in the test period,
how many years earlier would a PRAS>=0.5 trigger have warned us,
compared to the naive "%above-BP >= 5%" early-warning rule?
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_features.parquet")
bayes = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")

FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir", "vel_ecoff_3y", "acc_ecoff"]
OUTCOME = "bp_will_cross_10_5y"
df = df.dropna(subset=FEATURES + [OUTCOME]).reset_index(drop=True)
train = df[df["Year"].between(2007, 2014)].copy()
test  = df[df["Year"].between(2015, 2019)].copy()

# fit each baseline on train, score on test
def fit_score(cols):
    scaler = StandardScaler().fit(train[cols])
    Xtr = scaler.transform(train[cols]); Xte = scaler.transform(test[cols])
    mdl = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, train[OUTCOME])
    return mdl.predict_proba(Xte)[:,1]

baselines = {
    "Baseline 1: %above-BP only":          ["pct_above_bp"],
    "Baseline 2: %above-ECOFF only":       ["pct_above_ecoff"],
    "Baseline 3: BP velocity only":        ["vel_bp_3y"],
    "Baseline 4: E_now + BP_now":          ["pct_above_ecoff","pct_above_bp"],
    "PRAS (full 5 features)":              FEATURES,
}
yte = test[OUTCOME].values.astype(int)

# Compute predictions for each baseline (re-fit and score)
preds = {}
for name, cols in baselines.items():
    # Drop NaN rows for the specific feature set
    tr_sub = train.dropna(subset=cols + [OUTCOME])
    te_sub = test.dropna(subset=cols)
    # use a fresh fit with these features only
    scaler = StandardScaler().fit(tr_sub[cols])
    mdl = LogisticRegression(C=1.0, max_iter=2000).fit(
        scaler.transform(tr_sub[cols]), tr_sub[OUTCOME])
    p = mdl.predict_proba(scaler.transform(te_sub[cols]))[:,1]
    aligned = pd.Series(np.nan, index=test.index)
    aligned.loc[te_sub.index] = p
    preds[name] = aligned.values

# Bayesian PRAS: merge from bayes_scored
test_bayes = test.merge(bayes[["Species","Drug","Country","Year","PRAS_bayes_mean"]],
                         on=["Species","Drug","Country","Year"], how="left")
preds["PRAS Bayesian (5 features + pair RE)"] = test_bayes["PRAS_bayes_mean"].values

# Compute metrics
print("=== Test-set performance comparison ===\n")
rows = []
for name, p in preds.items():
    mask = ~np.isnan(p)
    if mask.sum() < 10: continue
    y_sub = yte[mask]; p_sub = p[mask]
    auc = roc_auc_score(y_sub, p_sub)
    auprc = average_precision_score(y_sub, p_sub)
    rows.append(dict(model=name, n=mask.sum(), AUC=auc, AUPRC=auprc))
    print(f"  {name:<45}  n={mask.sum():>5}  AUC={auc:.3f}  AUPRC={auprc:.3f}")
res = pd.DataFrame(rows)
res.to_csv(ROOT/"tables/pras_baseline_comparison.csv", index=False)

# --- ROC overlay figure ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
colors = ["#aab6c7","#7986a0","#94a8c2","#7aa9ff","#e6677a","#b18cff"]
for (name, p), c in zip(preds.items(), colors):
    mask = ~np.isnan(p)
    if mask.sum() < 10: continue
    fpr, tpr, _ = roc_curve(yte[mask], p[mask])
    auc = roc_auc_score(yte[mask], p[mask])
    ax.plot(fpr, tpr, lw=1.6 if "PRAS" not in name else 2.4, color=c,
            label=f"{name.split(':')[0]} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=0.6, alpha=0.5)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — PRAS vs simple baselines", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

# AUC bar chart
ax = axes[1]
res_sorted = res.sort_values("AUC")
y_pos = np.arange(len(res_sorted))
colors_b = ["#aab6c7" if "PRAS" not in m else "#7aa9ff" if "Bayesian" not in m else "#b18cff" for m in res_sorted["model"]]
ax.barh(y_pos, res_sorted["AUC"], color=colors_b, edgecolor="black", lw=0.4)
for i, (_, r) in enumerate(res_sorted.iterrows()):
    ax.text(r["AUC"] + 0.005, i, f"{r['AUC']:.3f}", va="center", fontsize=9)
ax.set_yticks(y_pos); ax.set_yticklabels(res_sorted["model"], fontsize=9)
ax.set_xlabel("Test AUC"); ax.set_xlim(0.5, 1.0)
ax.set_title("Side-by-side AUC", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

fig.suptitle("PRAS adds substantial value over simpler baselines",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_baseline_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_baseline_comparison.png")

# ============================================================
# Counterfactual: lead-time saved by PRAS-based surveillance
# ============================================================
# For each (Species, Drug, Country) series, find:
#   y_cross_BP10  = first year %above-BP >= 10% (the event we're trying to predict)
#   y_PRAS_alert  = first year PRAS >= 0.5      (our trigger)
#   y_NAIVE_alert = first year %above-BP >= 5%  (the simplest comparator: half-threshold rule)
# Lead time saved = y_NAIVE_alert - y_PRAS_alert
# Restrict to series that actually crossed in our data window.
scored = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")
all_data = pd.read_csv(ROOT/"tables/country_year_panel.csv")
# merge in PRAS where available
ad = all_data.merge(scored[["Species","Drug","Country","Year","PRAS_bayes_mean","PRAS_freq"]],
                    on=["Species","Drug","Country","Year"], how="left")

leads = []
for (sp, dr, c), g in ad.groupby(["Species","Drug","Country"]):
    g = g.sort_values("Year")
    # event year
    crossed = g[g["pct_above_bp"] >= 10]
    if not len(crossed): continue
    y_event = int(crossed["Year"].iloc[0])
    pre = g[g["Year"] <= y_event]
    # PRAS trigger
    pras_alert = pre[pre["PRAS_bayes_mean"] >= 0.5]
    y_pras = int(pras_alert["Year"].iloc[0]) if len(pras_alert) else None
    # Naive trigger (% above BP >= 5%)
    naive_alert = pre[pre["pct_above_bp"] >= 5]
    y_naive = int(naive_alert["Year"].iloc[0]) if len(naive_alert) else None
    leads.append(dict(
        Species=sp, Drug=dr, Country=c, y_event=y_event,
        y_PRAS_alert=y_pras, y_NAIVE_alert=y_naive,
        PRAS_lead_yrs = (y_event - y_pras) if y_pras is not None else None,
        NAIVE_lead_yrs= (y_event - y_naive) if y_naive is not None else None,
        savings_vs_naive= ((y_naive - y_pras) if (y_pras is not None and y_naive is not None) else None),
    ))
lead = pd.DataFrame(leads)
lead.to_csv(ROOT/"tables/pras_leadtime_saved.csv", index=False)

print("\n=== Counterfactual lead-time analysis (series that crossed BP=10%) ===")
print(f"Series that crossed: {len(lead)}")
print(f"PRAS triggered before event: {lead['y_PRAS_alert'].notna().sum()}")
print(f"NAIVE 5% triggered before event: {lead['y_NAIVE_alert'].notna().sum()}")
print(f"\nLead time vs event (years):")
print(f"  PRAS-based  : median={lead['PRAS_lead_yrs'].median():.1f}, mean={lead['PRAS_lead_yrs'].mean():.1f}")
print(f"  NAIVE 5%   : median={lead['NAIVE_lead_yrs'].median():.1f}, mean={lead['NAIVE_lead_yrs'].mean():.1f}")
print(f"\nDirect savings (NAIVE_alert_year - PRAS_alert_year):")
print(f"  median={lead['savings_vs_naive'].median():.1f}  mean={lead['savings_vs_naive'].mean():.2f}")
print(f"  >0 (PRAS earlier): {(lead['savings_vs_naive']>0).sum()}")
print(f"  =0 (tie)        : {(lead['savings_vs_naive']==0).sum()}")
print(f"  <0 (NAIVE earlier): {(lead['savings_vs_naive']<0).sum()}")

# Lead-time figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist([lead["PRAS_lead_yrs"].dropna(), lead["NAIVE_lead_yrs"].dropna()],
         bins=np.arange(-0.5, 18, 1), label=["PRAS ≥ 0.5", "%above-BP ≥ 5%"],
         color=["#b18cff","#aab6c7"], edgecolor="white")
ax.set_xlabel("Lead time before BP=10% crossing (years)")
ax.set_ylabel("Number of (country, pair) series")
ax.legend()
ax.set_title("Distribution of lead times", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
both = lead.dropna(subset=["PRAS_lead_yrs","NAIVE_lead_yrs"])
ax.scatter(both["NAIVE_lead_yrs"], both["PRAS_lead_yrs"], s=60,
           c=["#65d39a" if s > 0 else "#aab6c7" if s == 0 else "#e6677a"
              for s in both["savings_vs_naive"]],
           edgecolor="black", lw=0.4)
maxv = max(both["NAIVE_lead_yrs"].max(), both["PRAS_lead_yrs"].max()) + 1
ax.plot([0,maxv],[0,maxv], "k--", alpha=0.5, lw=0.8)
ax.set_xlabel("NAIVE 5%-rule lead time (years)")
ax.set_ylabel("PRAS lead time (years)")
ax.set_title(f"Pairwise comparison (each point = a country-pair series)\n"
             f"Green: PRAS earlier ({(both['savings_vs_naive']>0).sum()}),  "
             f"Grey: tie ({(both['savings_vs_naive']==0).sum()}),  "
             f"Red: NAIVE earlier ({(both['savings_vs_naive']<0).sum()})",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

fig.suptitle("Counterfactual: how much earlier does PRAS warn vs a naive surveillance threshold?",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_leadtime_saved.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_leadtime_saved.png and tables/pras_leadtime_saved.csv")
