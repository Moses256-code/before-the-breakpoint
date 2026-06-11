"""
Phase 4 — Proper counterfactual: precision vs recall at matched alert rates.

The previous lead-time framing was unfair because the naive "%above-BP >= 5%"
rule fires whenever the same metric we're predicting is already half-way to
the threshold — so it triggers in essentially every series that crosses,
but doesn't actually warn EARLY.

Proper comparison: at a fixed false-alert rate, what's the recall?
At a fixed alert volume, what's the precision and the actionable lead time
(years between alert and event among true positives)?

We pick alert rates such that 5%, 10%, 20% of test cells fire,
then compare PRAS vs naive baselines.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_features.parquet")
bayes = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")
freq = pd.read_parquet(ROOT/"data/pras_scored.parquet")

FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir", "vel_ecoff_3y", "acc_ecoff"]
OUTCOME = "bp_will_cross_10_5y"
df = df.dropna(subset=FEATURES + [OUTCOME]).reset_index(drop=True)

test = df[df["Year"].between(2015, 2019)].copy()
test = test.merge(freq[["Species","Drug","Country","Year","PRAS"]].rename(columns={"PRAS":"PRAS_freq"}),
                  on=["Species","Drug","Country","Year"], how="left")
test = test.merge(bayes[["Species","Drug","Country","Year","PRAS_bayes_mean"]],
                  on=["Species","Drug","Country","Year"], how="left")

# We'll compare 4 scoring functions:
scorers = {
    "PRAS (Bayesian)":            test["PRAS_bayes_mean"].values,
    "PRAS (frequentist)":         test["PRAS_freq"].values,
    "%above-BP (current level)":  test["pct_above_bp"].values,
    "%above-ECOFF (level)":       test["pct_above_ecoff"].values,
}
y = test[OUTCOME].values.astype(int)
n_total = len(y); n_pos = int(y.sum())
base_rate = n_pos / n_total
print(f"Test set: {n_total} cells, {n_pos} positives ({base_rate*100:.1f}%)")

# At each TARGET ALERT RATE (top X% of scores flagged), compute precision and recall
ALERT_RATES = [0.05, 0.10, 0.15, 0.20, 0.30]
rows = []
for name, scores in scorers.items():
    mask = ~np.isnan(scores)
    s = scores[mask]; y_m = y[mask]
    for rate in ALERT_RATES:
        thr = np.quantile(s, 1 - rate)
        pred = (s >= thr).astype(int)
        prec = precision_score(y_m, pred, zero_division=0)
        rec  = recall_score(y_m, pred, zero_division=0)
        f1   = f1_score(y_m, pred, zero_division=0)
        rows.append(dict(method=name, alert_rate=rate, n_alerts=pred.sum(),
                         precision=prec, recall=rec, f1=f1, threshold=thr))
prec_rec = pd.DataFrame(rows)
prec_rec.to_csv(ROOT/"tables/precision_recall_at_alert_rate.csv", index=False)
print("\n=== Precision @ alert rate (top X% of cells flagged) ===")
print(prec_rec.pivot_table(index="alert_rate", columns="method", values="precision").round(3).to_string())
print("\n=== Recall @ alert rate ===")
print(prec_rec.pivot_table(index="alert_rate", columns="method", values="recall").round(3).to_string())

# Proper lead-time analysis: among cells flagged at alert rate = 10%
# how early did each method flag relative to the event?
all_data = pd.read_csv(ROOT/"tables/country_year_panel.csv")
ad = all_data.merge(bayes[["Species","Drug","Country","Year","PRAS_bayes_mean"]],
                    on=["Species","Drug","Country","Year"], how="left")

# Pick a fixed PRAS threshold (0.5 = "high alert"), and a fixed naive threshold
# that produces a similar total flag count
PRAS_THR  = 0.5
NAIVE_THR = ad["pct_above_bp"].quantile(1 - (ad["PRAS_bayes_mean"]>=PRAS_THR).sum() / ad["PRAS_bayes_mean"].notna().sum())
print(f"\nUsing PRAS >= {PRAS_THR} (matched volume by quantile)")
print(f"Equivalent naive threshold: %above-BP >= {NAIVE_THR:.1f}%")

# For each (Species, Drug, Country) series that EVER crossed BP=10%, find:
#   y_event = first year %above-BP >= 10%
#   y_PRAS  = first year PRAS_bayes_mean >= PRAS_THR (must be < y_event)
#   y_NAIV  = first year pct_above_bp >= NAIVE_THR (must be < y_event)
leads = []
for (sp, dr, c), g in ad.groupby(["Species","Drug","Country"]):
    g = g.sort_values("Year")
    crossed = g[g["pct_above_bp"] >= 10]
    if not len(crossed): continue
    y_event = int(crossed["Year"].iloc[0])
    pre = g[g["Year"] < y_event]
    if not len(pre): continue
    pras_alert = pre[pre["PRAS_bayes_mean"] >= PRAS_THR]
    naive_alert = pre[pre["pct_above_bp"] >= NAIVE_THR]
    y_pras = int(pras_alert["Year"].iloc[0]) if len(pras_alert) else None
    y_naiv = int(naive_alert["Year"].iloc[0]) if len(naive_alert) else None
    leads.append(dict(
        Species=sp, Drug=dr, Country=c, y_event=y_event,
        y_PRAS=y_pras, y_NAIVE=y_naiv,
        PRAS_lead = (y_event - y_pras) if y_pras else None,
        NAIVE_lead= (y_event - y_naiv) if y_naiv else None,
    ))
lead = pd.DataFrame(leads)
n_series = len(lead)
n_pras_warned = lead["y_PRAS"].notna().sum()
n_naive_warned = lead["y_NAIVE"].notna().sum()
print(f"\n=== Among {n_series} country-pair series that crossed BP=10% in the data window ===")
print(f"PRAS warned in advance:   {n_pras_warned} ({100*n_pras_warned/n_series:.1f}%)")
print(f"NAIVE warned in advance:  {n_naive_warned} ({100*n_naive_warned/n_series:.1f}%)")
if n_pras_warned:
    print(f"\nLead time when warned (years before event):")
    print(f"  PRAS  median={lead['PRAS_lead'].median():.0f}  mean={lead['PRAS_lead'].mean():.1f}  max={lead['PRAS_lead'].max():.0f}")
if n_naive_warned:
    print(f"  NAIVE median={lead['NAIVE_lead'].median():.0f}  mean={lead['NAIVE_lead'].mean():.1f}  max={lead['NAIVE_lead'].max():.0f}")
both = lead.dropna(subset=["PRAS_lead","NAIVE_lead"])
if len(both):
    print(f"\nDirect comparison ({len(both)} series where both warned):")
    diff = both["PRAS_lead"] - both["NAIVE_lead"]
    print(f"  PRAS earlier than NAIVE: median diff={diff.median():.1f}y  mean={diff.mean():.2f}y")
    print(f"  PRAS earlier in {(diff>0).sum()} series")
    print(f"  Tie       in {(diff==0).sum()} series")
    print(f"  NAIVE earlier in {(diff<0).sum()} series")
lead.to_csv(ROOT/"tables/pras_vs_naive_leadtime.csv", index=False)

# Figure: 4-panel summary
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# A: Precision-at-alert-rate
ax = axes[0,0]
for name in scorers.keys():
    sub = prec_rec[prec_rec["method"]==name]
    style = "o-" if "PRAS" in name else "s--"
    color = "#b18cff" if "Bayes" in name else "#7aa9ff" if "freq" in name else "#aab6c7" if "BP" in name else "#f4a747"
    lw = 2 if "PRAS" in name else 1.2
    ax.plot(sub["alert_rate"], sub["precision"], style, lw=lw, color=color, label=name, ms=7)
ax.axhline(base_rate, color="k", lw=0.6, ls=":", label=f"Base rate ({base_rate*100:.0f}%)")
ax.set_xlabel("Alert rate (top X% of cells flagged)")
ax.set_ylabel("Precision (fraction of alerts that crossed)")
ax.set_title("(A) Precision at fixed alert rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(ALERT_RATES)

# B: Recall-at-alert-rate
ax = axes[0,1]
for name in scorers.keys():
    sub = prec_rec[prec_rec["method"]==name]
    style = "o-" if "PRAS" in name else "s--"
    color = "#b18cff" if "Bayes" in name else "#7aa9ff" if "freq" in name else "#aab6c7" if "BP" in name else "#f4a747"
    lw = 2 if "PRAS" in name else 1.2
    ax.plot(sub["alert_rate"], sub["recall"], style, lw=lw, color=color, label=name, ms=7)
ax.set_xlabel("Alert rate")
ax.set_ylabel("Recall (fraction of crossers caught)")
ax.set_title("(B) Recall at fixed alert rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(ALERT_RATES)

# C: Lead-time histograms
ax = axes[1,0]
data_p = lead["PRAS_lead"].dropna().values
data_n = lead["NAIVE_lead"].dropna().values
ax.hist([data_p, data_n], bins=np.arange(0.5, 17, 1),
        label=[f"PRAS (n={len(data_p)})",f"NAIVE %BP≥{NAIVE_THR:.0f}% (n={len(data_n)})"],
        color=["#b18cff","#aab6c7"], edgecolor="white")
ax.set_xlabel("Lead time (years before BP=10% crossing)")
ax.set_ylabel("Number of country-pair series")
ax.set_title("(C) Lead-time distribution among series that crossed\n(only counting cases where the rule actually fired before the event)",
             fontsize=10, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# D: pairwise direct comparison
ax = axes[1,1]
n_pras_better = n_tie = n_naive_better = 0
if len(both):
    diffs = both["PRAS_lead"] - both["NAIVE_lead"]
    n_pras_better = int((diffs > 0).sum())
    n_tie         = int((diffs == 0).sum())
    n_naive_better= int((diffs < 0).sum())
    ax.scatter(both["NAIVE_lead"], both["PRAS_lead"], s=80,
               c=["#65d39a" if d>0 else "#aab6c7" if d==0 else "#e6677a" for d in diffs],
               edgecolor="black", lw=0.4, alpha=0.8)
    maxv = max(both["NAIVE_lead"].max(), both["PRAS_lead"].max()) + 1
    ax.plot([0,maxv],[0,maxv], "k--", alpha=0.5)
else:
    ax.text(0.5, 0.5, f"At PRAS≥{PRAS_THR} / NAIVE≥{NAIVE_THR:.0f}% (matched alert volume):\n"
                       f"PRAS warned in advance: {n_pras_warned} series\n"
                       f"NAIVE warned in advance: {n_naive_warned} series\n\n"
                       f"No series had BOTH rules warn early —\n"
                       f"so PRAS provides the only early-warning signal\n"
                       f"at this alert volume.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("NAIVE rule lead time (years)")
ax.set_ylabel("PRAS lead time (years)")
ax.set_title(f"(D) Head-to-head ({len(both)} series where both warned)\n"
             f"Green: PRAS earlier ({n_pras_better}),  Grey: tie ({n_tie}),  Red: NAIVE earlier ({n_naive_better})",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

fig.suptitle("Counterfactual surveillance comparison — PRAS vs naive thresholds, matched alert volume",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_counterfactual.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_counterfactual.png")
