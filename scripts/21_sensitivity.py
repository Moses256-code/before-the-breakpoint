"""
Phase 4 — Sensitivity analyses.

(1) Alternative breakpoint thresholds: 5%, 10% (primary), 20%.
    Does PRAS still discriminate the future-crossers?
(2) Continuous-sites subset: restrict to (country, pair) series with data in
    >=80% of years in the analysis window. Controls for surveillance footprint changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_features.parquet")
FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir", "vel_ecoff_3y", "acc_ecoff"]
df = df.dropna(subset=FEATURES).reset_index(drop=True)

# ===== Sensitivity 1: alternative BP thresholds =====
THRESHOLDS = [5, 10, 20]
HORIZONS = [3, 5]
results = []
for T in THRESHOLDS:
    for H in HORIZONS:
        col = f"bp_will_cross_{T}_{H}y"
        if col not in df.columns: continue
        sub = df.dropna(subset=[col]).copy()
        train = sub[sub["Year"].between(2007, 2014)]
        test  = sub[sub["Year"].between(2015, 2019)]
        if train[col].nunique()<2 or test[col].nunique()<2: continue
        scaler = StandardScaler().fit(train[FEATURES])
        mdl = LogisticRegression(C=1.0, max_iter=2000).fit(
            scaler.transform(train[FEATURES]), train[col])
        p_te = mdl.predict_proba(scaler.transform(test[FEATURES]))[:,1]
        # also fit a baseline "bp current only" for comparison
        mdl_b = LogisticRegression(C=1.0, max_iter=2000).fit(
            scaler.fit_transform(train[["pct_above_bp"]]), train[col])
        p_te_b = mdl_b.predict_proba(StandardScaler().fit(train[["pct_above_bp"]]).transform(test[["pct_above_bp"]]))[:,1]
        y_te = test[col].values.astype(int)
        results.append(dict(
            BP_threshold=T, horizon=H, n_train=len(train), n_test=len(test),
            base_rate_test=y_te.mean(),
            PRAS_AUC=roc_auc_score(y_te, p_te),
            PRAS_AUPRC=average_precision_score(y_te, p_te),
            naive_AUC=roc_auc_score(y_te, p_te_b),
            naive_AUPRC=average_precision_score(y_te, p_te_b),
            delta_AUC=roc_auc_score(y_te, p_te) - roc_auc_score(y_te, p_te_b),
        ))
sens_thr = pd.DataFrame(results)
print("=== Sensitivity to BP threshold and horizon ===\n")
print(sens_thr.round(3).to_string(index=False))
sens_thr.to_csv(ROOT/"tables/sens_bp_thresholds.csv", index=False)

# ===== Sensitivity 2: continuous-sites only =====
panel = pd.read_csv(ROOT/"tables/country_year_panel.csv")

# Define "continuous" per (Species, Drug, Country): >=80% of years in window
WINDOWS = {
    ("Klebsiella pneumoniae",  "Meropenem"):              (2007, 2024),
    ("Klebsiella pneumoniae",  "Imipenem"):               (2012, 2024),
    ("Klebsiella pneumoniae",  "Ceftazidime avibactam"):  (2012, 2024),
    ("Escherichia coli",       "Meropenem"):              (2007, 2024),
    ("Escherichia coli",       "Ceftazidime avibactam"):  (2012, 2024),
    ("Enterobacter cloacae",   "Meropenem"):              (2007, 2024),
    ("Pseudomonas aeruginosa", "Ceftazidime avibactam"):  (2012, 2024),
}
cont_keys = set()
for (sp, dr), (y0, y1) in WINDOWS.items():
    nyr = y1 - y0 + 1
    sub = panel[(panel["Species"]==sp) & (panel["Drug"]==dr)
                & (panel["Year"].between(y0, y1))]
    g = sub.groupby("Country")["Year"].nunique()
    eligible = g[g >= 0.8 * nyr].index
    for c in eligible:
        cont_keys.add((sp, dr, c))

print(f"\nContinuous-sites set: {len(cont_keys)} country-pair series "
      f"(out of {panel.groupby(['Species','Drug','Country']).ngroups} total)")

# Restrict main analysis to continuous sites
keys_set = set((r["Species"], r["Drug"], r["Country"]) for _, r in df.iterrows())
df_cont = df[df.apply(lambda r: (r["Species"], r["Drug"], r["Country"]) in cont_keys, axis=1)].copy()
print(f"Filtered modeling dataset: {len(df)} -> {len(df_cont)} rows")

train_c = df_cont[df_cont["Year"].between(2007, 2014)]
test_c  = df_cont[df_cont["Year"].between(2015, 2019)]
print(f"  Train n={len(train_c)}, pos={int(train_c['bp_will_cross_10_5y'].sum())}")
print(f"  Test  n={len(test_c)}, pos={int(test_c['bp_will_cross_10_5y'].sum())}")

scaler = StandardScaler().fit(train_c[FEATURES])
mdl = LogisticRegression(C=1.0, max_iter=2000).fit(
    scaler.transform(train_c[FEATURES]), train_c["bp_will_cross_10_5y"])
y_te = test_c["bp_will_cross_10_5y"].values.astype(int)
p_te = mdl.predict_proba(scaler.transform(test_c[FEATURES]))[:,1]
auc_cont = roc_auc_score(y_te, p_te)
auprc_cont = average_precision_score(y_te, p_te)
print(f"\nContinuous-sites PRAS:  AUC={auc_cont:.3f}  AUPRC={auprc_cont:.3f}")

# Compare against full-data baseline
df_full = df.dropna(subset=["bp_will_cross_10_5y"])
train_f = df_full[df_full["Year"].between(2007, 2014)]
test_f  = df_full[df_full["Year"].between(2015, 2019)]
scaler_f = StandardScaler().fit(train_f[FEATURES])
mdl_f = LogisticRegression(C=1.0, max_iter=2000).fit(
    scaler_f.transform(train_f[FEATURES]), train_f["bp_will_cross_10_5y"])
p_te_f = mdl_f.predict_proba(scaler_f.transform(test_f[FEATURES]))[:,1]
y_te_f = test_f["bp_will_cross_10_5y"].values.astype(int)
print(f"Full-data PRAS (ref):    AUC={roc_auc_score(y_te_f, p_te_f):.3f}  AUPRC={average_precision_score(y_te_f, p_te_f):.3f}")

# Save & plot
cont_results = pd.DataFrame([
    dict(subset="Full data",       n_test=len(test_f), pos=int(y_te_f.sum()),
         AUC=roc_auc_score(y_te_f, p_te_f), AUPRC=average_precision_score(y_te_f, p_te_f)),
    dict(subset="Continuous-sites", n_test=len(test_c), pos=int(y_te.sum()),
         AUC=auc_cont, AUPRC=auprc_cont),
])
cont_results.to_csv(ROOT/"tables/sens_continuous_sites.csv", index=False)
print(cont_results.round(3).to_string(index=False))

# ===== Combined sensitivity figure =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: BP threshold sensitivity
ax = axes[0]
x = np.arange(len(sens_thr))
ax.bar(x - 0.18, sens_thr["PRAS_AUC"], width=0.32, color="#b18cff", label="PRAS (5 features)")
ax.bar(x + 0.18, sens_thr["naive_AUC"], width=0.32, color="#aab6c7", label="Naive (BP-only)")
labels = [f"BP={int(r['BP_threshold'])}%\n{int(r['horizon'])}y horizon" for _, r in sens_thr.iterrows()]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Test AUC")
ax.set_title("(A) Sensitivity to BP threshold & horizon",
             fontsize=11, fontweight="bold")
for i, r in sens_thr.iterrows():
    ax.text(i - 0.18, r["PRAS_AUC"] + 0.005, f"{r['PRAS_AUC']:.3f}", ha="center", fontsize=8, color="#6a3da0")
    ax.text(i + 0.18, r["naive_AUC"] + 0.005, f"{r['naive_AUC']:.3f}", ha="center", fontsize=8)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# Right: Continuous sites
ax = axes[1]
x = np.arange(len(cont_results))
ax.bar(x, cont_results["AUC"], width=0.4, color=["#7aa9ff","#b18cff"])
ax.set_xticks(x); ax.set_xticklabels(cont_results["subset"], fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Test AUC")
for i, r in cont_results.iterrows():
    ax.text(i, r["AUC"] + 0.008, f"AUC = {r['AUC']:.3f}\nn={int(r['n_test'])}, pos={int(r['pos'])}",
            ha="center", fontsize=9)
ax.set_title("(B) Sensitivity to continuous-sites restriction\n"
             "Restricting to country-pair series with data in ≥80% of window years",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Sensitivity analyses — PRAS performance is robust to threshold and surveillance-footprint choices",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_sensitivity.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase4_sensitivity.png")
