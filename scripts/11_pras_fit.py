"""
Phase 3 Step 2 — Pre-Resistance Alert Score (PRAS): fit + temporal validation.

Outcome:  bp_will_cross_10_5y  = does %above-breakpoint exceed 10% within
          the next 5 years? (binary)
Predictors (computed using only data <= year y):
  - pct_above_ecoff (level)
  - pct_above_bp    (level)
  - reservoir       (ecoff - bp gap)
  - vel_ecoff_3y    (recent trend)
  - acc_ecoff       (acceleration of trend)

Temporal split:
  TRAIN:  predictor years 2007–2014  (outcomes use years <=2019)
  TEST:   predictor years 2015–2019  (outcomes use years 2016–2024)
  No overlap on outcome years => proper out-of-time forecast validation.

Model: L2-regularized logistic regression with standardized features.

We also fit by (Species, Drug) to see if a single global model generalizes
or each pair has its own dynamics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_features.parquet")

FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir",
            "vel_ecoff_3y", "acc_ecoff"]
OUTCOME = "bp_will_cross_10_5y"

# Drop rows with missing predictors / outcome
df = df.dropna(subset=FEATURES + [OUTCOME]).reset_index(drop=True)
print(f"Modeling dataset: {len(df):,} rows from {df.groupby(['Species','Drug','Country']).ngroups} country-pair series")
print(f"Outcome class balance: {int(df[OUTCOME].sum())} positives / {len(df)} = {100*df[OUTCOME].mean():.1f}%\n")

# Temporal split
TRAIN_YEARS = (2007, 2014)
TEST_YEARS  = (2015, 2019)
train = df[df["Year"].between(*TRAIN_YEARS)].copy()
test  = df[df["Year"].between(*TEST_YEARS)].copy()
print(f"Train: years {TRAIN_YEARS}  n={len(train)}  pos={int(train[OUTCOME].sum())} ({100*train[OUTCOME].mean():.1f}%)")
print(f"Test : years {TEST_YEARS}  n={len(test)}  pos={int(test[OUTCOME].sum())} ({100*test[OUTCOME].mean():.1f}%)\n")

# --- Fit global model ---
scaler = StandardScaler().fit(train[FEATURES])
Xtr = scaler.transform(train[FEATURES]); ytr = train[OUTCOME].values
Xte = scaler.transform(test[FEATURES]);  yte = test[OUTCOME].values

mdl = LogisticRegression(C=1.0, max_iter=1000)
mdl.fit(Xtr, ytr)

p_tr = mdl.predict_proba(Xtr)[:,1]
p_te = mdl.predict_proba(Xte)[:,1]

print("=== Model coefficients (on standardized features) ===")
for f, c in zip(FEATURES, mdl.coef_[0]):
    print(f"  {f:<22} {c:+.3f}")
print(f"  intercept             {mdl.intercept_[0]:+.3f}\n")

print("=== Performance ===")
print(f"               AUC      AUPRC    Brier   class_balance")
print(f"  Train      {roc_auc_score(ytr,p_tr):.3f}   {average_precision_score(ytr,p_tr):.3f}   "
      f"{brier_score_loss(ytr,p_tr):.3f}   {ytr.mean():.3f}")
print(f"  Test       {roc_auc_score(yte,p_te):.3f}   {average_precision_score(yte,p_te):.3f}   "
      f"{brier_score_loss(yte,p_te):.3f}   {yte.mean():.3f}")

# Save predictions on full dataset
df["PRAS"] = mdl.predict_proba(scaler.transform(df[FEATURES]))[:,1]
df.to_parquet(ROOT/"data/pras_scored.parquet", index=False)

# Also per-pair AUC on test set (heterogeneity check)
print("\n=== Per-pair test AUC ===")
per_pair = []
for (sp, dr), g in test.groupby(["Species","Drug"]):
    if g[OUTCOME].nunique() < 2:
        print(f"  {sp} × {dr}: single-class outcome, skipping AUC")
        continue
    p = mdl.predict_proba(scaler.transform(g[FEATURES]))[:,1]
    auc = roc_auc_score(g[OUTCOME], p)
    auprc = average_precision_score(g[OUTCOME], p)
    per_pair.append(dict(Species=sp, Drug=dr, n=len(g), positives=int(g[OUTCOME].sum()),
                         AUC=auc, AUPRC=auprc))
    print(f"  {sp:<26} × {dr:<24}  n={len(g):>4}  pos={int(g[OUTCOME].sum()):>3}  AUC={auc:.3f}  AUPRC={auprc:.3f}")
pd.DataFrame(per_pair).to_csv(ROOT/"tables/pras_per_pair_test_metrics.csv", index=False)


# ===== Build validation figure =====
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ROC
ax = axes[0,0]
fpr_tr, tpr_tr, _ = roc_curve(ytr, p_tr)
fpr_te, tpr_te, _ = roc_curve(yte, p_te)
ax.plot(fpr_tr, tpr_tr, color="steelblue", lw=1.8, label=f"Train AUC={roc_auc_score(ytr,p_tr):.3f}")
ax.plot(fpr_te, tpr_te, color="crimson",   lw=2.4, label=f"Test  AUC={roc_auc_score(yte,p_te):.3f}")
ax.plot([0,1],[0,1], "k--", alpha=0.4, lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC — Pre-Resistance Alert Score", fontsize=10, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# PR curve
ax = axes[0,1]
prec_tr, rec_tr, _ = precision_recall_curve(ytr, p_tr)
prec_te, rec_te, _ = precision_recall_curve(yte, p_te)
ax.plot(rec_tr, prec_tr, color="steelblue", lw=1.8,
        label=f"Train AUPRC={average_precision_score(ytr,p_tr):.3f}")
ax.plot(rec_te, prec_te, color="crimson",   lw=2.4,
        label=f"Test  AUPRC={average_precision_score(yte,p_te):.3f}")
ax.axhline(yte.mean(), color="k", ls="--", alpha=0.4,
           label=f"Test base rate = {yte.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision–recall", fontsize=10, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# Calibration
ax = axes[1,0]
prob_te, true_te = calibration_curve(yte, p_te, n_bins=10, strategy="quantile")
ax.plot(prob_te, true_te, "o-", color="crimson", lw=2, ms=8, label="Test")
ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Perfect")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration (test set)", fontsize=10, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# Score distribution by outcome
ax = axes[1,1]
ax.hist(p_te[yte==0], bins=30, alpha=0.55, color="steelblue", label="No BP crossing (next 5y)")
ax.hist(p_te[yte==1], bins=30, alpha=0.55, color="crimson",   label="BP crossing within 5y")
ax.set_xlabel("PRAS score"); ax.set_ylabel("Count")
ax.set_title("Score distribution on held-out test set", fontsize=10, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Pre-Resistance Alert Score — out-of-time temporal validation\n"
             f"Train: predictor years {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} | "
             f"Test: predictor years {TEST_YEARS[0]}-{TEST_YEARS[1]} | "
             "Outcome: %above-BP ≥10% within 5 years",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT/"figures/phase3_pras_validation.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nWrote figures/phase3_pras_validation.png")

# Save coefficient table for reporting
coef_table = pd.DataFrame({
    "feature": FEATURES + ["intercept"],
    "coef_standardized": list(mdl.coef_[0]) + [mdl.intercept_[0]],
    "interpretation": [
        "Current %above-ECOFF (level)",
        "Current %above-breakpoint (level)",
        "Pre-resistance reservoir (ECOFF − BP)",
        "3-year velocity of %above-ECOFF",
        "Acceleration of %above-ECOFF",
        "Baseline log-odds"
    ]
})
coef_table.to_csv(ROOT/"tables/pras_coefficients.csv", index=False)
print("Wrote tables/pras_coefficients.csv")
print("\nDone.")
