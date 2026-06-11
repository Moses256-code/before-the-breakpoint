"""
Phase 4 — Leave-one-pair-out cross-validation.

For each of the 7 framework pairs, we hold it out, train on the other 6
(using 2007-2014 training years), and test on the held-out pair's
2015-2019 cells. This tests whether the model captures a *general*
pre-resistance phenomenon rather than memorizing pair-specific patterns.

We do this for BOTH the frequentist (sklearn LogReg) and the Bayesian
hierarchical version (PyMC). For the Bayesian model, when scoring a
held-out pair we use the GLOBAL intercept (mu_alpha) since we have
no posterior for that specific pair.
"""
import time, warnings, os
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

ROOT = Path("/home/claude/atlas")
df = pd.read_parquet(ROOT/"data/pras_features.parquet")
FEATURES = ["pct_above_ecoff", "pct_above_bp", "reservoir", "vel_ecoff_3y", "acc_ecoff"]
OUTCOME = "bp_will_cross_10_5y"
df = df.dropna(subset=FEATURES + [OUTCOME]).reset_index(drop=True)
df["pair"] = df["Species"] + " × " + df["Drug"]
PAIRS = sorted(df["pair"].unique())

results = []
for held in PAIRS:
    print(f"\n----- HOLD OUT: {held} -----")
    train_mask = (df["pair"] != held) & df["Year"].between(2007, 2014)
    test_mask  = (df["pair"] == held) & df["Year"].between(2015, 2019)
    tr = df[train_mask]
    te = df[test_mask]
    if tr[OUTCOME].nunique() < 2 or te[OUTCOME].nunique() < 2:
        print("  insufficient class balance — skipping")
        continue
    scaler = StandardScaler().fit(tr[FEATURES])
    Xtr = scaler.transform(tr[FEATURES]); Xte = scaler.transform(te[FEATURES])
    ytr = tr[OUTCOME].values.astype(int); yte = te[OUTCOME].values.astype(int)

    # ----- Frequentist -----
    mdl = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    p_te = mdl.predict_proba(Xte)[:,1]
    f_auc = roc_auc_score(yte, p_te)
    f_auprc = average_precision_score(yte, p_te)
    f_brier = brier_score_loss(yte, p_te)
    print(f"  Freq   n_train={len(tr):>4} pos={ytr.sum():>3}  n_test={len(te):>4} pos={yte.sum():>3}  "
          f"AUC={f_auc:.3f}  AUPRC={f_auprc:.3f}  Brier={f_brier:.3f}")

    # ----- Bayesian (single intercept since we're not modeling the held-out pair) -----
    t0 = time.time()
    with pm.Model() as m:
        beta = pm.Normal("beta", 0, 1.5, shape=len(FEATURES))
        alpha = pm.Normal("alpha", -1, 1.5)
        logit_p = alpha + pm.math.dot(Xtr, beta)
        pm.Bernoulli("y", logit_p=logit_p, observed=ytr)
        id_ = pm.sample(draws=600, tune=600, chains=2, cores=2,
                        target_accept=0.95, random_seed=42, progressbar=False)
    b_post = id_.posterior["beta"].stack(s=("chain","draw")).values  # (n_feat, n_samp)
    a_post = id_.posterior["alpha"].stack(s=("chain","draw")).values  # (n_samp,)
    eta = a_post[None,:] + Xte @ b_post  # (n_test, n_samp)
    p_b = 1.0 / (1.0 + np.exp(-eta))
    p_b_mean = p_b.mean(axis=1)
    p_b_lo, p_b_hi = np.quantile(p_b, [0.025, 0.975], axis=1)
    b_auc = roc_auc_score(yte, p_b_mean)
    b_auprc = average_precision_score(yte, p_b_mean)
    b_brier = brier_score_loss(yte, p_b_mean)
    print(f"  Bayes  AUC={b_auc:.3f}  AUPRC={b_auprc:.3f}  Brier={b_brier:.3f}  ({time.time()-t0:.1f}s)")

    results.append(dict(
        held_pair=held, n_train=len(tr), n_test=len(te), n_pos_test=int(yte.sum()),
        freq_AUC=f_auc, freq_AUPRC=f_auprc, freq_Brier=f_brier,
        bayes_AUC=b_auc, bayes_AUPRC=b_auprc, bayes_Brier=b_brier,
    ))

res = pd.DataFrame(results)
res.to_csv(ROOT/"tables/lopo_results.csv", index=False)
print("\n=== Leave-one-pair-out summary ===")
print(res[["held_pair","n_test","n_pos_test","freq_AUC","bayes_AUC","freq_AUPRC","bayes_AUPRC"]].to_string(index=False))

# Aggregate stats
print(f"\nMean LOPO Frequentist AUC: {res['freq_AUC'].mean():.3f}  (range {res['freq_AUC'].min():.3f}–{res['freq_AUC'].max():.3f})")
print(f"Mean LOPO Bayesian    AUC: {res['bayes_AUC'].mean():.3f}  (range {res['bayes_AUC'].min():.3f}–{res['bayes_AUC'].max():.3f})")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Per-pair LOPO AUC
ax = axes[0]
y_pos = np.arange(len(res))
ax.barh(y_pos-0.2, res["freq_AUC"], height=0.35, color="lightgray", label="Frequentist")
ax.barh(y_pos+0.2, res["bayes_AUC"], height=0.35, color="steelblue", label="Bayesian")
ax.set_yticks(y_pos)
labels = [p.split(" × ")[0].split()[0][0] + ". " + " ".join(p.split(" × ")[0].split()[1:]) + " × " + p.split(" × ")[1] for p in res["held_pair"]]
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Test AUC on held-out pair")
ax.set_xlim(0.5, 1.0)
ax.axvline(0.5, color="k", lw=0.5, ls="--", alpha=0.5)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
ax.set_title("Leave-one-pair-out: AUC on the held-out pair\n(Trained on the other 6 pairs only)",
             fontsize=10, fontweight="bold")

# In-set vs LOPO comparison
ax = axes[1]
within_aucs = pd.read_csv(ROOT/"tables/pras_per_pair_test_metrics.csv").set_index(["Species","Drug"])
res["Species"] = res["held_pair"].str.split(" × ").str[0]
res["Drug"]    = res["held_pair"].str.split(" × ").str[1]
res = res.set_index(["Species","Drug"]).join(within_aucs[["AUC","AUPRC"]], rsuffix="_within").reset_index()
ax.scatter(res["AUC"], res["bayes_AUC"], s=80, color="steelblue", edgecolor="navy")
for _, r in res.iterrows():
    ax.annotate(r["Species"].split()[0][0] + "." + r["Drug"][:6], (r["AUC"], r["bayes_AUC"]),
                xytext=(5,5), textcoords="offset points", fontsize=8)
ax.plot([0.5,1],[0.5,1], "k--", alpha=0.4)
ax.set_xlabel("Within-set AUC (pair seen in training)")
ax.set_ylabel("LOPO AUC (pair held out entirely)")
ax.set_xlim(0.6, 1.0); ax.set_ylim(0.5, 1.0)
ax.grid(alpha=0.3)
ax.set_title("Generalization gap: within-set vs LOPO\n(Close to diagonal = good generalization)",
             fontsize=10, fontweight="bold")

fig.suptitle("Leave-one-pair-out cross-validation\n"
             "Tests whether the pre-resistance signature is pair-generic vs pair-specific",
             y=1.02, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_lopo.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"\nWrote tables/lopo_results.csv and figures/phase4_lopo.png")
