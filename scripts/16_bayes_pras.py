"""
Phase 4 — Bayesian hierarchical PRAS in PyMC.

Frequentist v1 already in `data/pras_scored.parquet`:
  AUC test 0.902, AUPRC test 0.848 with L2 logistic regression.

Here we fit a hierarchical Bayesian logistic regression:
  y_i ~ Bernoulli(p_i)
  logit(p_i) = α_pair[i] + Σ β_k * x_ik
  α_pair ~ Normal(μ_α, σ_α)       # partial pooling across the 7 framework pairs
  β ~ Normal(0, 1.5)               # weakly informative priors
  μ_α ~ Normal(-1, 1.5);  σ_α ~ HalfNormal(0.5)

Sample with NUTS, then for every cell in the test set compute the posterior
predictive distribution of PRAS — giving credible intervals on every score.

Outputs:
  data/pras_bayes_scored.parquet  -- adds PRAS_mean, PRAS_lo, PRAS_hi, PRAS_std
  tables/pras_bayes_coefficients.csv
  figures/phase4_bayes_comparison.png
"""
import sys, time, warnings, os
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTENSOR_FLAGS","mode=FAST_RUN")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
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
pair_idx = {p: i for i, p in enumerate(PAIRS)}
df["pair_id"] = df["pair"].map(pair_idx)

train = df[df["Year"].between(2007, 2014)].copy()
test  = df[df["Year"].between(2015, 2019)].copy()

scaler = StandardScaler().fit(train[FEATURES])
Xtr = scaler.transform(train[FEATURES])
Xte = scaler.transform(test[FEATURES])
ytr = train[OUTCOME].values.astype(int)
yte = test[OUTCOME].values.astype(int)
ptr = train["pair_id"].values.astype(int)
pte = test["pair_id"].values.astype(int)

print(f"Train n={len(ytr)}  pos rate={ytr.mean():.3f}  pairs={len(PAIRS)}")
print(f"Test  n={len(yte)}  pos rate={yte.mean():.3f}\n")

t0 = time.time()
with pm.Model() as model:
    # priors
    beta    = pm.Normal("beta", 0, 1.5, shape=len(FEATURES))
    mu_a    = pm.Normal("mu_alpha", -1, 1.5)
    sigma_a = pm.HalfNormal("sigma_alpha", 0.5)
    # non-centered parametrization for the pair random intercepts
    alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=len(PAIRS))
    alpha = pm.Deterministic("alpha", mu_a + sigma_a * alpha_raw)
    # train likelihood
    logit_p = alpha[ptr] + pm.math.dot(Xtr, beta)
    pm.Bernoulli("y", logit_p=logit_p, observed=ytr)

print("Sampling NUTS (4 chains × 1000 draws + 1000 tune)...")
with model:
    idata = pm.sample(draws=1000, tune=1000, chains=4, cores=4,
                      target_accept=0.95, random_seed=42, progressbar=False)

elapsed = time.time() - t0
print(f"\nSampling completed in {elapsed:.1f}s")

# Quick diagnostics
summ = az.summary(idata, var_names=["beta","mu_alpha","sigma_alpha","alpha"], ci_prob=0.95)
print("\n=== Posterior summary ===")
print(summ.round(3).to_string())
summ.to_csv(ROOT/"tables/pras_bayes_summary.csv")

# Compute posterior predictive PRAS on all cells (train, test, and any other rows)
def predict_post(X, pair_ids, idata):
    """Return (n_rows, n_samples) of posterior PRAS predictions."""
    a = idata.posterior["alpha"].stack(s=("chain","draw")).values  # (pairs, samples)
    b = idata.posterior["beta"].stack(s=("chain","draw")).values   # (features, samples)
    # eta_i = a[pair_i] + X_i @ b
    eta = a[pair_ids,:] + X @ b   # broadcasting: (n, samples)
    return 1.0 / (1.0 + np.exp(-eta))

Xtr_full = scaler.transform(train[FEATURES])
Xte_full = scaler.transform(test[FEATURES])

print("Computing posterior predictions...")
p_tr_samples = predict_post(Xtr_full, ptr, idata)
p_te_samples = predict_post(Xte_full, pte, idata)
p_tr_mean = p_tr_samples.mean(axis=1)
p_te_mean = p_te_samples.mean(axis=1)
p_tr_lo, p_tr_hi = np.quantile(p_tr_samples, [0.025, 0.975], axis=1)
p_te_lo, p_te_hi = np.quantile(p_te_samples, [0.025, 0.975], axis=1)
p_tr_std = p_tr_samples.std(axis=1)
p_te_std = p_te_samples.std(axis=1)

print("\n=== Bayesian PRAS performance ===")
print(f"               AUC      AUPRC    Brier")
print(f"  Train      {roc_auc_score(ytr,p_tr_mean):.3f}   {average_precision_score(ytr,p_tr_mean):.3f}   {brier_score_loss(ytr,p_tr_mean):.3f}")
print(f"  Test       {roc_auc_score(yte,p_te_mean):.3f}   {average_precision_score(yte,p_te_mean):.3f}   {brier_score_loss(yte,p_te_mean):.3f}")

# Build the full-cell scored dataframe (everything in df, scored)
X_all = scaler.transform(df[FEATURES])
p_all = predict_post(X_all, df["pair_id"].values.astype(int), idata)
df["PRAS_bayes_mean"] = p_all.mean(axis=1).round(4)
df["PRAS_bayes_lo"]   = np.quantile(p_all, 0.025, axis=1).round(4)
df["PRAS_bayes_hi"]   = np.quantile(p_all, 0.975, axis=1).round(4)
df["PRAS_bayes_std"]  = p_all.std(axis=1).round(4)
df["PRAS_bayes_width"]= (df["PRAS_bayes_hi"] - df["PRAS_bayes_lo"]).round(4)

# also bring in the frequentist PRAS from before for comparison
freq = pd.read_parquet(ROOT/"data/pras_scored.parquet")
df = df.merge(freq[["Species","Drug","Country","Year","PRAS"]], on=["Species","Drug","Country","Year"], how="left")
df = df.rename(columns={"PRAS":"PRAS_freq"})

df.to_parquet(ROOT/"data/pras_bayes_scored.parquet", index=False)
print(f"\nWrote data/pras_bayes_scored.parquet  ({len(df)} rows)")

# Save coefficients table
beta_post = idata.posterior["beta"].stack(s=("chain","draw")).values
mu_a_post = idata.posterior["mu_alpha"].stack(s=("chain","draw")).values
sig_a_post = idata.posterior["sigma_alpha"].stack(s=("chain","draw")).values
alpha_post = idata.posterior["alpha"].stack(s=("chain","draw")).values
rows = []
for i, f in enumerate(FEATURES):
    rows.append(dict(parameter=f, mean=beta_post[i].mean(),
                     hdi_2p5=np.quantile(beta_post[i],0.025),
                     hdi_97p5=np.quantile(beta_post[i],0.975),
                     std=beta_post[i].std()))
rows.append(dict(parameter="mu_alpha (global intercept)", mean=mu_a_post.mean(),
                 hdi_2p5=np.quantile(mu_a_post,0.025), hdi_97p5=np.quantile(mu_a_post,0.975),
                 std=mu_a_post.std()))
rows.append(dict(parameter="sigma_alpha (pair-level SD)", mean=sig_a_post.mean(),
                 hdi_2p5=np.quantile(sig_a_post,0.025), hdi_97p5=np.quantile(sig_a_post,0.975),
                 std=sig_a_post.std()))
for i, p in enumerate(PAIRS):
    rows.append(dict(parameter=f"alpha[{p}]", mean=alpha_post[i].mean(),
                     hdi_2p5=np.quantile(alpha_post[i],0.025), hdi_97p5=np.quantile(alpha_post[i],0.975),
                     std=alpha_post[i].std()))
pd.DataFrame(rows).to_csv(ROOT/"tables/pras_bayes_coefficients.csv", index=False)

# Build comparison figure: Bayesian vs frequentist + CI width visualization
test_merged = test.copy()
test_merged["PRAS_bayes_mean"] = p_te_mean
test_merged["PRAS_bayes_lo"]   = p_te_lo
test_merged["PRAS_bayes_hi"]   = p_te_hi
freq_te = pd.read_parquet(ROOT/"data/pras_scored.parquet")
test_merged = test_merged.merge(freq_te[["Species","Drug","Country","Year","PRAS"]],
                                 on=["Species","Drug","Country","Year"], how="left").rename(
    columns={"PRAS":"PRAS_freq"})

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Bayesian vs frequentist agreement
ax = axes[0,0]
sc = ax.scatter(test_merged["PRAS_freq"], test_merged["PRAS_bayes_mean"],
                c=test_merged[OUTCOME], cmap="coolwarm", s=14, alpha=0.6, edgecolors="none")
ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("Frequentist PRAS"); ax.set_ylabel("Bayesian PRAS (posterior mean)")
ax.set_title("(A) Bayesian vs frequentist agreement\nColor = future BP crossing (red=yes)", fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label="BP crossed", fraction=0.04)

# Panel 2: Posterior CI width by score level
ax = axes[0,1]
ax.scatter(test_merged["PRAS_bayes_mean"], test_merged["PRAS_bayes_hi"] - test_merged["PRAS_bayes_lo"],
           c="steelblue", s=14, alpha=0.55, edgecolors="none")
ax.set_xlabel("PRAS (posterior mean)")
ax.set_ylabel("95% credible interval width")
ax.set_title("(B) Uncertainty profile\nWider intervals at PRAS≈0.5 (epistemic peak)",
             fontsize=10, fontweight="bold")
ax.grid(alpha=0.3)

# Panel 3: Per-pair Bayesian AUC
from sklearn.metrics import roc_auc_score
ax = axes[1,0]
pair_aucs = []
for p in PAIRS:
    g = test_merged[test_merged["pair"]==p]
    if g[OUTCOME].nunique() < 2: continue
    auc_f = roc_auc_score(g[OUTCOME], g["PRAS_freq"])
    auc_b = roc_auc_score(g[OUTCOME], g["PRAS_bayes_mean"])
    pair_aucs.append((p, auc_f, auc_b, len(g)))
pair_aucs.sort(key=lambda x: x[2])
labels = [p[0].split(" × ")[0].split()[0][0] + ". " + " ".join(p[0].split(" × ")[0].split()[1:]) + " × " + p[0].split(" × ")[1] for p in pair_aucs]
y_pos = np.arange(len(pair_aucs))
ax.barh(y_pos-0.2, [p[1] for p in pair_aucs], height=0.35, color="lightgray", label="Frequentist")
ax.barh(y_pos+0.2, [p[2] for p in pair_aucs], height=0.35, color="steelblue", label="Bayesian")
ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Test AUC"); ax.set_xlim(0.5, 1.0)
ax.axvline(0.5, color="k", lw=0.5, ls="--", alpha=0.5)
ax.legend(); ax.grid(axis="x", alpha=0.3)
ax.set_title("(C) Per-pair test AUC: Bayesian vs frequentist", fontsize=10, fontweight="bold")

# Panel 4: example Greece KP × CAZ-AVI Bayesian PRAS trajectory with credible band
ax = axes[1,1]
scored = pd.read_parquet(ROOT/"data/pras_bayes_scored.parquet")
greece = scored[(scored["Country"]=="Greece") & (scored["Species"]=="Klebsiella pneumoniae")
                 & (scored["Drug"]=="Ceftazidime avibactam")].sort_values("Year")
if len(greece):
    ax.fill_between(greece["Year"], greece["PRAS_bayes_lo"], greece["PRAS_bayes_hi"],
                    alpha=0.25, color="purple", label="Bayesian 95% credible")
    ax.plot(greece["Year"], greece["PRAS_bayes_mean"], "o-", color="purple",
            lw=2, ms=5, label="Bayesian PRAS (mean)")
    ax.plot(greece["Year"], greece["PRAS_freq"], "s--", color="black", lw=1, ms=3, label="Frequentist PRAS")
    ax.axhline(0.5, color="crimson", lw=0.8, ls=":", label="High-alert threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year"); ax.set_ylabel("PRAS")
    ax.set_title("(D) Greece × K. pneumoniae × CAZ-AVI\nBayesian PRAS with 95% credible band",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("Bayesian hierarchical PRAS — posterior intervals and comparison to frequentist",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(ROOT/"figures/phase4_bayes_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print("Wrote figures/phase4_bayes_comparison.png")
print(f"\nTotal time: {time.time()-t0:.1f}s")
