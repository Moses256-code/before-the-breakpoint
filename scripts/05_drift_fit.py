"""
Phase 2 Step 1 — overall + per-country MIC drift fits for each of the 17 pairs.

For each (Species, Drug) pair in recommended_pairs.csv:
  1. OVERALL FIT:
        log2(MIC) = beta0 + beta1 * (year - year_ref)
     Reports beta1 = annual drift in log2(MIC) (dilution steps / year), with 95% CI.
  2. PER-COUNTRY FITS:
        Same model fitted independently within each country that has
        >=5 years of data AND >=30 isolates in the window.
  3. DERIVED PER-CELL METRICS:
        For each (country, year) cell with >=10 isolates, store
          mu_hat, sigma_hat,
          pct_above_ECOFF, pct_above_BP (model-based),
          empirical resistant_pct (from ATLAS _I labels) for comparison.

Outputs:
  tables/drift_overall.csv          17 rows
  tables/drift_per_country.csv      ~ thousands of rows
  tables/cell_predictions.csv       country-year MIC distribution estimates
  figures/phase2_drift_overview.png
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from mic_model import IntervalNormalReg, get_thresholds

ROOT = Path("/home/claude/atlas")
LONG = ROOT / "data/atlas_long.parquet"
PAIRS = pd.read_csv(ROOT / "tables/recommended_pairs.csv")

YEAR_REF_MAP = {}  # filled per pair: midpoint of window

MIN_COUNTRY_YRS = 5
MIN_COUNTRY_N   = 30
MIN_CELL_N      = 10

# Load long table once
print("Loading long table ...")
long_cols = ["Species","Drug","Country","Year","Interp","log2_idx","log2_lower","log2_upper","cens_type"]
long_all = pq.read_table(LONG, columns=long_cols).to_pandas()
print(f"  {len(long_all):,} rows in long table")


def fit_overall(df, year_ref):
    """Fit log2_MIC = b0 + b1*(year - year_ref). Return (b1_est, b1_se, sigma, n, loglik)."""
    yrs = df["Year"].astype(float).values - year_ref
    X = np.column_stack([np.ones(len(df)), yrs])
    low = df["log2_lower"].astype(float).values
    upp = df["log2_upper"].astype(float).values
    # replace NaN-as-inf markers: parquet stored inf as float
    low = np.where(np.isfinite(low), low, -np.inf)
    upp = np.where(np.isfinite(upp), upp,  np.inf)
    try:
        m = IntervalNormalReg().fit(X, low, upp, ["intercept","year_c"])
    except Exception as e:
        return dict(error=str(e))
    se = m.se()
    return dict(
        b0=m.beta[0], b0_se=se[0] if se is not None else np.nan,
        b1=m.beta[1], b1_se=se[1] if se is not None else np.nan,
        sigma=m.sigma, log_sigma_se=se[2] if se is not None else np.nan,
        n=int(m.n), loglik=float(m.loglik),
    )


def fit_country(df, year_ref):
    """Fit per-country drift. Returns dict per country."""
    out = {}
    for country, g in df.groupby("Country"):
        if g["Year"].nunique() < MIN_COUNTRY_YRS or len(g) < MIN_COUNTRY_N:
            continue
        r = fit_overall(g, year_ref)
        if r.get("error"): continue
        r["country"] = country
        r["n"] = len(g)
        r["yrs"] = int(g["Year"].nunique())
        r["yr_min"] = int(g["Year"].min())
        r["yr_max"] = int(g["Year"].max())
        out[country] = r
    return out


def fit_cells(df, thresh):
    """For each (country, year) cell with >=MIN_CELL_N isolates, fit a Normal(mu, sigma)
    via interval-censored MLE with no covariates -> we get a per-cell mu_hat, sigma_hat.
    Compute derived stats: model-based P>ECOFF and P>BP, and empirical resistant %."""
    rows = []
    for (cy, yr), g in df.groupby(["Country","Year"]):
        if len(g) < MIN_CELL_N: continue
        X = np.ones((len(g),1))
        low = np.where(np.isfinite(g["log2_lower"].values), g["log2_lower"].values, -np.inf)
        upp = np.where(np.isfinite(g["log2_upper"].values), g["log2_upper"].values,  np.inf)
        try:
            m = IntervalNormalReg().fit(X, low, upp, ["intercept"])
        except Exception:
            continue
        mu, sig = m.beta[0], m.sigma
        # derived
        rec = {"Country": cy, "Year": int(yr), "n": len(g),
               "mu_log2": mu, "sigma_log2": sig,
               "median_MIC": 2**mu,
               "P95_MIC":    2**(mu + 1.645*sig),
               "P99_MIC":    2**(mu + 2.326*sig)}
        if thresh is not None:
            rec["pct_above_ECOFF_model"] = 100*(1 - norm.cdf((thresh["ECOFF_log2"] - mu)/sig))
            rec["pct_above_BP_model"]    = 100*(1 - norm.cdf((thresh["R_log2"]     - mu)/sig))
        # Empirical resistant % from ATLAS labels
        interps = g["Interp"].dropna()
        if len(interps):
            rec["pct_resistant_empirical"]    = 100*(interps=="Resistant").sum()/len(interps)
            rec["pct_intermediate_empirical"] = 100*(interps=="Intermediate").sum()/len(interps)
        rows.append(rec)
    return rows


# ----- Main loop -----
overall_rows = []
country_rows = []
cell_rows = []

for _, row in PAIRS.iterrows():
    sp, dr = row["Species"], row["Drug"]
    y0, y1 = int(row["window_start"]), int(row["window_end"])
    year_ref = (y0 + y1) // 2
    YEAR_REF_MAP[(sp, dr)] = year_ref

    sub = long_all[(long_all["Species"]==sp) & (long_all["Drug"]==dr)
                   & (long_all["Year"].between(y0, y1))].copy()
    if not len(sub):
        print(f"  SKIP {sp} × {dr} — no data")
        continue
    t0 = time.time()
    thresh = get_thresholds(sp, dr)

    # 1) overall fit
    ov = fit_overall(sub, year_ref)
    ov.update(dict(Species=sp, Drug=dr, Tier=int(row["Tier"]),
                   window_start=y0, window_end=y1, year_ref=year_ref))
    overall_rows.append(ov)

    # 2) per-country
    co = fit_country(sub, year_ref)
    for c, r in co.items():
        r.update(dict(Species=sp, Drug=dr, year_ref=year_ref))
        country_rows.append(r)

    # 3) cell predictions
    cells = fit_cells(sub, thresh)
    for c in cells:
        c["Species"] = sp; c["Drug"] = dr
        cell_rows.append(c)

    drift = ov.get("b1", float('nan'))
    drift_se = ov.get("b1_se", float('nan'))
    print(f"  {sp:<26} × {dr:<24}  drift={drift:+.4f}/yr (±{drift_se:.4f})  "
          f"countries fit={len(co)}  cells={len(cells)}  [{time.time()-t0:.1f}s]")

# Save
ov_df = pd.DataFrame(overall_rows)
co_df = pd.DataFrame(country_rows)
ce_df = pd.DataFrame(cell_rows)

ov_df.to_csv(ROOT/"tables/drift_overall.csv", index=False)
co_df.to_csv(ROOT/"tables/drift_per_country.csv", index=False)
ce_df.to_csv(ROOT/"tables/cell_predictions.csv", index=False)

print(f"\nWrote:")
print(f"  tables/drift_overall.csv         ({len(ov_df)} rows)")
print(f"  tables/drift_per_country.csv     ({len(co_df)} rows)")
print(f"  tables/cell_predictions.csv      ({len(ce_df)} rows)")
