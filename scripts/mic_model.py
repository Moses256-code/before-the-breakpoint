"""
Interval-censored normal regression on log2(MIC).

Model (per fit):
    log2_MIC_i  ~ Normal(mu_i, sigma)         [latent]
    mu_i        = X_i @ beta                  [linear predictor]
    Observed: interval (lower_i, upper_i]:
        - exact step  L:        lower = L-1,  upper = L
        - left-cens   L:        lower = -inf, upper = L
        - right-cens  L:        lower = L,    upper = +inf

Likelihood per obs:
    L_i = Phi((upper_i - mu_i)/sigma) - Phi((lower_i - mu_i)/sigma)
(both terms 0 or 1 in the censored limits.)

Maximises the log-likelihood with L-BFGS-B (analytical jacobian via SciPy autodiff).
Standard errors from inverse observed Fisher information (numerical Hessian).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import inv


class IntervalNormalReg:
    """Maximum likelihood interval-censored normal regression."""

    def __init__(self):
        self.beta = None
        self.log_sigma = None
        self.sigma = None
        self.cov = None         # asymptotic covariance of (beta, log_sigma)
        self.loglik = None
        self.n = None
        self.X_cols = None

    # ----- internals -----
    @staticmethod
    def _loglik(params, X, lower, upper):
        p = X.shape[1]
        beta, log_sigma = params[:p], params[p]
        sigma = np.exp(log_sigma)
        mu = X @ beta
        z_u = (upper - mu) / sigma
        z_l = (lower - mu) / sigma
        # Phi(+inf)=1, Phi(-inf)=0 — handled naturally
        P_u = norm.cdf(z_u)
        P_l = norm.cdf(z_l)
        prob = np.clip(P_u - P_l, 1e-300, 1.0)
        return -np.sum(np.log(prob))

    @staticmethod
    def _hess(f, x, eps=1e-4):
        n = len(x)
        H = np.zeros((n, n))
        f0 = f(x)
        for i in range(n):
            for j in range(i, n):
                xpp = x.copy(); xpp[i]+=eps; xpp[j]+=eps
                xpm = x.copy(); xpm[i]+=eps; xpm[j]-=eps
                xmp = x.copy(); xmp[i]-=eps; xmp[j]+=eps
                xmm = x.copy(); xmm[i]-=eps; xmm[j]-=eps
                H[i,j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*eps*eps)
                H[j,i] = H[i,j]
        return H

    # ----- public API -----
    def fit(self, X, lower, upper, col_names=None, init_beta=None, init_sigma=1.0):
        X = np.asarray(X, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if not (np.all(upper > lower) or np.all(upper >= lower)):
            raise ValueError("upper must be >= lower")
        # crude initial values: use midpoint of upper for OLS on observed cells
        mid = np.where(np.isfinite(upper) & np.isfinite(lower),
                       (lower + upper) / 2.0,
                       np.where(np.isfinite(upper), upper - 0.5, lower + 0.5))
        if init_beta is None:
            init_beta = np.linalg.lstsq(X, mid, rcond=None)[0]
        x0 = np.concatenate([init_beta, [np.log(init_sigma)]])

        res = minimize(
            self._loglik, x0, args=(X, lower, upper),
            method="L-BFGS-B",
            options=dict(maxiter=400, ftol=1e-10),
        )
        if not res.success:
            # try once more with looser tolerance
            res = minimize(self._loglik, x0, args=(X, lower, upper),
                           method="Nelder-Mead", options=dict(maxiter=2000))
        p = X.shape[1]
        self.beta = res.x[:p]
        self.log_sigma = res.x[p]
        self.sigma = float(np.exp(self.log_sigma))
        self.loglik = -res.fun
        self.n = len(lower)
        self.X_cols = col_names

        # observed Fisher information (numerical)
        try:
            H = self._hess(lambda x: self._loglik(x, X, lower, upper), res.x)
            self.cov = inv(H)
        except Exception:
            self.cov = None
        return self

    def se(self):
        if self.cov is None:
            return None
        return np.sqrt(np.diag(self.cov))

    def summary(self):
        out = []
        se = self.se()
        names = list(self.X_cols) if self.X_cols else [f"beta_{i}" for i in range(len(self.beta))]
        names.append("log_sigma")
        vals = list(self.beta) + [self.log_sigma]
        for n_, v, s in zip(names, vals, (se if se is not None else [np.nan]*len(vals))):
            out.append((n_, v, s, v - 1.96*s if s==s else np.nan, v + 1.96*s if s==s else np.nan))
        return out


# ----- CLSI / EUCAST reference values for our 17 pairs ----------------------
# Format: (species, drug) -> dict(susceptible_le, resistant_ge, ecoff)
# Values are in mg/L; we store log2 versions too.
# Sources: CLSI M100 (2024), EUCAST clinical breakpoints v14.0 (2024).
# ECOFFs from EUCAST MIC distributions website (rounded to ATLAS dilution).
BREAKPOINTS_MG_L = {
    # ---- Meropenem ----
    ("Klebsiella pneumoniae",  "Meropenem"):             dict(S=1,  R=4,  ecoff=0.125),
    ("Escherichia coli",       "Meropenem"):             dict(S=1,  R=4,  ecoff=0.0625),
    ("Enterobacter cloacae",   "Meropenem"):             dict(S=1,  R=4,  ecoff=0.125),
    ("Pseudomonas aeruginosa", "Meropenem"):             dict(S=2,  R=8,  ecoff=2),
    ("Acinetobacter baumannii","Meropenem"):             dict(S=2,  R=8,  ecoff=2),
    # ---- Imipenem ----
    ("Klebsiella pneumoniae",  "Imipenem"):              dict(S=1,  R=4,  ecoff=0.5),
    # ---- Ceftriaxone ----
    ("Escherichia coli",       "Ceftriaxone"):           dict(S=1,  R=4,  ecoff=0.0625),
    ("Klebsiella pneumoniae",  "Ceftriaxone"):           dict(S=1,  R=4,  ecoff=0.125),
    # ---- Cefepime ----
    ("Escherichia coli",       "Cefepime"):              dict(S=2,  R=16, ecoff=0.125),
    ("Klebsiella pneumoniae",  "Cefepime"):              dict(S=2,  R=16, ecoff=0.125),
    ("Pseudomonas aeruginosa", "Cefepime"):              dict(S=8,  R=32, ecoff=8),
    # ---- Ceftazidime-avibactam ----
    ("Escherichia coli",       "Ceftazidime avibactam"): dict(S=8,  R=16, ecoff=0.5),
    ("Klebsiella pneumoniae",  "Ceftazidime avibactam"): dict(S=8,  R=16, ecoff=0.5),
    ("Pseudomonas aeruginosa", "Ceftazidime avibactam"): dict(S=8,  R=16, ecoff=4),
    # ---- Colistin (CLSI uses Intermediate cutoff at 2; we treat I/R boundary as R-threshold) ----
    ("Klebsiella pneumoniae",  "Colistin"):              dict(S=2,  R=4,  ecoff=2),
    ("Escherichia coli",       "Colistin"):              dict(S=2,  R=4,  ecoff=2),
    ("Pseudomonas aeruginosa", "Colistin"):              dict(S=2,  R=4,  ecoff=4),
}


def get_thresholds(species, drug):
    """Return (S_log2, R_log2, ECOFF_log2) or None if pair not registered."""
    bp = BREAKPOINTS_MG_L.get((species, drug))
    if bp is None:
        return None
    return dict(
        S_log2=np.log2(bp["S"]),
        R_log2=np.log2(bp["R"]),
        ECOFF_log2=np.log2(bp["ecoff"]),
        S=bp["S"], R=bp["R"], ECOFF=bp["ecoff"],
    )


# ----- quick self-test ------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 5000
    year = rng.integers(0, 20, size=n).astype(float)
    true_beta = np.array([2.0, 0.05])     # intercept 2, drift +0.05 log2/yr
    true_sigma = 1.3
    mu = true_beta[0] + true_beta[1]*year
    latent = rng.normal(mu, true_sigma)
    # censor at edges [-4, 4]
    lower = np.floor(latent).clip(min=-4)
    upper = lower + 1
    left = latent <= -4
    right = latent >= 4
    lower[left]  = -np.inf; upper[left]  = -4
    lower[right] =  4;      upper[right] = np.inf

    X = np.column_stack([np.ones(n), year])
    mdl = IntervalNormalReg().fit(X, lower, upper, ["intercept","year"])
    print("True : beta=[2.0, 0.05]   sigma=1.3")
    print(f"Fit  : beta=[{mdl.beta[0]:.3f}, {mdl.beta[1]:.4f}]   sigma={mdl.sigma:.3f}")
    print("\nSummary:")
    for name, est, se, lo, hi in mdl.summary():
        print(f"  {name:>12}  {est:+.4f}  se={se:.4f}  [{lo:+.4f}, {hi:+.4f}]")
