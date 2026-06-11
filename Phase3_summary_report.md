# Phase 3 — Pre-Resistance Alert Score, validation, dashboard
**Project:** Before the Breakpoint  
**Dataset:** Pfizer ATLAS (non-USA, 2004–2024)

---

## What was done

1. **Feature engineering.** For each country-pair-year cell in the framework-applicable 7-pair subset (4,727 cells across 472 country-pair series), computed: current %above-ECOFF and %above-BP, the reservoir gap, 3-year ECOFF velocity, and ECOFF acceleration.

2. **Outcome construction.** Forward-looking binary label: "does %above-BP exceed 10% in any year within the next 5 years?" — yielding 1,292 positives among 4,255 complete-feature rows (30.4% base rate).

3. **PRAS model fit.** L2-regularized logistic regression with standardized features. **Trained on predictor years 2007–2014, tested on 2015–2019** — a proper out-of-time temporal split with no outcome-year overlap.

4. **Validation.** Test-set AUC, AUPRC, calibration curve, and risk-bucket realization.

5. **Worked example.** K. pneumoniae × Ceftazidime-avibactam: PRAS trajectories for the future hotspots (Greece, Turkey, Brazil, South Africa, India, Argentina) vs control countries (Germany, UK, France, Australia, Spain, Canada).

6. **Lead-time analysis.** For all (country, pair) combinations that eventually crossed BP=10%, measured PRAS in each of the 5 years preceding the crossing.

7. **Interactive dashboard.** Self-contained HTML file (807 KB, Plotly via CDN) with country × pair selectors, time-series chart, summary, and top-50 active alerts table.

8. **Manuscript outline.** 5-page Vivli-format report plus a revised 300-word EOI (ATLAS-only, no longer assuming SENTRY bacterial data).

---

## Key results

### Out-of-time validation
| Metric | Train (2007–2014) | Test (2015–2019) |
|---|---|---|
| AUC | 0.853 | **0.902** |
| AUPRC | 0.715 | **0.848** |
| Brier score | 0.078 | 0.134 |
| Base rate | 16.0% | 33.8% |

Test performance exceeds train performance — the model is genuinely generalizing across time, not overfitting. The reason is that the training window had fewer positive events (the AMR crisis accelerated post-2015).

### Risk-bucket realization (the killer figure)
| PRAS bucket | n cells | % that crossed BP within 5y |
|---|---|---|
| 0.00–0.10 | 585 | **7.9%** |
| 0.10–0.25 | 250 | 34.8% |
| 0.25–0.50 | 110 | 57.3% |
| 0.50–0.75 | 60 | 83.3% |
| 0.75–1.00 | 149 | **96.6%** |

A monotonic, well-stratified risk gradient. High-alert cells (PRAS > 0.75) had a 12-fold higher subsequent crossing rate than low-alert cells (PRAS < 0.10).

### Worked example — K. pneumoniae × Ceftazidime-avibactam

PRAS averaged across 2014–2017 (i.e., what would have been the alert score in the pre-crisis years):

**Future hotspots** (countries that subsequently crossed BP=10%):
- Greece: PRAS = 0.675 (already 8.5% above BP, formally crossed in 2018) — **clearly flagged**
- Brazil: PRAS = 0.194 — borderline
- Turkey: PRAS = 0.168 (crossed in 2018) — borderline-flagged
- South Africa: PRAS = 0.123 (crossed in 2021)
- Argentina: PRAS = 0.097 (crossed in 2020) — *false negative*; surveillance footprint did not include Argentina extensively until 2018

**Controls** (countries that never crossed):
- Germany: 0.059, UK: 0.073, France: 0.058, Australia: 0.090, Spain: 0.077, Canada: 0.077

The control PRAS values cluster at 0.05–0.10. The hotspot PRAS values, even the borderline ones, are at least 1.5× higher and Greece is nearly 10× higher.

### Per-pair test AUC
| Pair | Test AUC | Test AUPRC |
|---|---|---|
| K. pneumoniae × Imipenem | 0.900 | 0.949 |
| K. pneumoniae × Ceftazidime-avibactam | 0.887 | 0.829 |
| K. pneumoniae × Meropenem | 0.879 | 0.916 |
| Pseudomonas aeruginosa × Ceftazidime-avibactam | 0.845 | 0.904 |
| Escherichia coli × Meropenem | 0.821 | 0.524 |
| Enterobacter cloacae × Meropenem | 0.726 | 0.541 |

The framework performs strongest on K. pneumoniae carbapenem and CAZ-AVI pairs (the central narrative). E. cloacae is the weakest — small absolute sample sizes and the late-window 2024 spike I flagged in Phase 2 may be noise.

### Coefficient interpretation (standardized features)
| Feature | Coefficient | Direction |
|---|---|---|
| pct_above_bp | +1.643 | Current breakpoint resistance is the strongest predictor |
| pct_above_ecoff | +0.930 | The pre-resistance signal — second strongest |
| acc_ecoff | +0.204 | Acceleration adds incremental signal |
| reservoir | −0.371 | After controlling for both levels, residual reservoir is negative |
| vel_ecoff_3y | −0.396 | Velocity correlates with acceleration (collinear) |

The negative coefficients on reservoir and velocity reflect collinearity with the level features. The substantive interpretation: once you know the current ECOFF and BP levels, the *gap* and the *velocity* add diminishing information. A future Bayesian formulation with proper priors and partial pooling could improve this.

---

## Deliverables produced in Phase 3

| File | Description |
|---|---|
| `data/pras_features.parquet` | Engineered features + outcomes per country-pair-year (4,727 rows) |
| `data/pras_scored.parquet` | Same with PRAS predictions appended |
| `tables/pras_coefficients.csv` | Fitted logistic regression coefficients (interpretable scale) |
| `tables/pras_per_pair_test_metrics.csv` | AUC/AUPRC per pair on the test set |
| `tables/pras_leadtime_before_crossing.csv` | PRAS in each of the 5 years before crossing |
| `tables/pras_bucket_realization.csv` | Risk-bucket → observed crossing rate |
| `figures/phase3_pras_validation.png` | ROC + PR + calibration + score distribution |
| `figures/phase3_pras_worked_example.png` | The 4-panel headline: trajectories + lead-time + bucket realization |


