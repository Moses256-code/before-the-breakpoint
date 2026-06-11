# Before the Breakpoint

**A pre-resistance early-warning framework from longitudinal MIC distributions in 845,000 bacterial isolates.**

Vivli AMR Data Challenge 2026 · [Author], [Institution]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

---

## TL;DR

Antimicrobial-resistance surveillance reports `S / I / R` categories defined by clinical breakpoints. These fire *after* a clinically meaningful threshold has already been crossed. We track movement of isolates above the EUCAST **epidemiological cutoff (ECOFF)** but not yet above the clinical breakpoint — the *pre-resistance reservoir* — and combine current level, recent trend, and acceleration into a **Pre-Resistance Alert Score (PRAS)**.

Trained on Pfizer ATLAS predictor years 2007–2014 and validated out-of-time on 2015–2024:

| Metric | Full test set | Low-baseline subset (%BP < 5%) |
| --- | --- | --- |
| **PRAS Bayesian AUC** | **0.910** | **0.835** |
| Naive baseline (%above-BP only) | 0.872 | 0.716 |
| **PRAS advantage** | +0.04 | **+0.12** |

Mean leave-one-pair-out AUC: 0.821 → the signature is pair-generic.

The **dashboard** (`reports/dashboard_v2.html`) lets you explore every country × pathogen-drug pair × year cell interactively.

---

## Repository layout

```
.
├── README.md                   ← you are here
├── LICENSE                     ← MIT
├── requirements.txt            ← Python deps for reproducing every analysis
├── PROGRESS.md                 ← multi-session work log
│
├── scripts/                    ← all analysis code, numbered to indicate execution order
│   ├── mic_parser.py           ← interval-censored MIC parser (validated)
│   ├── mic_model.py            ← interval-censored normal regression + breakpoint tables
│   ├── 01_etl.py               ← ATLAS wide CSV → long parquet
│   ├── 02_panel_qc.py          ← panel-stability quality control
│   ├── 03_viability.py         ← pathogen-drug viability scoring
│   ├── 04_recommend_pairs.py   ← lock in 17-pair menu
│   ├── 05_drift_fit.py         ← baseline interval-censored drift fits
│   ├── 06_diagnostic_plots.py  ← per-pair MIC distribution heatmaps
│   ├── 07_country_year_panel.py← refined country-year-pair matrix
│   ├── 08_headline_figures.py  ← global trajectory, lead time, country heatmaps
│   ├── 09_genotype_overlay.py  ← β-lactamase carriage × MIC drift alignment
│   ├── 10_pras_features.py     ← engineer PRAS features + outcomes
│   ├── 11_pras_fit.py          ← fit + validate frequentist PRAS
│   ├── 12_pras_worked_example.py ← K. pneumoniae × CAZ-AVI worked example
│   ├── 13_dashboard.py         ← dashboard v1 (simple)
│   ├── 14_dashboard_data.py    ← precompute JSON payload for dashboard v2
│   ├── 15_build_dashboard.py   ← inject payload into HTML template
│   ├── 16_bayes_pras.py        ← hierarchical Bayesian PRAS in PyMC
│   ├── 17_lopo_cv.py           ← leave-one-pair-out cross-validation
│   ├── 18_baselines_and_leadtime.py ← naive baselines comparison
│   ├── 19_counterfactual.py    ← precision/recall at matched alert rates
│   ├── 20_early_warning_test.py← THE early-warning test (low-baseline subset)
│   ├── 21_sensitivity.py       ← BP threshold + continuous-sites sensitivity
│   ├── 22_mechanism_stratified.py ← NDM/KPC-stratified analysis
│   ├── 23_make_docx.js         ← generate Vivli_Report.docx
│   ├── 24_pediatric.py         ← pediatric subgroup
│   └── dashboard_template.html ← HTML scaffolding for the interactive dashboard
│
├── data/                       ← persistent intermediate outputs (parquet)
│   ├── atlas_isolates.parquet  ← one row per isolate, metadata + genotype (845K)
│   ├── atlas_long.parquet      ← one row per isolate-drug obs with parsed MICs (11.4M)
│   ├── pras_features.parquet   ← engineered features + outcomes
│   ├── pras_scored.parquet     ← frequentist PRAS predictions
│   ├── pras_bayes_scored.parquet ← Bayesian PRAS posterior summary
│   └── dashboard_payload.json  ← consolidated dashboard data
│
├── tables/                     ← all CSV deliverables (auditable)
│   ├── recommended_pairs.csv
│   ├── viable_focus.csv
│   ├── panel_stability_*.csv
│   ├── country_year_panel.csv  ← the primary surveillance matrix (10,010 rows)
│   ├── global_yearly_trajectory.csv
│   ├── lead_time_estimates.csv
│   ├── genotype_country_year_kp.csv
│   ├── drift_*.csv
│   ├── cell_predictions.csv
│   ├── pras_coefficients.csv
│   ├── pras_bayes_coefficients.csv
│   ├── pras_bayes_summary.csv
│   ├── pras_per_pair_test_metrics.csv
│   ├── pras_baseline_comparison.csv
│   ├── pras_leadtime_*.csv
│   ├── pras_bucket_realization.csv
│   ├── lopo_results.csv
│   ├── early_warning_low_baseline.csv
│   ├── early_warning_hits.csv
│   ├── precision_recall_at_alert_rate.csv
│   ├── sens_bp_thresholds.csv
│   ├── sens_continuous_sites.csv
│   ├── mechanism_summary_latest.csv
│   ├── mechanism_yearly.csv
│   ├── age_stratified_yearly.csv
│   └── pediatric_lead_test.csv
│
├── figures/                    ← PNG figures used in the report
│   ├── qc01_drug_year_volume.png
│   ├── qc02_panel_ranges.png
│   ├── phase1_recommended_pairs.png
│   ├── phase2_tier1_diagnostics.png
│   ├── phase2_tier2_diagnostics.png
│   ├── phase2_tier3_diagnostics.png
│   ├── phase2_figA_global_trajectories.png
│   ├── phase2_figB_lead_time.png
│   ├── phase2_figC_kp_cazavi_country.png
│   ├── phase2_figD_kp_meropenem_country.png
│   ├── phase2_figE_kp_cazavi_ecoff.png
│   ├── phase2_figF_genotype_kp.png
│   ├── phase3_pras_validation.png
│   ├── phase3_pras_worked_example.png
│   ├── phase4_bayes_comparison.png
│   ├── phase4_lopo.png
│   ├── phase4_baseline_comparison.png
│   ├── phase4_counterfactual.png
│   ├── phase4_early_warning.png
│   ├── phase4_leadtime_saved.png
│   ├── phase4_sensitivity.png
│   ├── phase4_mechanism_stratified.png
│   └── phase4_pediatric.png
│
└── reports/                    ← human-facing deliverables
    ├── Phase1_summary.md
    ├── Phase2_summary.md
    ├── Phase3_summary.md
    ├── Vivli_Report.docx        ← the headline Vivli submission
    ├── Vivli_Report.pdf         ← PDF rendering of the above
    ├── Vivli_Report_FullDraft.md← markdown source
    ├── OSF_preregistration.md   ← pre-registration template
    ├── Manuscript_outline.md
    ├── dashboard.html           ← dashboard v1 (simple)
    └── dashboard_v2.html        ← dashboard v2 (Palantir-style, recommended)
```

---

## Reproducing the analysis

### 1. Get the data
The Pfizer ATLAS surveillance database is accessible via the [Vivli AMR Register](https://amr.vivli.org). Request access and download the non-USA partition as CSV. Place at `/home/claude/atlas_vivli_2004_2024_nonUSA.csv` (or update the path constants in the scripts).

### 2. Set up the environment
Tested with Python 3.12, Ubuntu 24.04.
```bash
pip install -r requirements.txt
```

Node.js 22+ for the docx report generator:
```bash
npm install -g docx
```

### 3. Run the pipeline
The scripts are numbered to indicate execution order. From `scripts/`:
```bash
python 01_etl.py                  # ETL: CSV → parquet, ~3 min
python 02_panel_qc.py              # panel stability QC
python 03_viability.py             # viability scoring
python 04_recommend_pairs.py       # final 17 pairs
python 05_drift_fit.py             # interval-censored drift fits, ~3 min
python 06_diagnostic_plots.py      # per-pair diagnostics
python 07_country_year_panel.py    # build the primary surveillance matrix
python 08_headline_figures.py      # phase 2 headline figures
python 09_genotype_overlay.py      # mechanism alignment
python 10_pras_features.py         # features + outcomes
python 11_pras_fit.py              # frequentist PRAS
python 12_pras_worked_example.py   # worked-example figure
python 14_dashboard_data.py        # build dashboard payload
python 15_build_dashboard.py       # build dashboard HTML
python 16_bayes_pras.py            # Bayesian PRAS, ~1 min (NUTS)
python 17_lopo_cv.py               # leave-one-pair-out CV, ~1 min
python 18_baselines_and_leadtime.py # baseline comparison
python 19_counterfactual.py        # precision-recall analysis
python 20_early_warning_test.py    # THE early-warning test
python 21_sensitivity.py           # sensitivity analyses
python 22_mechanism_stratified.py  # NDM/KPC stratification
python 24_pediatric.py             # pediatric subgroup
node 23_make_docx.js               # generate Word report
```

Total runtime: ~15–20 minutes on a modern laptop.

### 4. Open the dashboard
```bash
# Just open in any browser:
open reports/dashboard_v2.html
```

---

## Key scientific results

| Result | Headline number |
| --- | --- |
| Out-of-time test AUC (Bayesian) | **0.910** |
| Out-of-time test AUPRC | 0.852 |
| Mean LOPO-CV AUC | 0.821 |
| **Early-warning AUC (low-baseline subset)** | **0.835** vs naive 0.716 |
| Bucket realization, PRAS > 0.75 → subsequently crossed | 97% |
| Bucket realization, PRAS < 0.10 → subsequently crossed | 8% |

For the worked example *K. pneumoniae* × Ceftazidime-avibactam: pre-2018 PRAS correctly elevated Greece (0.675), Turkey (0.168), Brazil (0.194), and South Africa (0.123) above control countries (Germany 0.059, UK 0.073, France 0.058) before any of them crossed the 10% breakpoint resistance threshold. Mechanistically anchored to rising NDM carriage.

---

## Citing this work

If you use this code or data products, please cite:

> [Author], [Year]. *Before the Breakpoint: a pre-resistance early-warning framework from longitudinal MIC distributions in 845,000 bacterial isolates.* Vivli AMR Data Challenge 2026. https://github.com/[user]/before-the-breakpoint

OSF pre-registration: [DOI to be added].

## License

MIT. The Pfizer ATLAS data itself is governed by Vivli's data sharing terms; this repository contains analysis code and *derived statistics*, never raw isolate-level data.

## Acknowledgements

Pfizer ATLAS surveillance programme. Vivli AMR Register for data access. CLSI and EUCAST for breakpoint/ECOFF definitions.
