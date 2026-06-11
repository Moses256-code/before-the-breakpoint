# Phase 2 — MIC drift estimation, lead-time analysis, and genotype anchoring
**Project:** Before the Breakpoint  
**Dataset:** Pfizer ATLAS (non-USA, 2004–2024), 17 pathogen-drug pairs

---

## What was done

1. **Built a maximum-likelihood interval-censored normal regression** from scratch (`scripts/mic_model.py`). Validated on synthetic data with known parameters — recovered intercept, drift, and dispersion within 1% of truth at n=5,000. This is the right statistical machinery for MIC data; off-the-shelf AFT models force a positive-time framing that obscures interpretation.

2. **Compiled a CLSI / EUCAST reference table** of clinical breakpoints (S, R) and epidemiological cutoffs (ECOFF) for each of the 17 pathogen-drug pairs, all converted to log₂ µg/mL for direct comparison with the model output.

3. **Fitted baseline drift models** (interval-censored normal) for all 17 pairs across their stable-panel windows. Overall annual drift coefficient β₁ (in log₂ dilution steps per year) with 95% CI, plus per-country fits for 981 (country × pair) combinations meeting minimum coverage.

4. **Built diagnostic plots** (cell-level MIC distribution heatmap + trajectory of % above ECOFF / breakpoint / ATLAS R-label) for every pair. These exposed two issues hidden by the regression:
   - **Cefepime 2008–2017 panel artifact**: pre-2012 panel floor at 0.5 mg/L meant every isolate ≤0.5 piled there, inflating the "% above ECOFF" denominator. The apparent decline from 95% to 40% is methodology, not biology. Cefepime analysis restricted to 2018+ in subsequent steps.
   - **Colistin methodology drift**: all three colistin pairs show large *downward* MIC drift (-0.13 to -0.23 log₂/yr), the opposite of the biological reality of rising *mcr*-mediated resistance. This is the documented 2016 CLSI/EUCAST broth-microdilution standardization. Colistin pairs flagged but retained for transparency; **excluded from the headline analysis**.

5. **Switched primary signal metric from model-based to empirical proportion above ECOFF and above breakpoint**, with Wilson 95% CIs. Rationale: several pairs (notably K. pneumoniae × Ceftriaxone, A. baumannii × Meropenem) have bimodal distributions — a wild-type peak plus a resistant peak — that a single-normal model misrepresents. Empirical proportions are robust to bimodality, directly interpretable, and directly comparable to standard global-surveillance metrics (WHO GLASS, ECDC EARS-Net).

6. **Generated the country × year × pair panel** — the 10,010-row matrix that becomes the input to Phase 3's Pre-Resistance Alert Score. For each cell with ≥10 isolates we have: empirical %above-ECOFF, %above-breakpoint, %resistant by ATLAS label, with Wilson CIs.

7. **Computed lead-time estimates** for each country-pair combination: how many years did %above-ECOFF cross 10% *before* %above-breakpoint did?

8. **Genotype overlay analysis** for K. pneumoniae: aligned MIC drift trajectories with KPC, NDM, OXA, VIM, IMP detection rates across country-years. Confirms that the MIC signal is biologically driven (not a measurement artifact) and identifies which β-lactamases drive resistance to which drug.

---

## Key findings

### Finding 1 — The pre-resistance signal works, but selectively

The lead-time analysis (% above ECOFF crossing 10% before % above breakpoint does) split the 17 pairs cleanly into two groups:

**Framework provides genuine lead time (median 1–6 years):**

| Pair | Median lead (yrs) | Max lead | n countries |
|---|---|---|---|
| E. coli × Meropenem | **6.0** | 12 | 5 |
| K. pneumoniae × Ceftazidime-avibactam | **3.0** | 12 | 25 |
| E. cloacae × Meropenem | **3.0** | 14 | 21 |
| K. pneumoniae × Imipenem | **2.0** | 12 | 37 |
| K. pneumoniae × Meropenem | **1.5** | 17 | 38 |
| P. aeruginosa × Ceftazidime-avibactam | **1.0** | 11 | 38 |

**Framework provides no useful lead time (median 0):**
- All cephalosporin pairs (ceftriaxone, cefepime) on Enterobacterales
- A. baumannii × Meropenem (already resistant — past the breakpoint era)
- P. aeruginosa × Cefepime, × Meropenem

**Interpretation.** The framework works where resistance is *incremental* — built from combinations of mechanisms (porin loss + efflux + low-activity carbapenemase) that produce intermediate MIC phenotypes. It fails where resistance comes from a *single-step* horizontal gene transfer event (most ESBLs against ceftriaxone jump straight from MIC <0.06 to >64, with nothing to "creep" through). This is a publishable finding in its own right, and it tells stewards and surveillance bodies *where* the early-warning approach is worth investing in.

### Finding 2 — The headline story: CAZ-AVI resistance in K. pneumoniae

A previously-stable distribution **broke after 2017**:

| Metric | 2012–2017 | 2024 |
|---|---|---|
| % above ECOFF | 10–14% | **24%** |
| % above breakpoint | <2% | **12%** |
| Mean lead time | — | **3.0 years** across 25 countries |

Country-level emergence concentrated in India (consistently 25–45% R from 2018 onward), Greece (rising from 8% to 33%), Argentina (0% → 34%), and Turkey (0% → 27%). Western European countries and East Asia (excluding India) remain mostly below 5%.

**Mechanistically confirmed**: NDM (and other metallo-β-lactamases) carriage rose nearly in lock-step with CAZ-AVI breakpoint resistance (Fig F panel 2). This is biologically correct — avibactam inhibits class A (KPC) and class C and class D (OXA-48) β-lactamases, but does not inhibit class B metallo-β-lactamases. So when NDM became dominant in K. pneumoniae globally, CAZ-AVI's clinical utility eroded.

### Finding 3 — The carbapenem story: K. pneumoniae × Meropenem

A textbook longitudinal resistance trajectory:

| Year | % above ECOFF | % above breakpoint |
|---|---|---|
| 2007 | 14% | 4% |
| 2014 | 14% | 9% |
| 2018 | 20% | 15% |
| 2024 | 27% | 22% |

The country heatmap reveals huge geographic heterogeneity: Greece (47–64% R throughout), India (≥50% from 2017 onward), Italy, Brazil, Argentina, Turkey all >40% by 2024. By contrast Germany, UK, France, Czech Republic, Australia stayed below 5% across the entire 17-year window. **Israel** is a documented public-health success: 22% R in 2007 → 3% by 2024 (the national post-2007 KPC eradication campaign).

The genotype overlay confirms the MIC signal: KPC, NDM, and OXA-48-type detection rates rose roughly in parallel with the MIC trajectory, with OXA-48 and NDM showing the steepest 2018+ acceleration.

### Finding 4 — Patterns that surprised me

- **P. aeruginosa × Meropenem is *improving***: % above ECOFF declined from 30% (2007) to 22% (2024). Possible drivers: better infection control in ICUs, displacement by newer drugs in the formulary, or genuine pressure-driven reduction. Worth a dedicated sensitivity analysis in Phase 3.
- **E. cloacae × Meropenem shows a late 2024 spike** (% above ECOFF jumped to 47%). Small-n likely; needs sample-size sensitivity check.
- **CTX-M-15 detection rates in K. pneumoniae stayed around 26–28%** across 2018–2024, reflecting plateaued ESBL prevalence after a decade of expansion.

---

## The country × year × pair matrix (input to Phase 3)

10,010 country-year cells across the 17 pairs and 81 countries, each carrying: n_isolates, % above ECOFF (+ Wilson CI), % above breakpoint (+ Wilson CI), % resistant per ATLAS labels. Phase 3 will use this matrix to build the **Pre-Resistance Alert Score** — a composite per (country, pair, year) that combines:

- Current % above ECOFF
- Recent rate of change (ΔECOFF over preceding 3 years)
- Distance from breakpoint exceedance (BP gap)

…and validate it by training on 2007–2014 cells and testing whether high-alert (country, pair) combos predicted subsequent 2015–2024 breakpoint-resistance escalation.

---

## Deliverables produced in Phase 2

| File | Description |
|---|---|
| `tables/drift_overall.csv` | Overall annual log₂-MIC drift per pair (17 rows) |
| `tables/drift_per_country.csv` | Country-specific drift estimates (981 rows) |
| `tables/cell_predictions.csv` | Model-based per (country, year) MIC distribution (11,292 rows) |
| `tables/country_year_panel.csv` | **Primary deliverable.** Empirical %above-ECOFF / %above-BP per cell, with Wilson CIs (10,010 rows) |
| `tables/global_yearly_trajectory.csv` | Globally-aggregated yearly trajectory per pair (222 rows) |
| `tables/lead_time_estimates.csv` | Lead-time per (pair, country): years that ECOFF crossing led BP crossing (524 rows) |
| `tables/genotype_country_year_kp.csv` | β-lactamase gene carriage rates for K. pneumoniae country-years (940 rows) |
| `figures/phase2_tier1_diagnostics.png` | Heatmap + trajectory per pair, Tier 1 (carbapenem) |
| `figures/phase2_tier2_diagnostics.png` | Same for Tier 2 (cephalosporins) — exposes the cefepime panel issue |
| `figures/phase2_tier3_diagnostics.png` | Same for Tier 3 (novel β-lactam/BLI + colistin) — exposes the colistin issue |
| `figures/phase2_figA_global_trajectories.png` | **Headline.** Per-pair % above ECOFF (orange) leads % above breakpoint (red) |
| `figures/phase2_figB_lead_time.png` | Lead-time box plots by pair — shows which pairs the framework works for |
| `figures/phase2_figC_kp_cazavi_country.png` | K. pneumoniae × CAZ-AVI: country × year R% heatmap |
| `figures/phase2_figD_kp_meropenem_country.png` | K. pneumoniae × Meropenem: same |
| `figures/phase2_figE_kp_cazavi_ecoff.png` | K. pneumoniae × CAZ-AVI: country × year ECOFF-exceeding % (the pre-resistance reservoir) |
| `figures/phase2_figF_genotype_kp.png` | MIC drift × β-lactamase gene carriage alignment — mechanism confirmation |

---
