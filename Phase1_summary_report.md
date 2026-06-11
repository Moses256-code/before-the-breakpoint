# Phase 1 — Data preparation, QC, and viability assessment
**Project:** Before the Breakpoint — Detecting Hidden AMR Escalation from Longitudinal MIC Drift  
**Dataset:** Pfizer ATLAS (non-USA, 2004–2024)

---

## What was done

1. **Parsed all MIC values** in ATLAS to a clean interval-censored representation.  ATLAS reports MIC as printed dilution labels with `≤` and `>` qualifiers at panel edges; these were mapped to integer log₂ indices so dilution arithmetic is exact and the interval-censored modeling framework can be used directly. The parser handled `≤0.06`, exact dilutions, and `>16`-style censored values, and was validated against ATLAS's full inventory of 18 distinct numeric labels (from 0.001 to 128).
2. **Reshaped** ATLAS from wide-by-drug (845 K isolates × 127 columns) to a tidy long format suitable for analysis. The resulting long table contains **11,428,940 isolate-drug observations** across 30 focus drugs, stored as a 63 MB Parquet file (5× smaller than the source CSV; loads in seconds).
3. **Panel-stability QC** by drug × year: for every drug we identified the minimum and maximum dilution observed each year, the censoring profile, and the *number of distinct panel configurations* used across the 21-year window. This is the test for the "panel reformulation" confounder that can mimic real MIC drift.
4. **Viability scoring** for every Species × Drug combination among 12 priority pathogens, applying explicit criteria: ≥10 years of data inside a stable panel window, median ≥30 isolates per year, and a censoring profile not so extreme that the distribution can't be estimated.
5. **Final menu of 17 pathogen-drug pairs** locked in for Phase 2 modeling, organised in three analytical tiers.

---

## Headline findings

### A. Overall censoring is workable
Across 11.4 M MIC observations: **64.5 % on-step exact**, 18.9 % right-censored, 16.6 % left-censored. The framework will use interval-censored regression (`lifelines` / PyMC), not naïve point estimates.

### B. Panel reformulations are real and need to be respected
Three categories of drug emerged from the QC:

| Pattern | Drugs (example) | Implication |
|---|---|---|
| Single stable panel across 21 years | Doripenem, Cefiderocol, Tigecycline | Free of panel-change confounding |
| 2–3 panel ranges, but boundaries change predictably | Meropenem, Cefepime, Amikacin, Colistin | **Usable** with a stable-panel time window |
| ≥4 panel configurations with erratic boundaries | Ceftazidime (9!), Ceftriaxone (4), Ciprofloxacin (3 with major 2018 jump) | Restricted to a single sub-window each |

The figure `qc02_panel_ranges.png` makes this immediately visual. Every recommended pair below has been windowed to its longest stable-panel period.

### C. Ceftriaxone testing was reduced in 2017–2018
Ceftriaxone disappeared from the ATLAS Gram-negative panel for most years after 2017 (the ceiling dropped from 5–6 to 2, and volume collapsed). The ESBL story therefore runs **2004–2017** using ceftriaxone; cefepime carries it through 2018–2024.

### D. Uganda is too sparse for a primary analytical unit
275 isolates, 2021–2023 only. Will use Uganda as a **case-study showcase** in the final report; the LMIC analytical arm should aggregate across all African and South Asian countries in the dataset.

### E. Carbapenem MIC distributions are heavily left-censored on Enterobacterales
- *E. coli* meropenem: 69.8 % left-censored (most isolates ≤0.06)
- *K. pneumoniae* meropenem: 55.7 %
- *Enterobacter cloacae* meropenem: 55.7 %

This is biologically expected — most isolates remain wild-type susceptible — and is exactly why the "pre-resistance" question is so interesting: the upper-tail movement is hidden inside that giant ≤0.06 pile. Interval-censored modeling estimates this tail correctly; naïve "median MIC" would not.

### F. Acinetobacter baumannii carbapenems are heavily right-censored
A. baumannii meropenem: 58.4 % right-censored (`>16`). This bug is mostly already resistant. It tracks *how high the upper tail has climbed*, not pre-resistance — but still valuable as a contrast case.

---

## Final pathogen-drug menu for Phase 2 (17 pairs)

### Tier 1 — Carbapenem story (headline; 6 pairs)
| Species | Drug | Window | N |
|---|---|---|---|
| K. pneumoniae | Meropenem | 2007–2024 | 86,731 |
| E. coli | Meropenem | 2007–2024 | 100,335 |
| Enterobacter cloacae | Meropenem | 2007–2024 | 33,614 |
| P. aeruginosa | Meropenem | 2007–2024 | 94,048 |
| A. baumannii | Meropenem | 2007–2024 | 35,478 |
| K. pneumoniae | Imipenem | 2012–2024 | 60,600 |

### Tier 2 — ESBL/cephalosporin story (5 pairs)
| Species | Drug | Window | N |
|---|---|---|---|
| E. coli | Ceftriaxone | 2004–2017 | 47,922 |
| K. pneumoniae | Ceftriaxone | 2004–2017 | 38,031 |
| E. coli | Cefepime | 2008–2024 | 97,448 |
| K. pneumoniae | Cefepime | 2008–2024 | 84,435 |
| P. aeruginosa | Cefepime | 2008–2024 | 91,731 |

### Tier 3 — Novel β-lactam/BLI combinations + last-resort (6 pairs)
| Species | Drug | Window | N |
|---|---|---|---|
| E. coli | Ceftazidime avibactam | 2012–2024 | 66,452 |
| K. pneumoniae | Ceftazidime avibactam | 2012–2024 | 60,601 |
| P. aeruginosa | Ceftazidime avibactam | 2012–2024 | 67,767 |
| K. pneumoniae | Colistin | 2014–2024 | 55,567 |
| E. coli | Colistin | 2014–2024 | 58,947 |
| P. aeruginosa | Colistin | 2014–2024 | 64,469 |

**Dropped (insufficient data):** E. coli × Ertapenem (only 2 years of data in ATLAS).

---

## Deliverables produced in Phase 1

| File | Description |
|---|---|
| `data/atlas_isolates.parquet` | One row per isolate; metadata + 23 β-lactamase genotype columns. 6.2 MB. |
| `data/atlas_long.parquet` | One row per isolate-drug observation with parsed MIC intervals. 63 MB, 11.4 M rows. |
| `tables/panel_stability_per_drug.csv` | Summary of panel-range stability per drug. |
| `tables/panel_stability_by_drug_year.csv` | Year-by-year panel range and censoring profile for every drug. |
| `tables/viable_focus.csv` | All 268 candidate Species × Drug combinations, with viability flag. |
| `tables/recommended_pairs.csv` | Final 17-pair menu for Phase 2 modeling. |
| `figures/qc01_drug_year_volume.png` | Heatmap: test volume per drug × year. |
| `figures/qc02_panel_ranges.png` | Per-drug panel range trajectories (the critical QC figure). |
| `figures/phase1_recommended_pairs.png` | Yearly volume for each recommended pair, showing in-window vs out-of-window data. |

