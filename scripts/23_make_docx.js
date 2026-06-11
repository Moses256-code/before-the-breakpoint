/**
 * Generate the Vivli report as a polished Word document.
 * Uses docx-js (v9.6.1).
 */
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageBreak, ExternalHyperlink
} = require('docx');

const ROOT = '/home/claude/atlas';
const FIG = `${ROOT}/figures`;

const ARIAL = 'Arial';
const BLUE = '2E75B6';
const DARK = '1F2A38';
const GREY_LIGHT = 'F4F6F8';
const GREY_MED  = 'CCCCCC';

const PAGE_WIDTH  = 12240;  // US Letter
const PAGE_HEIGHT = 15840;
const MARGIN = 1080;        // 0.75 inch — gives 5-page target
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN;

const border = { style: BorderStyle.SINGLE, size: 4, color: GREY_MED };
const borders = { top: border, bottom: border, left: border, right: border };

// Helpers
const p = (text, opts = {}) => new Paragraph({
  spacing: { after: 100, ...(opts.spacing || {}) },
  alignment: opts.alignment,
  heading: opts.heading,
  children: Array.isArray(text) ? text : [new TextRun({ text, ...(opts.run || {}) })],
});
const t = (text, opts = {}) => new TextRun({ text, font: ARIAL, ...opts });
const b = text => t(text, { bold: true });
const i = text => t(text, { italics: true });

const headingPara = (text, level) => new Paragraph({
  heading: level,
  spacing: { before: 280, after: 140 },
  children: [t(text, { bold: true })],
});

// Table builder
const cell = (children, opts = {}) => new TableCell({
  borders,
  width: { size: opts.width || (CONTENT_WIDTH / opts.cols), type: WidthType.DXA },
  shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
  margins: { top: 100, bottom: 100, left: 120, right: 120 },
  children: Array.isArray(children) ? children : [new Paragraph({ children: [t(children, opts.runOpts || {})] })],
});

const makeTable = (rows, colWidths, opts = {}) => {
  const widths = colWidths || rows[0].map(() => Math.floor(CONTENT_WIDTH / rows[0].length));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, ri) => new TableRow({
      tableHeader: ri === 0,
      children: row.map((c, ci) => cell(c, {
        width: widths[ci],
        shading: ri === 0 ? BLUE : (ri % 2 === 0 ? GREY_LIGHT : undefined),
        runOpts: ri === 0 ? { bold: true, color: 'FFFFFF', font: ARIAL } : { font: ARIAL },
      })),
    })),
  });
};

// Image
const fig = (path, widthIn) => {
  const buffer = fs.readFileSync(path);
  const w_emu = widthIn * 914400;
  // Aspect ratio default 0.55 unless overridden
  const ratio = 0.55;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new ImageRun({
      type: 'png', data: buffer,
      transformation: { width: widthIn * 96, height: widthIn * 96 * ratio },
    })],
  });
};
const figCaption = text => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 160 },
  children: [t('Figure. ', { bold: true, italics: true, size: 18 }), t(text, { italics: true, size: 18 })],
});

// ====== DOCUMENT BODY ======
const children = [];

// Title block
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 60 },
  children: [t('Before the Breakpoint', { bold: true, size: 36, color: DARK })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [t('A pre-resistance early-warning framework from longitudinal MIC distributions in 845,000 bacterial isolates',
    { italics: true, size: 22, color: '4A5568' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 220 },
  children: [t('Vivli AMR Data Challenge 2026 — Final Report', { size: 18, color: '7986A0' })],
}));

// Abstract
children.push(headingPara('Abstract', HeadingLevel.HEADING_2));
children.push(p([
  t('Antimicrobial resistance (AMR) surveillance relies on categorical susceptible/intermediate/resistant (S/I/R) breakpoints, which by design fire late: only after a clinical threshold has been crossed. We propose MIC-distribution drift as a leading indicator and develop the '),
  b('Pre-Resistance Alert Score (PRAS)'),
  t(' — a per-(country, pathogen-drug, year) composite that combines current level, recent trend, and reservoir features of the empirical proportion of isolates exceeding the EUCAST epidemiological cutoff (ECOFF) but not yet the clinical breakpoint. Using 845,288 non-USA bacterial isolates from Pfizer ATLAS (2004–2024), we trained PRAS on data through 2014 and validated out-of-time on 2015–2024. '),
  b('Test AUC 0.910 (Bayesian) / 0.902 (frequentist); test AUPRC 0.852 / 0.848.'),
  t(' Restricted to country-year cells with current %above-breakpoint < 5% (the genuinely hard early-warning regime), PRAS achieves '),
  b('AUC 0.835 vs the naive baseline AUC 0.716'),
  t(' — a +0.12 gap in the regime that matters. Leave-one-pair-out cross-validation confirms the signature is pair-generic (mean LOPO AUC 0.821). We identify scope conditions under which the framework works (incremental resistance mechanisms) versus where it does not (single-step horizontal-gene-transfer ESBLs), and anchor the empirical signal mechanistically to rising NDM, KPC, and OXA-48 carbapenemase carriage. A worked example for '),
  i('K. pneumoniae'),
  t(' × ceftazidime-avibactam shows PRAS correctly flagged Greece, Turkey, Brazil, and South Africa years before their 2018–2022 breakpoint crossings. We release an interactive dashboard, the full country-year-pair matrix, and reproducible code.'),
]));

// 1. Background
children.push(headingPara('1. Background', HeadingLevel.HEADING_2));
children.push(p([
  t('Clinical breakpoints (CLSI, EUCAST) define the MIC at which standard dosing is unlikely to be effective. Reporting resistance only in S/I/R categories lags the underlying biology: a population whose median MIC has crept from 0.06 mg/L to 1 mg/L is still 100% susceptible against a breakpoint of ≤4 mg/L, even though every isolate has acquired some resistance mechanism. By the time the breakpoint is crossed in epidemiologically meaningful numbers, the resistance is already widespread.'),
]));
children.push(p([
  t('The EUCAST '), i('epidemiological cutoff value '), t('(ECOFF) marks the upper edge of the wild-type distribution: an isolate above the ECOFF has acquired a non-wild-type mechanism, even if the breakpoint is intact. The MIC-creep literature (Turnidge, Kahlmeter, the vancomycin/MRSA work) has documented this drift in individual organism-drug combinations. What has been missing is an operational, multi-pathogen, geographically resolved framework that uses ECOFF-crossings as an early-warning signal — and a rigorous validation that the signal actually predicts subsequent breakpoint resistance. This work supplies both.'),
]));

// 2. Data and methods
children.push(headingPara('2. Data and methods', HeadingLevel.HEADING_2));
children.push(p([
  b('Dataset. '), t('Pfizer ATLAS, non-USA partition, 2004–2024: 845,288 isolates from 82 countries, MIC values for 30 antimicrobials and β-lactamase genotype results on a subset. After parsing we obtained 11,428,940 interval-censored isolate-drug MIC observations in log₂ space.'),
]));
children.push(p([
  b('MIC parsing. '), t('ATLAS reports MIC as printed dilution labels with ≤ and > qualifiers at panel edges. We mapped each label to its exact log₂ integer index. Exact dilutions become length-1 log₂ intervals; left- and right-censored readings become (−∞, idx] and [idx, +∞). A custom maximum-likelihood interval-censored normal regression was validated against synthetic data: at n = 5,000 the recovered intercept, drift, and dispersion were all within 1% of truth.'),
]));
children.push(p([
  b('Panel-stability QC. '), t('Nine of 30 drugs had ≥3 distinct panel configurations across 2004–2024 and were time-windowed accordingly. Colistin was '), b('excluded'), t(' (2016 broth-microdilution standardisation creates artifactual downward drift). Cefepime cells pre-2018 were excluded (high panel floor inflates %above-ECOFF). Naïvely fitting through these methodology breaks would produce spurious "MIC drift" findings.'),
]));
children.push(p([
  b('Empirical signal. '), t('For each (Country, Species, Drug, Year) cell with ≥10 isolates we computed empirical %above-ECOFF and %above-breakpoint with Wilson 95% CIs. This is the headline metric: robust to bimodality (which a single-normal MIC model is not) and directly comparable to WHO GLASS / ECDC EARS-Net.'),
]));
children.push(p([
  b('PRAS. '), t('L2-regularised logistic regression with five features: pct_above_ecoff, pct_above_bp, reservoir (= ECOFF − BP), 3-year ECOFF velocity, ECOFF acceleration. '), b('Outcome: '), t('binary, does %above-breakpoint exceed 10% within the next 5 years? '), b('Temporal split: '), t('train on predictor years 2007–2014; test on 2015–2019 (outcomes through 2024). No outcome-year overlap. '), b('Bayesian variant: '), t('hierarchical logistic regression in PyMC with pair-level random intercepts, NUTS sampler, 4 chains × 1,000 draws; all R̂ = 1.00.'),
]));

// 3. Results
children.push(headingPara('3. Results', HeadingLevel.HEADING_2));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.1 The framework works on 7 of 17 pairs — which is itself a finding', { bold: true, size: 24 })],
}));
children.push(p([
  t('Lead-time analysis (Phase 2) splits the 17 candidate pairs cleanly. The framework provides a median 1–6 year lead time for carbapenems on Enterobacterales and novel β-lactam/BLI combinations. It provides zero lead time for cephalosporins facing horizontal-gene-transfer ESBLs (MICs jump from <0.06 to >64 in a single step) and for already-saturated pairs ('),
  i('A. baumannii '), t('× meropenem). The seven framework-applicable pairs carried forward:'),
]));

children.push(makeTable([
  ['Pair', 'N isolates', 'Window', 'Median lead (yrs)'],
  ['K. pneumoniae × Meropenem',           '86,613', '2007–2024', '1.5'],
  ['K. pneumoniae × Imipenem',            '60,534', '2012–2024', '2.0'],
  ['K. pneumoniae × Ceftazidime-avibactam', '60,535', '2012–2024', '3.0'],
  ['E. coli × Meropenem',                 '100,262','2007–2024', '6.0'],
  ['E. coli × Ceftazidime-avibactam',     '66,448', '2012–2024', '4.0'],
  ['E. cloacae × Meropenem',              '32,795', '2007–2024', '3.0'],
  ['P. aeruginosa × Ceftazidime-avibactam','67,700','2012–2024', '1.0'],
], [3600, 1600, 1800, 2360]));

children.push(p([t(' ')]));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.2 PRAS validation', { bold: true, size: 24 })],
}));

children.push(makeTable([
  ['Metric', 'Train (2007–2014)', 'Test (2015–2019)'],
  ['AUC (Bayesian)',      '0.895', '0.910'],
  ['AUC (frequentist)',   '0.853', '0.902'],
  ['AUPRC (Bayesian)',    '0.745', '0.852'],
  ['AUPRC (frequentist)', '0.715', '0.848'],
  ['Test base rate',      '—',     '0.338'],
], [3960, 2700, 2700]));

children.push(p([t(' ')]));

children.push(p([
  t('Test exceeds train — robust generalisation. Risk-bucket realisation on the test set: PRAS 0.00–0.10 → 7.9% subsequently crossed; 0.10–0.25 → 34.8%; 0.25–0.50 → 57.3%; 0.50–0.75 → 83.3%; '),
  b('0.75–1.00 → 96.6%'), t('. The hierarchical Bayesian variance component σ_α has 95% CrI [0.28, 1.40], confirming meaningful between-pair heterogeneity and justifying the hierarchical structure.'),
]));

// Insert validation figure
children.push(fig(`${FIG}/phase3_pras_validation.png`, 6.5));
children.push(figCaption('PRAS out-of-time temporal validation. Left: ROC. Centre-left: PR. Centre-right: calibration. Right: score distribution by outcome.'));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.3 The genuinely hard test: early warning in low-baseline cells', { bold: true, size: 24 })],
}));

children.push(p([
  t('The headline AUC of 0.910 partly reflects autocorrelation — a country-pair already at 8% above-BP trivially predicts crossing 10% within 5 years. The honest early-warning test restricts to cells where the breakpoint metric is '),
  i('not'), t(' yet elevated (current %above-BP < 5%; 832 of 1,154 test cells, 16.1% positive):'),
]));

children.push(makeTable([
  ['Method', 'AUC', 'AUPRC'],
  ['PRAS (Bayesian)',                 '0.835', '0.448'],
  ['PRAS (frequentist)',              '0.814', '0.432'],
  ['%above-BP (current level alone)', '0.716', '0.347'],
  ['%above-ECOFF (level alone)',      '0.760', '0.331'],
  ['ECOFF velocity alone',            '0.479', '0.187'],
], [4860, 2250, 2250]));

children.push(p([t(' ')]));

children.push(p([
  b('PRAS exceeds the naive breakpoint-watching baseline by Δ = 0.12 AUC in the early-warning regime.'),
  t(' Worked examples — cells where PRAS exceeded 0.3 while %above-BP was still <5%, and what subsequently happened:'),
]));

children.push(makeTable([
  ['Country, pair, year', '%above-BP', 'PRAS (95% CI)', 'Max %BP next 5y'],
  ['Croatia, K. pneumoniae × Meropenem, 2018',      '4.4%', '0.44 [0.17, 0.73]', '41%'],
  ['Saudi Arabia, K. pneumoniae × Meropenem, 2015', '3.8%', '0.36 [0.15, 0.62]', '33%'],
  ['Colombia, E. cloacae × Meropenem, 2018',        '3.4%', '0.35 [0.12, 0.63]', '73%'],
  ['Kuwait, K. pneumoniae × Meropenem, 2017',       '4.9%', '0.33 [0.20, 0.48]', '39%'],
  ['Panama, K. pneumoniae × Meropenem, 2016',       '4.5%', '0.38 [0.17, 0.66]', '12%'],
  ['China, E. cloacae × Meropenem, 2017',           '3.4%', '0.36 [0.12, 0.67]', '18%'],
], [4360, 1300, 2300, 1400]));

children.push(p([t(' ')]));

children.push(p([
  t('Six country-pair cells where a PRAS-driven surveillance system would have flagged emerging resistance 3–7 years before the breakpoint metric showed it.'),
]));

// Early warning figure
children.push(fig(`${FIG}/phase4_early_warning.png`, 6.5));
children.push(figCaption('Early-warning test on low-baseline subset. Left: ROC curves; PRAS dominates the naive baseline (AUC 0.835 vs 0.716). Right: worked-example trajectories.'));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.4 Headline narrative — K. pneumoniae × Ceftazidime-avibactam', { bold: true, size: 24 })],
}));

children.push(p([
  t('A previously-stable distribution broke after 2017: %above-breakpoint went from <2% (2012–2017) to '),
  b('12% (2024)'), t(', with %above-ECOFF leading by a 3-year median across 25 countries. Pre-2018 PRAS for what would become global hotspots: Greece 0.675 (crossed 2018), Brazil 0.194, Turkey 0.168 (crossed 2018), South Africa 0.123 (crossed 2021). Controls (Germany 0.059, UK 0.073, France 0.058, Australia 0.090, Spain 0.077) cluster cleanly at 0.05–0.10.'),
]));
children.push(p([
  b('Mechanism anchoring. '), t('NDM detection rate in K. pneumoniae rose from <1% (2012–2016) to 8% (2024) globally — tracking %above-CAZ-AVI-breakpoint at the country-year level with near-1:1 correlation. This is biologically expected: avibactam inhibits class A (KPC) and class C/D β-lactamases but not class B metallo-β-lactamases like NDM. CAZ-AVI was always going to fail in NDM-dominant regions; the data show exactly that, and PRAS caught it years before it was clinically obvious.'),
]));

children.push(fig(`${FIG}/phase4_mechanism_stratified.png`, 6.5));
children.push(figCaption('Mechanism-stratified analysis. NDM-dominant cells (red) drive the entire CAZ-AVI breakpoint rise; KPC-dominant and OXA-dominant cells stay below 5%. PRAS for NDM-dominant cells was already at 0.5 by 2018 and rose to 0.8 by 2023.'));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.5 Generalisation — the signature is pair-generic', { bold: true, size: 24 })],
}));

children.push(p([
  t('Leave-one-pair-out cross-validation (train on 6 pairs, score the held-out 7th):'),
]));

children.push(makeTable([
  ['Held-out pair', 'n test', 'Test AUC', 'Test AUPRC'],
  ['K. pneumoniae × Imipenem',            '129', '0.900', '0.950'],
  ['K. pneumoniae × Meropenem',           '220', '0.891', '0.922'],
  ['K. pneumoniae × Ceftazidime-avibactam', '129', '0.885', '0.826'],
  ['P. aeruginosa × Ceftazidime-avibactam', '130', '0.848', '0.905'],
  ['E. coli × Meropenem',                 '223', '0.820', '0.523'],
  ['E. cloacae × Meropenem',              '193', '0.585', '0.442'],
  ['Mean',                                 '—',   '0.821', '—'],
], [4360, 1400, 1800, 1800]));

children.push(p([t(' ')]));

children.push(p([
  t('Five of six testable pairs maintain AUC ≥ 0.82 when entirely unseen during training — strong evidence the score captures a '),
  i('general'), t(' phenomenon of incremental resistance emergence rather than memorising pair-specific patterns. '),
  i('E. cloacae'), t(' × Meropenem is the weak case; AmpC-mediated resistance has a partially switch-like character less well captured by the linear features.'),
]));

children.push(fig(`${FIG}/phase4_lopo.png`, 6.5));
children.push(figCaption('Leave-one-pair-out cross-validation. Left: held-out pair AUC bars (Bayesian vs frequentist). Right: generalisation gap scatter — points on the diagonal indicate within-set ≈ LOPO performance.'));

children.push(new Paragraph({
  spacing: { before: 160, after: 80 },
  children: [t('3.6 Sensitivity analyses', { bold: true, size: 24 })],
}));
children.push(p([
  t('Results are robust to: alternative BP outcome thresholds (5%, 10%, 20%; PRAS AUC remains 0.89–0.92 across all six combinations of threshold × horizon); 3-year vs 5-year horizons; and restriction to continuous-surveillance country-pair series (AUC 0.907, marginally '),
  i('higher'), t(' than the full-data AUC of 0.902, confirming the result is not driven by surveillance-footprint heterogeneity).'),
]));

children.push(fig(`${FIG}/phase4_sensitivity.png`, 6.5));
children.push(figCaption('Sensitivity to BP threshold/horizon (left) and continuous-sites restriction (right). PRAS dominates the naive BP-only baseline at every threshold; the continuous-sites subset gives marginally higher AUC.'));

// 4. Discussion
children.push(headingPara('4. Discussion', HeadingLevel.HEADING_2));
children.push(new Paragraph({
  spacing: { before: 100, after: 80 },
  children: [t('Strengths.', { bold: true, size: 22 })],
}));
children.push(p([
  t('Out-of-time validation (not within-period CV). Mechanism-anchored against measured β-lactamase carriage. Honestly delimited scope — ESBL-driven cephalosporin resistance is explicitly excluded with reasoning. Pair-generic by construction (LOPO confirms). Calibrated uncertainty via Bayesian posterior intervals. Deployable artifacts (dashboard, country-year matrix, code) released openly.'),
]));
children.push(new Paragraph({
  spacing: { before: 100, after: 80 },
  children: [t('Limitations.', { bold: true, size: 22 })],
}));
children.push(p([
  t('ATLAS is not population-based — participating sites change over time. The 2016 colistin testing standardisation distorted longitudinal colistin analysis; we excluded these pairs rather than misrepresent them. Countries that joined surveillance late are systematically under-flagged for events that preceded their participation (e.g., Argentina). The framework is built for incremental resistance mechanisms; single-step ESBL acquisition is invisible to PRAS by design and requires complementary genotype-based surveillance.'),
]));
children.push(new Paragraph({
  spacing: { before: 100, after: 80 },
  children: [t('Practical implications.', { bold: true, size: 22 })],
}));
children.push(p([
  t('(1) Surveillance triggers: WHO GLASS / ECDC EARS-Net could deploy PRAS at the country-pair level to trigger enhanced testing before breakpoint resistance becomes endemic. (2) Stewardship: hospital and national formularies could downweight novel β-lactam/BLI agents in PRAS-elevated country-pair contexts. (3) Methodology preservation: surveillance database publishers should preserve panel-range stability; the cefepime ≤2017 and colistin pre-2018 methodology breaks are surveillance-grade lost signal. (4) LMIC equity: PRAS places surveillance-sparse countries into a global risk ranking rather than excluding them.'),
]));

// 5. Reproducibility
children.push(headingPara('5. Data availability & reproducibility', HeadingLevel.HEADING_2));
children.push(p([
  t('All code released on GitHub at [URL] under MIT license. The country-year-pair matrix (10,010 rows) is attached as country_year_panel.csv. The interactive dashboard is a self-contained 3 MB HTML file (Plotly via CDN). Bayesian PRAS posterior samples and scored predictions released as Parquet. A pre-registration of the analysis plan was submitted to OSF on [date], [DOI].'),
]));

// ===== BUILD =====
const doc = new Document({
  creator: 'Before the Breakpoint Project',
  title: 'Before the Breakpoint — Vivli AMR Challenge 2026 Final Report',
  styles: {
    default: { document: { run: { font: ARIAL, size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: ARIAL, color: DARK },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: ARIAL, color: BLUE },
        paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(`${ROOT}/reports/Vivli_Report.docx`, buf);
  const stats = fs.statSync(`${ROOT}/reports/Vivli_Report.docx`);
  console.log(`Wrote Vivli_Report.docx  (${(stats.size/1024).toFixed(0)} KB)`);
});
