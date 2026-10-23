# MAOA Project

Multi-electrode array (MEA) analysis pipeline developed in collaboration with Sandra Sanchez.

## Overview

Code for analyzing MEA recordings from the spike lists exported by the Axion software (spike detection is done in Axion): firing, burst, network and synchrony quantification, statistics, and visualization.

## Collaboration

Developed in collaboration with Sandra Sanchez.

*Co-author — manuscript in preparation.*

---

## Organoid MEA analysis (Exp1, Exp3)

Spiking, bursting, network and synchrony analysis of MAOA cortical organoids recorded on Axion MEA plates.
This README is the running record of **how the data were processed and analysed**, written so it can be
turned into the Methods section of the paper. Every analysis decision goes in the [decision log](#decision-log).

| Notebook | What it produces |
|---|---|
| `spk_burs_net_analysis.ipynb` | Builds `processed_df.pkl` from the raw Axion `*_spike_list.csv` files |
| `spk_burs_net_analysis_figure.ipynb` | Exp3 (7.4–8.5 months) paper figures → `figures/`, `figures/supplementary/` |
| `spk_burs_net_analysis_figure_exp1.ipynb` | Exp1 (2.4–4.8 months), same analysis → `figures/exp1/` |
| `spk_burs_net_analysis_figure_both.ipynb` | **Start here.** Both experiments, organised by feature (Exp1 plot above Exp3 plot), with every data-preparation step explained, a guide to reading each plot and each statistics output, and a summary table of the statistics. Display only, saves nothing |

Code: `helper_files/maoa_plate_loader.py` (loading, metrics, plate layouts, data cleaning) and
`helper_files/mea_feats_analysis.py` (plots, statistics).

---

## Methods (draft for the paper)

### Recordings and experiments
Organoids were plated on 12-well MEA plates (64 electrodes per well; Axion BioSystems) and recorded weekly.
Spike times were exported from the Axion software as spike lists (`*_spike_list.csv`) and analysed in Python.

| Experiment | Plates | Recording dates | Organoid age | Used for |
|---|---|---|---|---|
| Exp1 | MAOA1, MAOA2, MAOA3 | 2025-04-14 → 2025-09-19 | 73–231 days | analysed window 2025-04-14 → 2025-06-27, 2.4–4.8 months, 10 dates) |
| Exp2 | MAOA72, MAOA76, MAOA77 | 2025-06-27 → 2025-09-19 | 134–218 days | not used |
| Exp3 | MAOA72, MAOA76, MAOA77, re-plated into a new layout | 2025-09-25 → 2025-10-30 | 224–259 days (7.4–8.5 months) | all 5 dates analysed |

**Organoid age** = recording date − origin date. MAOA1–3 origin = 2025-01-31 (53 days old on 2025-03-25);
MAOA72/76/77 origin = 2025-02-13 (taken as 120 days old on 2025-06-13, from "4 months old"). Age in months = days / 30.44, rounded to 0.1.
The Exp1 and Exp3 data are different organoids on different plates; whether they are different differentiation
batches is to be confirmed (see below).

### Activity metrics
All metrics are computed per recording from the exported spike times (`maoa_plate_loader.py`).
Recording duration is taken as the time between the first and last spike in the file.

- **Electrode firing rate**: spikes / duration, for each electrode with ≥ 1 spike.
- **Electrode burst rate**: ISI-based bursts (≥ 5 consecutive spikes with every inter-spike interval ≤ 100 ms) / duration, for each electrode with ≥ 1 burst.
- **Well firing rate**: mean electrode firing rate over the well's electrodes with ≥ 1 spike.
- **Well burst rate**: mean electrode burst rate over the well's electrodes with ≥ 1 burst.
- **Network event rate**: spikes from all electrodes in a well are pooled; a network event is a run of spikes with consecutive ISIs ≤ 100 ms containing ≥ 50 spikes and involving ≥ 35 % of the well's spiking electrodes. Rate = events / duration.
- **Synchrony index**: spike trains of electrodes with ≥ 10 spikes are binned at 20 ms; the index is the mean zero-lag Pearson correlation over all electrode pairs in the well (NaN if < 2 such electrodes).
- **Electrode-level analyses** of firing rate include only electrodes firing ≥ 5 spikes/min (0.083 Hz). Electrode-level burst rate includes all bursting electrodes.

### Genotype groups and clones
| Group (figure label) | Clones |
|---|---|
| Xa:WT/Xi:MUT | C30 |
| Xa:MUT/Xi:WT | C71, C3 |
| MAOA/B⁻/Y (KO) | GM |

Clones not used in this analysis were removed per plate: H9, C12, C54, C90 from all plates, and additionally
C3 from MAOA2 and MAOA3. Wells analysed:

| | KO | Xa:MUT/Xi:WT | Xa:WT/Xi:MUT | Total |
|---|---|---|---|---|
| Exp1 (2.4–4.8 months) | 9 | 6 (C71 5, C3 1) | 5 | 20 wells × 10 dates |
| Exp3 (7.4–8.5 months) | 5 | 12 (C71 6, C3 6) | 4 | 21 wells × 5 dates (MAOA76 A4 excluded, see below) |

### Data cleaning and exclusions
1. **Mislabelled / duplicate files** (`clean_recordings()`). For four recordings, the plate folder and the file name disagreed.
   The true plate was identified by correlating each file's electrode-level firing pattern (768 electrodes,
   log firing rate) with each candidate plate's recordings in the neighbouring weeks: the same plate correlates at
   r ≈ 0.6–0.9, different plates at r ≈ 0.0–0.15. Corrections (all Exp1, outside the analysed window):
   - 2025-08-22: second file in the MAOA3 folder is plate 2 → reassigned to MAOA2.
   - 2025-09-06: file named "PLATE_02 (001)" in the MAOA3 folder is a second recording of plate 2 → reassigned, then dropped as a duplicate.
   - 2025-09-19: MAOA2 and MAOA3 files were swapped → swapped back.
   - 2025-10-16: MAOA77 file is named "plate 76" but its data is plate 77 → no change.

   Cell line and age are re-derived from the correct plate's layout. When a plate was recorded more than once on a
   day (2025-10-30, all three Exp3 plates; same plate confirmed, r = 0.74–0.88), only the first recording is kept.
   A fingerprint check of all 120 plate recordings after cleaning found every recording matching its own plate,
   except dates where the comparison is not valid (layout change on 2025-09-25; the artefact on 2025-05-09).
2. **Excluded recording days**:
   - 2025-09-26: bad recording.
   - 2025-05-09: noise artefact. Every electrode in every well fired at ~0.8 Hz (median electrode rate 0.84 Hz vs 0.002–0.017 Hz in the surrounding weeks), including normally silent KO wells, while synchrony fell to ~0. This is consistent with noise detection, not neuronal activity.

   **Excluded wells**: MAOA76 A4 (GM / KO) from 2025-09-25 on (all of Exp3). The organoid detached at the
   September re-plating. The well was near-silent at every Exp3 recording (0–5 of 64 electrodes active; well
   firing 0.004–0.038 Hz vs 0.011–0.161 Hz for the other KO well on the same plate). Its detachment had been
   noted in the plate layout, but the well had not been removed from the analysis until 2026-10-23.
3. **Good-data window** (`get_good_data_dates()`), the same rule for every experiment: a recording day is good if
   ≥ 40 % of wells are active (well firing rate ≥ 5 spikes/min), and the window runs from the first recording to the
   first day that fails. Exp3: all 5 days pass (lowest 52 %). Exp1: window 2.4–4.8 months (activity falls to 30 % at 5.1 months).
   The fall in active wells in Exp1 after 3.4 months is driven by KO wells becoming silent while
   both heterozygous groups stay active; the same KO silencing is seen in Exp3 from 7.8 months.

### Statistics (`run_mixed_model_stats()`)
The same wells are recorded every week, and electrodes are nested within wells, so observations are not independent.
All inferential statistics therefore use linear mixed-effects models (statsmodels `MixedLM`, maximum likelihood):

- **Well level**: `value ~ group × age + plate`, random intercept per well.
- **Electrode level**: the same fixed effects, random intercept per well and per electrode within well.
- Age is categorical (recording day); plate is a fixed effect.
- **Transform**: firing, burst and network rates are square-root transformed. This gave the most symmetric
  residuals in both experiments: raw values were right-skewed (Shapiro p < 1e-4 in Exp3), and a log transform
  over-corrected when wells fell silent in Exp1. The synchrony index can be slightly negative and is analysed untransformed.
- **Tests**:
  - group effect, age effect and group × age interaction ("trajectory") by likelihood-ratio tests between nested models;
  - overall pairwise group differences from the model without interaction (Wald z-tests on the fixed-effect contrasts), Holm-corrected over the three pairs;
  - pairwise differences at each age from the interaction model, Holm-corrected across all ages × pairs.
- When there is only one observation per well (single-day, well-level comparisons), ordinary least squares with plate as a covariate is used.
- Each model is fitted with several optimisers (L-BFGS, BFGS, Powell) and the best likelihood is kept.
- Significance: * p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001.

### Figures
- **Trajectory plots**: box plots per group and age (median, IQR, whiskers 1.5 × IQR, outlier markers hidden), with points for individual wells or electrodes.
  - Dashed lines: per-group linear fit of the raw values against age (with 95 % CI), evaluated at each recorded age. They are visual only and not used for inference.
  - Text box: the group × age test. Pairwise significance matrices show overall group differences and are drawn when the group effect is significant.
  - Electrode-level plots use a broken y-axis (lower panel capped at 11 Hz for firing and 0.25 Hz for burst).
- **Bar plots**: mean ± SEM of all well-day (or electrode-day) observations, with individual observations overlaid. Stars are the mixed-model overall pairwise tests. The SEM is descriptive; it does not account for repeated measures, but the stars do.
- **Supplementary figures**: the same analyses with each clone as its own group.

### Software
Python 3.9.7, pandas 1.5.3, numpy 1.23.5, statsmodels 0.13.5, scipy 1.10.1, matplotlib 3.7.3, seaborn 0.12.2.

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| ≤ 2026-10 | Exp3 used for the main paper figure; Sept 26 2025 excluded; unused clones removed per plate | Original analysis |
| 2026-10-20 | Added Exp1 as an additional analysis, using the same pipeline as Exp3 | Requested by Sandra |
| 2026-10-23 | Experiments labelled by name and analysed age range (Exp1: 2.4–4.8 months, Exp3: 7.4–8.5 months) instead of "2-month" / "7-month"; Exp1 notebook and figure folder renamed to `_exp1` / `figures/exp1/` | The shorthand did not match the ages analysed |
| 2026-10-20 | Age is taken from each plate's origin date (`organoid_age_days`) instead of a fixed Exp3 reference date | Needed so Exp1 ages are correct; identical ages for Exp3 |
| 2026-10-21 | Good-data rule: ≥ 40 % of wells active, window ends at the first failing day; same rule for all experiments | Consistent criterion that keeps all Exp3 days; Exp1 window = 2.4–4.8 months |
| 2026-10-23 | Mislabelled files reassigned and same-day duplicates removed (`clean_recordings`) | Data-integrity check; only the 2025-10-30 duplicate affects Exp3 |
| 2026-10-23 | OLS ANOVA + Tukey replaced with mixed models | Repeated measures of wells (and electrodes within wells) made p-values too small |
| 2026-10-23 | Square-root transform for rates; synchrony untransformed | Residual diagnostics (see Statistics) |
| 2026-10-23 | Holm correction for per-age comparisons; trend lines fit against actual age | Multiple testing; uneven gaps between recordings |
| 2026-10-23 | Trajectory "stars" = group × age test; pairwise matrix = overall group differences | Previously the overall Tukey tests were shown under the trajectory label |
| 2026-10-23 | 2025-05-09 excluded as a noise artefact | All electrodes in all wells active at ~0.8 Hz, ~80× neighbouring weeks |
| 2026-10-23 | MAOA76 A4 (KO) excluded from Exp3 | Organoid detached; it was marked "DETACHED" in the plate layout but had never been removed from the analysis |

**Effect on the Exp3 figures** of all 2026-10-23 changes together (well level, overall pairwise, old → new):

| | KO vs Xa:MUT/Xi:WT | KO vs Xa:WT/Xi:MUT | Xa:MUT/Xi:WT vs Xa:WT/Xi:MUT |
|---|---|---|---|
| Firing | **** → **** | **** → **** | **** → *** |
| Burst | * → ** | **** → **** | *** → ** |
| Network | ns → *** | **** → **** | **** → *** |
| Synchrony | ns → ns | *** → * | * → ns |

Group × age ("trajectory"): firing *, burst **, network **, synchrony ns.

---

## To confirm with Sandra
- [ ] Axion spike-detection settings (threshold, filter) and recording length, for the Methods.
- [ ] Are MAOA1–3 and MAOA72/76/77 the same differentiation batch? What is the exact start date of MAOA72/76/77? ("4 months old on 2025-06-13" was taken as 120 days; if the true start is 2025-01-31, all Exp3 ages shift +13 days, to 7.8–8.9 months.)
- [ ] The four mislabelled files (2025-08-22, 09-06, 09-19, 10-16): check against the recording notes.
- [ ] 2025-05-09: anything different on that recording day (settings, noise)?
- [ ] 2025-10-30: two recordings per plate; we keep the first. Was the second one intended?
- [x] MAOA76 well A4 (GM) detached → excluded from Exp3 (2026-10-23). Confirm with Sandra that it detached at or before the 2025-09-25 recording.
- [ ] Raw spike-list folder (`Sandra_MAOA_MEA/`) is no longer on this machine; keep a copy with the project.
