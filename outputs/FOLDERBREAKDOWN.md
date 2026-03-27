# outputs/ — Results and Figures

## results/

| File | What it contains |
|------|-----------------|
| `classification_results_excl-est.json` | **Primary results** — estimated tonic, bad-tonic tracks excluded (13 ragas, 46 tracks). This is what the report tables use. |
| `classification_results_est.json` | Estimated tonic, all tracks included (14 ragas, 53 tracks). Used for tonic ablation comparison in the appendix. |
| `classification_results_gt.json` | Ground-truth tonic results (13 ragas, 53 tracks). Upper bound on performance if tonic estimation were perfect. |
| `h4_melakarta_results.json` | H4 permutation test results — within/between family confusion rates (5.19x ratio), p-value (< 0.0001), family mapping, top confusions. |

### results/old_backup/

Archived results from earlier experimental runs.

| File | What it contains |
|------|-----------------|
| `classification_results_ESTIMATED.json` | Earlier estimated-tonic classification run (before tonic exclusion was added). |
| `classification_results_GT.json` | Earlier ground-truth tonic classification run. |
| `hierarchical_results_*.json` | Timestamped hierarchical classifier experiment runs. |
| `specialist_results_*.json` | Timestamped specialist (per-family) classifier experiment runs. |

---

## figures/

### classification/

Confusion matrices and model comparison charts, organised by tonic condition.

**Subfolders:** `est/`, `excl-est/`, `gt/` — each contains the same set of figures for that tonic condition:

- `cm_SVM_{feature_set}.png` — per-feature-set confusion matrices (e.g., `cm_SVM_dwell_vadi.png`, `cm_SVM_trigram.png`, `cm_SVM_mfcc.png`)
- `cm_fusion_pitch.png` — probability fusion confusion matrix
- `cm_LSTM_swara_seq.png` — BiLSTM confusion matrix
- `model_comparison.png` — bar chart comparing segment vs track accuracy across all models

Additional files in `excl-est/` only:

- `per_raga_f1_dwell_vadi.png` — per-raga F1 breakdown for the best model

**`gamaka/`** — gamaka feature experiment confusion matrices:

- `cm_SVM_dwell_gamaka.png`, `cm_SVM_gamaka_combined.png`, `cm_SVM_pitch_slope.png`, `cm_SVM_trigram_dwell_gamaka.png`, `cm_SVM_trigram_gamaka.png`

**`previous_runs/`** — archived confusion matrices from earlier feature combination experiments (e.g., `cm_SVM_combined.png`, `cm_SVM_transition.png`, `cm_SVM_full_combined.png`, etc.)

### features/

Per-raga feature visualisations from stage 05. Each raga has up to four plots:

- `pcd_{raga}.png` — pitch class distribution histogram (120-bin, 10-cent resolution)
- `transition_{raga}.png` — 12x12 swara transition matrix heatmap
- `arohana_{raga}.png` — ascending-only transition matrix
- `avarohana_{raga}.png` — descending-only transition matrix

Plots exist for all 16 ragas in the dataset (names use diacritics, e.g., `pcd_Kalyāṇi.png`).

### tonic_validation/

Tonic estimation error distributions from stage 03 and its experiment variants.

- `tonic_method_comparison.png` — bar chart comparing all tonic scoring methods
- `tonic_error_histogram.png` — default method error histogram
- `tonic_error_histogram_{method}.png` — per-method histograms for: `disambig`, `ensemble`, `harmonic`, `harmonic_pa_penalty`, `harmonic_triad`, `sa_mass`, `sa_mass_pa_penalty`, `sa_mass_triad`, `segment_voting`, `smoothed_harmonic`, `smoothed_harmonic_pa_penalty`, `triad`, `triad_smoothed_triad`, `triad_triad`

### pitch_debug/

Diagnostic plots from stage 02 — F0 contour and voicing confidence overlays for the first N segments of selected tracks.

- `{track_id}__{segment_index}.png` — combined F0 + confidence plot
- `{track_id}__{segment_index}_f0.png` — F0 contour only
- `{track_id}__{segment_index}_voicedprob.png` — voiced probability only

### swara_debug/

Diagnostic plots from stage 04 — pitch contour with colour-coded swara labels.

- `{track_id}__{segment_index}_swara.png` — cents contour (top) + swara label track (bottom)

### h4_melakarta/

H4 hypothesis test visualisations.

- `confusion_matrix_by_family.png` — confusion matrix grouped by melakarta family
- `within_vs_between_family.png` — box plot + histogram comparing within- vs between-family confusion rates

### hierarchical/

Hierarchical classifier experiment outputs.

- `block_comparison.png` — flat vs hierarchical vs oracle comparison chart
- `cm_A_flat.png` — flat classifier confusion matrix
- `cm_B_hierarchical.png` — hierarchical classifier confusion matrix
- `cm_B_specialist.png` — specialist (per-family) classifier confusion matrix
- `cm_C_oracle.png` — oracle (perfect family assignment) confusion matrix

### Root-level figures

- `pipeline_diagram.png` — system architecture diagram showing the six-stage pipeline
- `class_distribution.png` — dataset class balance chart (segment counts per raga, coloured by melakarta family)

---

## hindustani/

Parallel output structure for the Hindustani cross-tradition pilot. Mirrors the Carnatic layout:

- `results/classification_results.json` — Hindustani classification results
- `figures/classification/` — confusion matrices and model comparison for all feature sets
- `figures/features/` — per-raga PCDs, transitions, arohana/avarohana plots (5 ragas)
- `figures/pitch_debug/` — F0 diagnostic plots for selected Hindustani segments
- `figures/swara_debug/` — swara mapping overlays for selected Hindustani segments
- `figures/tonic_validation/tonic_error_histogram.png` — Hindustani tonic error distribution
