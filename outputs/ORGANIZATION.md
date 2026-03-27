# outputs/ — Results and Figures

## results/

| File | What it contains |
|------|-----------------|
| `classification_results_excl-est.json` | **Primary results** — estimated tonic with bad-tonic tracks excluded (13 ragas, 46 tracks). This is what the report tables use. |
| `classification_results_est.json` | Estimated tonic, all tracks included (14 ragas, 53 tracks). Used for tonic ablation comparison in the appendix. |
| `classification_results_gt.json` | Ground-truth tonic results (13 ragas, 53 tracks). Upper bound on performance if tonic estimation were perfect. |
| `h4_melakarta_results.json` | H4 permutation test results — within/between family confusion rates (5.44x ratio), p-value (< 0.0001), family mapping, top confusions |

## figures/

### classification/
Confusion matrices and model comparison charts from stage 06.
- `cm_SVM_{feature_set}.png` — per-feature-set confusion matrices (e.g., `cm_SVM_dwell_vadi.png`, `cm_SVM_trigram.png`, `cm_SVM_mfcc.png`)
- `cm_fusion_pitch.png` — probability fusion confusion matrix
- `cm_LSTM_swara_seq.png` — BiLSTM confusion matrix
- `model_comparison.png` — bar chart comparing segment vs track accuracy across all models
- Subfolder per run mode (e.g., `est/`, `excl-est/`, `gt/`) when multiple tonic conditions are evaluated

### features/
Per-raga feature visualisations from stage 05.
- `pcd_{raga}.png` — pitch class distribution histograms (120-bin, 10-cent resolution)
- `transition_{raga}.png` — 12x12 swara transition matrix heatmaps
- `arohana_{raga}.png` — ascending-only transition matrices
- `avarohana_{raga}.png` — descending-only transition matrices

### tonic_validation/
Tonic estimation error distributions from stage 03 and experiment variants.
- `tonic_error_histogram.png` — default method error histogram
- `tonic_error_histogram_{method}.png` — per-method histograms (sa_mass, harmonic, smoothed_harmonic)
- `tonic_method_comparison.png` — bar chart comparing all tonic scoring methods
- Additional histograms from experiments: ensemble, segment_voting, triad variants, etc.

### pitch_debug/
Diagnostic plots from stage 02 — F0 contour + voicing confidence overlays for first N segments.
- `{track_id}__{segment_index}.png` — combined F0 + confidence plot
- `{track_id}__{segment_index}_f0.png` — F0 contour only
- `{track_id}__{segment_index}_voicedprob.png` — voiced probability only

### swara_debug/
Diagnostic plots from stage 04 — pitch contour with colour-coded swara labels.
- `{track_id}__{segment_index}_swara.png` — cents contour (top) + swara label track (bottom)

### h4_melakarta/
H4 hypothesis test visualisations.
- `confusion_matrix_by_family.png` — confusion matrix annotated and grouped by melakarta family
- `within_vs_between_family.png` — box plot + histogram comparing within- vs between-family confusion rates

### hierarchical/
Hierarchical classifier experiment outputs.
- `block_comparison.png` — flat vs hierarchical vs oracle comparison

### class_distribution.png
Dataset class balance chart — segment counts per raga, coloured by melakarta family.

## hindustani/
Parallel output structure for the Hindustani cross-tradition pilot (figures and results follow the same layout as the Carnatic outputs).
