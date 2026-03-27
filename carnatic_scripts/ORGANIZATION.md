# carnatic_scripts/ — Pipeline and Experiments

## Main Pipeline (run in order)

| Script | Stage | What it does | Input | Output |
|--------|-------|-------------|-------|--------|
| `01_preprocess.py` | 1 | Loads SARAGA, filters tracks, segments audio into 45s overlapping windows | SARAGA dataset | `data/interim/*.wav`, `data/processed/segments_saraga_carnatic.csv` |
| `02_pitch_baseline.py` | 2 | Extracts F0 pitch contours using CREPE (GPU) or pYIN (CPU) | Segment .wav files | `data/processed/pitch/*.npz`, `data/processed/pitch_all.csv` |
| `03_tonic_baseline.py` | 3 | Estimates tonic (Sa) per track via pitch-class histogram scoring | Pitch .npz files | `data/processed/tonic_validation.csv` |
| `04_swara_mapping.py` | 4 | Maps continuous pitch to discrete 12-swara labels (40-cent tolerance) | Pitch + tonic | `data/processed/swara/*_swara.npz`, `data/processed/swara_all.csv` |
| `05_feature_extraction.py` | 5 | Extracts 7 feature families (trigrams, dwell, PCD, transitions, directional, auxiliary, MFCC) plus fourgrams and phrase-aware trigrams | Swara traces + audio | `data/processed/features/features_all.npz`, vocab files |
| `06_train_classifiers.py` | 6 | Trains SVMs + BiLSTM under GroupKFold CV, probability fusion, majority voting | Feature matrices | `outputs/results/classification_results_{mode}.json`, `models/*.pkl` |

Run modes for stage 06: `--run-mode est` (all tracks, estimated tonic), `--run-mode excl-est` (bad-tonic tracks excluded, **primary results**), `--run-mode gt` (ground-truth tonic).

## experiments/ — Experimental Variants and Negative Results

| Script | What it tests | Result |
|--------|--------------|--------|
| `00_saragatest.py` | One-time environment validation — downloads and verifies SARAGA dataset | Setup only |
| `03_tonic_baseline_with_penalty.py` | Sa+Ma+Pa triad scoring with ratio penalty | Marginal improvement over Sa+Pa |
| `03_tonic_ensemble.py` | Multi-method ensemble (sa_mass + harmonic, pick highest confidence) | No improvement over harmonic alone |
| `03_tonic_ensemble_v2.py` | Harmonic-first ensemble (prefer harmonic, fall back to sa_mass) | Marginal, not worth added complexity |
| `03_tonic_segment_voting.py` | Per-segment tonic estimation with majority voting across segments | No improvement over track-level pooling |
| `03_tonic_swara_validated.py` | Swara hit-rate as tonic validation signal (unsupervised consistency check) | Comparable to harmonic alone |
| `demucs_tonic_estimation.py` | Demucs vocal separation before pitch extraction | Did not surpass 87% baseline |
| `hpss_tonic_estimation.py` | Harmonic-percussive separation before pitch extraction | Marginal improvement on few tracks |
| `gamaka_experiment.py` | Continuous pitch contour features (pitch slope, per-swara gamaka profiles) | Pan-raga — no discriminative power |
| `07_hierarchical_classifier.py` | Two-level melakarta hierarchy (L1 family routing, L2 within-family) | Flat baseline matches or exceeds hierarchy |
| `h4_melakarta_confusion.py` | Permutation test for within-family confusion (H4 hypothesis) | Confirmed: 5.19x ratio, p < 0.0001 |

## Dependencies between stages

```
01 → 02 → 03 → 04 → 05 → 06
                          ↓
                    h4_melakarta_confusion.py
                    07_hierarchical_classifier.py
```

All experiment scripts in `experiments/` depend on outputs from stages 02-05 but do not modify the main pipeline.
