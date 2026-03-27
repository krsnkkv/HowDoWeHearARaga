# How Do We Hear a Raga?
### An Interpretable Cognitive Model of Melodic Recognition
**Final Year Project — MEng Computer Science and Software Engineering**
**University of Birmingham, 2025-26**

**Student:** Krishnasree Kelath (2540124)
**Supervisor:** Martin Escardo

**Final Report:** [`FYP.pdf`](FYP.pdf) | **Extended Appendix:** [`EXTENDED_APPENDIX.pdf`](EXTENDED_APPENDIX.pdf)

---

## Overview

This project investigates whether an interpretable, pitch-based machine learning pipeline can model Carnatic raga recognition in a way that reflects human melodic cognition. A six-stage pipeline processes raw audio through pitch extraction, tonic estimation, swara mapping, and feature extraction, producing representations where every feature dimension has a cognitive or musicological interpretation.

**Key results (primary setting: estimated tonic, bad-tonic tracks excluded, 13 ragas, 46 tracks, 1,648 segments):**
- 67.8% weighted F1 (69.6% track accuracy) using just 16 dwell-time and vadi features (SVM)
- Outperforms a BiLSTM baseline by over 31 percentage points in weighted F1
- Timbral features (MFCCs) score near random chance (4.4% track F1w), confirming raga identity is structural, not timbral
- Six hypothesis tests link classification results to music cognition predictions (5 confirmed, 1 informative negative)
- Within-family confusion is 5.19x higher than between-family (p < 0.0001), confirming the model's error structure mirrors melakarta kinship

## Classification Results

Results from the primary evaluation (estimated tonic, bad-tonic tracks excluded):

| Model | Dims | Seg Acc | Seg F1w | Trk Acc | Trk F1w |
|-------|------|---------|---------|---------|---------|
| SVM dwell_vadi | 16 | 50.55% | 51.36% | **69.57%** | **67.80%** |
| Fusion (pitch) | — | 53.88% | 50.86% | 65.22% | 64.01% |
| SVM trigram | 200 | 52.49% | 50.07% | 63.04% | 59.14% |
| SVM trigram_dwell | 216 | 52.79% | 50.41% | 60.87% | 57.20% |
| SVM pcd_trigram_aux | 330 | 49.45% | 46.48% | 56.52% | 52.73% |
| SVM pcd_trigram | 320 | 50.42% | 47.69% | 56.52% | 51.89% |
| SVM pcd | 120 | 45.51% | 43.56% | 52.17% | 49.61% |
| SVM directional_pcd | 408 | 47.82% | 45.79% | 52.17% | 49.08% |
| SVM directional_full | 418 | 47.15% | 44.97% | 52.17% | 49.08% |
| SVM fourgram | 200 | 43.45% | 41.00% | 43.48% | 38.79% |
| BiLSTM | — | 26.94% | 27.17% | 39.13% | 36.39% |
| SVM phrase_trigram | 200 | 28.82% | 20.97% | 8.70% | 2.80% |
| SVM mfcc | 58 | 25.85% | 16.05% | 10.87% | 4.39% |

Random chance baseline: 7.7% (1/13); majority-class baseline: 17.4%.

## Repository Structure

```
carnatic_scripts/
  01_preprocess.py           # Stage 1: Audio segmentation (SARAGA -> 45s segments + CSV)
  02_pitch_baseline.py       # Stage 2: Pitch extraction (CREPE on GPU, pYIN fallback)
  03_tonic_baseline.py       # Stage 3: Tonic (Sa) estimation (harmonic scoring, 87%)
  04_swara_mapping.py        # Stage 4: Pitch-to-swara mapping (12 bins, 40-cent tolerance)
  05_feature_extraction.py   # Stage 5: Feature extraction (7 families + 2 experimental)
  06_train_classifiers.py    # Stage 6: SVM + BiLSTM classification (GroupKFold, 5 folds)
  experiments/               # Tonic variants, gamaka, hierarchical, H4 permutation test

hindustani_scripts/          # Cross-tradition pilot pipeline (SARAGA Hindustani)
  run_hindustani.py          # Pipeline orchestrator with --from-stage resume

configs/
  preprocessing.yaml         # Audio: 22050 Hz, 45s segments, 15s hop
  tonic.yaml                 # Tonic: 80-300 Hz, 10-cent steps, harmonic scoring
  swara.yaml                 # Swara: 12 bins (0-1100 cents), 40-cent tolerance

data/
  raw/mirdata/               # SARAGA Carnatic dataset (via mirdata)
  processed/                 # Segments CSV, pitch/swara arrays, feature matrices

outputs/
  results/                   # Classification results JSON, H4 permutation results
  figures/                   # Confusion matrices, model comparison, feature plots

models/                      # Trained SVM models (.pkl)
docs/                        # Background research notes, weekly progress logs
FYP.pdf                      # Project report (compiled)
EXTENDED_APPENDIX.pdf        # Extended appendix (compiled)
```

## Running the Pipeline

```bash
# Prerequisites: Python 3.9+, CUDA GPU recommended for CREPE
pip install -r requirements.txt

# Run stages sequentially (each depends on the previous)
python carnatic_scripts/01_preprocess.py
python carnatic_scripts/02_pitch_baseline.py
python carnatic_scripts/03_tonic_baseline.py --method compare
python carnatic_scripts/04_swara_mapping.py
python carnatic_scripts/05_feature_extraction.py

# Primary results: estimated tonic, bad-tonic tracks excluded
python carnatic_scripts/06_train_classifiers.py --run-mode excl-est

# Alternative run modes:
# python carnatic_scripts/06_train_classifiers.py --run-mode est    # all tracks
# python carnatic_scripts/06_train_classifiers.py --run-mode gt     # ground-truth tonic
```

## Feature Families

Seven primary feature families, each with a cognitive or musicological analogue:

| Feature | Dims | Cognitive Analogue |
|---------|------|--------------------|
| Trigram | 200 | Pakad recognition via statistical learning (H1) |
| Dwell/vadi | 16 | Note prominence hierarchy (H5) |
| PCD | 120 | Tonal hierarchy |
| Transition | 144 | Implicit grammar learning |
| Directional | 288–552 | Arohana/avarohana distinction (H6) |
| Auxiliary | 10 | Ornamental sensitivity |
| MFCC | 58 | Control — should fail (H2) |

Two experimental variants (fourgram, phrase-aware trigram) are included as negative results (Section 4.3 of the report).

## Hypothesis Tests

| ID | Claim | Result |
|----|-------|--------|
| H1 | Trigrams encode pakad (characteristic phrases) | Confirmed |
| H2 | Raga is structural, not timbral (MFCCs fail) | Confirmed |
| H3 | Cognitive parsimony (fewer features = better) | Confirmed |
| H4 | Confusion follows melakarta kinship (5.19x ratio, p < 0.0001) | Confirmed |
| H5 | Dwell-time captures vadi emphasis | Confirmed |
| H6 | Directional encoding improves accuracy | Informative negative |

## Folder Documentation

Each major folder contains its own organisation document:

- [`carnatic_scripts/ORGANIZATION.md`](carnatic_scripts/ORGANIZATION.md) — pipeline stages 01–06, experiment scripts, dependencies
- [`docs/ORGANIZATION.md`](docs/ORGANIZATION.md) — background research, weekly progress logs
- [`outputs/ORGANIZATION.md`](outputs/ORGANIZATION.md) — results files, figures, and their interpretations

## Configuration

All hyperparameters are externalised in YAML config files under `configs/`, separating experimental settings from code.
