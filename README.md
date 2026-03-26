# How Do We Hear a Raga?
### An Interpretable Cognitive Model of Melodic Recognition

**Final Year Project — MEng Computer Science and Software Engineering**
**University of Birmingham, 2025-26**

**Student:** Krishnasree Kelath (2540124)
**Supervisor:** Martin Escardo

---

## Overview

This project investigates whether an interpretable, pitch-based machine learning pipeline can model Carnatic raga recognition in a way that reflects human melodic cognition. A six-stage pipeline processes raw audio through pitch extraction, tonic estimation, swara mapping, and feature extraction, producing representations where every feature dimension has a cognitive or musicological interpretation.

**Key results:**
- 69.6% track accuracy on 13 ragas using just 16 dwell-time features (SVM)
- Outperforms a BiLSTM baseline by over 24 percentage points
- Timbral features (MFCCs) score near random chance (10.9% track), confirming raga identity is structural not timbral
- Six hypothesis tests link results to music cognition predictions (5 confirmed, 1 informative negative)

## Repository Structure

```
carnatic_scripts/
  01_preprocess.py           # Stage 1: Audio segmentation (SARAGA -> .wav segments + CSV)
  02_pitch_baseline.py       # Stage 2: Pitch extraction (CREPE/pYIN)
  03_tonic_baseline.py       # Stage 3: Tonic (Sa) estimation (harmonic scoring, 87% accuracy)
  04_swara_mapping.py        # Stage 4: Pitch-to-swara mapping (12 bins, 40-cent tolerance)
  05_feature_extraction.py   # Stage 5: Feature extraction (7 families, 216d core representation)
  06_train_classifiers.py    # Stage 6: SVM + BiLSTM classification (GroupKFold, 5 folds)
  07_hierarchical_classifier.py  # Hierarchical melakarta-based classifier (H4)
  h4_melakarta_confusion.py  # Permutation test for within-family confusion (p=0.0004)
  experiments/               # Tonic estimation variants, diagnostics, gamaka experiments

configs/
  preprocessing.yaml         # Audio parameters (sample rate, segment length, hop)
  tonic.yaml                 # Tonic estimation parameters (candidate range, scoring)
  swara.yaml                 # Swara bin positions and tolerance

data/
  raw/mirdata/               # SARAGA Carnatic dataset (via mirdata)
  processed/                 # Segments CSV, pitch/swara arrays, features

outputs/
  results/                   # Classification results JSON files
  figures/                   # Confusion matrices, model comparison charts

models/                      # Trained SVM models (.pkl)
app/app.py                   # Streamlit interactive dashboard
docs/                        # Background research notes, weekly logs
FYP.tex                      # Project report (LaTeX source)
FYP.pdf                      # Project report (compiled)
```

## Running the Pipeline

```bash
# Prerequisites: Python 3.9+, CUDA GPU recommended for CREPE
pip install -r requirements.txt

# Run stages sequentially
python carnatic_scripts/01_preprocess.py
python carnatic_scripts/02_pitch_baseline.py
python carnatic_scripts/03_tonic_baseline.py --method compare
python carnatic_scripts/04_swara_mapping.py --use-estimated
python carnatic_scripts/05_feature_extraction.py
python carnatic_scripts/06_train_classifiers.py

# Interactive dashboard
streamlit run app/app.py
```

## Dataset

SARAGA Carnatic dataset (249 recordings, 73 usable tracks across 13 ragas after filtering) accessed via the `mirdata` Python library. Tracks require both raga labels and tonic annotations.

## Configuration

All hyperparameters are externalised in YAML config files under `configs/`, separating experimental settings from code. See Appendix C of the report for the full parameter listing.
