# **RagaRecognize**

**Automatic Analysis and Recognition of Rāga Structures from Musical Audio**

This repository contains the code, experiments, and documentation for my Final Year Project for the MEng Computer Science and Software Engineering degree at the University of Birmingham. The project focuses on analysing pitch-based representations of musical audio to study and model rāga structure in Indian classical music.

## Table of Contents
- [Project Description](#project-description)
- [Pipeline Overview](#pipeline-overview)
- [Getting Started](#getting-started)
- [Weekly Plan](#weekly-plan)
- [Project Report](#project-report)

## Project Description

RagaRecognize is a project aimed at developing a system for automatic recognition and analysis of rāgas in Indian classical music. The system takes musical audio as input, performs pitch-based analysis, and uses machine learning techniques to identify the underlying rāga structure. The project focuses on both Carnatic and Hindustani classical music traditions.

The main objectives of the project are:
- To develop a robust preprocessing pipeline for audio segmentation, noise reduction, and normalization.
- To extract pitch-based features such as pitch contours, tonic estimation, and swara mapping.
- To model the temporal structure and characteristic phrases of rāgas using machine learning techniques like Hidden Markov Models (HMMs) and Support Vector Machines (SVMs).
- To create an interactive frontend for users to upload audio, visualize analysis results, and explore rāga information.

## Pipeline Overview

The RagaRecognize pipeline consists of the following stages:

1. **Data Preprocessing**:
   - Audio segmentation: Divide long-form recordings into analysis-ready excerpts.
   - Noise reduction: Apply techniques to reduce background noise and enhance the signal.
   - Normalization: Normalize the audio signal to a consistent level.

2. **Feature Extraction**:
   - Pitch tracking: Extract pitch contours (F0) using librosa and custom algorithms.
   - Tonic estimation: Estimate the tonic (Sa) of the rāga using pitch distribution analysis.
   - Swara mapping: Convert pitch values to relative intervals with respect to the estimated tonic.
   - Pattern and sequence features: Compute transition matrices, n-gram features, and characteristic phrases.

3. **Raga Classification**:
   - Model training: Train HMM and SVM models using the extracted features.
   - Raga prediction: Classify the input audio into the most likely rāga based on the trained models.

4. **Frontend and Visualization**:
   - User interface: Develop an intuitive frontend using Streamlit for audio upload and analysis.
   - Visualization: Display pitch contours, tonic estimation, swara mapping, and rāga predictions.
   - Raga information: Provide educational content about the identified rāga, including characteristic phrases and cultural context.

## Getting Started

To run the RagaRecognize system locally, follow these steps:

1. Clone the repository:
```
   git clone https://github.com/kxk224/RagaRecognize.git
```

2. Install the required dependencies:
```
   pip install -r requirements.txt
```

3. Prepare the dataset:
   - Download the SARAGA Carnatic Music Dataset and place it in the `data/` directory.
   - (Optional) Add additional Hindustani classical music recordings to the `data/` directory.

4. Run the preprocessing pipeline:
```
   python preprocess.py
```

5. Extract features:
```
   python extract_features.py
```

6. Train the models:
```
   python train_models.py
```

7. Launch the frontend:
```
   streamlit run app.py
```

## Weekly Plan

### Week 1: Setup and Reproducibility

- Finalise the repository structure and documentation
- Define dataset selection criteria and shortlist suitable datasets
- Implement a preprocessing pipeline to load audio, trim recordings, segment clips, and store metadata

Deliverable: A reproducible preprocessing pipeline with sample outputs and metadata files

### Week 2: Pitch Tracking Baseline

- Implement pitch extraction using `librosa.pyin`
- Generate pitch contour plots and voiced probability visualisations
- Manually validate results on a small set of audio clips

Deliverable: A pitch tracking module with visual diagnostics for validation

### Week 3: Tonic (Sa) Estimation Baseline

- Implement a baseline tonic estimation method using pitch distribution analysis
- Introduce a confidence score for the estimated tonic
- Evaluate results on a small subset with manual sanity checks

Deliverable: A tonic estimation module producing `sa_hz` values with confidence scores

### Week 4: Swara Mapping (Version 1)

- Convert pitch values to relative intervals with respect to the estimated tonic
- Implement a soft mapping strategy with tolerance for ornamentation and pitch glides
- Generate time-aligned swara traces for selected clips

Deliverable: A swara mapping module with visual swara-time representations

### Week 5: Pattern and Sequence Features

- Compute transition matrices and n-gram features for swara sequences
- Extract basic phrase-level statistics such as frequent transitions
- Construct feature vectors for each audio clip

Deliverable: A feature extraction module and a saved dataset of extracted features

### Week 6: Baseline Classification

- Train a simple classification model such as logistic regression or SVM using extracted features
- Perform cross-validation and generate confusion matrices

Deliverable: A baseline rāga classifier with quantitative performance metrics

### Week 7: End-to-End Integration and Interface Demo

- Integrate all pipeline components into a single end-to-end system
- Build a Streamlit interface to upload audio and visualise tonic, pitch, swaras, patterns, and top candidate rāgas
- Handle low-confidence predictions gracefully

Deliverable: A stable end-to-end demo suitable for inspection and discussion

### Week 8: Evaluation and Report Refinement

- Conduct error analysis to identify common confusions between rāgas
- Refine segmentation and smoothing parameters based on evaluation results
- Update documentation and prepare a supervisor-ready progress summary

Deliverable: An evaluation summary and a refined baseline system

## Project Report

### Overleaf

The main project report is written in LaTeX and maintained on Overleaf: [Click here to view on Overleaf](https://www.overleaf.com/read/jfbjwcdtgppg#94c878) 

### Project Structure

The report follows a structured academic format aligned with University of Birmingham guidelines:

- Title Page  
- Abstract  
- Contents  
- Introduction  (Includes Motivation)
- Background and Literature Review  
- Design and Methodology  
- System Specification and Implementation  
- Testing and Evaluation  
- Conclusion and Future Work  
- Bibliography  
- Appendices

The report is designed to clearly communicate the motivation, design decisions, implementation details, and evaluation of the project in a way that supports discussion during the project inspection.