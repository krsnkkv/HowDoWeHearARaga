# Presentation Prep — "How Do We Hear a Rāga?"

> Use this to prepare for your update. Structured as: what you did, what you found, and how it connects to the title.

---

## PART 1: WHAT I'VE BUILT (Pipeline Walkthrough)

### Stage 1 — Audio Segmentation (`01_preprocess.py`)
- Takes 89 SARAGA Carnatic tracks, filters to 73 usable (16 have no tonic annotation)
- Cuts each track into **45-second windows** with **15-second hop** (66% overlap)
- Removes ragamalika (multi-raga) tracks
- **Output**: 4,487 segments with metadata (raga, tonic, track_id)
- **Key decision**: 45s is long enough to contain multiple characteristic phrases, short enough for homogeneous content

### Stage 2 — Pitch Extraction (`02_pitch_baseline.py`)
- Uses **CREPE** (deep learning pitch tracker) on GPU; falls back to librosa pyin on CPU
- Frame-level F0 at ~11.6ms resolution (hop=256 at 22050 Hz)
- Frames with confidence < 0.5 → unvoiced (NaN)
- **Output**: per-segment .npz with f0 arrays + voiced_ratio stats

### Stage 3 — Tonic (Sa) Estimation (`03_tonic_baseline.py`)
- Pools all pitch values per track, builds a 120-bin pitch class histogram
- Searches candidate tonics in 80–300 Hz range using log-frequency steps
- Scores each candidate by histogram mass near 0 cents (Sa peak) + 702 cents (Pa peak)
- **~20% octave error rate** — validates against ground-truth
- We use **ground-truth tonic** for downstream stages (honest about this limitation)

### Stage 4 — Swara Mapping (`04_swara_mapping.py`)
- Converts F0 → cents relative to tonic, folds into one octave [0, 1200)
- Maps each frame to nearest of **12 swara bins** (100-cent equal-tempered spacing)
- **Tolerance**: 40 cents — frames further away → labelled "glide" (gamaka)
- **Min duration filter**: runs < 100ms → "glide" (reduces label flickering)
- **Token collapse**: "Sa Sa Sa Ri2 Ri2 glide Ga2 Ga2" → [Sa, Ri2, Ga2]
  - This loses duration info — which we recover later with dwell-time features

### Stage 5 — Feature Extraction (`05_feature_extraction.py`)
**This is the core contribution.** 10+ handcrafted feature representations:

| Feature | Dims | What it captures | Cognitive analogy |
|---------|------|------------------|-------------------|
| **trigram** | 200 | Top-200 3-note patterns | "Characteristic phrases" (pakad) |
| **dwell_time** | 12 | Mean sustain per swara | "Which notes are held" |
| **vadi** | 4 | Most-prominent + 2nd-most note | "Dominant note" recognition |
| **transition** | 144 | 12×12 note-to-note probabilities | "Melodic grammar" |
| **pcd** | 120 | Pitch class distribution | "Scale/note presence" |
| **auxiliary** | 10 | Gamaka stats, entropy, token rate | "Ornament density" |
| **mfcc** | 58 | Timbral features (baseline) | "Voice/instrument quality" |
| **directional** | 288 | Ascending vs descending transitions | "Arohana/Avarohana" |

**Key composites**:
- `trigram_dwell` (216d) = trigram + dwell_time + vadi → **best performer**
- `dwell_vadi` (16d) = just emphasis features → surprisingly strong at 16 dims

### Stage 6 — Classification (`06_train_classifiers.py`)
- **SVM** (RBF kernel, balanced class weights, GridSearchCV over C/gamma)
- **LSTM** (bidirectional, 2-layer, on raw swara token sequences)
- **GroupKFold(5)** by track_id — prevents data leakage from overlapping segments
- **Track-level voting**: majority vote across all segment predictions per track
- **Probability fusion**: weighted average of SVM probability vectors
- **Hybrid fusion**: SVM + LSTM pooled majority vote at track level

---

## PART 2: WHAT I FOUND (Results & Analysis)

### Headline Numbers

| Setting | Best Model | Seg Acc | Trk Acc |
|---------|-----------|---------|---------|
| 14 ragas (2,107 segs, 53 tracks) | SVM_trigram_dwell | **0.6018** | **0.5849** |
| 10 ragas (1,974 segs, 40 tracks) | SVM_trigram_dwell | **0.6368** | **0.7000** |
| Random chance (14 ragas) | — | 0.071 | — |
| Majority class baseline | — | 0.174 | — |
| LSTM alone | LSTM_swara_seq | 0.4257 | 0.4528 |

### The Feature Ranking (14 ragas)

```
1. trigram_dwell  (216d)  0.6018  ← BEST (trigrams + sustain + emphasis)
2. trigram        (200d)  0.5795  ← strong baseline
3. pcd_trigram    (320d)  0.5676  ← adding PCD hurts
4. directional    (408d)  0.5600  ← too many dims
5. dwell_vadi     (16d)   0.5297  ← amazingly good for 16 dims
6. pcd            (120d)  0.5059  ← pitch distribution alone
7. LSTM           (seq)   0.4257  ← neural approach struggles
8. mfcc           (58d)   0.1191  ← timbral features useless
```

### Per-Raga Tiers (10-raga, SVM_trigram_dwell)

**Tier 1 — Easily recognisable** (F1 > 0.80):
- Bhairavi (0.89), Todi (0.86), Sanmukhapriya (0.84), Ritigaula (0.80)
- Distinctive scales, strong characteristic phrases

**Tier 2 — Moderate** (F1 0.35–0.65):
- Saveri (0.61), Begada (0.52), Mohanam (0.38), Kambhoji (0.37), Kamas (0.36)
- Share scales with related ragas, identity depends on subtle emphasis

**Tier 3 — Unrecognisable** (F1 < 0.15):
- Harikambhoji (0.13) — parent melakarta, child ragas steal its identity

### LSTM vs SVM: Why SVM Wins

| Factor | SVM | LSTM |
|--------|-----|------|
| Data efficiency | Thrives at 2,100 samples | Needs 10x+ more |
| Features | Domain-crafted (216d sweet spot) | Learns from raw tokens |
| Fold variance | Stable (0.49–0.75) | Wild (0.25–0.68) |
| Interpretability | Full — inspect trigram weights | Black box |
| Best single fold | 0.747 | 0.742 — comparable ceiling! |

**Key insight**: LSTM's best fold (0.742) nearly matches SVM's best (0.747). The neural model *can* learn raga patterns — it just needs more data. With 53 tracks, hand-crafted features give SVM an unfair advantage.

---

## PART 3: NEGATIVE RESULTS (What Failed and Why)

### 1. Phrase-Aware Trigrams — Catastrophic (0.58 → 0.18)
- **Idea**: Split token sequences at glide boundaries before computing trigrams
- **Why it failed**: In Carnatic music, gamakas (glides) *connect* notes — they're not phrase separators. Splitting at glides fragmented real phrases into single-note fragments.
- **Lesson**: Western music intuitions about phrasing don't apply to Carnatic

### 2. 4-Grams — Too Sparse (0.158)
- 12^4 = 20,736 possible 4-grams, only ~2,100 training samples
- Even top-200 were too sparse to be informative

### 3. MFCC — Below Random Chance (0.063 at 10 ragas)
- Random chance = 0.10, MFCC gets 0.063
- **This is a positive finding**: proves raga is purely structural, not timbral
- Same raga can be sung by any voice — timbre is irrelevant

### 4. More Dimensions = Worse
- 200d trigram: 0.5795
- 320d pcd_trigram: 0.5676 (−1.2%)
- 418d directional_full: 0.5510 (−2.9%)
- Classic curse of dimensionality with limited data

### 5. Probability Fusion Hurts Track Accuracy
- Dilutes the best model's confidence with weaker models
- Track accuracy drops even when segment accuracy is similar

---

## PART 4: CONNECTING TO THE TITLE

### "How Do We Hear a Rāga?" — The Cognitive Model Argument

**Central claim**: Our best-performing feature set (`trigram_dwell`) is also the most *cognitively plausible* model of raga recognition.

### How Humans Recognise Ragas (Musicological Literature)

| Human Process | Our Feature | Evidence |
|--------------|-------------|----------|
| Recognise characteristic phrases (pakad) | Trigrams | Best single feature (0.5795) |
| Hear which notes are emphasised (vadi/samvadi) | Vadi features | 0.5297 at just 16 dims |
| Feel sustain/duration of notes | Dwell-time | +2.2% when added to trigrams |
| NOT based on voice quality | MFCC = useless | 0.063 (below random) |
| Uses few salient cues, not all info | More features hurt | Peak at 216d, decline after |

### The 5 Evidence Points

1. **MFCC useless** → Raga is structure, not timbre (aligns with the fact that ragas are instrument-independent)
2. **Trigrams work best** → Sequential 3-note patterns = computational version of pakad recognition
3. **Dwell-time adds value** → Sustain emphasis = computational vadi/samvadi
4. **More features hurt** → Human cognition uses few salient cues (ecological rationality)
5. **Phrase-aware splitting failed** → Gamakas connect notes continuously (confirms Carnatic melodic flow model)

### The Interpretability Advantage

For any prediction, you can explain *why*:
- "This segment was classified as Todi because it has high counts of Ri1→Ga1→Ri1 trigrams and long dwell-time on Ri1 and Da1"
- This is something a spectrogram CNN can never do

### Comparison to Literature (Defence Points)
- Many papers get higher accuracy (80%+) BUT:
  - Use train/test split without GroupKFold → **data leakage**
  - Use fewer ragas (5–8)
  - Use Hindustani music (less gamaka = easier)
  - Use deep learning on spectrograms → **no interpretability**
- **Our contribution is the cognitive model, not raw accuracy**

---

## PART 5: ERROR ANALYSIS PLAN

### What I'm Currently Investigating

1. **Confusion matrix patterns**: Which ragas get confused with which? → Map to musicological raga family relationships (janya-melakarta)
2. **Per-fold variance**: Why does Fold 3 always perform worst? → Track composition analysis
3. **Harikambhoji deep dive**: Parent raga confused with its children → demonstrates limits of statistical features for ragas defined by "absence" rather than "presence"
4. **Track-level vs segment-level gap**: Track voting helps because bad segments get outvoted by good ones → validates 45s window + majority vote approach

### Planned Experiments

| Experiment | Purpose | Expected Outcome |
|-----------|---------|-----------------|
| Drop unlearnable ragas (Harikambhoji, Kamas) | Test ceiling with clean classes | 0.68–0.72 expected |
| Full SVM + LSTM + Hybrid fusion run | Test ensemble benefit | Hybrid may help at track level |
| Reduce hop to 7.5s | Double training data | Small accuracy bump |
| Per-raga confusion → family tree mapping | Explain error patterns | Visual for paper |

### Ablation Studies I Can Present

1. **Feature dimensionality curve**: Plot dims vs accuracy → shows 216d sweet spot
2. **Token collapse ablation**: With/without duration info
3. **Timbral vs structural**: MFCC vs pitch-based features
4. **Feature addition/subtraction**: What helps trigrams (dwell) vs what hurts (PCD)
5. **14 vs 10 raga**: Effect of removing ambiguous classes

---

## PART 6: QUICK REFERENCE — KEY NUMBERS TO KNOW

```
Best accuracy (14 ragas):     60.2% segment, 58.5% track
Best accuracy (10 ragas):     63.7% segment, 70.0% track
Random chance:                7.1% (14 ragas), 10% (10 ragas)
MFCC accuracy:                6.3% (below random — key finding)
LSTM accuracy:                42.6% (SVM wins by 18%)
Best feature dims:            216 (trigram_dwell)
Smallest useful feature:      16 dims (dwell_vadi → 53.0%)
Dataset:                      4,487 segments → 2,107 after filtering
Tracks:                       53 (14 ragas), 40 (10 ragas)
Voiced ratio filter:          0.25 (removes 52% of data)
GroupKFold:                   5 folds by track_id
Segment window:               45s with 15s hop
Swara tolerance:              40 cents
Trigram vocab:                200 (of 1,728 possible)
```

---

## PART 7: HYPOTHESES FOR PAPER

### H1: Raga identity is melodic-sequential, not timbral
- **Evidence**: MFCC ≈ 0.06 (below random); trigrams ≈ 0.58
- **Implication**: Aligns with musicological theory that ragas transcend instrument/voice

### H2: Short melodic motifs (3-note patterns) are the primary recognition cue
- **Evidence**: Trigrams (200d) outperform all other single features
- **Implication**: Computational analogue of pakad (characteristic phrase) recognition

### H3: Note emphasis (dwell-time) provides complementary information to sequence
- **Evidence**: trigram + dwell = 0.6018 vs trigram alone = 0.5795 (+2.2%)
- **Implication**: Humans hear both "what patterns occur" AND "which notes are held"

### H4: Cognitive recognition uses minimal salient cues, not exhaustive features
- **Evidence**: Accuracy peaks at 216 dims, then DECLINES with more features
- **Implication**: Mirrors ecological rationality — "less is more" in perception

### H5: Related ragas (janya-melakarta families) form the recognition boundary
- **Evidence**: Harikambhoji (0.13 F1) is parent of Kambhoji, Kamas, Behag — all confused
- **Implication**: The model's confusion patterns match human difficulty patterns
