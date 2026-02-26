# RagaRecognize — Complete Dissertation Context Window

> Use this document as a system prompt or context window when working on the project report,
> dissertation, and research paper. It contains every detail about the system, experiments,
> results, and findings.

---

## 1. Project Identity

- **Title**: "How Do We Hear a Raga? An Interpretable Cognitive Model of Melodic Recognition"
- **Type**: MEng Final Year Project (2025-26)
- **Domain**: Music Information Retrieval (MIR) — Automatic Raga Recognition
- **Music system**: South Indian Carnatic classical music
- **Dataset**: SARAGA Carnatic Music Dataset (CompMusic / MTG)
- **Core thesis**: Raga recognition can be modelled with interpretable, cognitively-motivated features (swara trigrams + dwell-time) rather than black-box deep learning on spectrograms

---

## 2. Dataset: SARAGA Carnatic

### Raw Data
- 89 total tracks in SARAGA Carnatic
- 73 usable tracks (16 excluded — no ground-truth tonic annotation, `ctonic: null`)
- Each track is a full concert performance recording with metadata: raga label, tonic frequency, artist

### Segmentation
- **Window**: 45 seconds, **Hop**: 15 seconds → 66% overlap between consecutive segments
- **Total segments**: 4,487 (before filtering)
- Config: `configs/preprocessing.yaml` → `segment_seconds: 45.0`, `hop_seconds: 15.0`, `sample_rate: 22050`

### Filtering Pipeline
1. **Voiced ratio filter** (threshold 0.25): removes segments with <25% melodic content → 2,145 remain (52% data loss — 2,342 removed)
2. **Raga count filter** (threshold 20 or 60 segments):
   - At threshold 20: 14 ragas, 2,107 segments, 53 tracks
   - At threshold 60: 10 ragas, 1,974 segments, 40 tracks

### Class Distribution (14 ragas, after voiced filter + min 20 segments)
```
366  Bhairavi           (4 tracks)
303  Sanmukhapriya      (5 tracks → 3 after filtering)
258  Kambhoji           (3 tracks)
248  Todi               (6 tracks)
174  Begada             (3 tracks)
170  Mohanam            (4 tracks)
135  Ritigaula          (4 tracks)
129  Harikambhoji       (3 tracks)
114  Kamas              (7 tracks)
 77  Saveri             (3 tracks)
 54  Kalyani            (4 tracks)
 29  Jonpuri            (2 tracks)
 25  Behag              (4 tracks)
 25  Sindhubhairavi     (3 tracks)
```

### Dropped Ragas (at threshold 20)
- Kanada: 10 segments
- Surati: 18 segments
- Sankarabharanam: 10 segments

### Additional Drops (at threshold 60, for 10-raga run)
- Behag: 25, Jonpuri: 29, Kalyani: 54, Sindhubhairavi: 25

### Dataset Challenges
- **Imbalance ratio**: 14.6:1 (Bhairavi 366 vs Behag/Sindhubhairavi 25)
- **16 unusable tracks**: all have `ctonic: null` in SARAGA metadata — no tonic annotation
- **Overlapping segments**: 66% overlap requires GroupKFold to prevent data leakage
- **Few tracks per raga**: 2-7 tracks each; GroupKFold with 5 folds means only 1-2 test tracks per fold per raga

---

## 3. Pipeline Architecture (6 Stages)

### Stage 1: Preprocessing (`carnatic_scripts/01_preprocess.py`)
- Loads SARAGA dataset via `mirdata`
- Filters tracks: must have raga label + tonic annotation + audio, no ragamalika
- Segments each track into overlapping 45s windows with 15s hop
- Resamples to 22050 Hz
- **Output**: `data/interim/*.wav` + `data/processed/segments_saraga_carnatic.csv`

### Stage 2: Pitch Extraction (`carnatic_scripts/02_pitch_baseline.py`)
- **Primary**: torchcrepe (CREPE deep learning pitch tracker) on GPU
- **Fallback**: librosa.pyin on CPU
- Hop length: 256 samples (~11.6ms per frame at 22050 Hz)
- Voiced frames: confidence >= 0.5; below → unvoiced (NaN)
- **Output**: `data/processed/pitch/*.npz` (f0, voiced_prob per segment) + `pitch_all.csv`

### Stage 3: Tonic Estimation (`carnatic_scripts/03_tonic_baseline.py`)
- Pools all F0 values per track
- Searches candidate tonics in log-frequency steps (10-cent resolution)
- Scores each candidate by pitch class histogram mass near 0 cents (Sa region)
- Validates against ground-truth tonic
- **~20% octave error rate** (estimated tonic off by exactly one octave)
- Ground-truth tonic used by default for downstream stages
- Config: `configs/tonic.yaml` → `candidate_min_hz: 80`, `candidate_max_hz: 300`, `scoring_method: "sa_mass"`
- **Output**: `data/processed/tonic_validation.csv`

### Stage 4: Swara Mapping (`carnatic_scripts/04_swara_mapping.py`)
- Converts F0 contour to cents relative to tonic (Sa), folded into [0, 1200) via mod
- Maps each frame to nearest of 12 swara bins (100-cent spacing):
  - Sa=0, Ri1=100, Ri2=200, Ga1=300, Ga2=400, Ma1=500, Ma2=600, Pa=700, Da1=800, Da2=900, Ni1=1000, Ni2=1100
- **Tolerance**: 40 cents — frames further than 40 cents from any bin → "glide"
- Unvoiced frames → "silence"
- **Min duration filter**: swara label runs shorter than 100ms → converted to "glide" (reduces label flickering)
- Config: `configs/swara.yaml` → `tolerance_cents: 40`, `min_duration_ms: 100`
- **Output**: `data/processed/swara/*.npz` (swara_labels, cents_mod, distances per segment) + `swara_all.csv`

### Stage 5: Feature Extraction (`carnatic_scripts/05_feature_extraction.py`)
Extracts 24 feature representations saved in `data/processed/features/features_all.npz`:

| Key | Shape | Description |
|-----|-------|-------------|
| `transition` | (4487, 144) | Row-normalised 12x12 swara transition matrix (flattened) |
| `arohana_transition` | (4487, 144) | Ascending transitions only (dst > src in swara index) |
| `avarohana_transition` | (4487, 144) | Descending transitions only (dst < src) |
| `directional_transition` | (4487, 288) | arohana + avarohana concatenated |
| `bigram` | (4487, 144) | Bigram frequency over swara tokens |
| `trigram` | (4487, 200) | Top-200 trigram vocabulary counts |
| `pcd` | (4487, 120) | Pitch class distribution (10-cent bins, one octave) |
| `mfcc` | (4487, 58) | MFCCs (20 mean + 20 std) + chroma (12) + spectral centroid/rolloff + ZCR |
| `auxiliary` | (4487, 10) | glide_ratio, silence_ratio, mean/std_dist_to_bin, pct_within_tol, n_tokens, tokens/sec, unique_swaras, transition_entropy, swara_entropy |
| `dwell_time` | (4487, 12) | Mean frame duration per swara (sustain profile) |
| `vadi` | (4487, 4) | vadi_idx, vadi_fraction, samvadi_idx, samvadi_fraction |
| `dwell_vadi` | (4487, 16) | dwell_time + vadi concatenated |
| `trigram_dwell` | (4487, 216) | trigram + dwell_vadi concatenated |
| `transition_aux` | (4487, 154) | transition + auxiliary |
| `combined` | (4487, 264) | transition + pcd |
| `combined_interp_aux` | (4487, 274) | transition + pcd + auxiliary |
| `combined_mfcc_pcd` | (4487, 178) | mfcc + pcd |
| `full_combined` | (4487, 332) | transition + pcd + mfcc + auxiliary |
| `directional_pcd` | (4487, 408) | directional_transition + pcd |
| `directional_aux` | (4487, 298) | directional_transition + auxiliary |
| `directional_full` | (4487, 418) | directional_transition + pcd + auxiliary |

**Token collapse**: `get_swara_token_sequence()` — consecutive repeated labels merged, silence and glide reset the previous label. "Sa Sa Sa Ri2 Ri2 glide Ga2 Ga2 Ga2" → ["Sa", "Ri2", "Ga2"]. Loses duration information.

**Trigram vocabulary**: Computed globally across all 4,487 segments, top 200 most frequent kept. Top trigrams include Sa→Sa→Sa, Pa→Pa→Pa, Ga2→Ga2→Ga2 (self-loops from segments with sustained notes that still produce repeated tokens).

**Dwell-time** (12 dims): For each swara, computes mean consecutive frame run length from raw swara_labels. Captures sustain information that token collapse destroys.

**Vadi features** (4 dims): Identifies most-prominent (vadi) and second-most-prominent (samvadi) swara by total frame time. Operational definition — not musicological vadi, but functionally similar.

### Stage 6: Classification (`carnatic_scripts/06_train_classifiers.py`)

**SVM Configuration:**
- Kernel: RBF
- `class_weight="balanced"` (inversely proportional to class frequency)
- GridSearchCV: C ∈ {0.1, 1, 10, 100}, gamma ∈ {"scale", 0.01, 0.1}
- Scoring: F1 weighted
- Best params (trigram_dwell): C=1, gamma="scale"
- Pipeline: StandardScaler → SVC

**Cross-Validation:**
- GroupKFold(n_splits=5) grouped by track_id
- Prevents data leakage from overlapping segments of the same track
- Without GroupKFold, accuracy would be artificially inflated

**LSTM Configuration:**
- Embedding: 32-dim, padding_idx=0
- Architecture: 2-layer bidirectional LSTM, hidden_dim=64, dropout=0.3
- Max sequence length: 200 tokens
- Optimizer: Adam, lr=0.001
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Early stopping: patience=10 epochs on validation accuracy
- Max epochs: 100
- Vocab: 14 symbols (12 swaras + glide + silence)

**Fusion Methods:**
- Probability fusion: F1-weighted average of SVM probability vectors across multiple models
- Hybrid track fusion: Pools segment-level predictions from SVM + LSTM per track, majority vote

**Track-level predictions**: Majority vote across all segment predictions for each track

---

## 4. Complete Results

### 14-Raga Run (min_segments_per_raga=20, 2107 segments, 53 tracks)

| Model | Seg Acc | Seg F1w | Trk Acc | Trk F1w | Dims |
|-------|---------|---------|---------|---------|------|
| SVM_trigram_dwell | 0.6018 | 0.5950 | 0.5849 | 0.5490 | 216 |
| fusion_pitch | 0.5909 | 0.5782 | 0.5283 | 0.4931 | — |
| SVM_trigram | 0.5795 | 0.5734 | 0.5660 | 0.5181 | 200 |
| SVM_pcd_trigram | 0.5676 | 0.5514 | 0.4906 | 0.4571 | 320 |
| SVM_directional_pcd | 0.5600 | 0.5356 | 0.5283 | 0.4908 | 408 |
| SVM_pcd_trigram_aux | 0.5586 | 0.5438 | 0.5094 | 0.4701 | 330 |
| SVM_directional_full | 0.5510 | 0.5274 | 0.5094 | 0.4780 | 418 |
| SVM_dwell_vadi | 0.5297 | 0.5303 | 0.5283 | 0.5270 | 16 |
| SVM_pcd | 0.5059 | 0.4947 | 0.4528 | 0.4025 | 120 |
| LSTM_swara_seq | 0.5107 | — | 0.4906 | — | seq |
| SVM_mfcc | 0.1191 | 0.0395 | 0.0943 | 0.0345 | 58 |

### 10-Raga Run (min_segments_per_raga=60, 1974 segments, 40 tracks)

| Model | Seg Acc | Seg F1w | Trk Acc | Trk F1w | Dims |
|-------|---------|---------|---------|---------|------|
| **SVM_trigram_dwell** | **0.6368** | **0.6383** | **0.7000** | **0.7051** | 216 |
| fusion_pitch | 0.6337 | 0.6334 | 0.6250 | 0.6267 | — |
| SVM_trigram | 0.6211 | 0.6194 | 0.7000 | 0.7089 | 200 |
| SVM_pcd_trigram | 0.6074 | 0.6032 | 0.6250 | 0.6281 | 320 |
| SVM_dwell_vadi | 0.6028 | 0.6003 | 0.6500 | 0.6541 | 16 |
| SVM_directional_pcd | 0.5988 | 0.5896 | 0.6000 | 0.5896 | 408 |
| SVM_pcd_trigram_aux | 0.5973 | 0.5936 | 0.6500 | 0.6522 | 330 |
| SVM_directional_full | 0.5861 | 0.5780 | 0.6000 | 0.5975 | 418 |
| SVM_pcd | 0.5390 | 0.5397 | 0.5500 | 0.5466 | 120 |
| SVM_mfcc | 0.0628 | 0.0525 | 0.0500 | 0.0229 | 58 |

### 14→10 Raga Improvement

| Model | 14-raga Seg | 10-raga Seg | Δ | 14-raga Trk | 10-raga Trk | Δ |
|---|---|---|---|---|---|---|
| trigram_dwell | 0.6018 | 0.6368 | +3.5% | 0.5849 | 0.7000 | +11.5% |
| trigram | 0.5795 | 0.6211 | +4.2% | 0.5660 | 0.7000 | +13.4% |
| dwell_vadi | 0.5297 | 0.6028 | +7.3% | 0.5283 | 0.6500 | +12.2% |
| pcd | 0.5059 | 0.5390 | +3.3% | 0.4528 | 0.5500 | +9.7% |

### Per-Raga F1 Scores (SVM_trigram_dwell, 10-raga run)

| Raga | F1 | Recall | Precision | Support | Tracks |
|------|-----|--------|-----------|---------|--------|
| Bhairavi | 0.89 | 85% | 94% | 366 | 4 |
| Todi | 0.86 | 92% | 81% | 248 | 6 |
| Sanmukhapriya | 0.84 | 89% | 79% | 303 | 3 |
| Ritigaula | 0.80 | 73% | 88% | 135 | 4 |
| Saveri | 0.61 | 47% | 88% | 77 | 3 |
| Begada | 0.52 | 55% | 49% | 174 | 3 |
| Mohanam | 0.38 | 35% | 42% | 170 | 4 |
| Kambhoji | 0.37 | 38% | 37% | 258 | 3 |
| Kamas | 0.36 | 36% | 36% | 114 | 7 |
| Harikambhoji | 0.13 | 15% | 12% | 129 | 3 |

### LSTM Notable Results
- Overall: 0.5107 segment accuracy
- **Fold 1: 0.7417** (best single fold ever observed)
- High variance across folds (0.74 vs ~0.45 in others)

### SVM Best Params (consistent across most feature sets)
- C=1, gamma="scale" (except dwell_vadi: C=1, gamma=0.01; mfcc: C=0.1, gamma=0.1)

### Fold Accuracies (SVM_trigram_dwell, 10-raga)
- Fold 1: 0.747, Fold 2: 0.668, Fold 3: 0.491, Fold 4: 0.718, Fold 5: 0.559

---

## 5. Failed Experiments (Negative Results)

### Phrase-Aware Trigrams — Catastrophic Failure
- **Hypothesis**: Split swara token sequences at glide boundaries to create cleaner melodic phrases before computing trigrams
- **Result**: Trigram accuracy collapsed from 0.580 → 0.180
- **Root cause**: In Carnatic music, gamakas (glides/oscillations) CONNECT swaras — they are not phrase boundaries. A typical phrase like "Sa-Ri2-Ga2" involves continuous gliding between notes. Splitting at glides fragmented phrases into isolated single-note "phrases", destroying the trigram signal.
- **Lesson**: Feature engineering must respect the musical domain. Western music intuitions about phrase boundaries don't apply to Carnatic music.

### 4-Gram Features — Too Sparse
- **Result**: 0.158 accuracy
- **Root cause**: Vocabulary explodes (12^4 = 20,736 possible 4-grams) with only ~2,100 training samples. Even top-200 4-grams were too sparse.

### MFCC Features — Completely Useless
- 14-raga: 0.119 (random chance = 1/14 = 0.071)
- 10-raga: 0.063 (random chance = 1/10 = 0.100, MFCC is BELOW random)
- **Implication**: Raga is a purely melodic/structural concept, not timbral. The same raga can be sung by any voice or played on any instrument.

### PCD + Trigrams — Redundancy Hurts
- trigram alone (200d): 0.5795
- pcd_trigram (320d): 0.5676 (-1.2%)
- Trigrams implicitly encode note frequency. PCD adds 120 dimensions of mostly redundant information.

### More Features = Worse
- directional_full (418d): 0.5510 < trigram (200d): 0.5795
- Clear curse of dimensionality with ~2,100 samples

---

## 6. Ablation Studies

### Feature Dimensionality vs Accuracy
| Dims | Feature | Seg Acc (14-raga) |
|------|---------|----------|
| 16 | dwell_vadi | 0.5297 |
| 58 | mfcc | 0.1191 |
| 120 | pcd | 0.5059 |
| 200 | trigram | 0.5795 |
| **216** | **trigram_dwell** | **0.6018** |
| 320 | pcd_trigram | 0.5676 |
| 330 | pcd_trigram_aux | 0.5586 |
| 408 | directional_pcd | 0.5600 |
| 418 | directional_full | 0.5510 |

Peak at 216 dims, then decline. Classic curse of dimensionality.

### Feature Redundancy (what helps vs hurts trigrams)
- trigram (baseline): 0.5795
- + PCD (320d): 0.5676 (-1.2%) — redundant
- + PCD + aux (330d): 0.5586 (-2.1%) — more noise
- + dwell (216d): 0.6018 (+2.2%) — genuinely new information

### Token Collapse Ablation
- With collapse (trigram): sequential melodic patterns captured, but duration lost
- Without collapse (dwell_time): raw frame-level durations, no sequence
- Both together (trigram_dwell): 0.6018 — combining sequence AND duration is optimal

### Voiced Ratio Filter
- At 0.25 threshold: 2,145/4,487 segments kept (52% data loss)
- Low-voiced segments are mostly drone/tanpura passages — including them adds noise

---

## 7. Case Studies

### Bhairavi — The Easy Raga (0.89 F1, 10-raga)
- Highest performance across all models. Distinctive scale and characteristic Ri2/Da2 gamakas.
- 366 segments (most data) but not just volume: Sanmukhapriya (303 segs) also scores 0.84.
- Musicologically considered one of the most recognisable ragas — model confirms this.

### Harikambhoji — The Impossible Raga (0.13 F1, 10-raga)
- 129 segments, 3 tracks, near-zero recall in ALL feature sets.
- Parent melakarta scale — shares its scale with derived ragas Kambhoji, Kamas, Behag.
- Interesting: dwell_vadi handles it better (0.25 F1) than trigrams (0.12). Identity may be more about note emphasis than sequence.

### Kalyani — Scale Overlap Problem (0% recall, 14-raga; dropped in 10-raga)
- Only 54 segments, 4 tracks. Distinctive Ma2 should help but model can't learn it.
- Possible tonic octave errors affecting Ma1/Ma2 distinction.

### Todi — Consistent High Performer (0.86 F1, 10-raga)
- 248 segments, 6 tracks (most tracks of any raga).
- Distinctive Ri1-Ga1 oscillation captured well by both trigrams and dwell-time.

### dwell_vadi at 16 Dimensions
- 0.5297 (14-raga), 0.6028 (10-raga) — with just 16 features.
- Validates the musicological principle that ragas are defined by which notes are emphasised and how long they're held.

### Kamas Paradox — Most Tracks, Poor Results
- 7 tracks (most of any raga) but only 0.36 F1.
- Kamas is a janya of Harikambhoji, shares notes with Kambhoji.
- More data doesn't help if the raga's identity is subtle.

### LSTM Fold 1 — 0.7417
- Shows the ceiling is much higher with more data.
- High variance (0.74 vs ~0.45) due to only 53 tracks / 5 folds.

---

## 8. Cognitive Modelling Argument

### How Humans Recognise Ragas
1. **Characteristic phrases (pakad)**: Short melodic motifs unique to each raga → captured by trigrams
2. **Vadi/Samvadi**: Most emphasised note and complement → captured by vadi features
3. **Gamaka patterns**: How notes are ornamented → partially captured by glide_ratio and dwell_time
4. **Note hierarchy**: Not just which notes, but emphasis through duration → captured by dwell_time
5. **Sequential context**: Notes derive meaning from neighbours → captured by trigrams and LSTM

### Our Best Model as a Cognitive Model
`trigram_dwell` (216 dims, 0.6018/0.6368 seg acc) combines:
- **Trigrams** → "which melodic patterns occur" (recognising characteristic phrases)
- **Dwell-time** → "which notes are sustained" (hearing emphasis/importance)

This is fully interpretable: for any segment you can examine trigram counts and dwell profiles to explain the prediction.

### What the Model Cannot Capture
- Gamaka quality (kampita, jaru, etc. — we only know "glide", not the type)
- Tempo/rhythm (stripped by swara mapping)
- Octave register (cents folded into [0, 1200))
- Performance style (artist-specific gamaka patterns)
- Emotional quality (rasa)

### Key Evidence
1. **MFCC useless** → raga is structure, not timbre
2. **Trigrams work** → sequential note patterns are primary cue
3. **Dwell-time adds value** → sustain/emphasis is secondary cue
4. **More features hurt** → cognitive recognition uses few salient cues, not all available info
5. **Phrase-aware splitting failed** → gamakas connect notes (continuous melodic flow), not separate phrases

---

## 9. Key Numbers

| Metric | Value |
|--------|-------|
| Best segment accuracy (14 ragas) | 0.6018 (SVM_trigram_dwell) |
| Best track accuracy (14 ragas) | 0.5849 (SVM_trigram_dwell) |
| Best segment accuracy (10 ragas) | 0.6368 (SVM_trigram_dwell) |
| Best track accuracy (10 ragas) | 0.7000 (SVM_trigram_dwell & SVM_trigram) |
| Random chance (14 classes) | 7.1% |
| Random chance (10 classes) | 10.0% |
| Majority class baseline (14 ragas) | 17.4% (Bhairavi) |
| MFCC accuracy | 6.3% (10-raga, below random) |
| Dataset segments | 4,487 total, 2,107 after filtering (14-raga) |
| Tracks | 73 usable / 89 total |
| Data loss from voiced filter | 52% |
| Class imbalance ratio | 14.6:1 |
| Segment overlap | 66% |
| Tonic octave error rate | ~20% |
| Swara tolerance | 40 cents |
| Trigram vocabulary size | 200 (of 1,728 possible) |
| Best LSTM single fold | 0.7417 |

---

## 10. Swara System Reference

12 semitones at equal-tempered 100-cent spacing relative to Sa (tonic):

| Swara | Cents | Western Approx | Notes |
|-------|-------|----------------|-------|
| Sa | 0 | Tonic | Fixed, always present |
| Ri1 | 100 | Minor 2nd | Shuddha Rishabham |
| Ri2 | 200 | Major 2nd | Chatushruti Rishabham |
| Ga1 | 300 | Minor 3rd | Sadharana Gandharam |
| Ga2 | 400 | Major 3rd | Antara Gandharam |
| Ma1 | 500 | Perfect 4th | Shuddha Madhyamam |
| Ma2 | 600 | Tritone | Prati Madhyamam |
| Pa | 700 | Perfect 5th | Fixed, always present |
| Da1 | 800 | Minor 6th | Shuddha Dhaivatam |
| Da2 | 900 | Major 6th | Chatushruti Dhaivatam |
| Ni1 | 1000 | Minor 7th | Kaisiki Nishadam |
| Ni2 | 1100 | Major 7th | Kakali Nishadam |

Additional labels: "glide" (gamaka/ornament, >40 cents from any bin), "silence" (unvoiced frame)

---

## 11. Configuration Files

### preprocessing.yaml
```yaml
dataset_name: saraga_carnatic
sample_rate: 22050
segment_seconds: 45.0
hop_seconds: 15.0
max_tracks: 20
trim_silence: false
```

### swara.yaml
```yaml
bins:
  Sa: 0, Ri1: 100, Ri2: 200, Ga1: 300, Ga2: 400, Ma1: 500,
  Ma2: 600, Pa: 700, Da1: 800, Da2: 900, Ni1: 1000, Ni2: 1100
tolerance_cents: 40
min_duration_ms: 100
```

### tonic.yaml
```yaml
candidate_min_hz: 80.0
candidate_max_hz: 300.0
step_cents: 10.0
bins_per_octave: 120
scoring_method: "sa_mass"
min_voiced_frames: 200
```

---

## 12. File Structure

```
kxk224/
├── carnatic_scripts/
│   ├── 01_preprocess.py          # Audio segmentation
│   ├── 02_pitch_baseline.py      # CREPE/pyin pitch extraction
│   ├── 03_tonic_baseline.py      # Tonic (Sa) estimation
│   ├── 04_swara_mapping.py       # Pitch → swara label mapping
│   ├── 05_feature_extraction.py  # All feature representations (667 lines)
│   └── 06_train_classifiers.py   # SVM/LSTM/Fusion training (959 lines)
├── app/
│   └── app.py                    # Streamlit dashboard (910 lines)
├── run_pipeline.py               # Pipeline orchestrator
├── configs/
│   ├── preprocessing.yaml
│   ├── swara.yaml
│   └── tonic.yaml
├── data/
│   ├── raw/mirdata/              # SARAGA dataset
│   ├── interim/                  # .wav segments
│   └── processed/
│       ├── segments_saraga_carnatic.csv
│       ├── pitch_all.csv
│       ├── pitch/*.npz
│       ├── tonic_validation.csv
│       ├── swara_all.csv
│       ├── swara/*.npz
│       └── features/
│           ├── features_all.npz  # 24 feature arrays, 13.4 MB
│           ├── features_meta.csv
│           └── trigram_vocab.txt
├── models/                       # Saved .pkl SVM models
├── outputs/
│   ├── results/classification_results.json
│   └── figures/
│       ├── classification/       # Confusion matrices
│       ├── features/             # Feature distribution plots
│       ├── pitch_debug/
│       ├── swara_debug/
│       └── tonic_validation/
└── CLAUDE.md
```

---

## 13. Top 30 Trigrams (from trigram_vocab.txt)

```
Sa→Sa→Sa, Pa→Pa→Pa, Ga2→Ga2→Ga2, Da2→Da2→Da2, Ri2→Ri2→Ri2,
Ri2→Sa→Sa, Ma1→Ma1→Ma1, Sa→Sa→Da2, Da2→Sa→Sa, Sa→Sa→Ri2,
Ri2→Ri2→Sa, Da1→Pa→Pa, Pa→Pa→Da2, Sa→Ri2→Ri2, Pa→Da1→Pa,
Da2→Pa→Pa, Sa→Da2→Sa, Sa→Ri2→Sa, Pa→Da2→Da2, Da2→Da2→Sa,
Sa→Sa→Pa, Ni1→Ni1→Ni1, Da2→Da2→Pa, Sa→Sa→Ni1, Pa→Sa→Sa,
Pa→Pa→Ma1, Pa→Pa→Da1, Ma1→Pa→Pa, Sa→Pa→Pa, Da2→Ni1→Da2
```

---

## 14. Discussion Points for Paper

### Why 60%/70% and Not Higher?
1. Data scarcity: 53 tracks, 5-fold CV means ~10 test tracks per fold
2. 4 ragas effectively unlearnable with 2-4 tracks each
3. Artist variation: 2-7 tracks per raga → may learn artist style not raga
4. Tonic errors: ~20% octave errors propagate through entire pipeline
5. Gamaka complexity: Carnatic music is heavily ornamented

### Comparison to Literature
- Many papers report higher accuracy but: use train/test splits without GroupKFold (data leakage), use fewer ragas (5-10), use Hindustani music (less gamaka), use deep learning on spectrograms (no interpretability)
- Our contribution is INTERPRETABILITY, not raw accuracy

### Future Work
1. Drop unclassifiable ragas → 0.68-0.72 expected
2. Reduce hop to 7.5s → double training data
3. Gamaka typing (classify glide types)
4. Attention-based LSTM to identify discriminative phrase regions
5. Cross-dataset validation (Hindustani ragas)
6. Tonic robustness study

---

## 15. Streamlit Dashboard (app/app.py)

7 pages:
1. **Run Pipeline** — Execute stages 1-6 with configurable options, live log, quick-action buttons
2. **Overview** — Dataset stats, pipeline diagram, raga distribution, voiced ratio histogram
3. **Model Comparison** — Results table, bar chart, fold variance boxplots, confusion matrix viewer
4. **Per-Raga Analysis** — Dwell-time profile, PCD curve, dominant swara distribution, swara traces
5. **Feature Explorer** — Feature inventory, side-by-side raga comparison, top-20 trigrams per raga
6. **Segment Inspector** — Individual segment swara trace with cents contour, token sequence
7. **Ablation Studies** — Dimensionality vs accuracy, feature redundancy, timbral vs structural
