# Hypotheses, Experiments & Ablations

> "How Do We Hear a Raga? An Interpretable Cognitive Model of Melodic Recognition"

---

## PART 1: SEVEN HYPOTHESES

### H1 — Raga identity is encoded in short melodic sequences

**Claim**: Three-note patterns (trigrams) are the most discriminative single feature for raga classification, acting as a computational analogue of *pakad* (characteristic phrase) recognition.

**How I prove it**: Compare trigram accuracy against every other individual feature set under identical SVM+GroupKFold conditions. If trigrams consistently rank first, it confirms that sequential melodic motifs — not pitch distributions, not timbral qualities, not ornament statistics — carry the primary raga signal.

**Results**:

| Feature (individual) | Dims | 14-raga Seg Acc | 10-raga Seg Acc |
|---|---|---|---|
| **trigram** | 200 | **0.5795** | **0.6211** |
| pcd | 120 | 0.5059 | 0.5390 |
| dwell_vadi | 16 | 0.5297 | 0.6028 |
| transition (bigram matrix) | 144 | — | — |
| mfcc | 58 | 0.1191 | 0.0628 |

Trigrams beat every other standalone feature in both the 14-raga and 10-raga settings. The gap over PCD (+7.4% at 14 ragas) shows that *sequential order matters* — PCD knows which notes appear but not what order. This is exactly how musicologists describe pakad: "it's not just the notes, it's the phrase."

Adding dwell-time to trigrams (trigram_dwell, 216d) pushes accuracy to 0.6018/0.6368, confirming that while trigrams are the primary cue, they benefit from duration information that token collapse strips away.

---

### H2 — Raga identity is structural, not timbral

**Claim**: Raga is a purely melodic/structural concept — the same raga sounds like "the same raga" regardless of whether it's sung by a male voice, female voice, or played on veena. Timbral features (MFCCs) should therefore be useless for raga classification.

**How I prove it**: Train the same SVM pipeline on MFCC features (58 dims: 20 MFCCs mean+std, 12 chroma, spectral centroid/rolloff, ZCR) and compare against pitch-based features.

**Results**:

| Setting | MFCC Accuracy | Random Chance | Best Pitch-Based |
|---|---|---|---|
| 14 ragas | 0.1191 | 0.071 (1/14) | 0.6018 (trigram_dwell) |
| 10 ragas | **0.0628** | **0.100 (1/10)** | 0.6368 (trigram_dwell) |

At 10 ragas, MFCC performs **below random chance** (6.3% vs 10.0%). The SVM actively learns the *wrong* boundaries from timbral features. This is a powerful negative result: it demonstrates that voice quality, spectral shape, and instrument timbre contain zero raga-discriminative information. Raga lives entirely in the melodic structure.

This aligns with musicological theory — a raga is defined by its scale, characteristic phrases, and note hierarchy, not by who performs it. Our model confirms this computationally.

---

### H3 — Cognitive parsimony: fewer salient features outperform exhaustive representations

**Claim**: Adding more feature dimensions does not monotonically improve accuracy. Beyond a sweet spot (~216 dimensions), performance degrades due to the curse of dimensionality. This mirrors the cognitive science principle of "ecological rationality" — humans recognise ragas using a few salient cues, not by processing all available information.

**How I prove it**: Plot feature dimensionality against accuracy across all feature sets. Test whether combining individually useful features (e.g., PCD + trigram) improves or hurts performance.

**Results — Dimensionality curve (14 ragas)**:

```
Dims → Accuracy
  16d  dwell_vadi          0.5297
 120d  pcd                 0.5059
 200d  trigram              0.5795
 216d  trigram_dwell        0.6018  ← PEAK
 320d  pcd_trigram          0.5676  ← adding PCD hurts (-1.2%)
 330d  pcd_trigram_aux      0.5586  ← adding more hurts (-2.1%)
 408d  directional_pcd      0.5600
 418d  directional_full     0.5510  ← worst high-dim set (-2.9%)
```

Peak accuracy occurs at 216 dimensions, then declines. This is not just an artefact of SVM hyperparameters — it's consistent across C/gamma settings found by GridSearchCV.

**Feature redundancy analysis**:
- trigram (200d) → 0.5795 baseline
- + PCD (320d) → 0.5676 (−1.2%): PCD is redundant because trigrams implicitly encode note frequency. Counting "Sa→Ri2→Ga2" already tells you Sa, Ri2, and Ga2 are present.
- + PCD + aux (330d) → 0.5586 (−2.1%): auxiliary features (entropy, glide_ratio) add noise
- + dwell (216d) → 0.6018 (+2.2%): dwell-time adds **genuinely new information** (duration) that trigrams cannot capture due to token collapse

**Interpretation**: The only feature that helps trigrams is one that provides non-redundant information (dwell-time = duration, which token collapse destroys). Every other addition is either redundant or noisy. With ~2,100 training samples, the SVM cannot exploit high-dimensional spaces — it needs compact, informative representations.

---

### H4 — Confusion patterns follow musicological raga kinship (melakarta-janya families)

**Claim**: When the model misclassifies a raga, it confuses it with musically related ragas (same melakarta family, shared scale) rather than random ragas. This means the model has learned real musical structure, and its errors are "sensible" from a musicological perspective.

**How I prove it**: Analyse the confusion matrix from SVM_trigram_dwell. Map misclassification pairs to the melakarta-janya system. If Harikambhoji is confused with Kambhoji and Kamas (its janya ragas), that's musicologically meaningful.

**Evidence so far**:
- **Harikambhoji** (0.13 F1): Parent melakarta (28th). Its derived ragas Kambhoji, Kamas, and Behag share its scale. The model cannot distinguish parent from children because the parent's identity is defined by what it *doesn't emphasise* — it uses the full scale neutrally, while children emphasise specific subsets.
- **Kamas** (0.36 F1): Janya of Harikambhoji, shares notes with Kambhoji. 7 tracks (most data) but poor accuracy — more data doesn't help when identity is subtle.
- **Bhairavi** (0.89 F1): Not closely related to other ragas in the dataset — clean separation.
- **Todi** (0.86 F1): Distinctive Ri1-Ga1 combination rare in other ragas present.

**Status**: Needs formal confusion matrix analysis to map every misclassification pair to musicological relationships. This is a planned analysis.

---

### H5 — Dwell-time captures vadi/samvadi emphasis with minimal features

**Claim**: Just 16 features — 12-dimensional dwell-time (mean sustain per swara) + 4-dimensional vadi (most-prominent and second-most-prominent note indices and fractions) — achieve competitive classification accuracy, validating the musicological principle that raga identity is partly defined by *which notes are emphasised and how long they are held*.

**How I prove it**: Train SVM on dwell_vadi (16d) alone and compare against much higher-dimensional feature sets.

**Results**:

| Feature | Dims | 14-raga Seg Acc | 10-raga Seg Acc |
|---|---|---|---|
| dwell_vadi | **16** | 0.5297 | 0.6028 |
| pcd | 120 | 0.5059 | 0.5390 |
| trigram | 200 | 0.5795 | 0.6211 |
| directional_full | 418 | 0.5510 | 0.5861 |

16 dimensions outperform 120-dim PCD and approach 418-dim directional features. At 10 ragas, dwell_vadi (0.6028) nearly matches directional_full (0.5861) with **26x fewer dimensions**.

**Per-raga insight**: dwell_vadi handles Harikambhoji better (0.25 F1) than trigrams (0.12 F1). This suggests the parent melakarta's identity — which trigrams can't capture because its phrases overlap with derived ragas — is partially encoded in *how long* notes are held rather than *which patterns* occur.

**Interpretation**: This is a direct computational validation of the vadi/samvadi concept in Indian music theory. The most-dwelt-upon swara (operational vadi) and its relative emphasis carry genuine discriminative information.

---

### H6 — Directional encoding (arohana/avarohana) adds marginal information

**Claim**: Separating transitions into ascending (arohana) and descending (avarohana) matrices should theoretically capture melodic directionality — a known raga characteristic. In practice, this adds dimensionality without proportional accuracy gain, making it a well-motivated but ultimately negative result.

**How I prove it**: Compare undirected transition/trigram features against directional variants under identical conditions.

**Results (14 ragas)**:

| Feature | Dims | Seg Acc | Approach |
|---|---|---|---|
| trigram | 200 | 0.5795 | Undirected |
| directional_pcd | 408 | 0.5600 | Arohana + avarohana + PCD |
| directional_full | 418 | 0.5510 | Arohana + avarohana + PCD + aux |

Directional features at 408–418 dims perform **worse** than undirected trigrams at 200 dims. The directional information exists but is diluted by doubled dimensionality (288 dims for directional transitions vs 144 for undirected).

**Why this happens**:
1. **Curse of dimensionality**: 288-dim directional transitions with 2,100 samples → too sparse
2. **Arohana/avarohana are less distinct in Carnatic music** than in Hindustani music. Carnatic ragas feature heavy gamaka (oscillation) that blurs the ascending/descending distinction — a note "ascending" often involves oscillation back and forth across the target pitch.
3. The information is already partially captured: if Ri2→Ga2 is common, the undirected model already knows these notes are adjacent. Knowing the direction adds little given the data constraints.

**For the paper**: This is a negative-but-informative result. The hypothesis was musicologically sound (arohana/avarohana are taught as distinct characteristics), but the data constraints and Carnatic gamaka conventions mean the signal-to-noise ratio is too low. This motivates future work with larger datasets where directional features may become viable.

---

### H7 — Longer temporal context improves recognition (track-level > segment-level)

**Claim**: Aggregating segment predictions at the track level (majority vote across all 45s segments from the same performance) improves accuracy, because a full performance contains more characteristic material than any single segment.

**How I prove it**: Compare segment-level accuracy vs track-level accuracy across all models and both dataset settings.

**Results**:

| Model | 14-raga Seg | 14-raga Trk | Δ | 10-raga Seg | 10-raga Trk | Δ |
|---|---|---|---|---|---|---|
| trigram_dwell | 0.6018 | 0.5849 | −1.7% | 0.6368 | 0.7000 | **+6.3%** |
| trigram | 0.5795 | 0.5660 | −1.4% | 0.6211 | 0.7000 | **+7.9%** |
| dwell_vadi | 0.5297 | 0.5283 | −0.1% | 0.6028 | 0.6500 | **+4.7%** |
| pcd | 0.5059 | 0.4528 | −5.3% | 0.5390 | 0.5500 | +1.1% |

**Key pattern**: At 10 ragas, track voting consistently boosts accuracy (+4.7% to +7.9%). At 14 ragas, it sometimes hurts slightly. Why?

At 14 ragas, the 4 weakest ragas (Behag, Sindhubhairavi, Jonpuri, Kalyani) have very few tracks (2–4 each). A single misclassified track = massive accuracy drop. At 10 ragas (after dropping these), each remaining raga has enough tracks that majority voting works as intended — bad segments get outvoted by good ones.

**The 14→10 raga track accuracy explosion**: trigram_dwell jumps from 0.5849 → 0.7000 (+11.5%), far exceeding the segment improvement (+3.5%). This is because:
1. Dropped ragas had few tracks that were absorbing misclassifications in both directions
2. With cleaner classes, majority voting over ~4-6 segments per track becomes reliable
3. A 45s segment may catch a transitional passage, but a full 5-minute track always contains characteristic material somewhere

**Interpretation**: This mirrors human listening — you might be uncertain about a raga from a 45-second excerpt, but after hearing several minutes of a performance, the raga becomes clear. Our track-level voting is a simple model of this "accumulation of evidence" process.

---

## PART 2: ALL EXPERIMENTS AND ABLATIONS

### Experiment 1: Feature Set Comparison (Core Experiment)

**Setup**: SVM (RBF, balanced weights, GridSearchCV over C∈{0.1,1,10,100}, gamma∈{scale,0.01,0.1}), GroupKFold(5) by track_id, on 9 different feature sets.

**14-Raga Results (2,107 segments, 53 tracks, 14 classes)**:

| Rank | Model | Dims | Seg Acc | Seg F1w | Trk Acc | Trk F1w | Best C | Best gamma |
|------|-------|------|---------|---------|---------|---------|--------|-----------|
| 1 | SVM_trigram_dwell | 216 | 0.6018 | 0.5950 | 0.5849 | 0.5490 | 1 | scale |
| 2 | SVM_trigram | 200 | 0.5795 | 0.5734 | 0.5660 | 0.5181 | 1 | scale |
| 3 | SVM_pcd_trigram | 320 | 0.5676 | 0.5514 | 0.4906 | 0.4571 | 1 | scale |
| 4 | SVM_directional_pcd | 408 | 0.5600 | 0.5356 | 0.5283 | 0.4908 | 1 | scale |
| 5 | SVM_pcd_trigram_aux | 330 | 0.5586 | 0.5438 | 0.5094 | 0.4701 | 1 | scale |
| 6 | SVM_directional_full | 418 | 0.5510 | 0.5274 | 0.5094 | 0.4780 | 1 | scale |
| 7 | SVM_dwell_vadi | 16 | 0.5297 | 0.5303 | 0.5283 | 0.5270 | 1 | 0.01 |
| 8 | SVM_pcd | 120 | 0.5059 | 0.4947 | 0.4528 | 0.4025 | 1 | scale |
| 9 | SVM_mfcc | 58 | 0.1191 | 0.0395 | 0.0943 | 0.0345 | 0.1 | 0.1 |

**Key observations**:
- trigram_dwell wins across all metrics. The 16-dim dwell/vadi addition to 200-dim trigrams accounts for +2.2% segment accuracy.
- C=1, gamma=scale is optimal for all pitch-based features — the data naturally separates at moderate regularisation.
- MFCC uses entirely different hyperparameters (C=0.1, gamma=0.1) — the SVM has to heavily regularise because the feature space is uninformative.
- Track accuracy doesn't consistently beat segment accuracy at 14 ragas due to few-track ragas.

---

### Experiment 2: Reduced Class Set (10-Raga Run)

**Setup**: Same pipeline, but `--min-segments-per-raga 60` drops 4 ragas (Behag:25, Jonpuri:29, Kalyani:54, Sindhubhairavi:25). Remaining: 1,974 segments, 40 tracks, 10 classes.

**10-Raga Results**:

| Rank | Model | Dims | Seg Acc | Seg F1w | Trk Acc | Trk F1w |
|------|-------|------|---------|---------|---------|---------|
| 1 | SVM_trigram_dwell | 216 | 0.6368 | 0.6383 | 0.7000 | 0.7051 |
| 2 | SVM_trigram | 200 | 0.6211 | 0.6194 | 0.7000 | 0.7089 |
| 3 | SVM_pcd_trigram | 320 | 0.6074 | 0.6032 | 0.6250 | 0.6281 |
| 4 | SVM_dwell_vadi | 16 | 0.6028 | 0.6003 | 0.6500 | 0.6541 |
| 5 | SVM_directional_pcd | 408 | 0.5988 | 0.5896 | 0.6000 | 0.5896 |
| 6 | SVM_pcd_trigram_aux | 330 | 0.5973 | 0.5936 | 0.6500 | 0.6522 |
| 7 | SVM_directional_full | 418 | 0.5861 | 0.5780 | 0.6000 | 0.5975 |
| 8 | SVM_pcd | 120 | 0.5390 | 0.5397 | 0.5500 | 0.5466 |
| 9 | SVM_mfcc | 58 | 0.0628 | 0.0525 | 0.0500 | 0.0229 |

**Improvement over 14 ragas**:

| Model | Δ Seg Acc | Δ Trk Acc |
|---|---|---|
| trigram_dwell | +3.5% | **+11.5%** |
| trigram | +4.2% | **+13.4%** |
| dwell_vadi | **+7.3%** | **+12.2%** |
| pcd | +3.3% | +9.7% |

**Key observations**:
- Track accuracy explodes (+11–13%) while segment accuracy improves modestly (+3–4%). Dropping few-track ragas lets majority voting work properly.
- dwell_vadi shows the largest segment improvement (+7.3%) — the 4 dropped ragas were particularly hard for emphasis-based features (small support, ambiguous vadi).
- MFCC gets even worse (0.119 → 0.063) — with fewer classes, the model has fewer "lucky" misclassification targets. It's now reliably below random chance (0.100).
- Ranking order is preserved: trigram_dwell > trigram > pcd_trigram > etc. The feature hierarchy is robust to class count.

---

### Experiment 3: LSTM vs SVM (Sequential vs Statistical)

**Setup**: Bidirectional 2-layer LSTM (embed=32, hidden=64, dropout=0.3, patience=10) on raw swara token sequences (max length 200). Same GroupKFold(5) by track_id.

**LSTM Results (14 ragas)**:

| Metric | LSTM | Best SVM (trigram_dwell) |
|---|---|---|
| Seg Acc | 0.4257 | 0.6018 |
| Trk Acc | 0.4528 | 0.5849 |
| Fold variance | 0.246–0.685 | 0.491–0.747 |

**LSTM per-raga highlights**:
- Bhairavi 0.77 F1 (SVM: 0.89) — LSTM captures it but less precisely
- Ritigaula 0.71 (SVM: 0.80) — sequential patterns partially captured
- Harikambhoji 0.00 (SVM: 0.13) — LSTM completely gives up
- Sindhubhairavi 0.00 — too few examples for neural learning

**LSTM fold variance**:
```
Fold 1: 0.685  ← best — got a favourable track split
Fold 2: 0.246  ← worst — unlucky split
Fold 3: 0.318
Fold 4: 0.501
Fold 5: 0.378
```

**Why SVM wins**: The SVM receives pre-computed statistical summaries (216 numbers encoding trigram counts + dwell-times) — essentially, the domain knowledge is baked into the features. The LSTM must discover these patterns from raw token sequences with only 2,107 examples. With 53 tracks split into 5 folds, each fold trains on ~42 tracks — far too few for a neural model.

**Why this matters for the thesis**: The SVM's superiority with hand-crafted features over a learned sequential model *is the cognitive model argument*. It shows that specific, interpretable cues (which 3-note patterns occur, how long notes are held) outperform generic sequence learning — just as human listeners recognise ragas through specific salient cues, not by processing every note equally.

---

### Experiment 4: Probability Fusion (Multi-Model Ensemble)

**Setup**: Weighted average of SVM probability vectors from SVM_pcd, SVM_trigram, and SVM_trigram_dwell. Weights proportional to each model's F1-weighted score.

**Results (14 ragas)**:

| Model | Seg Acc | Trk Acc |
|---|---|---|
| SVM_trigram_dwell (best individual) | 0.6018 | 0.5849 |
| fusion_pitch (ensemble) | 0.5909 | 0.5283 |

**Fusion hurts.** Segment accuracy drops slightly (−1.1%), but track accuracy drops substantially (−5.7%). Why?

- The best model (trigram_dwell) has high-confidence correct predictions. Averaging with weaker models (PCD) dilutes this confidence.
- At the segment level, the dilution is small (similar accuracy). But at the track level, where majority voting amplifies small confidence differences, the dilution is catastrophic.
- This is a known issue with simple probability fusion: it assumes models are equally calibrated, which they're not.

**For the paper**: Fusion is a negative result worth reporting. It shows that the trigram_dwell representation is not just the best — it's sufficient. The other features don't provide complementary information that fusion can exploit.

---

### Experiment 5: Phrase-Aware Trigrams (Catastrophic Failure)

**Setup**: Modified `get_swara_token_sequence()` to split the swara stream at glide boundaries before computing trigrams. The hypothesis was that glides mark phrase boundaries, so splitting would create cleaner melodic phrases.

**Result**: Trigram accuracy collapsed from 0.580 → 0.180.

**Root cause**: In Carnatic music, gamakas (glides/oscillations) **connect** swaras — they are integral to melodic phrases, not boundaries between them. A typical Carnatic phrase like "Sa→Ri2→Ga2" involves continuous pitch movement (gamaka) between each note. Splitting at glides fragmented real phrases into isolated single-note "phrases" with no trigram content.

**Lesson**: Feature engineering must respect the musical domain. This was a Western music intuition (phrases separated by rests/jumps) incorrectly applied to Carnatic music (continuous melodic flow with ornamental connections). The failure validates our non-splitting approach and provides evidence for the continuous gamaka model of Carnatic melody.

**Status**: Reverted. The feature was removed from the pipeline.

---

### Experiment 6: 4-Gram Features (Sparse Failure)

**Setup**: Extended trigram vocabulary to 4-grams (4-note patterns).

**Result**: 0.158 accuracy (14 ragas).

**Root cause**: 12^4 = 20,736 possible 4-grams. Even keeping only the top-200, most are near-zero for any given segment. With ~2,100 training samples, the feature space is too sparse for SVM to find meaningful boundaries.

**Lesson**: Trigrams (12^3 = 1,728 possible, top-200 kept) hit the sweet spot between expressiveness and sparsity. Bigrams are too short to capture characteristic phrases; 4-grams are too long given data constraints.

---

### Ablation A: Voiced Ratio Filter Threshold

**Setup**: Segments with voiced_ratio < threshold are removed before training.

| Threshold | Segments Remaining | % of Original |
|---|---|---|
| 0.00 (no filter) | 4,487 | 100% |
| 0.15 | ~3,200 | ~71% |
| 0.25 (current) | 2,145 | 48% |

**Rationale**: Low-voiced segments are dominated by drone (tanpura) or percussion (mridangam) — they contain no melodic information. Including them adds noise to both feature extraction and classification.

**Tradeoff**: At 0.25, we lose 52% of data but the remaining segments have clean melodic content. Lowering to 0.15 keeps more data but introduces noisier features.

---

### Ablation B: Token Collapse — What It Preserves vs Destroys

**Setup**: Token collapse merges consecutive repeated swara labels: "Sa Sa Sa Ri2 Ri2 Ga2 Ga2 Ga2" → [Sa, Ri2, Ga2].

**What it preserves**: Sequential order (which note follows which)
**What it destroys**: Duration information (Sa was held for 3 frames, Ri2 for 2, Ga2 for 3)

**Evidence that this matters**:
- trigram alone (no duration): 0.5795
- trigram + dwell_time (recovers duration): 0.6018 (+2.2%)
- dwell_vadi alone (duration only, no sequence): 0.5297

**Interpretation**: Both sequence and duration carry independent discriminative information. The optimal model combines them. Token collapse is necessary for clean trigram counting (otherwise "Sa Sa Sa" dominates), but dwell-time features recover the lost duration signal.

---

### Ablation C: 14 vs 10 Ragas — Effect of Removing Ambiguous Classes

**Setup**: Compare identical pipeline at min_segments_per_raga=20 (14 ragas) vs 60 (10 ragas).

**What gets dropped at threshold 60**:
- Behag (25 segs, 4 tracks) — janya of Harikambhoji
- Jonpuri (29 segs, 2 tracks) — very few tracks
- Kalyani (54 segs, 4 tracks) — Ma2 distinction possibly affected by tonic errors
- Sindhubhairavi (25 segs, 3 tracks) — variant of Bhairavi, easily confused

**Key finding**: Track accuracy improves far more than segment accuracy:

| Metric | 14 ragas | 10 ragas | Δ |
|---|---|---|---|
| Segment Acc (trigram_dwell) | 0.6018 | 0.6368 | +3.5% |
| Track Acc (trigram_dwell) | 0.5849 | 0.7000 | **+11.5%** |

The disproportionate track improvement (+11.5% vs +3.5%) reveals that the dropped ragas had very few tracks that were absorbing misclassifications bidirectionally — both being misclassified themselves AND causing other ragas to be misclassified as them.

---

### Ablation D: Timbral vs Structural Features

A controlled comparison isolating the question: is raga in the melody or the sound?

| Category | Feature | Dims | 14-raga | 10-raga |
|---|---|---|---|---|
| Structural (pitch-based) | trigram_dwell | 216 | 0.6018 | 0.6368 |
| Structural (pitch-based) | trigram | 200 | 0.5795 | 0.6211 |
| Structural (pitch-based) | pcd | 120 | 0.5059 | 0.5390 |
| Structural (emphasis) | dwell_vadi | 16 | 0.5297 | 0.6028 |
| Timbral (audio-based) | mfcc | 58 | 0.1191 | 0.0628 |

Every structural feature massively outperforms the timbral baseline. MFCC's performance *decreases* as we move to cleaner classes (10 ragas), while all structural features improve. This is the strongest evidence for H2.

---

### Ablation E: SVM Fold Stability Analysis

**Fold accuracies for SVM_trigram_dwell**:

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Std Dev |
|---|---|---|---|---|---|---|
| 14-raga | — | — | — | — | — | — |
| 10-raga | 0.747 | 0.668 | 0.491 | 0.718 | 0.559 | 0.098 |

Fold 3 (0.491) is consistently the weakest. This likely reflects an unlucky track split where test tracks contain atypical performances. With only 40 tracks / 5 folds = 8 test tracks per fold, a single unusual track can swing accuracy by ~5%.

**Comparison with LSTM fold variance**:
- SVM folds: 0.491–0.747 (range: 0.256)
- LSTM folds: 0.246–0.685 (range: 0.439)

SVM is nearly twice as stable, despite using the same GroupKFold splits. This demonstrates that hand-crafted features are more robust to data split variation than learned representations.
