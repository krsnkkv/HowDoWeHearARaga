# Swaras (Notes) in Carnatic Music

This document defines the 12-swara system used in Carnatic music, explains how the pipeline converts continuous pitch (Hz) to discrete swara labels using logarithmic cent-based distance, details the tolerance and minimum-duration parameters that govern the mapping, and describes the gamaka (ornament) and vadi/samvadi (note hierarchy) concepts that underpin the feature extraction stage. Every mathematical operation in the pipeline — cents conversion, octave folding, circular distance, and the tolerance threshold — is derived from the cognitive and musicological principles described here.

## Mathematical Foundations

### The Cents Scale

The cent is a logarithmic unit of pitch interval. One octave = 1200 cents. The conversion from Hz to cents relative to a reference frequency (the tonic, Sa) is:

```
cents = 1200 × log₂(F0 / Sa_Hz)
```

For example, if Sa = 150 Hz and the current pitch is 225 Hz:
```
cents = 1200 × log₂(225 / 150) = 1200 × log₂(1.5) = 1200 × 0.585 = 702 cents
```
This is Pa (the perfect fifth). The logarithmic scale is essential because human pitch perception is logarithmic — equal Hz differences sound different depending on register, but equal cent differences sound the same at any register.

### Octave Folding

All pitches are folded into a single octave using modular arithmetic:
```
cents_mod = cents % 1200
```
This implements octave equivalence: Sa at 150 Hz and Sa at 300 Hz both map to 0 cents. Octave equivalence is a fundamental property of human pitch perception (Burns & Ward, 1978).

### Circular Distance

Swara bins are arranged on a circle (0–1200 cents wrapping around). The distance from a pitch to the nearest bin uses circular distance:
```
d = min(|cents_mod - bin|, 1200 - |cents_mod - bin|)
```
This ensures that a pitch at 1190 cents is correctly identified as 10 cents from Sa (0/1200), not 1190 cents away.

### Harmonic Scoring (Tonic Estimation)

The tonic scoring function used in Stage 3:
```
score(candidate) = mass(Sa, 0 cents) + 0.5 × mass(Pa, 702 cents)
```
where mass(note, target) is the sum of the pitch-class histogram within ±30 cents of the target. The 0.5 weight on Pa reflects the tanpura drone structure: Sa is the primary tone, Pa is the secondary reinforcement.

## The 12 Swaras

Carnatic music uses 12 swara positions within one octave, measured in cents relative to Sa (the tonic). The system is roughly equivalent to the 12 chromatic notes in Western music, but with different tuning and different musical function.

| Swara | Cents | Western Approx. | Category | Notes |
|-------|-------|-----------------|----------|-------|
| Sa | 0 | C (tonic) | Fixed | Always present. Established by drone. |
| Ri1 | 100 | Db | Shuddha Rishabham | Used in Todi, Saveri. Rare in dataset. |
| Ri2 | 200 | D | Chatushruti Rishabham | Most common Ri. Used in Bhairavi, Harikambhoji family. |
| Ga1 | 300 | Eb | Sadharana Gandharam | Used in Todi, Bhairavi, Sanmukhapriya. |
| Ga2 | 400 | E | Antara Gandharam | Used in Harikambhoji family, Kalyani, Begada. |
| Ma1 | 500 | F | Shuddha Madhyamam | Present in most ragas in the dataset. |
| Ma2 | 600 | F# | Prati Madhyamam | Only Sanmukhapriya and Kalyani use this. Distinctive. |
| Pa | 700 | G | Panchamam | Fixed. Always present (reinforced by drone). |
| Da1 | 800 | Ab | Shuddha Dhaivatam | Used in Todi, Saveri, Sanmukhapriya. |
| Da2 | 900 | A | Chatushruti Dhaivatam | Common. Bhairavi, Harikambhoji family. |
| Ni1 | 1000 | Bb | Kaisiki Nishadam | Used in Todi, Bhairavi, Ritigaula. |
| Ni2 | 1100 | B | Kakali Nishadam | Used in Harikambhoji family, Kalyani. |

## Important Properties

**Sa and Pa are fixed.** Every raga includes them, and the tanpura drone reinforces both. Sa is at 0 cents, Pa is at 702 cents (just fifth — slightly different from the 700-cent equal temperament approximation). Our tonic estimation (Stage 3) uses both Sa and Pa mass for scoring.

**Ri/Ga share interval space.** Ri2 (200 cents) and Ga1 (300 cents) are only 100 cents apart. In some ragas they coexist; in others, only one is used. The specific Ri/Ga pairing defines the raga's character more than almost any other factor.

**Same for Da/Ni.** Da2 (900) and Ni1 (1000) are 100 cents apart. Similar logic applies.

**Enharmonic pairs:** In some theoretical formulations, Ri2 (200 cents) and Ga1 (300 cents) can overlap when ragas use non-standard tuning, but in our equal-tempered pipeline they are treated as separate bins 100 cents apart. Their musical function differs depending on the raga context.

## Swara Mapping in the Pipeline (Stage 4)

Our pipeline converts continuous pitch (F0 in Hz) to discrete swara labels:

1. Convert F0 to cents relative to Sa: `cents = 1200 × log2(F0 / Sa_Hz)`
2. Fold into one octave: `cents_mod = cents % 1200`
3. Find nearest swara bin (0, 100, 200, ..., 1100)
4. If distance to nearest bin ≤ 40 cents → assign that swara
5. If distance > 40 cents → label as "glide"
6. If no valid pitch → label as "silence"

**The 40-cent tolerance** is the key parameter. Each bin is 100 cents wide, so 40 cents means a pitch must be within 40% of the bin centre. This leaves a 20-cent "gap" between adjacent bins where pitches are labelled as glides. The tolerance models categorical pitch perception — the perceptual boundaries between notes are fuzzy, not sharp.

**Minimum duration filter (100ms):** After initial mapping, any swara label lasting less than 100ms is relabelled as "glide." This prevents rapid flickering from gamaka oscillations being misinterpreted as separate note events. 100ms roughly corresponds to the temporal integration window of human pitch perception.

## Gamakas (Ornaments)

Gamakas are oscillatory pitch movements that are central to Carnatic music. Unlike Western vibrato (which ornaments a note), Carnatic gamakas often span the full distance between two swaras. For example, a gamaka on Ri2 in Bhairavi might oscillate between 150 and 250 cents — covering the full range from Ri1 to midway between Ri2 and Ga1.

**Impact on the pipeline:**
- ~28% of voiced frames in our dataset are labelled "glide" — this is the gamaka contribution
- Gamakas are not noise or error — they carry raga identity (see: phrase-aware trigram failure)
- The token collapse operation in Stage 5 strips sustained notes but preserves transitions through gamakas
- Dwell-time features (Stage 5) capture how long a note is sustained between gamakas

**Types of gamakas relevant to our features:**
- Kampita: oscillation around a swara (creates repeated same-swara frames → longer dwell)
- Jaru: slide from one swara to another (creates glide frames between notes)
- Nokku: quick touch on adjacent swara (creates brief frames → filtered by 100ms threshold)

## Vadi and Samvadi

The vadi is the most prominent swara in a raga — the note that gets the most emphasis through frequency of appearance, duration of sustain, and placement at phrase endpoints. The samvadi is second most prominent, typically a fourth or fifth away from the vadi.

**How we compute it (Stage 5):**
- Count total frame-time per swara from the raw (pre-collapse) swara labels
- Vadi = swara with highest total frame count
- Samvadi = swara with second highest
- We store vadi_idx, vadi_fraction, samvadi_idx, samvadi_fraction (4 features)

This is an "operational" definition — it measures emphasis through duration rather than through musicological analysis. But for most ragas, the operationally measured vadi aligns with the theoretically defined one, because performers naturally emphasise the vadi.

## Swaras as Features

Our feature extraction uses swaras in several ways:

| Feature | How swaras are used | Dimension |
|---------|-------------------|-----------|
| Transition matrix | P(swara_j \| swara_i) for all 12×12 pairs | 144 |
| Trigrams | Count of top-200 three-swara sequences | 200 |
| PCD | Continuous pitch distribution in 120 bins (10-cent resolution) | 120 |
| Dwell-time | Mean sustain duration per swara | 12 |
| Vadi | Most/second-most prominent swara + their fractions | 4 |
| Directional | Transition matrix split by ascending/descending | 288 |

The swara vocabulary (12 symbols + glide + silence = 14 tokens) is small enough that trigrams (12³ = 1,728 possible) are tractable, unlike raw pitch which would have infinite vocabulary.
