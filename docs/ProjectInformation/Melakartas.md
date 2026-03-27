# Melakarta System

The melakarta system is the taxonomic hierarchy that organises all Carnatic ragas into 72 parent scales. This document explains how the 72 melakartas arise from combinatorial enumeration of swara variants, maps the melakartas present in this project's dataset (SARAGA Carnatic), and describes the parent-child (melakarta-janya) relationships that predict where the classifier should struggle. The melakarta structure directly underpins hypothesis H4 (confusion follows musicological kinship) and the hierarchical classification experiment. The combinatorial formula is: 6 Ri/Ga pairings × 6 Da/Ni pairings × 2 Ma variants = 72 melakartas.

Every raga in Carnatic music is either a melakarta (parent) or a janya (derived from a parent).

## Structure

A melakarta raga must use all 7 scale degrees (sampoorna) in both ascent and descent. The 72 scales arise from the combination of:

- Sa and Pa are fixed (present in all ragas)
- 6 variants of Ri/Ga (Ri and Ga share interval space, giving 6 valid pairings)
- 6 variants of Da/Ni (same logic)
- 2 variants of Ma (Ma1 = shuddha, Ma2 = prati)

6 × 6 × 2 = 72 melakartas.

They're numbered 1–72 and split into two halves:
- 1–36 use Ma1 (shuddha madhyama)
- 37–72 use Ma2 (prati madhyama)

Each group of 6 consecutive melakartas shares the same Ri/Ga pair but varies Da/Ni. These groups of 6 are called chakras — there are 12 chakras total.

## Melakartas in This Project's Dataset

| # | Melakarta | Ri/Ga | Ma | Da/Ni | Tracks | Notes |
|---|-----------|-------|----|-------|--------|-------|
| 8 | Hanumatodi (Todi) | Ri1/Ga1 | Ma1 | Da1/Ni1 | 6 | Very distinctive. Ri1-Ga1 oscillation is rare. |
| 20 | Kharaharapriya (Bhairavi) | Ri2/Ga1 | Ma1 | Da2/Ni1 | ~8 | Most common raga in dataset. Heavy gamaka usage. |
| 22 | Kharaharapriya | Ri2/Ga1 | Ma1 | Da2/Ni1 | — | Parent of Bhairavi. Bhairavi is often described as a janya despite being sampoorna in practice. |
| 28 | Harikambhoji | Ri2/Ga2 | Ma1 | Da2/Ni2 | 3 | Parent of Kambhoji, Kamas, Behag. Hardest to classify (0% F1 in 13-raga run). |
| 56 | Sanmukhapriya | Ri2/Ga1 | Ma2 | Da1/Ni1 | 4 | Ma2 usage makes it distinctive. |

Note: Bhairavi's melakarta classification is debated. It's technically janya of Natabhairavi (20th) but in practice uses all 12 swaras freely. For this project it's treated as its own class.

## Janya Derivation

A janya raga is derived from a melakarta by:
- Dropping swaras (varja — e.g. Mohanam drops Ga and Ni → pentatonic)
- Using different swaras in ascent vs descent (vakra)
- Both

The parent-child relationship matters for classification because janyas of the same melakarta share a note pool. If two ragas use the same notes, distinguishing them requires detecting phrase patterns and emphasis — not just which notes appear.

## Melakarta-Janya Relationships in the Dataset

**Harikambhoji family (28th melakarta) — the problem cluster:**
- Harikambhoji (parent): Sa Ri2 Ga2 Ma1 Pa Da2 Ni2 — uses full scale
- Kambhoji (janya): drops Ni2 in descent, emphasises Da2
- Kamas (janya): uses Ni2 sparingly, emphasis on Ga2-Ma1
- Behag (janya): vakra raga, zigzag patterns in ascent

These four share essentially the same note material. The parent (Harikambhoji) has no distinctive phrase because its children "own" the characteristic patterns. This is why our classifier scores 0.13 F1 on Harikambhoji.

**Todi family (8th melakarta):**
- Todi: Sa Ri1 Ga1 Ma1 Pa Da1 Ni1
- No other ragas from this family appear in the dataset
- Ri1-Ga1 combination is unique in the dataset → clean separation → 0.86 F1

**Kharaharapriya family (22nd melakarta):**
- Bhairavi uses notes from multiple melakartas in practice (Ri1, Ri2, Ga1, Ga2 all appear)
- Sindhubhairavi is a Bhairavi variant — these two are often confused

## Why This Matters for the Project

The melakarta system predicts where the classifier should struggle:
- Same family = shared notes = hard to separate with pitch distributions alone
- Different families = different note pools = easy to separate

If our confusion matrix shows that misclassifications cluster within melakarta families (H4), it confirms the model has learned genuine musical structure rather than arbitrary patterns.
