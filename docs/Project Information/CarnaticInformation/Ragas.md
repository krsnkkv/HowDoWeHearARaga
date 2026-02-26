# Ragas in the Dataset

Reference for the 14 ragas retained after filtering (≥20 segments, ≥3 tracks).

## Quick Reference

| Raga | Melakarta | Type | Tracks | Segments | F1 (14-raga) | Tier |
|------|-----------|------|--------|----------|-------------|------|
| Bhairavi | 20/Natabhairavi | Janya* | ~8 | 366 | 0.89 | 1 |
| Todi | 8/Hanumatodi | Melakarta | 6 | 248 | 0.86 | 1 |
| Sanmukhapriya | 56 | Melakarta | 4 | 303 | 0.84 | 1 |
| Ritigaula | 20/Natabhairavi | Janya | 3 | 135 | 0.80 | 1 |
| Saveri | 15/Mayamalavagowla | Janya | 3 | 77 | 0.61 | 2 |
| Begada | 29/Shankarabharanam | Janya | 4 | 174 | 0.52 | 2 |
| Mohanam | 28/Harikambhoji | Janya | 4 | 170 | 0.38 | 2 |
| Kambhoji | 28/Harikambhoji | Janya | 5 | 258 | 0.37 | 2 |
| Kamas | 28/Harikambhoji | Janya | 7 | 114 | 0.36 | 2 |
| Harikambhoji | 28 | Melakarta | 3 | 129 | 0.13 | 3 |
| Kalyani | 65 | Melakarta | 4 | 54 | — | Dropped at 10 |
| Behag | 28/Harikambhoji | Janya | 4 | 25 | — | Dropped at 10 |
| Jonpuri | — | Hindustani origin | 2 | 29 | — | Dropped at 10 |
| Sindhubhairavi | 20/Natabhairavi | Janya | 3 | 25 | — | Dropped at 10 |

*Bhairavi is technically a janya but uses all 12 swaras in practice.

## Detailed Raga Profiles

### Bhairavi — F1: 0.89

**Scale (approximate):** Sa Ri2 Ga1 Ma1 Pa Da2 Ni1 Sa (but freely uses Ri1, Ga2, Da1, Ni2 as ornamental swaras)

**Character:** Considered the "queen of ragas." Extremely expressive, heavy gamaka on Ri2 and Da2. Often performed as the closing piece of a concert. Known for its emotional depth (karuna rasa — pathos).

**Why it classifies well:** The gamaka patterns on Ri2 and Da2 create distinctive trigram signatures. It's not closely related to any other raga in the dataset. Largest data support (366 segments) also helps.

**Vadi/Samvadi:** Ma1 and Sa are prominent. Da2 is heavily ornamented.

---

### Todi (Hanumatodi) — F1: 0.86

**Scale:** Sa Ri1 Ga1 Ma1 Pa Da1 Ni1 Sa

**Character:** A morning raga with a serious, contemplative mood. The Ri1-Ga1 oscillation (100–300 cents) is its defining characteristic. Strong ascending phrases starting from Sa.

**Why it classifies well:** Ri1 and Ga1 are rare in the dataset — only Todi uses them. The classifier effectively detects the presence of these swaras through trigrams. 6 tracks give reasonable coverage.

**Vadi/Samvadi:** Da1 (vadi), Ga1 (samvadi).

---

### Sanmukhapriya — F1: 0.84

**Scale:** Sa Ri2 Ga1 Ma2 Pa Da1 Ni1 Sa

**Character:** Uses prati madhyama (Ma2) which is relatively rare across the dataset's ragas. Creates a distinctive tonal colour. Sometimes described as having an eerie or intense quality.

**Why it classifies well:** Ma2 combined with Ga1 and Da1 produces unique trigram patterns. The scale combination doesn't overlap significantly with other ragas present.

---

### Ritigaula — F1: 0.80

**Scale:** Sa Ri2 Ga1 Ma1 Pa Da2 Ni1 Sa (vakra — uses zigzag patterns)

**Character:** A rare, serious raga. Its characteristic lies in specific phrase shapes rather than just the note set. Phrases typically involve characteristic movements around Ga1 and Ni1.

**Why it classifies well:** Despite only 3 tracks (135 segments), the phrase patterns are distinctive enough for trigrams to capture. The vakra (zigzag) movements create unusual trigram combinations.

---

### Saveri — F1: 0.61

**Scale:** Sa Ri1 Ma1 Pa Da1 Sa (ascent, no Ga or Ni) / Sa Ni1 Da1 Pa Ma1 Ga1 Ri1 Sa (descent uses Ga1)

**Character:** Asymmetric ascent/descent. The ascent skips Ga and Ni entirely. Morning raga with a devotional quality. Ri1 is shared with Todi, causing some confusion.

**Why it partially fails:** Shares Da1-Ni1 phrases with Todi in descent. The asymmetric scale should help (different ascending patterns), but our directional features don't capture this well enough (H6).

---

### Begada — F1: 0.52

**Scale:** Sa Ga2 Ri2 Ga2 Ma1 Pa Da2 Ni2 Sa (vakra — Ga2 appears before and after Ri2)

**Character:** A bright, optimistic raga. The vakra movement around Ga2-Ri2-Ga2 is its defining feature. Often performed in the evening.

**Why it partially fails:** Shares Ri2, Ga2, Da2, Ni2 with several other ragas (Harikambhoji family). The identity lies in the specific Ga2-Ri2-Ga2 zigzag, which trigrams can partially capture but not always distinguish from simpler Ri2-Ga2 movements in other ragas.

---

### Mohanam — F1: 0.38

**Scale:** Sa Ri2 Ga2 Pa Da2 Sa (pentatonic — only 5 notes)

**Character:** One of the most popular ragas, bright and pleasant. A pentatonic scale (audava) — drops Ma and Ni entirely. Often one of the first ragas taught to students.

**Why it struggles:** Every note in Mohanam also appears in Harikambhoji, Kambhoji, Kamas, Begada, and others. With only 5 notes, the trigram space is small. The classifier has to distinguish Mohanam purely by what's absent (no Ma, no Ni), but our features encode presence, not absence.

---

### Kambhoji — F1: 0.37

**Scale:** Sa Ri2 Ga2 Ma1 Pa Da2 Sa (ascent) / Sa Ni2 Da2 Pa Ma1 Ga2 Ri2 Sa (descent with Ni2)

**Character:** A major raga, considered the "king of ragas" by some. Ni2 is used sparingly and mainly in descent. The identity comes from how Da2 is emphasised and how Ni2 is introduced.

**Why it struggles:** Shares notes with parent Harikambhoji. Distinguished mainly by Ni2 usage patterns and Da2 emphasis — subtle features that need more data to learn reliably.

---

### Kamas — F1: 0.36

**Scale:** Sa Ri2 Ga2 Ma1 Pa Da2 Ni2 Sa (uses all notes of Harikambhoji)

**Character:** Lighter than Kambhoji, with Ga2-Ma1 emphasis. Often used for lighter compositions. 7 tracks in the dataset (most of any raga), yet still poor accuracy.

**Why it struggles:** Uses the full Harikambhoji scale. Identity lies in very subtle phrasing differences — the emphasis on Ga2-Ma1 movements and specific ornamental patterns. 7 tracks doesn't help when the signal is this subtle. More data ≠ better accuracy when the discriminative features are nuanced.

---

### Harikambhoji — F1: 0.13

**Scale:** Sa Ri2 Ga2 Ma1 Pa Da2 Ni2 Sa (all 7 notes)

**Character:** The 28th melakarta. Parent scale of many popular ragas. As a parent, it uses all notes without strongly favouring any particular phrases — its identity is essentially "the default" from which children deviate.

**Why it fails:** This is the most musically interesting failure. The parent melakarta has no distinctive phrase because its children "own" the characteristic patterns. Kambhoji owns the Da2 emphasis, Kamas owns the Ga2-Ma1 movement, Mohanam owns the pentatonic subset. Harikambhoji is what's left — the "average" of its children. Even human experts find Harikambhoji hard to identify in isolation.

---

## Ragas Dropped at 10-Raga Threshold

**Kalyani** (65th melakarta, 54 segments): Uses Ma2 like Sanmukhapriya but with Ga2, Ri2. Only 4 tracks. Identity depends on the bright Ma2-Pa movement. Dropped due to segment count.

**Behag** (janya of 28/Harikambhoji, 25 segments): Vakra raga with zigzag patterns in ascent. Only 4 tracks, very few segments. Part of the Harikambhoji problem cluster.

**Jonpuri** (Hindustani origin, 29 segments): Not originally a Carnatic raga — borrowed from Hindustani tradition. Only 2 tracks, making GroupKFold unreliable (1 track per fold at best).

**Sindhubhairavi** (variant of Bhairavi, 25 segments): Very close to Bhairavi. Uses similar swaras with slightly different emphasis. Easily confused with Bhairavi — only 3 tracks.

## Observations for the Dissertation

1. Ragas with unique note combinations classify well regardless of data quantity (Todi: 6 tracks, 0.86 F1)
2. Ragas with shared note pools struggle regardless of data quantity (Kamas: 7 tracks, 0.36 F1)
3. The melakarta parent (Harikambhoji) is harder to identify than its children — counterintuitive but musicologically sound
4. Pentatonic ragas (Mohanam) are hard because every note they use also appears in many other ragas
5. Ragas with heavy gamaka (Bhairavi) classify well because gamakas create distinctive trigram patterns
