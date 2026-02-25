"""
05_feature_extraction.py — Stage 5: Feature Extraction (v2)

Extracts FOUR feature representations:
  1. Transition matrix (12×12 = 144 features)
  2. N-gram counts (bigram + top-200 trigrams)
  3. Pitch Class Distribution (PCD) — 120-bin histogram
  4. MFCCs + chroma + spectral features (standard audio ML baseline)

Usage:
    python scripts/05_feature_extraction.py
    python scripts/05_feature_extraction.py --plot-n 5
"""
import argparse
from pathlib import Path
from itertools import product as iter_product

import numpy as np
import pandas as pd
import yaml
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Swara definitions ────────────────────────────────────────────

SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
SWARA_TO_IDX = {s: i for i, s in enumerate(SWARA_NAMES)}
N_SWARAS = len(SWARA_NAMES)


# ── Swara-based feature functions ────────────────────────────────

def get_swara_token_sequence(swara_labels: np.ndarray) -> list:
    tokens = []
    prev = None
    for label in swara_labels:
        if label in ("silence", "glide"):
            prev = None
            continue
        if label != prev:
            tokens.append(label)
            prev = label
    return tokens


def compute_transition_matrix(tokens: list) -> np.ndarray:
    mat = np.zeros((N_SWARAS, N_SWARAS), dtype=np.float64)
    for i in range(len(tokens) - 1):
        src = SWARA_TO_IDX.get(tokens[i])
        dst = SWARA_TO_IDX.get(tokens[i + 1])
        if src is not None and dst is not None:
            mat[src, dst] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat /= row_sums
    return mat


def compute_directional_transition_matrices(tokens: list) -> tuple:
    """
    Split transitions by melodic direction (swara index order).

    Arohana  (ascending):  dst_idx > src_idx
    Avarohana (descending): dst_idx < src_idx
    Equal-index repeats are excluded from both matrices.

    Returns row-normalised (12x12) arohana and avarohana matrices.
    """
    arohana = np.zeros((N_SWARAS, N_SWARAS), dtype=np.float64)
    avarohana = np.zeros((N_SWARAS, N_SWARAS), dtype=np.float64)

    for i in range(len(tokens) - 1):
        src = SWARA_TO_IDX.get(tokens[i])
        dst = SWARA_TO_IDX.get(tokens[i + 1])
        if src is None or dst is None:
            continue
        if dst > src:
            arohana[src, dst] += 1
        elif dst < src:
            avarohana[src, dst] += 1

    for mat in (arohana, avarohana):
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat /= row_sums

    return arohana, avarohana


def compute_ngrams(tokens: list, n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        if all(t in SWARA_TO_IDX for t in gram):
            counts[gram] = counts.get(gram, 0) + 1
    total = sum(counts.values())
    if total > 0:
        counts = {k: v / total for k, v in counts.items()}
    return counts


def ngrams_to_fixed_vector(ngram_counts: dict, n: int) -> np.ndarray:
    if n == 2:
        vec = np.zeros(N_SWARAS * N_SWARAS, dtype=np.float64)
        for (s1, s2), count in ngram_counts.items():
            i = SWARA_TO_IDX.get(s1)
            j = SWARA_TO_IDX.get(s2)
            if i is not None and j is not None:
                vec[i * N_SWARAS + j] = count
        return vec
    else:
        return ngram_counts


def compute_pcd(cents_mod: np.ndarray, bins_per_octave: int = 120) -> np.ndarray:
    valid = cents_mod[np.isfinite(cents_mod)]
    if valid.size == 0:
        return np.zeros(bins_per_octave, dtype=np.float64)
    edges = np.linspace(0.0, 1200.0, bins_per_octave + 1)
    hist, _ = np.histogram(valid, bins=edges)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ── Auxiliary feature functions (gamaka + sequence richness) ─────

def compute_auxiliary_features(swara_labels: np.ndarray,
                                distances: np.ndarray,
                                cents_mod: np.ndarray,
                                tokens: list,
                                segment_duration_s: float = 45.0) -> np.ndarray:
    """
    Compute auxiliary features capturing ornament/gamaka information
    and sequence richness — things the transition matrix loses.

    Returns 10 features:
      [0] glide_ratio          — fraction of voiced frames labelled "glide"
      [1] silence_ratio        — fraction of all frames labelled "silence"
      [2] mean_dist_to_bin     — avg cents distance to nearest swara (voiced)
      [3] std_dist_to_bin      — std of cents distance (ornament spread)
      [4] pct_within_tolerance — fraction of voiced frames within 40c of a bin
      [5] n_tokens             — number of collapsed swara tokens
      [6] tokens_per_second    — token rate (melodic activity)
      [7] unique_swaras        — count of distinct swaras used
      [8] transition_entropy   — entropy of transition matrix (complexity)
      [9] swara_entropy        — entropy of swara usage distribution
    """
    total_frames = len(swara_labels)

    # Glide and silence ratios
    n_glide = int(np.sum(swara_labels == "glide"))
    n_silence = int(np.sum(swara_labels == "silence"))
    n_voiced = total_frames - n_silence

    glide_ratio = float(n_glide / n_voiced) if n_voiced > 0 else 0.0
    silence_ratio = float(n_silence / total_frames) if total_frames > 0 else 0.0

    # Distance to nearest swara bin
    valid_dist = distances[np.isfinite(distances)]
    mean_dist = float(np.mean(valid_dist)) if valid_dist.size > 0 else 0.0
    std_dist = float(np.std(valid_dist)) if valid_dist.size > 0 else 0.0
    pct_within = float(np.mean(valid_dist <= 40.0)) if valid_dist.size > 0 else 0.0

    # Token-based features
    n_tokens = len(tokens)
    tokens_per_sec = n_tokens / segment_duration_s if segment_duration_s > 0 else 0.0
    unique_swaras = len(set(t for t in tokens if t in SWARA_TO_IDX))

    # Transition entropy
    trans_mat = compute_transition_matrix(tokens)
    flat = trans_mat.flatten()
    flat = flat[flat > 0]
    transition_entropy = float(-np.sum(flat * np.log2(flat))) if flat.size > 0 else 0.0

    # Swara usage entropy
    if n_tokens > 0:
        swara_counts = {}
        for t in tokens:
            if t in SWARA_TO_IDX:
                swara_counts[t] = swara_counts.get(t, 0) + 1
        total_t = sum(swara_counts.values())
        if total_t > 0:
            probs = np.array(list(swara_counts.values()), dtype=np.float64) / total_t
            swara_entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            swara_entropy = 0.0
    else:
        swara_entropy = 0.0

    return np.array([
        glide_ratio, silence_ratio, mean_dist, std_dist, pct_within,
        n_tokens, tokens_per_sec, unique_swaras, transition_entropy, swara_entropy,
    ], dtype=np.float64)


# ── MFCC / Audio feature functions ──────────────────────────────

def extract_audio_features(audio_path: str, sr: int = 22050,
                            n_mfcc: int = 20) -> np.ndarray:
    """
    Extract standard audio ML features from a segment:
      - MFCCs (mean + std of each coefficient)
      - Chroma (mean of 12 chroma bins)
      - Spectral centroid (mean + std)
      - Spectral rolloff (mean + std)
      - Zero crossing rate (mean + std)

    Returns a fixed-length feature vector.
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # MFCCs: n_mfcc coefficients, take mean and std over time
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfccs.mean(axis=1)       # (n_mfcc,)
    mfcc_std = mfccs.std(axis=1)          # (n_mfcc,)

    # Chroma: 12 bins, mean over time
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)     # (12,)

    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sc_mean = np.array([spec_cent.mean()])
    sc_std = np.array([spec_cent.std()])

    # Spectral rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sr_mean = np.array([spec_roll.mean()])
    sr_std = np.array([spec_roll.std()])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.array([zcr.mean()])
    zcr_std = np.array([zcr.std()])

    # Concatenate all: 20+20+12+2+2+2 = 58 features (with n_mfcc=20)
    feature_vec = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean,
        sc_mean, sc_std, sr_mean, sr_std,
        zcr_mean, zcr_std,
    ])

    return feature_vec


# ── Visualization ────────────────────────────────────────────────

def plot_transition_matrix(mat, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(mat, xticklabels=SWARA_NAMES, yticklabels=SWARA_NAMES,
                cmap="YlOrRd", annot=False, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel("To swara")
    ax.set_ylabel("From swara")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pcd(pcd, title, out_path):
    bins = np.linspace(0, 1200, len(pcd), endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bins, pcd, width=10, color="steelblue", alpha=0.8)
    for sname, sval in zip(SWARA_NAMES, range(0, 1200, 100)):
        ax.axvline(sval, color="red", linewidth=0.5, alpha=0.5)
        ax.text(sval, ax.get_ylim()[1] * 0.95, sname, fontsize=6,
                ha="center", color="red")
    ax.set_xlabel("Cents (mod 1200)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Extract features from swara traces + audio")
    parser.add_argument("--plot-n", type=int, default=5)
    args = parser.parse_args()

    root = repo_root()

    # --- Input: swara traces
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    if not swara_csv.exists():
        raise FileNotFoundError(f"Missing: {swara_csv}. Run 04_swara_mapping.py first.")
    df = pd.read_csv(swara_csv)

    # Also need segments CSV for audio paths (for MFCC extraction)
    seg_csv = root / "data" / "processed" / "segments_saraga_carnatic.csv"
    seg_df = pd.read_csv(seg_csv)

    # Merge audio paths and duration into df
    df = df.merge(
        seg_df[["track_id", "segment_index", "path", "duration_s"]].rename(
            columns={"path": "audio_path"}),
        on=["track_id", "segment_index"],
        how="left"
    )

    print(f"Loaded {len(df)} segments")

    # --- Output dirs
    feat_dir = root / "data" / "processed" / "features"
    fig_dir = root / "outputs" / "figures" / "features"
    ensure_dir(feat_dir)
    ensure_dir(fig_dir)

    # --- Extract all features
    all_transition_vecs = []
    all_bigram_vecs = []
    all_pcd_vecs = []
    all_trigram_dicts = []
    all_mfcc_vecs = []
    all_auxiliary_vecs = []
    all_arohana_vecs = []
    all_avarohana_vecs = []
    meta_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
        # -- Swara-based features --
        swara_npz_path = root / row["swara_npz"]
        sdata = np.load(swara_npz_path, allow_pickle=True)
        swara_labels = sdata["swara_labels"]
        cents_mod = sdata["cents_mod"]
        distances = sdata["distances"]

        tokens = get_swara_token_sequence(swara_labels)

        # Transition matrix
        trans_mat = compute_transition_matrix(tokens)
        all_transition_vecs.append(trans_mat.flatten())

        # Directional transition matrices (arohana / avarohana split)
        aro_mat, ava_mat = compute_directional_transition_matrices(tokens)
        all_arohana_vecs.append(aro_mat.flatten())
        all_avarohana_vecs.append(ava_mat.flatten())

        # Bigrams
        bigram_counts = compute_ngrams(tokens, 2)
        all_bigram_vecs.append(ngrams_to_fixed_vector(bigram_counts, 2))

        # Trigrams (dict for now)
        all_trigram_dicts.append(compute_ngrams(tokens, 3))

        # PCD
        all_pcd_vecs.append(compute_pcd(cents_mod, bins_per_octave=120))

        # -- Auxiliary features (gamaka + sequence richness) --
        seg_dur = float(row.get("duration_s", 45.0))
        aux_vec = compute_auxiliary_features(
            swara_labels, distances, cents_mod, tokens,
            segment_duration_s=seg_dur
        )
        all_auxiliary_vecs.append(aux_vec)

        # -- Audio-based features (MFCC + chroma + spectral) --
        audio_path = str(root / row["audio_path"])
        mfcc_vec = extract_audio_features(audio_path, sr=22050, n_mfcc=20)
        all_mfcc_vecs.append(mfcc_vec)

        meta_rows.append({
            "track_id": row["track_id"],
            "raga": row["raga"],
            "segment_index": row["segment_index"],
            "n_tokens": len(tokens),
        })

    # --- Build trigram vectors (shared vocabulary)
    print("Building trigram vocabulary...")
    global_trigrams = {}
    for td in all_trigram_dicts:
        for gram, count in td.items():
            global_trigrams[gram] = global_trigrams.get(gram, 0) + count

    TOP_K_TRIGRAMS = 200
    top_trigrams = sorted(global_trigrams.items(), key=lambda x: -x[1])[:TOP_K_TRIGRAMS]
    trigram_vocab = [gram for gram, _ in top_trigrams]
    trigram_to_idx = {gram: i for i, gram in enumerate(trigram_vocab)}

    all_trigram_vecs = []
    for td in all_trigram_dicts:
        vec = np.zeros(TOP_K_TRIGRAMS, dtype=np.float64)
        for gram, count in td.items():
            if gram in trigram_to_idx:
                vec[trigram_to_idx[gram]] = count
        all_trigram_vecs.append(vec)

    # --- Assemble matrices
    meta_df = pd.DataFrame(meta_rows)

    transition_mat = np.array(all_transition_vecs)
    bigram_mat = np.array(all_bigram_vecs)
    trigram_mat = np.array(all_trigram_vecs)
    pcd_mat = np.array(all_pcd_vecs)
    mfcc_mat = np.array(all_mfcc_vecs)
    auxiliary_mat = np.array(all_auxiliary_vecs)
    arohana_mat = np.array(all_arohana_vecs)
    avarohana_mat = np.array(all_avarohana_vecs)
    directional_mat = np.hstack([arohana_mat, avarohana_mat])

    # Combined feature sets
    transition_aux = np.hstack([transition_mat, auxiliary_mat])
    combined_interp = np.hstack([transition_mat, pcd_mat])
    combined_interp_aux = np.hstack([transition_mat, pcd_mat, auxiliary_mat])
    combined_mfcc_pcd = np.hstack([mfcc_mat, pcd_mat])
    full_combined = np.hstack([transition_mat, pcd_mat, mfcc_mat, auxiliary_mat])
    directional_pcd = np.hstack([directional_mat, pcd_mat])
    directional_aux = np.hstack([directional_mat, auxiliary_mat])
    directional_full = np.hstack([directional_mat, pcd_mat, auxiliary_mat])

    print(f"\nFeature dimensions:")
    print(f"  Transition:            {transition_mat.shape[1]}")
    print(f"  Arohana transition:    {arohana_mat.shape[1]}")
    print(f"  Avarohana transition:  {avarohana_mat.shape[1]}")
    print(f"  Directional (aro+ava): {directional_mat.shape[1]}")
    print(f"  Bigram:                {bigram_mat.shape[1]}")
    print(f"  Trigram (top-{TOP_K_TRIGRAMS}):     {trigram_mat.shape[1]}")
    print(f"  PCD:                   {pcd_mat.shape[1]}")
    print(f"  MFCC+audio:            {mfcc_mat.shape[1]}")
    print(f"  Auxiliary (gamaka+seq): {auxiliary_mat.shape[1]}")
    print(f"  Transition+aux:        {transition_aux.shape[1]}")
    print(f"  Combined (interp):     {combined_interp.shape[1]}")
    print(f"  Combined (interp+aux): {combined_interp_aux.shape[1]}")
    print(f"  Combined (MFCC+PCD):   {combined_mfcc_pcd.shape[1]}")
    print(f"  Full combined:         {full_combined.shape[1]}")
    print(f"  Directional+PCD:       {directional_pcd.shape[1]}")
    print(f"  Directional+aux:       {directional_aux.shape[1]}")
    print(f"  Directional+PCD+aux:   {directional_full.shape[1]}")

    # --- Save
    np.savez_compressed(
        feat_dir / "features_all.npz",
        transition=transition_mat,
        arohana_transition=arohana_mat,
        avarohana_transition=avarohana_mat,
        directional_transition=directional_mat,
        bigram=bigram_mat,
        trigram=trigram_mat,
        pcd=pcd_mat,
        mfcc=mfcc_mat,
        auxiliary=auxiliary_mat,
        transition_aux=transition_aux,
        combined=combined_interp,
        combined_interp_aux=combined_interp_aux,
        combined_mfcc_pcd=combined_mfcc_pcd,
        full_combined=full_combined,
        directional_pcd=directional_pcd,
        directional_aux=directional_aux,
        directional_full=directional_full,
        track_ids=meta_df["track_id"].values,
        ragas=meta_df["raga"].values,
        segment_indices=meta_df["segment_index"].values,
    )

    meta_df.to_csv(feat_dir / "features_meta.csv", index=False)

    with open(feat_dir / "trigram_vocab.txt", "w", encoding="utf-8") as f:
        for gram in trigram_vocab:
            f.write("→".join(gram) + "\n")

    print(f"\nSaved to: {feat_dir}")

    # --- Per-raga plots
    ragas_to_plot = meta_df["raga"].unique()[:args.plot_n]
    for raga in ragas_to_plot:
        mask = meta_df["raga"].values == raga
        if mask.sum() == 0:
            continue
        avg_trans = transition_mat[mask].mean(axis=0).reshape(N_SWARAS, N_SWARAS)
        plot_transition_matrix(avg_trans, f"Avg Transition — {raga}",
                               fig_dir / f"transition_{raga.replace(' ', '_')}.png")
        avg_aro = arohana_mat[mask].mean(axis=0).reshape(N_SWARAS, N_SWARAS)
        plot_transition_matrix(avg_aro, f"Avg Arohana — {raga}",
                               fig_dir / f"arohana_{raga.replace(' ', '_')}.png")
        avg_ava = avarohana_mat[mask].mean(axis=0).reshape(N_SWARAS, N_SWARAS)
        plot_transition_matrix(avg_ava, f"Avg Avarohana — {raga}",
                               fig_dir / f"avarohana_{raga.replace(' ', '_')}.png")
        avg_pcd = pcd_mat[mask].mean(axis=0)
        plot_pcd(avg_pcd, f"Avg PCD — {raga}",
                 fig_dir / f"pcd_{raga.replace(' ', '_')}.png")

    # --- Summary
    print(f"\nDone. {len(meta_df)} segments.")
    print(f"Tokens per segment — mean: {meta_df['n_tokens'].mean():.1f}")
    print(f"\nAuxiliary feature means:")
    aux_names = ["glide_ratio", "silence_ratio", "mean_dist_to_bin",
                 "std_dist_to_bin", "pct_within_tol", "n_tokens",
                 "tokens_per_sec", "unique_swaras", "trans_entropy",
                 "swara_entropy"]
    for j, name in enumerate(aux_names):
        print(f"  {name:25s}: {auxiliary_mat[:, j].mean():.4f}")
    print(f"\nSegments per raga:")
    print(meta_df["raga"].value_counts().to_string())


if __name__ == "__main__":
    main()