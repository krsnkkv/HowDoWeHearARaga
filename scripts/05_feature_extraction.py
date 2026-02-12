"""
05_feature_extraction.py — Stage 5: Sequence & Distribution Features

Extracts three feature representations from swara traces:
  1. Transition matrix (12×12 = 144 features)
  2. N-gram counts (bigram, trigram, 4-gram)
  3. Pitch Class Distribution (PCD) — 120-bin histogram

Usage:
    python scripts/05_feature_extraction.py
"""
import argparse
from pathlib import Path
from itertools import product as iter_product

import numpy as np
import pandas as pd
import yaml
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


# ── Swara names (must match swara.yaml) ──────────────────────────

SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
SWARA_TO_IDX = {s: i for i, s in enumerate(SWARA_NAMES)}
N_SWARAS = len(SWARA_NAMES)


# ── Feature extraction functions ─────────────────────────────────

def get_swara_token_sequence(swara_labels: np.ndarray) -> list:
    """
    Convert frame-level swara labels into a token sequence.
    Collapses consecutive identical labels into single tokens.
    Removes 'silence' and 'glide' tokens for clean n-gram analysis.
    """
    tokens = []
    prev = None
    for label in swara_labels:
        if label in ("silence", "glide"):
            prev = None  # break continuity
            continue
        if label != prev:
            tokens.append(label)
            prev = label
    return tokens


def compute_transition_matrix(tokens: list) -> np.ndarray:
    """
    Compute a 12×12 swara transition matrix from a token sequence.
    Normalized per row (transition probabilities).
    """
    mat = np.zeros((N_SWARAS, N_SWARAS), dtype=np.float64)

    for i in range(len(tokens) - 1):
        src = SWARA_TO_IDX.get(tokens[i])
        dst = SWARA_TO_IDX.get(tokens[i + 1])
        if src is not None and dst is not None:
            mat[src, dst] += 1

    # Normalize rows to probabilities
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    mat /= row_sums

    return mat


def compute_ngrams(tokens: list, n: int) -> dict:
    """Count n-grams in the token sequence."""
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        # Only count if all tokens are valid swaras
        if all(t in SWARA_TO_IDX for t in gram):
            counts[gram] = counts.get(gram, 0) + 1

    # Normalize
    total = sum(counts.values())
    if total > 0:
        counts = {k: v / total for k, v in counts.items()}
    return counts


def ngrams_to_fixed_vector(ngram_counts: dict, n: int) -> np.ndarray:
    """
    Convert n-gram counts to a fixed-length vector.
    For bigrams: 12×12 = 144 features
    For trigrams and 4-grams: too sparse for full enumeration,
    so we keep top-K most common across the dataset.
    For bigrams we use full enumeration.
    """
    if n == 2:
        # Full 12×12 enumeration for bigrams
        vec = np.zeros(N_SWARAS * N_SWARAS, dtype=np.float64)
        for (s1, s2), count in ngram_counts.items():
            i = SWARA_TO_IDX.get(s1)
            j = SWARA_TO_IDX.get(s2)
            if i is not None and j is not None:
                vec[i * N_SWARAS + j] = count
        return vec
    else:
        # For higher-order n-grams, return the dict (we'll build a shared vocab later)
        return ngram_counts


def compute_pcd(cents_mod: np.ndarray, bins_per_octave: int = 120) -> np.ndarray:
    """
    Compute Pitch Class Distribution — histogram of cents values
    folded into one octave. This is the feature used in Gulati et al. 2016.
    """
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


def plot_transition_matrix(mat: np.ndarray, title: str, out_path: Path):
    """Plot a transition matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(mat, xticklabels=SWARA_NAMES, yticklabels=SWARA_NAMES,
                cmap="YlOrRd", annot=False, fmt=".2f", ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("To swara")
    ax.set_ylabel("From swara")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pcd(pcd: np.ndarray, title: str, out_path: Path):
    """Plot a Pitch Class Distribution."""
    bins = np.linspace(0, 1200, len(pcd), endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bins, pcd, width=10, color="steelblue", alpha=0.8)

    # Mark swara positions
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
    parser = argparse.ArgumentParser(description="Extract features from swara traces")
    parser.add_argument("--plot-n", type=int, default=5,
                        help="Number of per-raga aggregate plots to generate")
    args = parser.parse_args()

    root = repo_root()

    # --- Input
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    if not swara_csv.exists():
        raise FileNotFoundError(f"Missing: {swara_csv}. Run 04_swara_mapping.py first.")

    df = pd.read_csv(swara_csv)
    print(f"Loaded {len(df)} segments")

    # --- Output dirs
    feat_dir = root / "data" / "processed" / "features"
    fig_dir = root / "outputs" / "figures" / "features"
    ensure_dir(feat_dir)
    ensure_dir(fig_dir)

    # --- Extract features per segment
    all_transition_vecs = []
    all_bigram_vecs = []
    all_pcd_vecs = []
    all_trigram_dicts = []
    meta_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
        # Load swara trace
        swara_npz_path = root / row["swara_npz"]
        sdata = np.load(swara_npz_path, allow_pickle=True)
        swara_labels = sdata["swara_labels"]
        cents_mod = sdata["cents_mod"]

        # Token sequence (collapsed, no silence/glide)
        tokens = get_swara_token_sequence(swara_labels)

        # 1. Transition matrix → flatten to 144-dim vector
        trans_mat = compute_transition_matrix(tokens)
        trans_vec = trans_mat.flatten()
        all_transition_vecs.append(trans_vec)

        # 2. Bigram counts → 144-dim vector
        bigram_counts = compute_ngrams(tokens, 2)
        bigram_vec = ngrams_to_fixed_vector(bigram_counts, 2)
        all_bigram_vecs.append(bigram_vec)

        # 3. Trigram counts (as dict, we'll build shared vocab later)
        trigram_counts = compute_ngrams(tokens, 3)
        all_trigram_dicts.append(trigram_counts)

        # 4. PCD — 120-dim vector
        pcd = compute_pcd(cents_mod, bins_per_octave=120)
        all_pcd_vecs.append(pcd)

        meta_rows.append({
            "track_id": row["track_id"],
            "raga": row["raga"],
            "segment_index": row["segment_index"],
            "n_tokens": len(tokens),
        })

    # --- Build trigram feature vectors (shared top-K vocabulary)
    print("Building trigram vocabulary...")
    # Collect all trigrams across dataset
    global_trigrams = {}
    for td in all_trigram_dicts:
        for gram, count in td.items():
            if gram not in global_trigrams:
                global_trigrams[gram] = 0
            global_trigrams[gram] += count

    # Keep top 200 most frequent trigrams
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

    # --- Assemble feature matrices
    meta_df = pd.DataFrame(meta_rows)

    transition_mat = np.array(all_transition_vecs)
    bigram_mat = np.array(all_bigram_vecs)
    trigram_mat = np.array(all_trigram_vecs)
    pcd_mat = np.array(all_pcd_vecs)

    # Combined: transition + PCD
    combined_mat = np.hstack([transition_mat, pcd_mat])

    # Full combined: transition + bigram + trigram + PCD
    full_combined_mat = np.hstack([transition_mat, bigram_mat, trigram_mat, pcd_mat])

    print(f"\nFeature dimensions:")
    print(f"  Transition matrix:  {transition_mat.shape[1]}")
    print(f"  Bigram:             {bigram_mat.shape[1]}")
    print(f"  Trigram (top-{TOP_K_TRIGRAMS}):  {trigram_mat.shape[1]}")
    print(f"  PCD:                {pcd_mat.shape[1]}")
    print(f"  Combined (trans+PCD): {combined_mat.shape[1]}")
    print(f"  Full combined:      {full_combined_mat.shape[1]}")

    # --- Save feature matrices as .npz
    np.savez_compressed(
        feat_dir / "features_all.npz",
        transition=transition_mat,
        bigram=bigram_mat,
        trigram=trigram_mat,
        pcd=pcd_mat,
        combined=combined_mat,
        full_combined=full_combined_mat,
        track_ids=meta_df["track_id"].values,
        ragas=meta_df["raga"].values,
        segment_indices=meta_df["segment_index"].values,
    )

    # Save metadata
    meta_df.to_csv(feat_dir / "features_meta.csv", index=False)

    # Save trigram vocabulary for reference
    with open(feat_dir / "trigram_vocab.txt", "w", encoding="utf-8") as f:
        for gram in trigram_vocab:
            f.write("→".join(gram) + "\n")

    print(f"\nFeature files saved to: {feat_dir}")

    # --- Per-raga aggregate plots
    print(f"\nGenerating per-raga plots...")
    ragas_to_plot = meta_df["raga"].unique()[:args.plot_n]

    for raga in ragas_to_plot:
        mask = meta_df["raga"].values == raga
        if mask.sum() == 0:
            continue

        # Average transition matrix for this raga
        avg_trans = transition_mat[mask].mean(axis=0).reshape(N_SWARAS, N_SWARAS)
        plot_transition_matrix(
            avg_trans,
            f"Avg Transition Matrix — {raga}",
            fig_dir / f"transition_{raga.replace(' ', '_')}.png"
        )

        # Average PCD for this raga
        avg_pcd = pcd_mat[mask].mean(axis=0)
        plot_pcd(
            avg_pcd,
            f"Avg Pitch Class Distribution — {raga}",
            fig_dir / f"pcd_{raga.replace(' ', '_')}.png"
        )

    # --- Summary stats
    print(f"\nDone. Extracted features for {len(meta_df)} segments.")
    print(f"Ragas in dataset: {meta_df['raga'].nunique()}")
    print(f"\nTokens per segment — mean: {meta_df['n_tokens'].mean():.1f}, "
          f"median: {meta_df['n_tokens'].median():.1f}")
    print(f"\nSegments per raga:")
    print(meta_df["raga"].value_counts().to_string())


if __name__ == "__main__":
    main()
