"""
gamaka_experiment.py — Standalone Gamaka Feature Experiment

WHAT THIS SCRIPT DOES:
    Tests whether continuous pitch contour features (gamaka representations)
    improve raga classification beyond the discrete swara features used in the
    main pipeline. Computes pitch slope distributions (histogram of frame-to-
    frame cent differences) and per-swara gamaka profiles (how each swara is
    ornamented). Runs its own SVM classification under the same GroupKFold
    protocol for comparison. Does not modify the main pipeline.

LOGIC AND DESIGN:
    The main pipeline's token collapse step discards continuous gamaka
    information — a deliberate trade-off that works for most ragas but fails
    for the Harikambhoji family, where ragas differ only in ornamental contour.
    This experiment tests the complementary approach: preserving continuous pitch
    movements as features. Pitch slopes capture the rate and direction of
    ornamental movements. Per-swara gamaka profiles capture how each note is
    ornamented (e.g., Ma with a wide oscillation vs. Ma with a smooth glide).

COGNITIVE / MUSICOLOGICAL RATIONALE:
    Gamakas are the continuous pitch movements that Krishnaswamy (2003) showed
    are central to Carnatic expression. Expert listeners distinguish ragas
    partly by how notes are ornamented — the same note in two ragas may have
    different gamaka shapes. This experiment tests whether this ornamental
    information is computationally discriminative. Result: gamaka features do
    not improve accuracy. Pitch slope distributions are pan-raga (all Carnatic
    music uses heavy ornamentation), and per-swara profiles are too sparse with
    73 tracks. The discrete swara representation already captures the dominant
    classification-relevant patterns; gamaka discrimination would require a
    substantially larger dataset.

INPUT:
    - data/processed/features/features_all.npz — precomputed feature arrays
    - data/processed/features/features_meta.csv — segment metadata
    - data/processed/pitch_all.csv — pitch extraction results with voiced ratio
    - data/processed/swara_all.csv — swara-level npz paths per segment

OUTPUT:
    - outputs/results/gamaka_experiment_results.json — per-feature-set accuracy

Usage:
    python carnatic_scripts/experiments/gamaka_experiment.py
    python carnatic_scripts/experiments/gamaka_experiment.py --exclude-tracks "188_,91_,..."
"""
# cli argument parsing
import argparse
# json serialization for saving results
import json
# filesystem path handling
from pathlib import Path

# numerical array operations
import numpy as np
# tabular data manipulation
import pandas as pd
# support vector machine classifier
from sklearn.svm import SVC
# label encoding and feature scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler
# group-aware cross-validation splits
from sklearn.model_selection import GroupKFold
# classification evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report
# sklearn pipeline for chaining scaler and classifier
from sklearn.pipeline import Pipeline
# clone sklearn estimators for per-fold training
from sklearn.base import clone
# progress bar for iteration
from tqdm import tqdm


# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
SWARA_TO_IDX = {s: i for i, s in enumerate(SWARA_NAMES)}
N_SWARAS = len(SWARA_NAMES)


# ── Gamaka feature functions ──────────────────────────────────────

# build a normalized histogram of frame-to-frame pitch velocity in cents
def compute_pitch_slope_distribution(cents_mod, n_bins=50, max_slope=100.0):
    """Histogram of frame-to-frame pitch velocity (cents/frame)."""
    valid_cents = cents_mod.copy()
    valid_cents[~np.isfinite(cents_mod)] = np.nan
    slopes = np.diff(valid_cents)
    slopes_valid = slopes[np.isfinite(slopes)]
    if slopes_valid.size < 10:
        return np.zeros(n_bins, dtype=np.float64)
    slopes_valid = np.where(slopes_valid > 600, slopes_valid - 1200, slopes_valid)
    slopes_valid = np.where(slopes_valid < -600, slopes_valid + 1200, slopes_valid)
    slopes_clipped = np.clip(slopes_valid, -max_slope, max_slope)
    edges = np.linspace(-max_slope, max_slope, n_bins + 1)
    hist, _ = np.histogram(slopes_clipped, bins=edges)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# compute per-swara approach slope, departure slope, oscillation, and glide density
def compute_per_swara_gamaka(cents_mod, swara_labels):
    """Per-swara gamaka profiles: approach slope, departure slope, oscillation, density."""
    n_frames = len(cents_mod)
    features = np.zeros(N_SWARAS * 4, dtype=np.float64)
    if n_frames < 3:
        return features
    slopes = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames - 1):
        if np.isfinite(cents_mod[i]) and np.isfinite(cents_mod[i + 1]):
            s = cents_mod[i + 1] - cents_mod[i]
            if s > 600: s -= 1200
            elif s < -600: s += 1200
            slopes[i] = s
    for swara_idx, swara_name in enumerate(SWARA_NAMES):
        mask = swara_labels == swara_name
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        approach = [slopes[idx - 1] for idx in indices
                    if idx > 0 and np.isfinite(slopes[idx - 1])]
        departure = [slopes[idx] for idx in indices
                     if idx < n_frames - 1 and np.isfinite(slopes[idx])]
        cents_valid = cents_mod[mask]
        cents_valid = cents_valid[np.isfinite(cents_valid)]
        glide_count = 0
        total_nearby = 0
        for idx in indices:
            for offset in range(-2, 3):
                ni = idx + offset
                if 0 <= ni < n_frames and ni != idx:
                    total_nearby += 1
                    if swara_labels[ni] == "glide":
                        glide_count += 1
        base = swara_idx * 4
        features[base + 0] = float(np.mean(approach)) if approach else 0.0
        features[base + 1] = float(np.mean(departure)) if departure else 0.0
        features[base + 2] = float(np.std(cents_valid)) if cents_valid.size > 1 else 0.0
        features[base + 3] = float(glide_count / total_nearby) if total_nearby > 0 else 0.0
    return features


# ── Data loading ──────────────────────────────────────────────────

# load feature arrays and metadata, filtering by voiced ratio and raga count
def load_data(root, min_voiced_ratio=0.25, min_segments_per_raga=20,
              exclude_tracks=None):
    feat_dir = root / "data" / "processed" / "features"
    fdata = np.load(feat_dir / "features_all.npz", allow_pickle=True)
    meta_df = pd.read_csv(feat_dir / "features_meta.csv")
    pitch_df = pd.read_csv(root / "data" / "processed" / "pitch_all.csv")
    meta_df = meta_df.merge(
        pitch_df[["track_id", "segment_index", "voiced_ratio"]],
        on=["track_id", "segment_index"], how="left")

    mask = meta_df["voiced_ratio"] >= min_voiced_ratio
    mask_np = mask.values
    ragas = fdata["ragas"][mask_np]
    track_ids = meta_df.loc[mask, "track_id"].values
    meta_df = meta_df[mask].reset_index(drop=True)

    raga_counts = pd.Series(ragas).value_counts()
    keep_ragas = set(raga_counts[raga_counts >= min_segments_per_raga].index)
    raga_mask = np.array([r in keep_ragas for r in ragas])
    ragas = ragas[raga_mask]
    track_ids = track_ids[raga_mask]
    meta_df = meta_df[raga_mask].reset_index(drop=True)

    orig_indices = np.where(mask_np)[0]
    final_indices = orig_indices[raga_mask]
    n_before = len(fdata["ragas"])
    features = {}
    for key in fdata.files:
        arr = fdata[key]
        if arr.ndim >= 1 and arr.shape[0] == n_before:
            features[key] = arr[final_indices]

    if exclude_tracks:
        prefixes = [p.strip() for p in exclude_tracks.split(",")]
        exc = np.array([any(tid.startswith(p) for p in prefixes) for tid in track_ids])
        keep = ~exc
        for key in features:
            features[key] = features[key][keep]
        ragas = ragas[keep]
        track_ids = track_ids[keep]
        meta_df = meta_df[keep].reset_index(drop=True)
        raga_counts = pd.Series(ragas).value_counts()
        keep_ragas = set(raga_counts[raga_counts >= min_segments_per_raga].index)
        raga_keep = np.array([r in keep_ragas for r in ragas])
        for key in features:
            features[key] = features[key][raga_keep]
        ragas = ragas[raga_keep]
        track_ids = track_ids[raga_keep]
        meta_df = meta_df[raga_keep].reset_index(drop=True)

    return features, ragas, track_ids, meta_df


# aggregate segment predictions to track-level by majority vote
def track_level_voting(y_true, y_pred, groups, label_names):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "track_id": groups})
    t_true, t_pred = [], []
    for _, group in df.groupby("track_id"):
        t_true.append(group["y_true"].iloc[0])
        t_pred.append(group["y_pred"].value_counts().index[0])
    return np.array(t_true), np.array(t_pred)


# train an svm with groupkfold cross-validation and return segment accuracy
def train_svm(X, y, groups, name, cv_folds=5):
    n_groups = len(np.unique(groups))
    actual_folds = min(cv_folds, n_groups)
    gkf = GroupKFold(n_splits=actual_folds)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1, gamma=0.01, probability=True,
                     random_state=42, class_weight="balanced")),
    ])
    all_true, all_pred, all_groups = [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[train_idx], y[train_idx])
        pred = fold_pipe.predict(X[val_idx])
        all_true.extend(y[val_idx])
        all_pred.extend(pred)
        all_groups.extend(groups[val_idx])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_groups = np.array(all_groups)
    seg_acc = accuracy_score(all_true, all_pred)
    return seg_acc, all_true, all_pred, all_groups


# run the full gamaka experiment: load data, extract features, train, and report
def main():
    parser = argparse.ArgumentParser(description="Gamaka feature experiment")
    parser.add_argument("--exclude-tracks", type=str, default=None)
    args = parser.parse_args()

    root = repo_root()
    results_dir = root / "outputs" / "results"
    ensure_dir(results_dir)

    print("Loading data...")
    features, ragas, track_ids, meta_df = load_data(
        root, exclude_tracks=args.exclude_tracks)

    le = LabelEncoder()
    y = le.fit_transform(ragas)
    label_names = list(le.classes_)
    groups = track_ids

    print(f"Dataset: {len(y)} segments, {len(label_names)} ragas, "
          f"{len(np.unique(groups))} tracks")

    # Extract gamaka features from swara npz files
    print("Extracting gamaka features...")
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    swara_df = pd.read_csv(swara_csv)

    all_slope = []
    all_swara_gamaka = []
    matched = 0

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Gamaka"):
        swara_match = swara_df[
            (swara_df["track_id"] == row["track_id"]) &
            (swara_df["segment_index"] == row["segment_index"])
        ]
        if len(swara_match) == 0:
            all_slope.append(np.zeros(50))
            all_swara_gamaka.append(np.zeros(N_SWARAS * 4))
            continue

        swara_npz_path = root / swara_match.iloc[0]["swara_npz"]
        sdata = np.load(swara_npz_path, allow_pickle=True)
        cents_mod = sdata["cents_mod"]
        swara_labels = sdata["swara_labels"]

        all_slope.append(compute_pitch_slope_distribution(cents_mod))
        all_swara_gamaka.append(compute_per_swara_gamaka(cents_mod, swara_labels))
        matched += 1

    print(f"Matched {matched}/{len(meta_df)} segments")

    slope_mat = np.array(all_slope)
    swara_gamaka_mat = np.array(all_swara_gamaka)
    gamaka_combined = np.hstack([slope_mat, swara_gamaka_mat])

    # Build feature sets to test
    test_sets = {
        "dwell_vadi (baseline)": features["dwell_vadi"],
        "trigram_dwell (baseline)": features["trigram_dwell"],
        "pitch_slope": slope_mat,
        "gamaka_combined": gamaka_combined,
        "dwell_gamaka": np.hstack([features["dwell_vadi"], gamaka_combined]),
        "trigram_dwell_gamaka": np.hstack([features["trigram_dwell"], gamaka_combined]),
    }

    print(f"\n{'='*70}")
    print(f"GAMAKA FEATURE EXPERIMENT")
    print(f"{'='*70}")
    print(f"{'Feature':<30} {'Dims':>5} {'Seg Acc':>8} {'Trk Acc':>8}")
    print(f"{'-'*55}")

    results = {}
    for name, X in test_sets.items():
        seg_acc, y_true, y_pred, cv_groups = train_svm(X, y, groups, name)
        t_true, t_pred = track_level_voting(y_true, y_pred, cv_groups, label_names)
        trk_acc = accuracy_score(t_true, t_pred)

        print(f"{name:<30} {X.shape[1]:>5} {seg_acc:>7.2%} {trk_acc:>7.2%}")
        results[name] = {
            "dims": int(X.shape[1]),
            "segment_accuracy": float(seg_acc),
            "track_accuracy": float(trk_acc),
        }

    # Save results
    out_json = results_dir / "gamaka_experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
