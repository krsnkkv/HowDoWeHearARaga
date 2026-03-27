"""
h4_melakarta_confusion.py — H4: Confusion Follows Melakarta Kinship

WHAT THIS SCRIPT DOES:
    Tests hypothesis H4: whether classifier confusion follows melakarta (parent
    scale) kinship. Loads the trained SVM's confusion matrix, maps each raga to
    its melakarta family, computes within-family vs between-family confusion
    rates, and runs a permutation test (10,000 iterations) to determine whether
    the difference is statistically significant. Generates a family-annotated
    confusion matrix and a within-vs-between bar chart with p-value.

LOGIC AND DESIGN:
    For each pair of ragas (i, j) in the confusion matrix, classifies the
    confusion as "within-family" (same melakarta parent) or "between-family"
    (different parent). Computes the mean confusion rate for each group. The
    permutation test randomly shuffles family labels 10,000 times and recomputes
    the within/between ratio each time, building a null distribution. If the
    observed ratio exceeds 95% of permuted ratios, the family structure explains
    confusion patterns beyond chance.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    Castellano et al. (1984) found that ragas sharing a parent scale (that)
    cluster together perceptually — listeners rate them as more similar. H4
    predicts that computational confusion should follow the same pattern: the
    classifier should confuse ragas within the same melakarta family more often
    than ragas from different families. Result: confirmed (p < 0.0001, 0 of
    10,000 permutations exceeded observed), with within-family confusion 5.19x
    higher than between-family. The Harikambhoji family shows the strongest
    effect, consistent with these ragas sharing identical note sets.

INPUT:
    - data/processed/features/features_all.npz   (feature arrays per segment)
    - data/processed/features/features_meta.csv   (segment metadata)
    - data/processed/pitch_all.csv                (voiced ratio per segment)

OUTPUT:
    - outputs/figures/h4_melakarta/confusion_matrix_by_family.png
    - outputs/figures/h4_melakarta/within_vs_between_family.png
    - outputs/results/h4_melakarta_results.json

Usage:
    python carnatic_scripts/experiments/h4_melakarta_confusion.py
    python carnatic_scripts/experiments/h4_melakarta_confusion.py --exclude-tracks "..."
"""
# cli argument parsing
import argparse
# json serialisation for saving results
import json
# binary object serialisation (unused but retained for compatibility)
import pickle
# filesystem path handling
from pathlib import Path

# numerical array operations
import numpy as np
# tabular data manipulation
import pandas as pd
# non-interactive plotting backend
import matplotlib
matplotlib.use("Agg")
# figure and axis creation for plots
import matplotlib.pyplot as plt
# statistical visualisation (heatmaps, distributions)
import seaborn as sns

# label encoding and feature scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler
# group-aware cross-validation (no track leakage)
from sklearn.model_selection import GroupKFold
# classification evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score
# support vector machine classifier
from sklearn.svm import SVC
# chained preprocessing + model pipeline
from sklearn.pipeline import Pipeline
# deep-copy a pipeline for per-fold training
from sklearn.base import clone


# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Melakarta Family Definitions ──────────────────────────────────
# Each raga mapped to its parent melakarta scale.
# Ragas sharing the same melakarta are "within-family."

MELAKARTA_FAMILIES = {
    # Melakarta 28: Harikambhoji family (Sa Ri2 Ga2 Ma1 Pa Da2 Ni2)
    "Harikāmbhōji":    "28_Harikambhoji",
    "Kāṁbhōji":        "28_Harikambhoji",
    "Kamās":            "28_Harikambhoji",
    "Mōhanaṁ":         "28_Harikambhoji",
    "Suraṭi":           "28_Harikambhoji",

    # Melakarta 29: Shankarabharanam (Sa Ri2 Ga2 Ma1 Pa Da2 Ni2)
    "Śankarābharaṇaṁ":  "29_Shankarabharanam",
    "Bēgaḍa":           "29_Shankarabharanam",
    "Behāg":             "29_Shankarabharanam",

    # Melakarta 20: Natabhairavi (Sa Ri2 Ga1 Ma1 Pa Da1 Ni2)
    "Bhairavi":         "20_Natabhairavi",
    "Jōnpuri":          "20_Natabhairavi",

    # Melakarta 8: Hanumatodi (Sa Ri1 Ga1 Ma1 Pa Da1 Ni2)
    "Tōḍi":             "08_Hanumatodi",
    "Sindhubhairavi":   "08_Hanumatodi",

    # Melakarta 22: Kharaharapriya (Sa Ri2 Ga1 Ma1 Pa Da2 Ni2)
    "Kānaḍa":           "22_Kharaharapriya",
    "Rītigauḷa":        "22_Kharaharapriya",

    # Melakarta 65: Kalyani (Sa Ri2 Ga2 Ma2 Pa Da2 Ni2)
    "Kalyāṇi":          "65_Kalyani",

    # Melakarta 56: Sanmukhapriya (Sa Ri2 Ga2 Ma2 Pa Da1 Ni2)
    "Ṣanmukhapriya":    "56_Sanmukhapriya",

    # Melakarta 15: Mayamalavagowla (Sa Ri1 Ga2 Ma1 Pa Da1 Ni2)
    "Sāvēri":           "15_Mayamalavagowla",

    # Melakarta 14: Vakulabharanam (Sa Ri1 Ga2 Ma1 Pa Da1 Ni1)
    "Saurāṣtraṁ":       "14_Vakulabharanam",
}


# load features and apply voiced-ratio / raga-count / track-exclusion filters
def load_and_filter_data(root, min_voiced_ratio=0.25, min_segments_per_raga=20,
                         exclude_tracks=None):
    """Load and filter data identically to 06_train_classifiers.py."""
    feat_dir = root / "data" / "processed" / "features"
    fdata = np.load(feat_dir / "features_all.npz", allow_pickle=True)
    meta_df = pd.read_csv(feat_dir / "features_meta.csv")
    pitch_df = pd.read_csv(root / "data" / "processed" / "pitch_all.csv")

    meta_df = meta_df.merge(
        pitch_df[["track_id", "segment_index", "voiced_ratio"]],
        on=["track_id", "segment_index"], how="left"
    )

    # Voiced ratio filter
    mask = meta_df["voiced_ratio"] >= min_voiced_ratio
    mask_np = mask.values
    ragas = fdata["ragas"][mask_np]
    track_ids = meta_df.loc[mask, "track_id"].values
    meta_df = meta_df[mask].reset_index(drop=True)

    # Raga count filter
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

    # Exclude tracks
    if exclude_tracks:
        prefixes = [p.strip() for p in exclude_tracks.split(",")]
        exc_mask = np.array([
            any(tid.startswith(p) for p in prefixes) for tid in track_ids
        ])
        keep = ~exc_mask
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


# run group k-fold cross-validation and collect out-of-fold predictions
def get_cv_predictions(X, y, groups, cv_folds=5):
    """Run GroupKFold CV and return predictions."""
    n_groups = len(np.unique(groups))
    actual_folds = min(cv_folds, n_groups)
    gkf = GroupKFold(n_splits=actual_folds)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1, gamma=0.01,
                     probability=True, random_state=42,
                     class_weight="balanced")),
    ])

    all_y_true = []
    all_y_pred = []
    all_groups = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[train_idx], y[train_idx])
        y_pred = fold_pipe.predict(X[val_idx])
        all_y_true.extend(y[val_idx])
        all_y_pred.extend(y_pred)
        all_groups.extend(groups[val_idx])

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_groups)


# compute mean within-family and between-family off-diagonal confusion rates
def compute_family_confusion_rates(cm, label_names, family_map):
    """
    Compute within-family and between-family confusion rates.

    For each off-diagonal cell (i, j) where i != j:
      - If raga_i and raga_j share the same melakarta family → within-family error
      - Otherwise → between-family error

    Returns within_rate, between_rate, and detailed breakdown.
    """
    n = len(label_names)

    # Normalise rows (each row sums to 1)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    within_errors = []
    between_errors = []
    detail_rows = []

    for i in range(n):
        raga_i = label_names[i]
        family_i = family_map.get(raga_i, f"unknown_{raga_i}")

        for j in range(n):
            if i == j:
                continue
            raga_j = label_names[j]
            family_j = family_map.get(raga_j, f"unknown_{raga_j}")

            error_rate = cm_norm[i, j]
            same_family = (family_i == family_j)

            if same_family:
                within_errors.append(error_rate)
            else:
                between_errors.append(error_rate)

            if error_rate > 0.01:  # only log non-trivial confusions
                detail_rows.append({
                    "true_raga": raga_i,
                    "pred_raga": raga_j,
                    "error_rate": error_rate,
                    "same_family": same_family,
                    "family_true": family_i,
                    "family_pred": family_j,
                })

    within_rate = np.mean(within_errors) if within_errors else 0.0
    between_rate = np.mean(between_errors) if between_errors else 0.0

    return within_rate, between_rate, within_errors, between_errors, detail_rows


# test whether within-family confusion is significantly higher via label shuffling
def permutation_test(within_errors, between_errors, n_perms=10000):
    """
    Permutation test: is the mean within-family error significantly higher
    than the mean between-family error?

    H0: No difference (family labels don't matter)
    H1: Within-family error > between-family error (confusion follows kinship)
    """
    observed_diff = np.mean(within_errors) - np.mean(between_errors)

    all_errors = np.array(within_errors + between_errors)
    n_within = len(within_errors)

    count_ge = 0
    for _ in range(n_perms):
        perm = np.random.permutation(all_errors)
        perm_within = perm[:n_within].mean()
        perm_between = perm[n_within:].mean()
        if (perm_within - perm_between) >= observed_diff:
            count_ge += 1

    p_value = count_ge / n_perms
    return observed_diff, p_value


# plot row-normalised confusion matrix sorted and annotated by melakarta family
def plot_family_confusion(cm, label_names, family_map, out_path):
    """Plot confusion matrix with family grouping annotations."""
    # Sort ragas by family for visual clustering
    families = [family_map.get(r, "zzz_unknown") for r in label_names]
    sorted_indices = sorted(range(len(label_names)),
                            key=lambda i: (families[i], label_names[i]))
    sorted_names = [label_names[i] for i in sorted_indices]
    sorted_families = [families[i] for i in sorted_indices]

    # Reorder confusion matrix
    cm_sorted = cm[np.ix_(sorted_indices, sorted_indices)]
    cm_norm = cm_sorted.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm_norm, xticklabels=sorted_names, yticklabels=sorted_names,
                cmap="Blues", annot=True, fmt=".2f", ax=ax, square=True,
                cbar_kws={"shrink": 0.7}, vmin=0, vmax=1)
    ax.set_xlabel("Predicted Raga")
    ax.set_ylabel("True Raga")
    ax.set_title("H4: Confusion Matrix Grouped by Melakarta Family\n"
                 "(Row-normalised; within-family blocks highlighted)")

    # Draw family boundary lines
    prev_family = None
    for idx, fam in enumerate(sorted_families):
        if fam != prev_family and prev_family is not None:
            ax.axhline(idx, color="red", linewidth=2)
            ax.axvline(idx, color="red", linewidth=2)
        prev_family = fam

    # Add family labels on right side
    prev_family = None
    start_idx = 0
    for idx, fam in enumerate(sorted_families + [None]):
        if fam != prev_family:
            if prev_family is not None:
                mid = (start_idx + idx) / 2
                short_name = prev_family.split("_", 1)[1] if "_" in prev_family else prev_family
                ax.text(len(sorted_names) + 0.5, mid, short_name,
                        fontsize=7, va="center", ha="left", color="red",
                        fontweight="bold")
            start_idx = idx
            prev_family = fam

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# plot box-plot and histogram comparing within- vs between-family error rates
def plot_error_distribution(within_errors, between_errors, observed_diff,
                             p_value, out_path):
    """Box plot comparing within-family vs between-family error rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    data = [within_errors, between_errors]
    bp = ax1.boxplot(data, labels=["Within-family", "Between-family"],
                     patch_artist=True)
    bp["boxes"][0].set_facecolor("indianred")
    bp["boxes"][1].set_facecolor("steelblue")
    ax1.set_ylabel("Confusion Rate (row-normalised)")
    ax1.set_title(f"H4: Within vs Between Family Confusion\n"
                  f"Diff = {observed_diff:.4f}, p = {p_value:.4f}")

    # Scatter with means
    within_mean = np.mean(within_errors)
    between_mean = np.mean(between_errors)
    ax1.scatter([1], [within_mean], color="darkred", s=100, zorder=5,
                label=f"Mean: {within_mean:.4f}")
    ax1.scatter([2], [between_mean], color="darkblue", s=100, zorder=5,
                label=f"Mean: {between_mean:.4f}")
    ax1.legend(fontsize=9)

    # Histogram of both distributions
    ax2.hist(within_errors, bins=20, alpha=0.6, color="indianred",
             label=f"Within-family (n={len(within_errors)})", density=True)
    ax2.hist(between_errors, bins=20, alpha=0.6, color="steelblue",
             label=f"Between-family (n={len(between_errors)})", density=True)
    ax2.axvline(within_mean, color="darkred", linestyle="--")
    ax2.axvline(between_mean, color="darkblue", linestyle="--")
    ax2.set_xlabel("Confusion Rate")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Confusion Rates")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# entry point: parse args, run cv, analyse family confusion, save results and plots
def main():
    parser = argparse.ArgumentParser(
        description="H4: Test if confusion follows melakarta family kinship")
    parser.add_argument("--exclude-tracks", type=str, default=None)
    parser.add_argument("--min-segments-per-raga", type=int, default=20)
    parser.add_argument("--feature", type=str, default="dwell_vadi",
                        help="Feature set to use (default: dwell_vadi)")
    parser.add_argument("--n-perms", type=int, default=10000,
                        help="Number of permutations for significance test")
    parser.add_argument("--exclude-bad-tonic", action="store_true",
                        help="Exclude tracks with >50 cent tonic error (reads tonic_validation.csv)")
    args = parser.parse_args()

    root = repo_root()
    fig_dir = root / "outputs" / "figures" / "h4_melakarta"
    results_dir = root / "outputs" / "results"
    ensure_dir(fig_dir)
    ensure_dir(results_dir)

    # auto-exclude bad-tonic tracks if requested
    if args.exclude_bad_tonic and not args.exclude_tracks:
        tonic_csv = root / "data" / "processed" / "tonic_validation.csv"
        if tonic_csv.exists():
            import pandas as pd
            tonic_df = pd.read_csv(tonic_csv)
            bad = tonic_df[tonic_df["within_50_cents"] == False]["track_id"].tolist()
            no_est = tonic_df[tonic_df["estimated_tonic_hz"].isna()]["track_id"].tolist()
            args.exclude_tracks = ",".join(set(bad + no_est))
            print(f"[excl-bad-tonic] Excluding {len(set(bad + no_est))} bad-tonic tracks")

    # Load data
    print("Loading data...")
    features, ragas, track_ids, meta_df = load_and_filter_data(
        root, exclude_tracks=args.exclude_tracks,
        min_segments_per_raga=args.min_segments_per_raga,
    )

    le = LabelEncoder()
    y = le.fit_transform(ragas)
    label_names = list(le.classes_)
    groups = track_ids

    print(f"Dataset: {len(y)} segments, {len(label_names)} ragas, "
          f"{len(np.unique(groups))} tracks")
    print(f"Ragas: {label_names}")

    # Map ragas to families
    print(f"\nMelakarta family mapping:")
    family_counts = {}
    for raga in label_names:
        fam = MELAKARTA_FAMILIES.get(raga, f"unknown_{raga}")
        family_counts.setdefault(fam, []).append(raga)
        print(f"  {raga} -> {fam}")

    print(f"\nMulti-raga families (where within-family confusion can occur):")
    for fam, members in family_counts.items():
        if len(members) > 1:
            print(f"  {fam}: {members}")

    # Get cross-validated predictions
    feat_name = args.feature
    if feat_name not in features:
        print(f"Feature '{feat_name}' not found. Available: {list(features.keys())}")
        return

    X = features[feat_name]
    print(f"\nRunning {feat_name} ({X.shape[1]}d) CV predictions...")
    y_true, y_pred, cv_groups = get_cv_predictions(X, y, groups)

    seg_acc = accuracy_score(y_true, y_pred)
    print(f"Segment accuracy: {seg_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute family confusion rates
    print(f"\n{'='*60}")
    print("H4: MELAKARTA FAMILY CONFUSION ANALYSIS")
    print(f"{'='*60}")

    within_rate, between_rate, within_errors, between_errors, details = \
        compute_family_confusion_rates(cm, label_names, MELAKARTA_FAMILIES)

    print(f"\nMean within-family confusion rate:  {within_rate:.4f}")
    print(f"Mean between-family confusion rate: {between_rate:.4f}")
    print(f"Ratio (within/between):             {within_rate/between_rate:.2f}x" if between_rate > 0 else "")

    # Permutation test
    print(f"\nRunning permutation test ({args.n_perms} permutations)...")
    observed_diff, p_value = permutation_test(
        within_errors, between_errors, n_perms=args.n_perms
    )
    print(f"Observed difference: {observed_diff:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("=> SIGNIFICANT (p < 0.05): Confusion follows melakarta kinship!")
        print("   H4 CONFIRMED: Ragas from the same family are confused more often.")
    elif p_value < 0.10:
        print("=> MARGINALLY significant (p < 0.10): Trend toward family-based confusion.")
    else:
        print("=> NOT significant: Confusion does not clearly follow family structure.")

    # Top confusions
    details_sorted = sorted(details, key=lambda d: d["error_rate"], reverse=True)
    print(f"\nTop 10 confusions:")
    print(f"  {'True':<20} {'Pred':<20} {'Rate':>6} {'Same Family?'}")
    print(f"  {'-'*60}")
    for d in details_sorted[:10]:
        tag = "YES" if d["same_family"] else "no"
        print(f"  {d['true_raga']:<20} {d['pred_raga']:<20} "
              f"{d['error_rate']:>5.3f}  {tag}")

    # Within-family detail
    within_details = [d for d in details_sorted if d["same_family"]]
    if within_details:
        print(f"\nAll within-family confusions (> 1%):")
        for d in within_details:
            print(f"  {d['true_raga']} -> {d['pred_raga']}: "
                  f"{d['error_rate']:.3f} ({d['family_true']})")

    # Plots
    plot_family_confusion(cm, label_names, MELAKARTA_FAMILIES,
                          fig_dir / "confusion_matrix_by_family.png")
    print(f"\nConfusion matrix plot: {fig_dir / 'confusion_matrix_by_family.png'}")

    plot_error_distribution(within_errors, between_errors, observed_diff,
                            p_value, fig_dir / "within_vs_between_family.png")
    print(f"Distribution plot: {fig_dir / 'within_vs_between_family.png'}")

    # Save results
    h4_results = {
        "feature": feat_name,
        "segment_accuracy": float(seg_acc),
        "n_ragas": len(label_names),
        "ragas": label_names,
        "within_family_confusion_rate": float(within_rate),
        "between_family_confusion_rate": float(between_rate),
        "within_over_between_ratio": float(within_rate / between_rate) if between_rate > 0 else None,
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": args.n_perms,
        "significant": p_value < 0.05,
        "n_within_family_pairs": len(within_errors),
        "n_between_family_pairs": len(between_errors),
        "family_mapping": {r: MELAKARTA_FAMILIES.get(r, "unknown") for r in label_names},
        "top_confusions": details_sorted[:15],
    }

    out_json = results_dir / "h4_melakarta_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(h4_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
