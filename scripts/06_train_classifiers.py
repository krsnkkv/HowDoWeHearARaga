"""
06_train_classifiers.py — Stage 6: Raga Classification (v2)

Improvements over v1:
  - Filters low-melody segments (voiced_ratio < threshold)
  - Uses GroupKFold (by track_id) to prevent data leakage
  - Class-weighted SVM to handle imbalance
  - LSTM properly uses GPU
  - Each model variant saved with unique filename

Usage:
    python scripts/06_train_classifiers.py
    python scripts/06_train_classifiers.py --skip-lstm
    python scripts/06_train_classifiers.py --min-voiced-ratio 0.2
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Swara definitions ───────────────────────────────────────────

SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
SWARA_TO_IDX = {s: i + 1 for i, s in enumerate(SWARA_NAMES)}
SWARA_TO_IDX["glide"] = len(SWARA_NAMES) + 1
SWARA_TO_IDX["silence"] = 0
VOCAB_SIZE = len(SWARA_NAMES) + 2


# ── Data Loading & Filtering ────────────────────────────────────

def load_and_filter_data(root: Path, min_voiced_ratio: float = 0.15,
                         min_segments_per_raga: int = 20):
    """
    Load feature matrices and filter out:
      1. Low-melody segments (voiced_ratio < threshold)
      2. Rāgas with too few segments after filtering
    """
    feat_dir = root / "data" / "processed" / "features"
    feat_path = feat_dir / "features_all.npz"
    meta_csv = feat_dir / "features_meta.csv"
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"

    fdata = np.load(feat_path, allow_pickle=True)
    meta_df = pd.read_csv(meta_csv)
    pitch_df = pd.read_csv(pitch_csv)

    # Merge voiced_ratio into meta
    meta_df = meta_df.merge(
        pitch_df[["track_id", "segment_index", "voiced_ratio"]],
        on=["track_id", "segment_index"],
        how="left"
    )

    # Filter 1: keep segments with enough melodic content
    mask = meta_df["voiced_ratio"] >= min_voiced_ratio
    n_before = len(meta_df)
    n_after = int(mask.sum())
    print(f"Segment filtering (min voiced ratio = {min_voiced_ratio}):")
    print(f"  Before: {n_before}")
    print(f"  After:  {n_after} ({n_before - n_after} removed)")

    # Apply voiced filter to all arrays
    mask_np = mask.values
    ragas_filtered = fdata["ragas"][mask_np]
    track_ids_filtered = meta_df.loc[mask, "track_id"].values
    meta_filtered = meta_df[mask].reset_index(drop=True)

    # Filter 2: drop rāgas with too few segments
    raga_counts = pd.Series(ragas_filtered).value_counts()
    keep_ragas = set(raga_counts[raga_counts >= min_segments_per_raga].index)
    dropped_ragas = set(raga_counts.index) - keep_ragas

    if dropped_ragas:
        print(f"\nDropping rāgas with < {min_segments_per_raga} segments:")
        for r in sorted(dropped_ragas):
            print(f"  {r}: {raga_counts[r]} segments")

    raga_mask = np.array([r in keep_ragas for r in ragas_filtered])
    ragas_final = ragas_filtered[raga_mask]
    track_ids_final = track_ids_filtered[raga_mask]
    meta_final = meta_filtered[raga_mask].reset_index(drop=True)

    # Combined boolean mask for indexing feature arrays
    # (voiced filter applied to original indices, then raga filter on top)
    orig_indices = np.where(mask_np)[0]
    final_indices = orig_indices[raga_mask]

    filtered = {}
    for key in fdata.files:
        arr = fdata[key]
        if arr.ndim >= 1 and arr.shape[0] == n_before:
            filtered[key] = arr[final_indices]

    print(f"\nClass distribution after all filtering:")
    final_counts = pd.Series(ragas_final).value_counts()
    for raga, count in final_counts.items():
        print(f"  {count:5d}  {raga}")
    print()

    return filtered, ragas_final, track_ids_final, meta_final


# ── Track-level majority voting ──────────────────────────────────

def track_level_voting(y_true_seg, y_pred_seg, groups_seg, label_names):
    """
    Aggregate segment-level predictions to track-level via majority vote.
    Returns track-level y_true, y_pred, and per-track details.
    """
    df = pd.DataFrame({
        "y_true": y_true_seg,
        "y_pred": y_pred_seg,
        "track_id": groups_seg,
    })

    track_true = []
    track_pred = []
    track_details = []

    for track_id, group in df.groupby("track_id"):
        true_label = group["y_true"].iloc[0]  # all segs same raga
        # Majority vote
        pred_counts = group["y_pred"].value_counts()
        pred_label = pred_counts.index[0]
        confidence = pred_counts.iloc[0] / len(group)

        track_true.append(true_label)
        track_pred.append(pred_label)
        track_details.append({
            "track_id": track_id,
            "true_raga": label_names[true_label],
            "pred_raga": label_names[pred_label],
            "confidence": confidence,
            "n_segments": len(group),
            "correct": true_label == pred_label,
        })

    return np.array(track_true), np.array(track_pred), track_details


# ── SVM Training with GroupKFold ─────────────────────────────────

def train_svm(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              label_names: list, name: str, cv_folds: int = 5,
              model_dir: Path = None):
    """
    Train SVM with class weighting and GroupKFold cross-validation.
    """
    print(f"\n{'='*60}")
    print(f"SVM: {name} — {X.shape[1]} features, {len(label_names)} classes")
    print(f"{'='*60}")

    # Check we have enough groups for CV
    n_groups = len(np.unique(groups))
    actual_folds = min(cv_folds, n_groups)
    if actual_folds < cv_folds:
        print(f"  Warning: only {n_groups} groups, reducing to {actual_folds} folds")

    # Pipeline: scale → SVM with class balancing
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42,
                     class_weight="balanced")),  # KEY: handles imbalance
    ])

    # Grid search with GroupKFold
    gkf = GroupKFold(n_splits=actual_folds)

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.1],
    }

    print(f"Running {actual_folds}-fold GroupKFold grid search...")
    grid = GridSearchCV(
        pipe, param_grid, cv=gkf, scoring="f1_weighted",
        n_jobs=-1, verbose=0, refit=True
    )
    grid.fit(X, y, groups=groups)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1 (weighted): {grid.best_score_:.4f}")

    # Full cross-validation predictions for detailed metrics
    best_pipe = grid.best_estimator_

    all_y_true = []
    all_y_pred = []
    all_groups = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipe = clone(best_pipe)
        fold_pipe.fit(X_train, y_train)

        y_pred = fold_pipe.predict(X_val)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_groups.extend(groups[val_idx])
        fold_accuracies.append(accuracy_score(y_val, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_groups = np.array(all_groups)

    # Segment-level metrics
    acc = accuracy_score(all_y_true, all_y_pred)
    f1_w = f1_score(all_y_true, all_y_pred, average="weighted")
    f1_m = f1_score(all_y_true, all_y_pred, average="macro")

    print(f"\n--- Segment-level ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 (weighted):    {f1_w:.4f}")
    print(f"F1 (macro):       {f1_m:.4f}")
    print(f"Fold accuracies:  {[f'{a:.3f}' for a in fold_accuracies]}")

    report = classification_report(all_y_true, all_y_pred,
                                    target_names=label_names, zero_division=0)
    print(f"\n{report}")

    # Track-level majority voting
    track_true, track_pred, track_details = track_level_voting(
        all_y_true, all_y_pred, all_groups, label_names
    )
    track_acc = accuracy_score(track_true, track_pred)
    track_f1_w = f1_score(track_true, track_pred, average="weighted", zero_division=0)
    track_f1_m = f1_score(track_true, track_pred, average="macro", zero_division=0)

    print(f"--- Track-level (majority vote) ---")
    print(f"Tracks:           {len(track_true)}")
    print(f"Accuracy:         {track_acc:.4f}")
    print(f"F1 (weighted):    {track_f1_w:.4f}")
    print(f"F1 (macro):       {track_f1_m:.4f}")

    track_report = classification_report(track_true, track_pred,
                                          target_names=label_names, zero_division=0)
    print(f"\n{track_report}")

    # Save model with unique name
    if model_dir:
        model_path = model_dir / f"svm_{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(grid.best_estimator_, f)
        print(f"Model saved: {model_path}")

    return {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "track_accuracy": track_acc,
        "track_f1_weighted": track_f1_w,
        "track_f1_macro": track_f1_m,
        "fold_accuracies": fold_accuracies,
        "best_params": grid.best_params_,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "groups": all_groups,
        "track_details": track_details,
        "report": report,
    }


# ── LSTM Training ────────────────────────────────────────────────

def get_swara_token_sequence(swara_labels: np.ndarray) -> list:
    tokens = []
    prev = None
    for label in swara_labels:
        if label == "silence":
            prev = None
            continue
        if label != prev:
            tokens.append(label)
            prev = label
    return tokens


def pad_sequences(sequences: list, max_len: int) -> np.ndarray:
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


def train_lstm(meta_df: pd.DataFrame, le: LabelEncoder,
               groups: np.ndarray, cv_folds: int = 5,
               model_dir: Path = None, root: Path = None):
    """Train LSTM with GroupKFold and GPU support."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}")
    print(f"LSTM CLASSIFIER — {len(meta_df)} segments, {len(le.classes_)} classes")
    print(f"{'='*60}")

    # Force GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build token sequences
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    swara_df = pd.read_csv(swara_csv)

    # Merge swara paths
    lstm_meta = meta_df.merge(
        swara_df[["track_id", "segment_index", "swara_npz"]],
        on=["track_id", "segment_index"],
        how="inner"
    )

    all_sequences = []
    all_labels = []
    valid_groups = []

    for idx, row in lstm_meta.iterrows():
        swara_npz_path = root / row["swara_npz"]
        if not swara_npz_path.exists():
            continue

        sdata = np.load(swara_npz_path, allow_pickle=True)
        swara_labels = sdata["swara_labels"]
        tokens = get_swara_token_sequence(swara_labels)
        token_indices = [SWARA_TO_IDX.get(t, 0) for t in tokens]

        if len(token_indices) < 5:  # skip very short sequences
            continue

        all_sequences.append(token_indices)
        all_labels.append(le.transform([row["raga"]])[0])
        valid_groups.append(row["track_id"])

    max_len = min(200, max(len(s) for s in all_sequences))
    X_seq = pad_sequences(all_sequences, max_len)
    y_seq = np.array(all_labels)
    groups_seq = np.array(valid_groups)

    print(f"Sequences: {len(X_seq)}, Max length: {max_len}")

    # Model
    class RagaLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                bidirectional=True, dropout=dropout, num_layers=2)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, n_classes)

        def forward(self, x):
            emb = self.embedding(x)
            _, (hidden, _) = self.lstm(emb)
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            out = self.dropout(hidden_cat)
            return self.fc(out)

    # GroupKFold CV
    n_groups = len(np.unique(groups_seq))
    actual_folds = min(cv_folds, n_groups)
    gkf = GroupKFold(n_splits=actual_folds)
    n_classes = len(le.classes_)

    all_y_true = []
    all_y_pred = []
    all_lstm_groups = []
    fold_accuracies = []

    # Class weights for loss function
    class_counts = np.bincount(y_seq, minlength=n_classes).astype(np.float32)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y_seq, groups_seq)):
        print(f"\n  Fold {fold + 1}/{actual_folds}")

        X_train = torch.LongTensor(X_seq[train_idx]).to(device)
        y_train = torch.LongTensor(y_seq[train_idx]).to(device)
        X_val = torch.LongTensor(X_seq[val_idx]).to(device)
        y_val_np = y_seq[val_idx]

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = RagaLSTM(
            vocab_size=VOCAB_SIZE, embed_dim=32,
            hidden_dim=64, n_classes=n_classes, dropout=0.3
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        best_val_acc = 0
        patience_counter = 0
        best_preds = None

        for epoch in range(100):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val).argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val_np, val_preds)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds = val_preds.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                print(f"    Early stop epoch {epoch + 1}, best acc: {best_val_acc:.4f}")
                break

        if patience_counter < 10:
            print(f"    100 epochs done, best acc: {best_val_acc:.4f}")

        all_y_true.extend(y_val_np)
        all_y_pred.extend(best_preds)
        all_lstm_groups.extend(groups_seq[val_idx])
        fold_accuracies.append(best_val_acc)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_lstm_groups = np.array(all_lstm_groups)

    acc = accuracy_score(all_y_true, all_y_pred)
    f1_w = f1_score(all_y_true, all_y_pred, average="weighted")
    f1_m = f1_score(all_y_true, all_y_pred, average="macro")

    print(f"\n--- Segment-level ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 (weighted):    {f1_w:.4f}")
    print(f"F1 (macro):       {f1_m:.4f}")
    print(f"Fold accuracies:  {[f'{a:.3f}' for a in fold_accuracies]}")

    report = classification_report(all_y_true, all_y_pred,
                                    target_names=le.classes_, zero_division=0)
    print(f"\n{report}")

    # Track-level majority voting
    track_true, track_pred, track_details = track_level_voting(
        all_y_true, all_y_pred, all_lstm_groups, list(le.classes_)
    )
    track_acc = accuracy_score(track_true, track_pred)
    track_f1_w = f1_score(track_true, track_pred, average="weighted", zero_division=0)
    track_f1_m = f1_score(track_true, track_pred, average="macro", zero_division=0)

    print(f"--- Track-level (majority vote) ---")
    print(f"Tracks:           {len(track_true)}")
    print(f"Accuracy:         {track_acc:.4f}")
    print(f"F1 (weighted):    {track_f1_w:.4f}")
    print(f"F1 (macro):       {track_f1_m:.4f}")

    return {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "track_accuracy": track_acc,
        "track_f1_weighted": track_f1_w,
        "track_f1_macro": track_f1_m,
        "fold_accuracies": fold_accuracies,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "groups": all_lstm_groups,
        "report": report,
    }


# ── Visualization ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, xticklabels=label_names, yticklabels=label_names,
                cmap="Blues", annot=True, fmt=".2f", ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_bar(results_dict: dict, out_path: Path):
    names = list(results_dict.keys())
    seg_acc = [r["accuracy"] for r in results_dict.values()]
    track_acc = [r.get("track_accuracy", 0) for r in results_dict.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, seg_acc, width, label="Segment Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, track_acc, width, label="Track Accuracy (vote)", color="darkorange")

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Segment vs Track-level")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train raga classifiers (v2)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--min-voiced-ratio", type=float, default=0.15,
                        help="Filter segments below this voiced ratio (default: 0.15)")
    parser.add_argument("--min-segments-per-raga", type=int, default=20,
                        help="Drop rāgas with fewer segments than this (default: 20)")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    root = repo_root()

    # --- Load and filter data
    features, ragas, track_ids, meta_df = load_and_filter_data(
        root,
        min_voiced_ratio=args.min_voiced_ratio,
        min_segments_per_raga=args.min_segments_per_raga,
    )

    le = LabelEncoder()
    y = le.fit_transform(ragas)
    label_names = list(le.classes_)
    groups = track_ids

    print(f"Dataset: {len(y)} segments, {len(label_names)} ragas, "
          f"{len(np.unique(groups))} tracks")

    # --- Output dirs
    model_dir = root / "models"
    fig_dir = root / "outputs" / "figures" / "classification"
    results_dir = root / "outputs" / "results"
    ensure_dir(model_dir)
    ensure_dir(fig_dir)
    ensure_dir(results_dir)

    # --- Train SVMs on all feature sets
    all_results = {}

    feature_sets = [
        "transition", "pcd", "mfcc", "auxiliary", "transition_aux",
        "combined", "combined_interp_aux", "combined_mfcc_pcd", "full_combined",
    ]

    for feat_name in feature_sets:
        if feat_name not in features:
            print(f"\nSkipping {feat_name} — not in features_all.npz")
            continue

        X = features[feat_name]
        results = train_svm(X, y, groups, label_names, feat_name,
                           cv_folds=args.cv_folds, model_dir=model_dir)
        all_results[f"SVM_{feat_name}"] = results

        plot_confusion_matrix(
            results["y_true"], results["y_pred"], label_names,
            f"Confusion Matrix — SVM {feat_name}\n"
            f"(Seg Acc: {results['accuracy']:.3f}, Track Acc: {results['track_accuracy']:.3f})",
            fig_dir / f"cm_SVM_{feat_name}.png"
        )

    # --- Train LSTM
    if not args.skip_lstm:
        lstm_results = train_lstm(
            meta_df, le, groups,
            cv_folds=args.cv_folds,
            model_dir=model_dir,
            root=root,
        )
        all_results["LSTM_swara_seq"] = lstm_results

        plot_confusion_matrix(
            lstm_results["y_true"], lstm_results["y_pred"], label_names,
            f"Confusion Matrix — LSTM\n"
            f"(Seg Acc: {lstm_results['accuracy']:.3f}, "
            f"Track Acc: {lstm_results.get('track_accuracy', 0):.3f})",
            fig_dir / "cm_LSTM_swara_seq.png"
        )

    # --- Comparison chart
    if len(all_results) > 1:
        plot_comparison_bar(all_results, fig_dir / "model_comparison.png")

    # --- Save results
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "segment_accuracy": float(res["accuracy"]),
            "segment_f1_weighted": float(res["f1_weighted"]),
            "segment_f1_macro": float(res["f1_macro"]),
            "track_accuracy": float(res.get("track_accuracy", 0)),
            "track_f1_weighted": float(res.get("track_f1_weighted", 0)),
            "track_f1_macro": float(res.get("track_f1_macro", 0)),
            "fold_accuracies": [float(a) for a in res["fold_accuracies"]],
        }
        if "best_params" in res:
            summary[name]["best_params"] = {k: str(v) for k, v in res["best_params"].items()}

    results_json = results_dir / "classification_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save label encoder
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # --- Final summary table
    print(f"\n\n{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Seg Acc':>8} {'Seg F1w':>8} {'Trk Acc':>8} {'Trk F1w':>8}")
    print(f"{'-'*62}")
    for name, res in all_results.items():
        print(f"{name:<30} {res['accuracy']:>8.4f} {res['f1_weighted']:>8.4f} "
              f"{res.get('track_accuracy', 0):>8.4f} {res.get('track_f1_weighted', 0):>8.4f}")

    print(f"\nResults: {results_json}")
    print(f"Plots:   {fig_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()