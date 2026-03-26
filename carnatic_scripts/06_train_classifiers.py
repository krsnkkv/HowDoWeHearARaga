"""
06_train_classifiers.py — Stage 6: Raga Classification

WHAT THIS SCRIPT DOES:
    Trains multiple classifiers across all feature representations and evaluates
    them under GroupKFold cross-validation. Filters low-melody segments
    (voiced_ratio < 0.25), drops ragas with fewer than 20 segments, then trains:
    (a) independent SVMs (RBF kernel, balanced class weights) on each feature
    set via GridSearchCV scored by weighted F1; (b) a probability-level fusion
    model weighting per-class vectors from top SVMs by their CV F1 scores; (c) a
    bidirectional LSTM baseline (2 layers, 64 hidden, 32-dim embedding) on swara
    token sequences. Track-level predictions are obtained via majority voting
    across segment predictions. Outputs trained models, confusion matrices, a
    model comparison chart, and a comprehensive JSON results file.

LOGIC AND DESIGN:
    GroupKFold partitions data by track_id so that all segments from a single
    recording appear in the same fold — preventing the classifier from exploiting
    within-track similarity (Sturm, 2014). Class-weighted SVMs (inverse
    frequency) and weighted cross-entropy for the LSTM handle the 14.6:1 class
    imbalance. GridSearchCV searches C in {0.1, 1, 10, 100} and gamma in
    {scale, 0.01, 0.1}. Track-level majority voting aggregates noisy segment
    predictions — dwell_vadi gains 19 percentage points from segment to track
    because random tonic-induced label errors are correctable by voting, while
    trigram errors are systematic. The LSTM uses early stopping (10 epochs
    patience) and provides a learned-representation comparison.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    The evaluation design tests six hypotheses linking feature performance to
    cognitive predictions. GroupKFold prevents artefact learning — ensuring the
    classifier learns raga-level patterns, not recording-specific signatures.
    The comparison between interpretable features (trigram, dwell) and the LSTM
    tests whether hand-crafted cognitive representations outperform learned ones
    under data scarcity — the cognitive parsimony prediction (Gigerenzer, 1999)
    that simpler models using fewer, theoretically motivated cues outperform
    exhaustive ones in noisy, low-data environments. The MFCC control baseline
    tests Peretz and Coltheart's (2003) modularity prediction. Majority voting
    mirrors cognitive evidence accumulation: listeners integrate melodic evidence
    over time to converge on a raga identification.

Usage:
    python carnatic_scripts/06_train_classifiers.py
    python carnatic_scripts/06_train_classifiers.py --skip-lstm
    python carnatic_scripts/06_train_classifiers.py --min-voiced-ratio 0.2
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # (unused rn) kept for config consistency with other stages
import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC                    # The SVM classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Label encoding + feature scaling
from sklearn.model_selection import GroupKFold, GridSearchCV     # CV + hyperparameter search
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline           # Chain scaler + SVM together
from sklearn.base import clone                  # Copy a model for each fold

from tqdm import tqdm  # (unused rn) but handy if you add progress bars later


def repo_root() -> Path:
    # repo root = parent of carnatic_scripts/
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    # mkdir safely
    p.mkdir(parents=True, exist_ok=True)


# ── Swara definitions ───────────────────────────────────────────

# swara token vocab for lstm input (integer-encoded)
SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
# note: +1 so 0 can be reserved for padding / silence
SWARA_TO_IDX = {s: i + 1 for i, s in enumerate(SWARA_NAMES)}
# extra symbols
SWARA_TO_IDX["glide"] = len(SWARA_NAMES) + 1
SWARA_TO_IDX["silence"] = 0
# vocab size for embedding layer
VOCAB_SIZE = len(SWARA_NAMES) + 2


# ── Data Loading & Filtering ────────────────────────────────────

def load_and_filter_data(root: Path, min_voiced_ratio: float = 0.15,
                         min_segments_per_raga: int = 20):
    """
    Load feature matrices and filter out:
      1. Low-melody segments (voiced_ratio < threshold)
      2. Rāgas with too few segments after filtering
    """
    # features created in stage 5
    feat_dir = root / "data" / "processed" / "features"
    feat_path = feat_dir / "features_all.npz"
    meta_csv = feat_dir / "features_meta.csv"
    # voiced ratio comes from pitch extraction stage output
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"

    fdata = np.load(feat_path, allow_pickle=True)
    meta_df = pd.read_csv(meta_csv)
    pitch_df = pd.read_csv(pitch_csv)

    # merge voiced_ratio into meta
    meta_df = meta_df.merge(
        pitch_df[["track_id", "segment_index", "voiced_ratio"]],
        on=["track_id", "segment_index"],
        how="left"
    )

    # filter 1: keep segments with enough melodic content
    mask = meta_df["voiced_ratio"] >= min_voiced_ratio
    n_before = len(meta_df)
    n_after = int(mask.sum())
    print(f"Segment filtering (min voiced ratio = {min_voiced_ratio}):")
    print(f"  Before: {n_before}")
    print(f"  After:  {n_after} ({n_before - n_after} removed)")

    # apply voiced filter to all arrays
    mask_np = mask.values
    ragas_filtered = fdata["ragas"][mask_np]
    track_ids_filtered = meta_df.loc[mask, "track_id"].values
    meta_filtered = meta_df[mask].reset_index(drop=True)

    # filter 2: drop rāgas with too few segments
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

    # combined boolean mask for indexing feature arrays
    # (voiced filter applied to original indices, then raga filter on top)
    orig_indices = np.where(mask_np)[0]
    final_indices = orig_indices[raga_mask]

    filtered = {}
    for key in fdata.files:
        arr = fdata[key]
        # only filter arrays that are aligned by segment dimension
        if arr.ndim >= 1 and arr.shape[0] == n_before:
            filtered[key] = arr[final_indices]

    # print class balance after filtering (important for cv stability)
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
    # each segment belongs to a track; group by track_id then vote
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
        # majority vote
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
    # main svm trainer for one feature set
    print(f"\n{'='*60}")
    print(f"SVM: {name} — {X.shape[1]} features, {len(label_names)} classes")
    print(f"{'='*60}")

    # check we have enough groups for cv (groupkfold needs >= n_splits tracks)
    n_groups = len(np.unique(groups))
    actual_folds = min(cv_folds, n_groups)
    if actual_folds < cv_folds:
        print(f"  Warning: only {n_groups} groups, reducing to {actual_folds} folds")

    # pipeline: scale → svm with class balancing
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42,
                     class_weight="balanced")),  # key: handles imbalance
    ])

    # grid search with groupkfold
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

    # full cv predictions for detailed metrics (manual loop so we can store proba + fold accs)
    best_pipe = grid.best_estimator_

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_groups = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # split by tracks (no leakage across performances)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # clone so each fold is independent
        fold_pipe = clone(best_pipe)
        fold_pipe.fit(X_train, y_train)

        y_pred = fold_pipe.predict(X_val)
        y_proba = fold_pipe.predict_proba(X_val)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_groups.extend(groups[val_idx])
        fold_accuracies.append(accuracy_score(y_val, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    # handle variable-size proba arrays (folds may see different class counts)
    try:
        all_y_proba = np.array(all_y_proba)
    except ValueError:
        n_classes = max(p.shape[-1] for p in all_y_proba)
        padded = []
        for p in all_y_proba:
            if p.ndim == 1 and p.shape[0] < n_classes:
                padded.append(np.pad(p, (0, n_classes - p.shape[0])))
            else:
                padded.append(p)
        all_y_proba = np.array(padded)
    all_groups = np.array(all_groups)

    # segment-level metrics
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

    # track-level majority voting
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

    # save model with unique name
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
        "y_proba": all_y_proba,
        "groups": all_groups,
        "track_details": track_details,
        "report": report,
    }


# ── LSTM Training ────────────────────────────────────────────────

def get_swara_token_sequence(swara_labels: np.ndarray) -> list:
    # same idea as stage 5 token collapse but here silence resets only
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
    # pad / truncate sequences to a fixed length for batching
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

    # choose device (gpu if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # build token sequences
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    swara_df = pd.read_csv(swara_csv)

    # merge swara paths
    lstm_meta = meta_df.merge(
        swara_df[["track_id", "segment_index", "swara_npz"]],
        on=["track_id", "segment_index"],
        how="inner"
    )

    all_sequences = []
    all_labels = []
    valid_groups = []

    for idx, row in lstm_meta.iterrows():
        # load swara npz for each segment
        swara_npz_path = root / row["swara_npz"]
        if not swara_npz_path.exists():
            continue

        sdata = np.load(swara_npz_path, allow_pickle=True)
        swara_labels = sdata["swara_labels"]
        tokens = get_swara_token_sequence(swara_labels)
        token_indices = [SWARA_TO_IDX.get(t, 0) for t in tokens]

        # skip ultra-short sequences (basically no info)
        if len(token_indices) < 5:
            continue

        all_sequences.append(token_indices)
        all_labels.append(le.transform([row["raga"]])[0])
        valid_groups.append(row["track_id"])

    # set max_len cap for efficiency (avoid insane padding)
    max_len = min(200, max(len(s) for s in all_sequences))
    X_seq = pad_sequences(all_sequences, max_len)
    y_seq = np.array(all_labels)
    groups_seq = np.array(valid_groups)

    print(f"Sequences: {len(X_seq)}, Max length: {max_len}")

    # model
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
            # concat last forward + backward hidden states
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            out = self.dropout(hidden_cat)
            return self.fc(out)

    # groupkfold cv
    n_groups = len(np.unique(groups_seq))
    actual_folds = min(cv_folds, n_groups)
    gkf = GroupKFold(n_splits=actual_folds)
    n_classes = len(le.classes_)

    all_y_true = []
    all_y_pred = []
    all_lstm_groups = []
    fold_accuracies = []

    # class weights for loss function (inverse frequency)
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

        # fresh model per fold
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

            # validate each epoch
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val).argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val_np, val_preds)

            # early stopping on accuracy
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

    # segment-level metrics
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

    # track-level majority voting
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


# ── Probability Fusion ───────────────────────────────────────────

def probability_fusion(results_subset: dict, label_names: list) -> dict:
    """
    Late fusion: weighted average of per-class probability vectors from
    multiple SVMs. Weights are proportional to each model's CV weighted F1.

    All models in results_subset must have been evaluated with the same
    GroupKFold split (same y and groups), so their y_true orderings match.
    """
    # fuse per-class probabilities across models (late fusion)
    names = list(results_subset.keys())
    f1_scores = np.array([results_subset[n]["f1_weighted"] for n in names])
    weights = f1_scores / f1_scores.sum()

    print(f"\n{'='*60}")
    print(f"PROBABILITY FUSION — {len(names)} models")
    print(f"{'='*60}")
    for name, w, f1 in zip(names, weights, f1_scores):
        print(f"  {name:<35} weight={w:.3f}  (CV F1w={f1:.4f})")

    # reference ordering to ensure all models aligned
    ref_y_true = results_subset[names[0]]["y_true"]
    ref_groups = results_subset[names[0]]["groups"]

    fused_proba = np.zeros_like(results_subset[names[0]]["y_proba"])
    for name, w in zip(names, weights):
        # assert is a safety check: your fusion breaks if ordering differs
        assert np.array_equal(results_subset[name]["y_true"], ref_y_true), (
            f"Segment ordering mismatch for {name} — GroupKFold must use "
            "identical y and groups across all models."
        )
        fused_proba += w * results_subset[name]["y_proba"]

    fused_pred = fused_proba.argmax(axis=1)

    acc = accuracy_score(ref_y_true, fused_pred)
    f1_w = f1_score(ref_y_true, fused_pred, average="weighted")
    f1_m = f1_score(ref_y_true, fused_pred, average="macro")

    print(f"\n--- Segment-level ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 (weighted):    {f1_w:.4f}")
    print(f"F1 (macro):       {f1_m:.4f}")

    report = classification_report(ref_y_true, fused_pred,
                                   target_names=label_names, zero_division=0)
    print(f"\n{report}")

    # track-level majority voting on fused preds
    track_true, track_pred, track_details = track_level_voting(
        ref_y_true, fused_pred, ref_groups, label_names
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

    return {
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "track_accuracy": track_acc,
        "track_f1_weighted": track_f1_w,
        "track_f1_macro": track_f1_m,
        "fold_accuracies": [],
        "y_true": ref_y_true,
        "y_pred": fused_pred,
        "groups": ref_groups,
        "track_details": track_details,
        "report": report,
    }


# ── Hybrid LSTM+SVM Fusion ───────────────────────────────────────

def hybrid_track_fusion(svm_result: dict, lstm_result: dict,
                        label_names: list) -> dict:
    """
    Combine SVM and LSTM via track-level pooled majority vote.
    Pools segment predictions from both models per track, then votes.
    """
    # fuse segment predictions at track level by pooling votes from both models
    print(f"\n{'='*60}")
    print(f"HYBRID FUSION — SVM + LSTM (track-level pooled vote)")
    print(f"{'='*60}")

    svm_df = pd.DataFrame({
        "track_id": svm_result["groups"],
        "y_true": svm_result["y_true"],
        "y_pred": svm_result["y_pred"],
        "source": "svm",
    })
    lstm_df = pd.DataFrame({
        "track_id": lstm_result["groups"],
        "y_true": lstm_result["y_true"],
        "y_pred": lstm_result["y_pred"],
        "source": "lstm",
    })

    print(f"  SVM segments:  {len(svm_df)}")
    print(f"  LSTM segments: {len(lstm_df)}")

    combined = pd.concat([svm_df, lstm_df], ignore_index=True)

    track_true = []
    track_pred = []
    track_details = []

    for track_id, group in combined.groupby("track_id"):
        true_label = group["y_true"].iloc[0]
        pred_counts = group["y_pred"].value_counts()
        pred_label = pred_counts.index[0]
        confidence = pred_counts.iloc[0] / len(group)

        n_svm = (group["source"] == "svm").sum()
        n_lstm = (group["source"] == "lstm").sum()

        track_true.append(true_label)
        track_pred.append(pred_label)
        track_details.append({
            "track_id": track_id,
            "true_raga": label_names[true_label],
            "pred_raga": label_names[pred_label],
            "confidence": confidence,
            "n_svm_segs": int(n_svm),
            "n_lstm_segs": int(n_lstm),
            "correct": true_label == pred_label,
        })

    track_true = np.array(track_true)
    track_pred = np.array(track_pred)

    track_acc = accuracy_score(track_true, track_pred)
    track_f1_w = f1_score(track_true, track_pred, average="weighted", zero_division=0)
    track_f1_m = f1_score(track_true, track_pred, average="macro", zero_division=0)

    print(f"\n--- Track-level (pooled majority vote) ---")
    print(f"Tracks:           {len(track_true)}")
    print(f"Accuracy:         {track_acc:.4f}")
    print(f"F1 (weighted):    {track_f1_w:.4f}")
    print(f"F1 (macro):       {track_f1_m:.4f}")

    report = classification_report(track_true, track_pred,
                                   target_names=label_names, zero_division=0)
    print(f"\n{report}")

    # how often svm and lstm agree/disagree at track level
    svm_track = {}
    for d in svm_result.get("track_details", []):
        svm_track[d["track_id"]] = d["pred_raga"]
    lstm_track = {}
    for d in lstm_result.get("track_details", []):
        lstm_track[d["track_id"]] = d["pred_raga"]

    common = set(svm_track.keys()) & set(lstm_track.keys())
    agree = sum(1 for t in common if svm_track[t] == lstm_track[t])
    print(f"\nSVM/LSTM agreement on {len(common)} common tracks: "
          f"{agree}/{len(common)} ({agree/len(common)*100:.1f}%)")

    return {
        "accuracy": 0.0,  # no segment-level metric for hybrid
        "f1_weighted": 0.0,
        "f1_macro": 0.0,
        "track_accuracy": track_acc,
        "track_f1_weighted": track_f1_w,
        "track_f1_macro": track_f1_m,
        "fold_accuracies": [],
        "track_details": track_details,
        "report": report,
    }


# ── Visualization ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, title, out_path):
    # normalised confusion matrix plot (row-normalised)
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
    # bar chart comparing segment vs track accuracy across models
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

    # label bars with values (quick scan)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────

def main():
    # cli flags for ablations + filtering
    parser = argparse.ArgumentParser(description="Train raga classifiers (v2)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-svm", action="store_true",
                        help="Skip all SVM classifiers (LSTM only)")
    parser.add_argument("--min-voiced-ratio", type=float, default=0.25,
                        help="Filter segments below this voiced ratio (default: 0.25)")
    parser.add_argument("--min-segments-per-raga", type=int, default=20,
                        help="Drop rāgas with fewer segments than this (default: 20)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--exclude-tracks", type=str, default=None,
                        help="Comma-separated track_id prefixes to exclude "
                             "(e.g. '157_,166_,91_' to drop bad-tonic tracks)")
    args = parser.parse_args()

    root = repo_root()

    # --- load and filter data
    features, ragas, track_ids, meta_df = load_and_filter_data(
        root,
        min_voiced_ratio=args.min_voiced_ratio,
        min_segments_per_raga=args.min_segments_per_raga,
    )

    # --- optional: exclude specific tracks (e.g. bad tonic estimates)
    if args.exclude_tracks:
        prefixes = [p.strip() for p in args.exclude_tracks.split(",")]
        exclude_mask = np.array([
            any(tid.startswith(p) for p in prefixes) for tid in track_ids
        ])
        n_excl = exclude_mask.sum()
        keep = ~exclude_mask
        for key in features:
            features[key] = features[key][keep]
        ragas = ragas[keep]
        track_ids = track_ids[keep]
        meta_df = meta_df[keep].reset_index(drop=True)
        # re-filter ragas that may now have too few segments
        raga_counts = pd.Series(ragas).value_counts()
        keep_ragas = set(raga_counts[raga_counts >= args.min_segments_per_raga].index)
        raga_keep = np.array([r in keep_ragas for r in ragas])
        for key in features:
            features[key] = features[key][raga_keep]
        ragas = ragas[raga_keep]
        track_ids = track_ids[raga_keep]
        meta_df = meta_df[raga_keep].reset_index(drop=True)
        print(f"Excluded {n_excl} segments from {len(prefixes)} track prefixes")

    # encode raga labels into ints for classifiers
    le = LabelEncoder()
    y = le.fit_transform(ragas)
    label_names = list(le.classes_)
    groups = track_ids  # groups = track ids for groupkfold

    print(f"Dataset: {len(y)} segments, {len(label_names)} ragas, "
          f"{len(np.unique(groups))} tracks")

    # --- output dirs
    model_dir = root / "models"
    fig_dir = root / "outputs" / "figures" / "classification"
    results_dir = root / "outputs" / "results"
    ensure_dir(model_dir)
    ensure_dir(fig_dir)
    ensure_dir(results_dir)

    # --- build trigram-based composite features (not precomputed in 05)
    # these are quick combos that are cheap to stack here
    if "trigram" in features and "pcd" in features:
        features["pcd_trigram"] = np.hstack([features["pcd"], features["trigram"]])
    if "trigram" in features and "pcd" in features and "auxiliary" in features:
        features["pcd_trigram_aux"] = np.hstack([
            features["pcd"], features["trigram"], features["auxiliary"]
        ])
    if "trigram" in features and "mfcc" in features and "pcd" in features:
        features["pcd_trigram_mfcc"] = np.hstack([
            features["pcd"], features["trigram"], features["mfcc"]
        ])

    # --- train svms on all feature sets
    all_results = {}

    feature_sets = [
        # pure swara-based (pitch structure)
        "pcd",
        "trigram",
        "pcd_trigram",
        "pcd_trigram_aux",
        # new: sustain & emphasis features
        "dwell_vadi",
        "trigram_dwell",
        # audio baseline
        "mfcc",
        # directional transition features
        "directional_pcd",
        "directional_full",
    ]

    if not args.skip_svm:
        for feat_name in feature_sets:
            if feat_name not in features:
                print(f"\nSkipping {feat_name} — not in features_all.npz")
                continue

            X = features[feat_name]
            results = train_svm(X, y, groups, label_names, feat_name,
                               cv_folds=args.cv_folds, model_dir=model_dir)
            all_results[f"SVM_{feat_name}"] = results

            # save confusion matrix for this model
            plot_confusion_matrix(
                results["y_true"], results["y_pred"], label_names,
                f"Confusion Matrix — SVM {feat_name}\n"
                f"(Seg Acc: {results['accuracy']:.3f}, Track Acc: {results['track_accuracy']:.3f})",
                fig_dir / f"cm_SVM_{feat_name}.png"
            )

        # --- probability fusion over best pitch-structure svms
        fusion_candidates = ["SVM_pcd", "SVM_trigram", "SVM_trigram_dwell"]
        fusion_subset = {
            k: all_results[k] for k in fusion_candidates
            if k in all_results and "y_proba" in all_results[k]
        }
        if len(fusion_subset) >= 2:
            fusion_results = probability_fusion(fusion_subset, label_names)
            all_results["fusion_pitch"] = fusion_results
            plot_confusion_matrix(
                fusion_results["y_true"], fusion_results["y_pred"], label_names,
                f"Confusion Matrix — Probability Fusion\n"
                f"(Seg Acc: {fusion_results['accuracy']:.3f}, "
                f"Track Acc: {fusion_results['track_accuracy']:.3f})",
                fig_dir / "cm_fusion_pitch.png",
            )
    else:
        print("\n⏭ Skipping all SVM classifiers (--skip-svm)")

    # --- train lstm
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

        # --- hybrid lstm + svm fusion (track-level)
        best_svm_name = max(
            [k for k in all_results if k.startswith("SVM_")],
            key=lambda k: all_results[k]["accuracy"],
            default=None,
        )
        if best_svm_name and "track_details" in lstm_results:
            print(f"\nUsing {best_svm_name} for hybrid fusion...")
            hybrid_results = hybrid_track_fusion(
                all_results[best_svm_name], lstm_results, label_names
            )
            all_results[f"hybrid_{best_svm_name}+LSTM"] = hybrid_results

    # --- comparison chart
    if len(all_results) > 1:
        plot_comparison_bar(all_results, fig_dir / "model_comparison.png")

    # --- save results
    summary = {}
    for name, res in all_results.items():
        # keep json light: store only scalar metrics + fold accs
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

    # save label encoder
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # --- final summary table
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
    # entrypoint
    main()