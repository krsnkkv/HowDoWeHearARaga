"""
07_hierarchical_classifier.py — Stage 7 (Experimental): Hierarchical Raga Classification

WHAT THIS SCRIPT DOES:
    Implements a two-level hierarchical classifier grounded in melakarta (parent
    scale) theory. Level 1 routes segments to a melakarta family using
    directional_pcd features (408d, SVM). Level 2 disambiguates within-family
    using either directional_transition SVM or a bidirectional LSTM on swara
    token sequences. Three evaluation blocks under identical GroupKFold splits:
    (A) flat baseline, (B) hierarchical with specialist L2 models, (C) oracle
    upper bound with perfect family routing. Also supports flat fusion and
    single-family specialist analysis.

LOGIC AND DESIGN:
    Motivated by H4's prediction that confusion clusters within melakarta
    families. The hierarchy exploits this structure: L1 separates families
    (which differ in note set), then L2 handles the harder within-family
    disambiguation (where ragas share notes but differ in melodic grammar). The
    SVM L2 is weak for the Harikambhoji family because all members share
    identical notes, making bigrams near-identical. The LSTM L2 can capture
    phrase order and gamaka patterns that distinguish same-family ragas.
    Soft routing with a confidence threshold prevents L1 errors from
    propagating. The oracle block (C) measures the ceiling if L1 were perfect.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    The melakarta system is a taxonomic hierarchy: 72 parent scales, each
    generating a family of derivative ragas (janya ragas). Listeners likely
    use a similar coarse-to-fine strategy — first identifying the scale family
    from the note set, then narrowing to the specific raga from phrase patterns
    and ornamental contour. This experiment tests whether encoding this
    hierarchical structure computationally improves classification. Result:
    hierarchy does not improve over flat classification, because 60.7% of errors
    already come from within-family confusion where L1 routing cannot help.

Three evaluation blocks (same GroupKFold splits, fair comparison):

  A) Flat baseline
       Single SVM over all ragas — no routing, no hierarchy.
       With --flat-fusion: F1-weighted average of predict_proba() from
       SVM(directional_pcd) + SVM(directional_transition).

  B) Hierarchical predicted  (soft routing)
       L1 predicts family. Segments with L1 max_prob >= --conf-threshold are routed
       to L2. Uncertain segments (max_prob < threshold) fall back to flat SVM.
       --conf-threshold 0.0 = always route (hard routing).

  C) Hierarchical oracle  (honest upper bound)
       True family used for routing to L2 for MULTI-RAGA families only.
       Flat SVM used for SINGLETON families (no trivial assignment inflation).
       Gap (C - A) = pure benefit of perfect routing for hard multi-raga cases.

Specialist mode (--family-only):
       Keeps flat SVM everywhere; adds a binary L1 + specialist L2 only for one
       focused family. Isolates per-family gains without a full hierarchy.
       Block B: flat + specialist for L1-predicted-focused segments.
       Block C: flat + specialist for TRULY focused segments (oracle upper bound).

INPUT:
    - data/processed/features/features_all.npz   (feature arrays per segment)
    - data/processed/features/features_meta.csv   (segment metadata)
    - data/processed/pitch_all.csv                (voiced ratio per segment)
    - data/processed/swara_all.csv                (swara npz paths, for lstm mode)

OUTPUT:
    - outputs/figures/hierarchical/cm_A_flat.png
    - outputs/figures/hierarchical/cm_B_hierarchical.png  (or cm_B_specialist.png)
    - outputs/figures/hierarchical/cm_C_oracle.png
    - outputs/figures/hierarchical/block_comparison.png
    - outputs/results/hierarchical_results_<timestamp>.json
      (or specialist_results_<timestamp>.json)

Usage:
    python carnatic_scripts/experiments/07_hierarchical_classifier.py
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --l2-model lstm
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --l2-model lstm --conf-threshold 0.6
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --conf-threshold 0.6
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --flat-fusion
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --family-only Harikāmbhōji_mela
    python carnatic_scripts/experiments/07_hierarchical_classifier.py --flat-fusion --family-only Harikāmbhōji_mela
"""
# cli argument parsing
import argparse
# json serialisation for saving results
import json
# timestamp generation for output filenames
from datetime import datetime
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
# statistical visualisation (heatmaps)
import seaborn as sns

# support vector machine classifier
from sklearn.svm import SVC
# label encoding and feature scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler
# group-aware cross-validation (no track leakage)
from sklearn.model_selection import GroupKFold
# classification evaluation metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
# chained preprocessing + model pipeline
from sklearn.pipeline import Pipeline

# progress bar for cv folds
from tqdm import tqdm
# pytorch tensor operations and gpu support
import torch
# neural network layer definitions
import torch.nn as nn
# dataset and batch loading utilities
from torch.utils.data import DataLoader, Dataset
# gradient-based optimisers
import torch.optim as optim

# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Swara vocabulary (shared with 06) ─────────────────────────────

SWARA_NAMES = ["Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2",
               "Pa", "Da1", "Da2", "Ni1", "Ni2"]
SWARA_TO_IDX = {s: i + 1 for i, s in enumerate(SWARA_NAMES)}
SWARA_TO_IDX["glide"]   = len(SWARA_NAMES) + 1
SWARA_TO_IDX["silence"] = 0
VOCAB_SIZE = len(SWARA_NAMES) + 2


# ── Melakarta family membership ───────────────────────────────────

RAGA_TO_FAMILY = {
    # Melakarta 28 (Harikāmbhōji): Sa Ri2 Ga2 Ma1 Pa Da2 Ni1
    "Harikāmbhōji":  "Harikāmbhōji_mela",
    "Kāṁbhōji":      "Harikāmbhōji_mela",
    "Mōhanaṁ":       "Harikāmbhōji_mela",
    "Bēgaḍa":        "Harikāmbhōji_mela",
    "Kamās":         "Harikāmbhōji_mela",
    # Melakarta 20 (Natabhairavi): Sa Ri1 Ga1 Ma1 Pa Da1 Ni1
    "Bhairavi":       "Natabhairavi_mela",
    "Sindhubhairavi": "Natabhairavi_mela",
    # Singletons — sole raga in their family within this dataset
    "Tōḍi":          "Subhapantuvarali_mela",
    "Kalyāṇi":       "Mechakalyani_mela",
    "Ṣanmukhapriya": "Shanmukhapriya_mela",
    "Rītigauḷa":     "Ritigaula_group",
    "Sāvēri":        "Saveri_group",
    "Jōnpuri":       "Jonpuri_group",
    "Behāg":         "Behag_group",
}


# ── Data loading ──────────────────────────────────────────────────

# load features and apply voiced-ratio and raga-count filters
def load_and_filter_data(root: Path, min_voiced_ratio: float = 0.25,
                         min_segments_per_raga: int = 20):
    feat_dir = root / "data" / "processed" / "features"
    fdata    = np.load(feat_dir / "features_all.npz", allow_pickle=True)
    meta_df  = pd.read_csv(feat_dir / "features_meta.csv")
    pitch_df = pd.read_csv(root / "data" / "processed" / "pitch_all.csv")

    meta_df = meta_df.merge(
        pitch_df[["track_id", "segment_index", "voiced_ratio"]],
        on=["track_id", "segment_index"], how="left"
    )

    mask = meta_df["voiced_ratio"] >= min_voiced_ratio
    n_before = len(meta_df)
    print(f"Segment filter (voiced ≥ {min_voiced_ratio}): {n_before} → {int(mask.sum())}")

    mask_np = mask.values
    ragas_filtered    = fdata["ragas"][mask_np]
    track_ids_filtered = meta_df.loc[mask, "track_id"].values
    meta_filtered     = meta_df[mask].reset_index(drop=True)

    raga_counts = pd.Series(ragas_filtered).value_counts()
    keep_ragas  = set(raga_counts[raga_counts >= min_segments_per_raga].index)
    dropped     = set(raga_counts.index) - keep_ragas
    if dropped:
        print(f"Dropping sparse rāgas: {sorted(dropped)}")

    raga_mask     = np.array([r in keep_ragas for r in ragas_filtered])
    ragas_final   = ragas_filtered[raga_mask]
    track_ids_final = track_ids_filtered[raga_mask]
    meta_final    = meta_filtered[raga_mask].reset_index(drop=True)

    orig_indices  = np.where(mask_np)[0]
    final_indices = orig_indices[raga_mask]

    filtered = {}
    for key in fdata.files:
        arr = fdata[key]
        if arr.ndim >= 1 and arr.shape[0] == n_before:
            filtered[key] = arr[final_indices]

    counts = pd.Series(ragas_final).value_counts()
    print(f"\nClass distribution ({len(counts)} ragas, {len(ragas_final)} segments):")
    for raga, count in counts.items():
        fam = RAGA_TO_FAMILY.get(raga, "unknown")
        print(f"  {count:5d}  {raga:<30}  [{fam}]")
    print()

    return filtered, ragas_final, track_ids_final, meta_final


# ── Swara sequence utilities ──────────────────────────────────────

# collapse consecutive identical swara labels into a deduplicated token list
def get_swara_token_sequence(swara_labels: np.ndarray) -> list:
    tokens, prev = [], None
    for label in swara_labels:
        if label == "silence":
            prev = None
            continue
        if label != prev:
            tokens.append(label)
            prev = label
    return tokens


# zero-pad variable-length token sequences to a fixed max length
def pad_sequences(sequences: list, max_len: int) -> np.ndarray:
    out = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        n = min(len(seq), max_len)
        out[i, :n] = seq[:n]
    return out


# load per-segment swara token index sequences from npz files
def load_swara_sequences(root: Path, meta_df: pd.DataFrame) -> list:
    """
    Load swara token sequences aligned with meta_df rows.
    Returns a list[list[int]] — one token index sequence per segment.
    """
    swara_csv = root / "data" / "processed" / "swara_all.csv"
    if not swara_csv.exists():
        raise FileNotFoundError(f"Missing {swara_csv}. Run 04_swara_mapping.py first.")

    swara_df = pd.read_csv(swara_csv)
    merged   = meta_df.merge(
        swara_df[["track_id", "segment_index", "swara_npz"]],
        on=["track_id", "segment_index"], how="left"
    )

    sequences = []
    for _, row in merged.iterrows():
        if pd.isna(row.get("swara_npz")):
            sequences.append([])
            continue
        npz_path = root / row["swara_npz"]
        if not npz_path.exists():
            sequences.append([])
            continue
        data   = np.load(npz_path, allow_pickle=True)
        tokens = get_swara_token_sequence(data["swara_labels"])
        sequences.append([SWARA_TO_IDX.get(t, 0) for t in tokens])

    return sequences


# ── Within-family LSTM (L2) ───────────────────────────────────────

class WithinFamilyLSTM:
    """
    Small bidirectional LSTM for within-family raga discrimination.
    Operates on swara token sequences (phrase order captured).

    Input : list of token-index sequences
    Output: global raga indices (same encoding as y throughout the CV loop)
    """

    def __init__(self, unique_global_ragas: list, device):
        self.local_to_global = list(unique_global_ragas)
        self.global_to_local = {g: i for i, g in enumerate(unique_global_ragas)}
        self.n_local = len(unique_global_ragas)
        self.device  = device
        self.model   = None
        self._max_len_fitted = 200

    def _build_model(self):
        import torch.nn as nn

        n_classes = self.n_local

        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb  = nn.Embedding(VOCAB_SIZE, 32, padding_idx=0)
                self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True,
                                    num_layers=2, dropout=0.3)
                self.fc   = nn.Linear(128, n_classes)

            def forward(self, x):
                e = self.emb(x)
                _, (h, _) = self.lstm(e)
                h = torch.cat((h[-2], h[-1]), dim=1)
                return self.fc(h)

        return _LSTM().to(self.device)

    def fit(self, sequences: list, y_global: np.ndarray,
            epochs: int = 80, patience: int = 10, batch_size: int = 32) -> bool:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        valid = [(s, g) for s, g in zip(sequences, y_global) if len(s) >= 3]
        if len(valid) < 4 or len(set(g for _, g in valid)) < 2:
            return False

        seqs_v, y_v = zip(*valid)
        seqs_v = list(seqs_v)
        y_local = np.array([self.global_to_local[g] for g in y_v])

        max_len = min(200, max(len(s) for s in seqs_v))
        self._max_len_fitted = max_len
        X = pad_sequences(seqs_v, max_len)

        X_t = torch.LongTensor(X).to(self.device)
        y_t = torch.LongTensor(y_local).to(self.device)

        counts = np.bincount(y_local, minlength=self.n_local).astype(np.float32)
        counts[counts == 0] = 1
        w = 1.0 / counts
        w = w / w.sum() * self.n_local
        w_t = torch.FloatTensor(w).to(self.device)

        self.model = self._build_model()
        opt        = torch.optim.Adam(self.model.parameters(), lr=0.001)
        crit       = nn.CrossEntropyLoss(weight=w_t)

        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        best_loss, patience_count, best_state = float("inf"), 0, None

        for _ in range(epochs):
            self.model.train()
            ep_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            ep_loss /= len(dl)

            if ep_loss < best_loss:
                best_loss = ep_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)
        return True

    def predict(self, sequences: list) -> np.ndarray:
        """Returns global raga indices."""
        import torch

        if self.model is None:
            return np.array([self.local_to_global[0]] * len(sequences))

        seqs_v = [s if s else [0] for s in sequences]
        X = pad_sequences(seqs_v, self._max_len_fitted)
        X_t = torch.LongTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            local_preds = self.model(X_t).argmax(dim=1).cpu().numpy()

        return np.array([self.local_to_global[lp] for lp in local_preds])


# ── SVM pipeline factory ──────────────────────────────────────────

# create a standardscaler + rbf svm pipeline with balanced class weights
def make_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])


# ── Track-level majority voting ───────────────────────────────────

# aggregate segment predictions to track level via majority voting
def track_level_voting(y_true_seg, y_pred_seg, groups_seg, label_names):
    df = pd.DataFrame({"y_true": y_true_seg, "y_pred": y_pred_seg, "track_id": groups_seg})
    track_true, track_pred, details = [], [], []
    for track_id, grp in df.groupby("track_id"):
        true_label  = grp["y_true"].iloc[0]
        pred_counts = grp["y_pred"].value_counts()
        pred_label  = pred_counts.index[0]
        track_true.append(true_label)
        track_pred.append(pred_label)
        details.append({
            "track_id":   track_id,
            "true_raga":  label_names[true_label],
            "pred_raga":  label_names[pred_label],
            "n_segments": len(grp),
            "correct":    true_label == pred_label,
        })
    return np.array(track_true), np.array(track_pred), details


# ── Level-2 routing (SVM or LSTM) ────────────────────────────────

# route validation segments to per-family l2 classifiers and collect predictions
def apply_l2_routing(X_l2_val, sequences_val, routed_families,
                     l2_classifiers, family_to_ragas):
    """
    Route val segments to L2 classifiers, batched by family for efficiency.
    l2_classifiers values may be SVC Pipelines or WithinFamilyLSTM instances.
      SVC:             predict(X_l2_val[indices]) → global raga indices
      WithinFamilyLSTM: predict([sequences_val[i] for i in indices]) → global raga indices
    sequences_val may be None when all L2 classifiers are SVMs.
    """
    n = len(routed_families)
    pred = np.empty(n, dtype=int)

    family_indices = {}
    for i, fam in enumerate(routed_families):
        family_indices.setdefault(fam, []).append(i)

    for fam, indices in family_indices.items():
        idx_arr   = np.array(indices)
        fam_ragas = family_to_ragas.get(fam, [])

        if len(fam_ragas) == 1:
            pred[idx_arr] = fam_ragas[0]
        elif fam in l2_classifiers:
            clf = l2_classifiers[fam]
            if isinstance(clf, WithinFamilyLSTM):
                seqs = [sequences_val[i] for i in indices]
                pred[idx_arr] = clf.predict(seqs)
            else:
                pred[idx_arr] = clf.predict(X_l2_val[idx_arr])
        elif fam_ragas:
            pred[idx_arr] = fam_ragas[0]
        else:
            pred[idx_arr] = 0

    return pred


# ── Probability fusion helper ─────────────────────────────────────

# train two svms on different features and fuse predict_proba weighted by f1
def _fuse_flat(X_a_tr, X_b_tr, X_a_v, X_b_v, y_tr):
    """
    Train two SVMs on different feature sets, fuse predict_proba() weighted by
    training F1. Returns fused class predictions for the validation set.
    """
    pipe_a = make_svm_pipeline()
    pipe_b = make_svm_pipeline()
    pipe_a.fit(X_a_tr, y_tr)
    pipe_b.fit(X_b_tr, y_tr)
    w_a = f1_score(y_tr, pipe_a.predict(X_a_tr), average="weighted", zero_division=0)
    w_b = f1_score(y_tr, pipe_b.predict(X_b_tr), average="weighted", zero_division=0)
    w_sum = max(w_a + w_b, 1e-9)
    fused = (w_a / w_sum) * pipe_a.predict_proba(X_a_v) + \
            (w_b / w_sum) * pipe_b.predict_proba(X_b_v)
    return fused.argmax(axis=1)


# ── Hierarchical GroupKFold CV ────────────────────────────────────

# run groupkfold cv producing flat, hierarchical, and oracle predictions
def run_hierarchical_cv(
    X_l1: np.ndarray,
    X_l2: np.ndarray,
    y_raga: np.ndarray,
    families: np.ndarray,
    groups: np.ndarray,
    raga_names: list,
    unique_families: list,
    family_to_ragas: dict,
    singleton_families: frozenset,
    cv_folds: int = 5,
    conf_threshold: float = 0.0,
    l2_model: str = "svm",
    sequences_all: list = None,
    device=None,
    flat_fusion: bool = False,
    X_fusion_secondary: np.ndarray = None,
):
    """
    Single GroupKFold pass producing predictions for all three blocks.

    Block A: flat SVM (or F1-weighted fusion if flat_fusion=True).
    Block B: soft routing with conf_threshold. If threshold=0 → always route.
    Block C: oracle routing for multi-raga families; flat SVM for singletons.
    """
    n = len(y_raga)
    flat_pred   = np.empty(n, dtype=int)
    hier_pred   = np.empty(n, dtype=int)
    oracle_pred = np.empty(n, dtype=int)
    l1_pred_all = np.empty(n, dtype=object)
    l1_conf_all = np.zeros(n, dtype=float)

    le_family = LabelEncoder().fit(unique_families)
    y_fam_enc = le_family.transform(families)

    actual_folds = min(cv_folds, len(np.unique(groups)))
    gkf          = GroupKFold(n_splits=actual_folds)
    fold_stats   = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(list(gkf.split(X_l1, y_raga, groups)), desc="Hierarchical CV")
    ):
        X_l1_tr, X_l1_v = X_l1[train_idx], X_l1[val_idx]
        X_l2_tr, X_l2_v = X_l2[train_idx], X_l2[val_idx]
        y_r_tr,  y_r_v  = y_raga[train_idx], y_raga[val_idx]
        y_f_tr,  y_f_v  = families[train_idx], families[val_idx]
        y_fe_tr, y_fe_v = y_fam_enc[train_idx], y_fam_enc[val_idx]

        # Block A: flat SVM or F1-weighted probability fusion
        if flat_fusion and X_fusion_secondary is not None:
            flat_pred[val_idx] = _fuse_flat(
                X_l1_tr, X_fusion_secondary[train_idx],
                X_l1_v,  X_fusion_secondary[val_idx],
                y_r_tr,
            )
        else:
            flat_pipe = make_svm_pipeline()
            flat_pipe.fit(X_l1_tr, y_r_tr)
            flat_pred[val_idx] = flat_pipe.predict(X_l1_v)

        # L1: family classifier
        l1_pipe     = make_svm_pipeline()
        l1_pipe.fit(X_l1_tr, y_fe_tr)
        l1_proba    = l1_pipe.predict_proba(X_l1_v)
        l1_enc_pred = l1_proba.argmax(axis=1)
        l1_max_prob = l1_proba.max(axis=1)
        l1_fam_pred = le_family.inverse_transform(l1_enc_pred)
        l1_pred_all[val_idx] = l1_fam_pred
        l1_conf_all[val_idx] = l1_max_prob

        # L2: per multi-raga family classifier (SVM or LSTM)
        l2_classifiers = {}
        n_lstm_trained = 0

        for fam, fam_raga_indices in family_to_ragas.items():
            if len(fam_raga_indices) <= 1:
                continue
            fam_mask_tr = y_f_tr == fam
            if fam_mask_tr.sum() < 2:
                continue
            if len(np.unique(y_r_tr[fam_mask_tr])) < 2:
                continue

            if l2_model == "lstm" and sequences_all is not None:
                tr_global_indices = train_idx[fam_mask_tr]
                seqs_tr = [sequences_all[i] for i in tr_global_indices]
                lstm = WithinFamilyLSTM(fam_raga_indices, device)
                if lstm.fit(seqs_tr, y_r_tr[fam_mask_tr]):
                    l2_classifiers[fam] = lstm
                    n_lstm_trained += 1
                else:
                    # Fallback: SVM when LSTM can't train (too little data)
                    svm = make_svm_pipeline()
                    svm.fit(X_l2_tr[fam_mask_tr], y_r_tr[fam_mask_tr])
                    l2_classifiers[fam] = svm
            else:
                svm = make_svm_pipeline()
                svm.fit(X_l2_tr[fam_mask_tr], y_r_tr[fam_mask_tr])
                l2_classifiers[fam] = svm

        # Prepare val sequences if LSTM is active
        sequences_val = None
        if l2_model == "lstm" and sequences_all is not None:
            sequences_val = [sequences_all[i] for i in val_idx]

        # Block B: soft routing
        confident = l1_max_prob >= conf_threshold
        hier_l2   = apply_l2_routing(X_l2_v, sequences_val, l1_fam_pred,
                                     l2_classifiers, family_to_ragas)
        hier_pred[val_idx] = np.where(confident, hier_l2, flat_pred[val_idx])
        n_routed = int(confident.sum())

        # Block C: oracle routing for multi-raga; flat SVM for singletons
        is_singleton_v = np.array([fam in singleton_families for fam in y_f_v])
        oracle_l2      = apply_l2_routing(X_l2_v, sequences_val, y_f_v,
                                          l2_classifiers, family_to_ragas)
        oracle_pred[val_idx] = np.where(is_singleton_v, flat_pred[val_idx], oracle_l2)

        fold_stats.append({
            "fold":       fold_idx + 1,
            "flat_acc":   float(accuracy_score(y_r_v, flat_pred[val_idx])),
            "hier_acc":   float(accuracy_score(y_r_v, hier_pred[val_idx])),
            "oracle_acc": float(accuracy_score(y_r_v, oracle_pred[val_idx])),
            "l1_acc":     float(accuracy_score(y_fe_v, l1_enc_pred)),
            "n_routed":   n_routed,
            "n_val":      len(val_idx),
            "routed_pct": round(n_routed / len(val_idx) * 100, 1),
            "n_lstm":     n_lstm_trained,
        })

    return flat_pred, hier_pred, oracle_pred, l1_pred_all, l1_conf_all, fold_stats


# ── Specialist GroupKFold CV ──────────────────────────────────────

# run groupkfold cv with a focused single-family specialist replacing only that family's predictions
def run_specialist_cv(
    X_l1: np.ndarray,
    X_l2: np.ndarray,
    y_raga: np.ndarray,
    families: np.ndarray,
    groups: np.ndarray,
    raga_names: list,
    unique_families: list,
    family_to_ragas: dict,
    focused_family: str,
    cv_folds: int = 5,
    conf_threshold: float = 0.0,
    l2_model: str = "svm",
    sequences_all: list = None,
    device=None,
    flat_fusion: bool = False,
    X_fusion_secondary: np.ndarray = None,
):
    """
    Specialist mode: flat everywhere, with a focused binary L1 + specialist L2
    replacing predictions only for segments routed to focused_family.

    Block A: flat SVM or F1-weighted fusion.
    Block B: flat + specialist for L1-predicted-focused segments (with threshold).
    Block C: flat + specialist for TRULY focused segments (oracle upper bound).
    """
    n = len(y_raga)
    flat_pred   = np.empty(n, dtype=int)
    spec_pred   = np.empty(n, dtype=int)
    oracle_pred = np.empty(n, dtype=int)
    l1_pred_all = np.empty(n, dtype=object)
    l1_conf_all = np.zeros(n, dtype=float)

    family_ragas = family_to_ragas.get(focused_family, [])

    actual_folds = min(cv_folds, len(np.unique(groups)))
    gkf          = GroupKFold(n_splits=actual_folds)
    fold_stats   = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(list(gkf.split(X_l1, y_raga, groups)),
             desc=f"Specialist CV [{focused_family}]")
    ):
        X_l1_tr, X_l1_v = X_l1[train_idx], X_l1[val_idx]
        X_l2_tr, X_l2_v = X_l2[train_idx], X_l2[val_idx]
        y_r_tr,  y_r_v  = y_raga[train_idx], y_raga[val_idx]
        y_f_tr,  y_f_v  = families[train_idx], families[val_idx]

        # Block A: flat SVM or F1-weighted probability fusion
        if flat_fusion and X_fusion_secondary is not None:
            flat_pred[val_idx] = _fuse_flat(
                X_l1_tr, X_fusion_secondary[train_idx],
                X_l1_v,  X_fusion_secondary[val_idx],
                y_r_tr,
            )
        else:
            flat_pipe = make_svm_pipeline()
            flat_pipe.fit(X_l1_tr, y_r_tr)
            flat_pred[val_idx] = flat_pipe.predict(X_l1_v)

        # Binary L1: focused_family (1) vs rest (0)
        y_bin_tr = (y_f_tr == focused_family).astype(int)
        y_bin_v  = (y_f_v  == focused_family).astype(int)

        if len(np.unique(y_bin_tr)) >= 2:
            bin_pipe  = make_svm_pipeline()
            bin_pipe.fit(X_l1_tr, y_bin_tr)
            bin_proba = bin_pipe.predict_proba(X_l1_v)
            col1      = int(np.where(bin_pipe.named_steps["svm"].classes_ == 1)[0][0])
            l1_focused_prob = bin_proba[:, col1]
        else:
            l1_focused_prob = np.zeros(len(val_idx), dtype=float)

        l1_pred_focused = l1_focused_prob >= 0.5
        l1_pred_all[val_idx] = np.where(l1_pred_focused, "focused", "rest")
        l1_conf_all[val_idx] = l1_focused_prob

        # Specialist L2: trained on focused_family training segments only
        fam_mask_tr  = y_f_tr == focused_family
        specialist_l2 = None

        if (len(family_ragas) >= 2 and fam_mask_tr.sum() >= 2
                and len(np.unique(y_r_tr[fam_mask_tr])) >= 2):
            if l2_model == "lstm" and sequences_all is not None:
                local_fam_pos = np.where(fam_mask_tr)[0]
                seqs_tr = [sequences_all[train_idx[i]] for i in local_fam_pos]
                lstm    = WithinFamilyLSTM(family_ragas, device)
                if lstm.fit(seqs_tr, y_r_tr[fam_mask_tr]):
                    specialist_l2 = lstm
                else:
                    svm = make_svm_pipeline()
                    svm.fit(X_l2_tr[fam_mask_tr], y_r_tr[fam_mask_tr])
                    specialist_l2 = svm
            else:
                svm = make_svm_pipeline()
                svm.fit(X_l2_tr[fam_mask_tr], y_r_tr[fam_mask_tr])
                specialist_l2 = svm

        def _predict_specialist(local_idx_arr):
            """Predict for local_idx_arr (positions within val arrays)."""
            if len(local_idx_arr) == 0:
                return np.array([], dtype=int)
            if specialist_l2 is None or len(family_ragas) == 0:
                return flat_pred[val_idx[local_idx_arr]]
            if len(family_ragas) == 1:
                return np.full(len(local_idx_arr), family_ragas[0])
            if isinstance(specialist_l2, WithinFamilyLSTM):
                seqs = [sequences_all[val_idx[i]] for i in local_idx_arr]
                return specialist_l2.predict(seqs)
            return specialist_l2.predict(X_l2_v[local_idx_arr])

        # Block B: route confidently-predicted-focused segments to specialist
        confident_focused = l1_pred_focused & (l1_focused_prob >= conf_threshold)
        n_routed          = int(confident_focused.sum())
        spec_arr          = flat_pred[val_idx].copy()
        routed_local      = np.where(confident_focused)[0]
        if len(routed_local) > 0:
            spec_arr[routed_local] = _predict_specialist(routed_local)
        spec_pred[val_idx] = spec_arr

        # Block C: oracle routing — truly focused segments get specialist
        oracle_arr         = flat_pred[val_idx].copy()
        true_focused_local = np.where(y_f_v == focused_family)[0]
        if len(true_focused_local) > 0:
            oracle_arr[true_focused_local] = _predict_specialist(true_focused_local)
        oracle_pred[val_idx] = oracle_arr

        bin_acc = float(accuracy_score(y_bin_v, l1_pred_focused.astype(int)))

        fold_stats.append({
            "fold":       fold_idx + 1,
            "flat_acc":   float(accuracy_score(y_r_v, flat_pred[val_idx])),
            "spec_acc":   float(accuracy_score(y_r_v, spec_arr)),
            "oracle_acc": float(accuracy_score(y_r_v, oracle_arr)),
            "l1_bin_acc": bin_acc,
            "n_routed":   n_routed,
            "n_val":      len(val_idx),
            "routed_pct": round(n_routed / len(val_idx) * 100, 1),
        })

    return flat_pred, spec_pred, oracle_pred, l1_pred_all, l1_conf_all, fold_stats


# ── Evaluation ────────────────────────────────────────────────────

# print and return segment/track accuracy, f1, and classification reports for one block
def evaluate_block(label: str, y_true, y_pred, groups, label_names):
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)

    print(f"\n{'='*60}")
    print(f"BLOCK {label}")
    print(f"{'='*60}")
    print(f"Segment accuracy:    {acc:.4f}")
    print(f"Segment F1 weighted: {f1w:.4f}")
    print(f"Segment F1 macro:    {f1m:.4f}")
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    print(report)

    trk_true, trk_pred, _ = track_level_voting(y_true, y_pred, groups, label_names)
    trk_acc = accuracy_score(trk_true, trk_pred)
    trk_f1w = f1_score(trk_true, trk_pred, average="weighted", zero_division=0)
    trk_f1m = f1_score(trk_true, trk_pred, average="macro",    zero_division=0)

    print(f"Track accuracy:      {trk_acc:.4f}")
    print(f"Track F1 weighted:   {trk_f1w:.4f}")
    print(f"Track F1 macro:      {trk_f1m:.4f}")
    trk_report = classification_report(trk_true, trk_pred,
                                        target_names=label_names, zero_division=0)
    print(trk_report)

    return {
        "seg_acc": acc, "seg_f1w": f1w, "seg_f1m": f1m,
        "trk_acc": trk_acc, "trk_f1w": trk_f1w, "trk_f1m": trk_f1m,
        "report": report, "trk_report": trk_report,
        "y_true": y_true, "y_pred": y_pred,
    }


# attribute block b errors to misrouting, within-family l2 mistakes, or fallback
def error_attribution(y_raga, hier_pred, flat_pred,
                       true_fams, pred_fams, l1_conf, conf_threshold):
    n_total    = len(y_raga)
    routed     = l1_conf >= conf_threshold
    n_routed   = int(routed.sum())
    n_fallback = n_total - n_routed
    errors     = y_raga != hier_pred
    n_errors   = int(errors.sum())

    print(f"\n{'='*60}")
    print(f"ERROR ATTRIBUTION (Block B)")
    print(f"{'='*60}")
    print(f"Total segments:           {n_total}")
    if conf_threshold > 0.0:
        print(f"Confidence threshold τ:   {conf_threshold:.2f}")
        print(f"Routed through L1/L2:     {n_routed}  ({n_routed / n_total * 100:.1f}%)")
        print(f"Fell back to flat SVM:    {n_fallback}  ({n_fallback / n_total * 100:.1f}%)")
    print(f"Total errors:             {n_errors}  ({n_errors / n_total * 100:.1f}%)")

    if n_errors == 0:
        return {"n_total": n_total, "n_errors": 0}

    fallback_errors = 0
    if conf_threshold > 0.0:
        fallback_errors = int((errors & ~routed).sum())
        print(f"\nAmong FALLBACK segments ({n_fallback}):")
        print(f"  Flat SVM errors: {fallback_errors}  "
              f"({fallback_errors / max(n_fallback, 1) * 100:.1f}% of fallback, "
              f"{fallback_errors / n_errors * 100:.1f}% of all errors)")

    misrouted         = routed & (true_fams != pred_fams)
    n_misrouted       = int(misrouted.sum())
    misroute_errors   = int((errors & misrouted).sum())
    within_fam_errors = int((errors & routed & (true_fams == pred_fams)).sum())
    routed_errors     = n_errors - fallback_errors

    print(f"\nAmong ROUTED segments ({n_routed}):")
    print(f"  Misrouting events:       {n_misrouted}  "
          f"({n_misrouted / max(n_routed, 1) * 100:.1f}% of routed segs)")
    print(f"  Errors from misrouting:  {misroute_errors}  "
          f"({misroute_errors / max(routed_errors, 1) * 100:.1f}% of routed errors)")
    print(f"  Within-family L2 errors: {within_fam_errors}  "
          f"({within_fam_errors / max(routed_errors, 1) * 100:.1f}% of routed errors)")

    return {
        "n_total": n_total,       "n_errors": n_errors,
        "n_routed": n_routed,     "n_fallback": n_fallback,
        "fallback_errors": fallback_errors,
        "n_misrouted": n_misrouted,
        "misroute_errors": misroute_errors,
        "within_family_errors": within_fam_errors,
        "pct_routed": round(n_routed / n_total * 100, 1),
        "pct_misroute_errors": round(misroute_errors / max(n_errors, 1) * 100, 1),
        "pct_within_fam_errors": round(within_fam_errors / max(n_errors, 1) * 100, 1),
    }


# print per-raga accuracy deltas (hier vs flat, oracle vs flat) for multi-raga families
def within_family_delta(y_true, flat_pred, hier_pred, oracle_pred,
                        label_names, family_to_ragas):
    print(f"\n{'='*60}")
    print(f"WITHIN-FAMILY Δ ACCURACY (multi-raga families only)")
    print(f"{'='*60}")
    rows = []
    for fam in sorted(family_to_ragas):
        fam_raga_indices = family_to_ragas[fam]
        if len(fam_raga_indices) <= 1:
            continue
        print(f"\n  Family: {fam}")
        print(f"  {'Raga':<25} {'Flat':>6} {'Hier':>6} {'Oracle':>7} {'ΔHier':>7} {'ΔOracle':>8}")
        print(f"  {'-'*67}")
        for ridx in fam_raga_indices:
            mask = y_true == ridx
            if not mask.any():
                continue
            flat_acc   = accuracy_score(y_true[mask], flat_pred[mask])
            hier_acc   = accuracy_score(y_true[mask], hier_pred[mask])
            oracle_acc = accuracy_score(y_true[mask], oracle_pred[mask])
            dh, do = hier_acc - flat_acc, oracle_acc - flat_acc
            print(f"  {label_names[ridx]:<25} {flat_acc:>6.3f} {hier_acc:>6.3f} "
                  f"{oracle_acc:>7.3f} {'+'if dh>=0 else ''}{dh:>6.3f} "
                  f"{'+'if do>=0 else ''}{do:>7.3f}")
            rows.append({
                "family": fam, "raga": label_names[ridx],
                "flat_acc":     round(float(flat_acc),   4),
                "hier_acc":     round(float(hier_acc),   4),
                "oracle_acc":   round(float(oracle_acc), 4),
                "delta_hier":   round(float(dh), 4),
                "delta_oracle": round(float(do), 4),
                "n_segments":   int(mask.sum()),
            })
    return rows


# ── Plotting ──────────────────────────────────────────────────────

# plot a row-normalised confusion matrix heatmap and save to disk
def plot_confusion_matrix(y_true, y_pred, label_names, title, out_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, xticklabels=label_names, yticklabels=label_names,
                cmap="Blues", annot=True, fmt=".2f", ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# plot grouped bar chart comparing segment and track accuracy across blocks
def plot_block_comparison(block_results: dict, out_path: Path):
    labels   = list(block_results.keys())
    seg_accs = [block_results[k]["seg_acc"] for k in labels]
    trk_accs = [block_results[k]["trk_acc"] for k in labels]
    x, width = np.arange(len(labels)), 0.35
    fig, ax  = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, seg_accs, width, label="Segment Accuracy",       color="steelblue")
    b2 = ax.bar(x + width / 2, trk_accs, width, label="Track Accuracy (vote)",  color="darkorange")
    ax.set_ylabel("Accuracy")
    ax.set_title("Hierarchical vs Flat — Block Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

# entry point: parse args, load data, run hierarchical or specialist cv, save results
def main():
    parser = argparse.ArgumentParser(description="Hierarchical raga classifier")
    parser.add_argument("--min-voiced-ratio",      type=float, default=0.25)
    parser.add_argument("--min-segments-per-raga", type=int,   default=20)
    parser.add_argument("--cv-folds",              type=int,   default=5)
    parser.add_argument("--l1-feat",               default="directional_pcd",
                        help="Feature key for L1 family classifier (default: directional_pcd)")
    parser.add_argument("--l2-feat",               default="directional_transition",
                        help="Feature key for L2 SVM (ignored when --l2-model lstm)")
    parser.add_argument("--l2-model",              default="svm", choices=["svm", "lstm"],
                        help="L2 within-family model: svm (fast) or lstm (phrase-order aware). "
                             "lstm uses swara token sequences from data/processed/swara/. "
                             "(default: svm)")
    parser.add_argument("--conf-threshold",        type=float, default=0.0,
                        help="L1 confidence threshold for routing. Segments with L1 "
                             "max_prob < threshold fall back to flat SVM. 0.0 = always route. "
                             "Try 0.55–0.70 to reduce misrouting damage. (default: 0.0)")
    parser.add_argument("--flat-fusion",           action="store_true",
                        help="Block A uses F1-weighted probability fusion of "
                             "SVM(directional_pcd) + SVM(directional_transition) "
                             "instead of a single SVM. (default: off)")
    parser.add_argument("--family-only",           default=None,
                        help="Specialist mode: flat + specialist L2 for one family. "
                             "Example: --family-only Harikāmbhōji_mela")
    parser.add_argument("--exclude-bad-tonic", action="store_true",
                        help="Exclude tracks with >50 cent tonic error (reads tonic_validation.csv)")
    args = parser.parse_args()

    root = repo_root()

    features, ragas, track_ids, meta_df = load_and_filter_data(
        root,
        min_voiced_ratio=args.min_voiced_ratio,
        min_segments_per_raga=args.min_segments_per_raga,
    )

    # exclude bad-tonic tracks (same logic as 06_train_classifiers --run-mode excl-est)
    if args.exclude_bad_tonic:
        tonic_csv = root / "data" / "processed" / "tonic_validation.csv"
        if tonic_csv.exists():
            tonic_df = pd.read_csv(tonic_csv)
            bad = tonic_df[tonic_df["within_50_cents"] == False]["track_id"].tolist()
            no_est = tonic_df[tonic_df["estimated_tonic_hz"].isna()]["track_id"].tolist()
            bad_ids = set(bad + no_est)
            keep_mask = np.array([tid not in bad_ids for tid in track_ids])
            n_excl = int((~keep_mask).sum())
            print(f"[excl-bad-tonic] Excluding {len(bad_ids)} bad-tonic tracks ({n_excl} segments)")
            for key in features:
                if features[key].shape[0] == len(ragas):
                    features[key] = features[key][keep_mask]
            ragas = ragas[keep_mask]
            track_ids = track_ids[keep_mask]
            meta_df = meta_df[keep_mask].reset_index(drop=True)
        else:
            print("Warning: tonic_validation.csv not found, skipping bad-tonic exclusion")

    if args.l1_feat not in features:
        raise KeyError(f"L1 feature '{args.l1_feat}' not found. "
                       f"Available: {[k for k in features if k != 'ragas']}")
    if args.l2_model == "svm" and args.l2_feat not in features:
        raise KeyError(f"L2 feature '{args.l2_feat}' not found. "
                       f"Available: {[k for k in features if k != 'ragas']}")

    le_raga    = LabelEncoder().fit(ragas)
    y          = le_raga.transform(ragas)
    raga_names = list(le_raga.classes_)
    groups     = track_ids

    families        = np.array([RAGA_TO_FAMILY.get(r, f"unknown_{r}") for r in ragas])
    unique_families = sorted(set(families.tolist()))
    family_to_ragas = {
        fam: sorted(set(y[families == fam].tolist()))
        for fam in unique_families
    }
    singleton_families = frozenset(
        fam for fam, ridxs in family_to_ragas.items() if len(ridxs) == 1
    )

    print(f"\nDataset: {len(y)} segments, {len(raga_names)} ragas, "
          f"{len(unique_families)} families, {len(np.unique(groups))} tracks")
    print(f"\nFamily structure:")
    for fam in sorted(family_to_ragas, key=lambda f: -len(family_to_ragas[f])):
        names = [raga_names[i] for i in family_to_ragas[fam]]
        tag   = "(multi)" if len(names) > 1 else "(singleton)"
        print(f"  {fam:<30} {tag}  {', '.join(names)}")

    # Load swara sequences for LSTM L2
    sequences_all = None
    device        = None
    if args.l2_model == "lstm":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading swara sequences for LSTM L2 (device: {device})...")
        sequences_all = load_swara_sequences(root, meta_df)
        n_valid = sum(1 for s in sequences_all if len(s) >= 3)
        print(f"  {n_valid}/{len(sequences_all)} segments have valid sequences (≥3 tokens)")

    fig_dir     = root / "outputs" / "figures" / "hierarchical"
    results_dir = root / "outputs" / "results"
    ensure_dir(fig_dir)
    ensure_dir(results_dir)

    X_l1 = features[args.l1_feat]
    X_l2 = features[args.l2_feat] if args.l2_model == "svm" else features[args.l1_feat]

    # Fusion secondary feature (directional_transition) for --flat-fusion
    X_fusion_secondary = None
    if args.flat_fusion:
        fusion_feat = "directional_transition"
        if fusion_feat not in features:
            raise KeyError(f"'{fusion_feat}' not in features; required for --flat-fusion. "
                           f"Available: {[k for k in features if k != 'ragas']}")
        X_fusion_secondary = features[fusion_feat]
        print(f"\nFlat fusion: SVM({args.l1_feat}) + SVM({fusion_feat}), "
              f"weighted by training F1")

    routing_mode = (f"soft (τ={args.conf_threshold:.2f})"
                    if args.conf_threshold > 0 else "hard (no fallback)")
    l2_desc = (f"LSTM (swara sequences)" if args.l2_model == "lstm"
               else f"SVM ({args.l2_feat}, {X_l2.shape[1]} dims)")

    print(f"\nL1 feature:    {args.l1_feat} ({X_l1.shape[1]} dims)")
    print(f"L2 model:      {l2_desc}")
    print(f"CV folds:      {args.cv_folds}")
    print(f"Routing mode:  {routing_mode}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Specialist mode ───────────────────────────────────────────
    if args.family_only:
        focused_family = args.family_only
        if focused_family not in family_to_ragas:
            raise ValueError(
                f"Family '{focused_family}' not found. "
                f"Available families: {sorted(family_to_ragas)}"
            )
        fam_raga_names = [raga_names[i] for i in family_to_ragas[focused_family]]
        print(f"\nSpecialist mode: focused family = {focused_family}")
        print(f"Family ragas ({len(fam_raga_names)}): {', '.join(fam_raga_names)}")

        flat_pred, spec_pred, oracle_pred, l1_pred_all, l1_conf_all, fold_stats = (
            run_specialist_cv(
                X_l1, X_l2, y, families, groups,
                raga_names, unique_families, family_to_ragas,
                focused_family=focused_family,
                cv_folds=args.cv_folds,
                conf_threshold=args.conf_threshold,
                l2_model=args.l2_model,
                sequences_all=sequences_all,
                device=device,
                flat_fusion=args.flat_fusion,
                X_fusion_secondary=X_fusion_secondary,
            )
        )

        flat_tag    = "fusion" if args.flat_fusion else f"SVM ({args.l1_feat})"
        block_a_lbl = f"A — Flat {flat_tag}"
        block_b_lbl = (f"B — Specialist [{focused_family}] soft τ={args.conf_threshold:.2f}"
                       if args.conf_threshold > 0
                       else f"B — Specialist [{focused_family}] hard")
        block_c_lbl = f"C — Oracle [{focused_family}] + flat (rest)"

        block_A = evaluate_block(block_a_lbl, y, flat_pred,   groups, raga_names)
        block_B = evaluate_block(block_b_lbl, y, spec_pred,   groups, raga_names)
        block_C = evaluate_block(block_c_lbl, y, oracle_pred, groups, raga_names)

        block_results = {
            f"A: Flat {flat_tag}": block_A,
            "B: Specialist":       block_B,
            "C: Oracle":           block_C,
        }

        delta_rows = within_family_delta(
            y, flat_pred, spec_pred, oracle_pred, raga_names,
            {focused_family: family_to_ragas[focused_family]}
        )

        plot_confusion_matrix(
            y, flat_pred, raga_names,
            f"Block A — Flat {flat_tag}\nSeg Acc: {block_A['seg_acc']:.3f}",
            fig_dir / "cm_A_flat.png"
        )
        plot_confusion_matrix(
            y, spec_pred, raga_names,
            f"Block B — Specialist [{focused_family}] | L2: {args.l2_model}\n"
            f"Seg Acc: {block_B['seg_acc']:.3f}",
            fig_dir / "cm_B_specialist.png"
        )
        plot_confusion_matrix(
            y, oracle_pred, raga_names,
            f"Block C — Oracle [{focused_family}] + flat (rest)\n"
            f"L2: {args.l2_model} | Seg Acc: {block_C['seg_acc']:.3f}",
            fig_dir / "cm_C_oracle.png"
        )
        plot_block_comparison(block_results, fig_dir / "block_comparison.png")

        print(f"\n\n{'='*70}")
        print(f"SPECIALIST SUMMARY  [family: {focused_family}, L2: {args.l2_model}]")
        print(f"{'='*70}")
        print(f"{'Block':<45} {'Seg Acc':>8} {'Seg F1w':>8} {'Trk Acc':>8} {'Trk F1w':>8}")
        print(f"{'-'*77}")
        for blabel, res in block_results.items():
            print(f"{blabel:<45} {res['seg_acc']:>8.4f} {res['seg_f1w']:>8.4f} "
                  f"{res['trk_acc']:>8.4f} {res['trk_f1w']:>8.4f}")

        print(f"\nFold breakdown:")
        for fs in fold_stats:
            print(f"  Fold {fs['fold']}: flat={fs['flat_acc']:.4f}  "
                  f"spec={fs['spec_acc']:.4f}  oracle={fs['oracle_acc']:.4f}  "
                  f"L1bin={fs['l1_bin_acc']:.4f}  routed={fs['routed_pct']:.1f}%")

        summary = {
            "run_timestamp": timestamp,
            "mode": "specialist",
            "config": {
                "focused_family":        focused_family,
                "flat_fusion":           args.flat_fusion,
                "l1_feature":            args.l1_feat,
                "l2_model":              args.l2_model,
                "l2_feature":            args.l2_feat if args.l2_model == "svm" else "swara_sequences",
                "cv_folds":              args.cv_folds,
                "conf_threshold":        args.conf_threshold,
                "routing_mode":          routing_mode,
                "min_voiced_ratio":      args.min_voiced_ratio,
                "min_segments_per_raga": args.min_segments_per_raga,
            },
            "dataset": {
                "n_segments":  int(len(y)),
                "n_ragas":     len(raga_names),
                "n_families":  len(unique_families),
                "n_tracks":    int(len(np.unique(groups))),
                "focused_family_ragas": fam_raga_names,
            },
            "results": {
                "block_A": {k: float(v) for k, v in block_A.items()
                            if isinstance(v, (float, int, np.floating, np.integer))},
                "block_B": {k: float(v) for k, v in block_B.items()
                            if isinstance(v, (float, int, np.floating, np.integer))},
                "block_C": {k: float(v) for k, v in block_C.items()
                            if isinstance(v, (float, int, np.floating, np.integer))},
            },
            "within_family_delta": delta_rows,
            "fold_stats": fold_stats,
        }

        out_json = results_dir / f"specialist_results_{timestamp}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResults JSON: {out_json}")
        print(f"Confusion matrices: {fig_dir}")
        return

    # ── Full hierarchical mode ────────────────────────────────────
    flat_pred, hier_pred, oracle_pred, l1_pred_all, l1_conf_all, fold_stats = (
        run_hierarchical_cv(
            X_l1, X_l2, y, families, groups,
            raga_names, unique_families, family_to_ragas,
            singleton_families=singleton_families,
            cv_folds=args.cv_folds,
            conf_threshold=args.conf_threshold,
            l2_model=args.l2_model,
            sequences_all=sequences_all,
            device=device,
            flat_fusion=args.flat_fusion,
            X_fusion_secondary=X_fusion_secondary,
        )
    )

    flat_tag    = "fusion" if args.flat_fusion else f"SVM ({args.l1_feat})"
    block_b_label = (f"B — Hier. soft (τ={args.conf_threshold:.2f})"
                     if args.conf_threshold > 0 else "B — Hierarchical hard routing")
    block_c_label = "C — Oracle (multi-raga) + flat (singletons)"

    block_A = evaluate_block(f"A — Flat {flat_tag}",
                              y, flat_pred,   groups, raga_names)
    block_B = evaluate_block(block_b_label,
                              y, hier_pred,   groups, raga_names)
    block_C = evaluate_block(block_c_label,
                              y, oracle_pred, groups, raga_names)

    block_results = {
        f"A: Flat {flat_tag}": block_A,
        "B: Hierarchical":     block_B,
        "C: Oracle":           block_C,
    }

    attr = error_attribution(
        y, hier_pred, flat_pred, families, l1_pred_all, l1_conf_all,
        conf_threshold=args.conf_threshold
    )
    delta_rows = within_family_delta(
        y, flat_pred, hier_pred, oracle_pred, raga_names, family_to_ragas
    )

    plot_confusion_matrix(
        y, flat_pred, raga_names,
        f"Block A — Flat {flat_tag}\nSeg Acc: {block_A['seg_acc']:.3f}",
        fig_dir / "cm_A_flat.png"
    )
    plot_confusion_matrix(
        y, hier_pred, raga_names,
        f"Block B — Hierarchical {routing_mode} | L2: {args.l2_model}\n"
        f"Seg Acc: {block_B['seg_acc']:.3f}",
        fig_dir / "cm_B_hierarchical.png"
    )
    plot_confusion_matrix(
        y, oracle_pred, raga_names,
        f"Block C — Oracle (multi-raga) + Flat (singletons)\n"
        f"L2: {args.l2_model} | Seg Acc: {block_C['seg_acc']:.3f}",
        fig_dir / "cm_C_oracle.png"
    )
    plot_block_comparison(block_results, fig_dir / "block_comparison.png")

    print(f"\n\n{'='*70}")
    print(f"SUMMARY TABLE  [L2: {args.l2_model}  |  Flat: {flat_tag}]")
    print(f"{'='*70}")
    print(f"{'Block':<40} {'Seg Acc':>8} {'Seg F1w':>8} {'Trk Acc':>8} {'Trk F1w':>8}")
    print(f"{'-'*72}")
    for blabel, res in block_results.items():
        print(f"{blabel:<40} {res['seg_acc']:>8.4f} {res['seg_f1w']:>8.4f} "
              f"{res['trk_acc']:>8.4f} {res['trk_f1w']:>8.4f}")

    print(f"\nFold accuracy breakdown:")
    hdr = f"  {'Fold':>5} {'Flat':>8} {'Hier':>8} {'Oracle':>8} {'L1 Acc':>8}"
    if args.conf_threshold > 0:
        hdr += f" {'Routed%':>9}"
    print(hdr)
    for fs in fold_stats:
        line = (f"  {fs['fold']:>5} {fs['flat_acc']:>8.4f} {fs['hier_acc']:>8.4f} "
                f"{fs['oracle_acc']:>8.4f} {fs['l1_acc']:>8.4f}")
        if args.conf_threshold > 0:
            line += f"  {fs['routed_pct']:>7.1f}%"
        print(line)

    summary = {
        "run_timestamp": timestamp,
        "mode": "hierarchical",
        "config": {
            "flat_fusion":       args.flat_fusion,
            "l1_feature":        args.l1_feat,
            "l2_model":          args.l2_model,
            "l2_feature":        args.l2_feat if args.l2_model == "svm" else "swara_sequences",
            "cv_folds":          args.cv_folds,
            "conf_threshold":    args.conf_threshold,
            "routing_mode":      routing_mode,
            "svm_C": 1.0, "svm_gamma": "scale",
            "min_voiced_ratio":      args.min_voiced_ratio,
            "min_segments_per_raga": args.min_segments_per_raga,
        },
        "dataset": {
            "n_segments": int(len(y)),
            "n_ragas":    len(raga_names),
            "n_families": len(unique_families),
            "n_tracks":   int(len(np.unique(groups))),
            "family_structure": {
                fam: [raga_names[i] for i in ridxs]
                for fam, ridxs in family_to_ragas.items()
            },
        },
        "results": {
            "block_A": {k: float(v) for k, v in block_A.items()
                        if isinstance(v, (float, int, np.floating, np.integer))},
            "block_B": {k: float(v) for k, v in block_B.items()
                        if isinstance(v, (float, int, np.floating, np.integer))},
            "block_C": {k: float(v) for k, v in block_C.items()
                        if isinstance(v, (float, int, np.floating, np.integer))},
        },
        "error_attribution": attr,
        "within_family_delta": delta_rows,
        "fold_stats": fold_stats,
    }

    out_json = results_dir / f"hierarchical_results_{timestamp}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults JSON: {out_json}")
    print(f"Confusion matrices: {fig_dir}")


if __name__ == "__main__":
    main()
