"""
02_pitch_extraction.py — Stage 2: Pitch Tracking (torchcrepe, GPU)

WHAT THIS SCRIPT DOES:
    Extracts fundamental frequency (F0) contours from each audio segment using
    CREPE, a deep-learning pitch tracker, with GPU acceleration via torchcrepe.
    Falls back to librosa's pYIN on CPU when CUDA is unavailable. For each
    segment, outputs a compressed .npz file containing the F0 array, confidence
    scores, and voicing flags. Also writes a summary CSV (pitch_all.csv) with
    one row per segment recording track ID, raga label, path to the pitch file,
    voiced ratio, and median F0. Generates diagnostic plots overlaying the
    extracted pitch curve and confidence over time for visual sanity checking.

LOGIC AND DESIGN:
    CREPE was selected over pYIN because it achieves substantially lower raw
    pitch error on non-Western vocal music, where heavy gamaka ornamentation
    causes rapid pitch variation that challenges autocorrelation-based methods.
    The hop length of 256 samples (~11.6 ms at 22,050 Hz) provides sufficient
    temporal resolution to capture gamaka contours. A confidence threshold of
    0.5 (CREPE's recommended default) masks low-confidence frames as unvoiced
    (NaN), preventing downstream swara mapping from assigning labels to noise.
    Batch GPU processing (2,048 frames per batch) enables efficient extraction
    across the full dataset.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    This stage corresponds to the auditory periphery: the cochlea performs
    frequency analysis on incoming sound, and pitch tracking extracts the
    melodic contour that listeners perceive. The confidence threshold mirrors
    the perceptual phenomenon that listeners do not assign pitch to noise or
    unvoiced sounds. Preserving the continuous F0 contour (rather than
    immediately discretising) retains gamaka detail for downstream analysis—
    gamakas are the continuous pitch movements that give Carnatic melody its
    characteristic expressiveness (Krishnaswamy, 2003).

OUTPUTS:
    - data/processed/pitch/pitch_all.csv  # Summary of all segments + pitch stats
    - data/processed/pitch/*.npz          # Per-segment pitch arrays
    - outputs/figures/pitch_debug/*.png   # Diagnostic plots for first N segments

Usage:
    python carnatic_scripts/02_pitch_extraction.py
    python carnatic_scripts/02_pitch_extraction.py --method pyin
    python carnatic_scripts/02_pitch_extraction.py --plot-n 5
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

#Repo Roo finder
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


#Makes sure directory exists
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

#Loading YAML for config values
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# TorchCREPE - Estimation of pitch and periodicity; converts output tensors to numpy arrays 
# Masks the F0 with confidence greater than or equal to 0.5
def extract_pitch_crepe(y: np.ndarray, sr: int, hop_length: int = 256,
                        fmin: float = 65.0, fmax: float = 2100.0,
                        device: str = "cuda"):
    """
    Extract pitch using torchcrepe on GPU.
    Returns: f0 (Hz, NaN for unvoiced), voiced_flag, confidence
    """
    import torchcrepe

    # Convert audio to GPU tensor
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device) 

    # hop_length in samples → step_size in seconds isn't needed;
    # torchcrepe uses hop_length directly
    # Model sizes: 'tiny', 'small', 'medium', 'large', 'full'
    # 'full' is most accurate, still fast on GPU

    pitch, periodicity = torchcrepe.predict(
    audio_tensor, sr,
    hop_length=256,      # Extract pitch every 256 samples (~11.6ms at 22050Hz = ~86 frames/sec)
    fmin=65, fmax=2100,  # C2 to C7 — covers full vocal range
    model="full",        # Most accurate CREPE model
    return_periodicity=True,  # Also give confidence scores
    batch_size=2048,     # Process 2048 frames at once on GPU
    device=device,
)


    # Move to numpy
    f0 = pitch.squeeze(0).cpu().numpy()
    confidence = periodicity.squeeze(0).cpu().numpy()

    # Apply voiced threshold: low confidence → unvoiced (NaN)
    voiced_threshold = 0.5
    voiced_flag = confidence >= voiced_threshold
    f0_masked = f0.copy()
    f0_masked[~voiced_flag] = np.nan

    return f0_masked, voiced_flag.astype(np.bool_), confidence

# If GPU not present, then using librsoa ; a baseline pitch tracker for comparison and reproducibility.
def extract_pitch_pyin(y: np.ndarray, sr: int, hop_length: int = 256,
                       fmin: float = 65.0, fmax: float = 2100.0):
    """Fallback: extract pitch using librosa pyin (CPU)."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    return f0, voiced_flag, voiced_prob

# Visual sanity check for the verification of how the pitch contours look
def plot_pitch_diagnostic(root, row, hop_length, sr_target, fig_dir):
    """Generate a diagnostic plot for one segment."""
    npz_path = root / row["pitch_npz"]
    data = np.load(npz_path)
    f0 = data["f0"]
    voiced_prob = data["voiced_prob"]
    sr = int(data["sr"])

    t_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t_f0, f0, linewidth=0.8, color="steelblue")
    ax1.set_ylabel("F0 (Hz)")
    ax1.set_title(f"{row['track_id']} seg {row['segment_index']} | {row['raga']}")

    ax2.fill_between(t_f0, voiced_prob, alpha=0.5, color="orange")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
    plt.savefig(fig_dir / f"{out_base}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Extract pitch from segments")
    parser.add_argument("--plot-n", type=int, default=20,
                        help="Diagnostic plots for first N segments")
    parser.add_argument("--max-per-raga", type=int, default=None,
                        help="Max segments per raga (for quick dev runs)")
    parser.add_argument("--method", choices=["crepe", "pyin"], default="crepe",
                        help="Pitch extraction method (default: crepe)")
    args = parser.parse_args()

    root = repo_root()

    # --- Config
    pitch_cfg_path = root / "configs" / "pitch.yaml"
    if pitch_cfg_path.exists():
        pcfg = load_cfg(str(pitch_cfg_path))
    else:
        pcfg = {"hop_length": 256, "fmin_note": "C2", "fmax_note": "C7"}

    hop_length = int(pcfg.get("hop_length", 256))
    fmin = librosa.note_to_hz(pcfg.get("fmin_note", "C2"))
    fmax = librosa.note_to_hz(pcfg.get("fmax_note", "C7"))

    # --- Device setup
    if args.method == "crepe" and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CREPE on GPU: {torch.cuda.get_device_name(0)}")
    elif args.method == "crepe":
        print("WARNING: CUDA not available, falling back to pyin (CPU)")
        args.method = "pyin"
        device = "cpu"
    else:
        device = "cpu"
        print("Using pyin (CPU)")

    # --- Input
    segments_csv = root / "data" / "processed" / "segments_saraga_carnatic.csv"
    if not segments_csv.exists():
        raise FileNotFoundError(f"Missing: {segments_csv}. Run 01_preprocess.py first.")

    df = pd.read_csv(segments_csv)

    # --- Optional: limit segments per raga
    if args.max_per_raga is not None:
        df = df.groupby("raga").head(args.max_per_raga).reset_index(drop=True)
        print(f"Limited to {args.max_per_raga} segments per raga -> {len(df)} segments")
    else:
        print(f"Loaded {len(df)} segments (full dataset)")

    # --- Output dirs
    pitch_dir = root / "data" / "processed" / "pitch"
    fig_dir = root / "outputs" / "figures" / "pitch_debug"
    ensure_dir(pitch_dir)
    ensure_dir(fig_dir)

    sr_target = int(df["sample_rate"].iloc[0])

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Pitch extraction"):
        seg_path = root / row["path"]
        y, sr = librosa.load(str(seg_path), sr=sr_target, mono=True)

        if args.method == "crepe":
            f0, voiced_flag, confidence = extract_pitch_crepe(
                y, sr, hop_length=hop_length, fmin=fmin, fmax=fmax, device=device
            )
        else:
            f0, voiced_flag, confidence = extract_pitch_pyin(
                y, sr, hop_length=hop_length, fmin=fmin, fmax=fmax
            )

        # Save arrays (same format regardless of method)
        out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
        npy_path = pitch_dir / f"{out_base}.npz"
        np.savez_compressed(
            npy_path,
            f0=f0,
            voiced_flag=voiced_flag,
            voiced_prob=confidence,  # keep same key name for compatibility
            sr=sr,
            hop_length=hop_length,
            method=args.method,
        )

        # Summary stats
        valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]
        voiced_ratio = float(voiced_flag.mean()) if voiced_flag is not None else 0.0
        f0_median = float(np.median(valid_f0)) if valid_f0.size > 0 else float("nan")

        results.append({
            "track_id": row["track_id"],
            "raga": row["raga"],
            "gt_tonic_hz": row["gt_tonic_hz"],
            "segment_index": int(row["segment_index"]),
            "segment_path": row["path"],
            "pitch_npz": str(npy_path.relative_to(root)).replace("\\", "/"),
            "voiced_ratio": voiced_ratio,
            "f0_median_hz": f0_median,
            "method": args.method,
        })

    # --- Write summary CSV
    out_csv = root / "data" / "processed" / "pitch_all.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    # --- Diagnostic plots
    if args.plot_n > 0:
        print(f"Generating {min(args.plot_n, len(res_df))} diagnostic plots...")
        for i in range(min(args.plot_n, len(res_df))):
            row = res_df.iloc[i]
            plot_pitch_diagnostic(root, row, hop_length, sr_target, fig_dir)

    # --- Summary
    print(f"\nDone. Processed {len(results)} segments using {args.method}.")
    print(f"Pitch CSV: {out_csv}")
    print(f"Voiced ratio — mean: {res_df['voiced_ratio'].mean():.3f}, "
          f"median: {res_df['voiced_ratio'].median():.3f}")


if __name__ == "__main__":
    main()


    ##Updated