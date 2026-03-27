import os, sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
02_pitch_extraction.py -- Stage 2: Pitch Tracking (torchcrepe, GPU)

Uses CREPE (deep learning pitch tracker) on GPU for fast extraction.
Falls back to librosa.pyin on CPU if CUDA is unavailable.

Audio segments from preprocessing are taken as input.
Pitch contour (F0 over time) is extracted per segment.
Voicing/confidence scores indicate presence of melody.
torchcrepe used for GPU acceleration; falls back to librosa pyin if GPU not present.
Each segment gets a compressed .npz file with pitch arrays and metadata.
Writes a summary CSV with one row per segment including track ID, raga label,
path to pitch file, and voiced ratio.
Generates diagnostic plots showing extracted pitch curve and confidence over time.

OUTPUTS:
    - data/processed/pitch/pitch_all.csv  # Summary of all segments + pitch stats
    - data/processed/pitch/*.npz          # Per-segment pitch arrays (f0, voiced_prob, etc)
    - outputs/figures/pitch_debug/*.png   # Diagnostic plots for first N segments   
    
Commands:
    python hindustani_scripts/02_pitch_extraction.py                          # full dataset, GPU
    python hindustani_scripts/02_pitch_extraction.py --max-per-raga 10        # quick dev run
    python hindustani_scripts/02_pitch_extraction.py --method pyin            # force pyin (CPU)
    python hindustani_scripts/02_pitch_extraction.py --plot-n 5               # diagnostic plots
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
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

    # torchcrepe expects a batch of audio tensors
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

    # hop_length in samples → step_size in seconds isn't needed;
    # torchcrepe uses hop_length directly
    # Model sizes: 'tiny', 'small', 'medium', 'large', 'full'
    # 'full' is most accurate, still fast on GPU

    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model="full",
        return_periodicity=True,
        batch_size=2048,
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
    if args.method == "crepe" and HAS_TORCH and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CREPE on GPU: {torch.cuda.get_device_name(0)}")
    elif args.method == "crepe":
        print("WARNING: CUDA/torch not available, falling back to pyin (CPU)")
        args.method = "pyin"
        device = "cpu"
    else:
        device = "cpu"
        print("Using pyin (CPU)")

    # --- Input
    segments_csv = root / "data" / "hindustani" / "processed" / "segments_saraga_hindustani.csv"
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
    pitch_dir = root / "data" / "hindustani" / "processed" / "pitch"
    fig_dir = root / "outputs" / "hindustani" / "figures" / "pitch_debug"
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
    out_csv = root / "data" / "hindustani" / "processed" / "pitch_all.csv"
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