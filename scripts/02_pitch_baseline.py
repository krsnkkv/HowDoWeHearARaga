import os
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_pitch(y, sr, f0, voiced_prob, out_png: Path, title: str):
    # Time axes
    t_audio = np.arange(len(y)) / sr
    hop_length = 256  # must match pyin below
    t_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    plt.figure(figsize=(10, 4))
    plt.plot(t_audio, y, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Pitch + voiced prob
    plt.figure(figsize=(10, 4))
    # f0 has NaNs for unvoiced frames; matplotlib will break the line nicely.
    plt.plot(t_f0, f0, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(title + " | F0 (pyin)")
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_f0.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(t_f0, voiced_prob, linewidth=1.0)
    plt.ylim(0, 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Voiced prob")
    plt.title(title + " | Voiced probability")
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_voicedprob.png"), dpi=150)
    plt.close()


def main():
    root = repo_root()

    # Input produced by Week 1
    segments_csv = root / "data" / "processed" / "segments_saraga_carnatic.csv"
    if not segments_csv.exists():
        raise FileNotFoundError(f"Missing: {segments_csv}. Run scripts/01_preprocess.py first.")

    df = pd.read_csv(segments_csv)

    # Output dirs (these are large-ish, keep them gitignored)
    pitch_dir = root / "data" / "processed" / "pitch"
    fig_dir = root / "outputs" / "figures" / "pitch_debug"
    ensure_dir(pitch_dir)
    ensure_dir(fig_dir)

    # Limit for tonight (manual validation set)
    N = 20
    df_small = df.head(N).copy()

    results = []

    # pyin settings (reasonable defaults)
    sr_target = int(df_small["sample_rate"].iloc[0])
    hop_length = 256
    fmin = librosa.note_to_hz("C2")   # ~65 Hz
    fmax = librosa.note_to_hz("C7")   # ~2093 Hz

    print("Reading segments from:", segments_csv)
    print("Writing pitch arrays to:", pitch_dir)
    print("Writing plots to:", fig_dir)
    print("Processing N =", N)

    for i, row in tqdm(df_small.iterrows(), total=len(df_small), desc="Segments"):
        seg_path = root / row["path"]
        y, sr = librosa.load(seg_path, sr=sr_target, mono=True)

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )

        # Save arrays
        out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
        npy_path = pitch_dir / f"{out_base}.npz"
        np.savez_compressed(
            npy_path,
            f0=f0,
            voiced_flag=voiced_flag,
            voiced_prob=voiced_prob,
            sr=sr,
            hop_length=hop_length,
        )

        # Simple summary stats
        voiced_ratio = float(np.nanmean(voiced_flag.astype(float)))
        f0_median = float(np.nanmedian(f0)) if np.any(~np.isnan(f0)) else float("nan")

        results.append(
            {
                "track_id": row["track_id"],
                "segment_index": int(row["segment_index"]),
                "segment_path": row["path"],
                "pitch_npz": str(npy_path.relative_to(root)).replace("\\", "/"),
                "voiced_ratio": voiced_ratio,
                "f0_median_hz": f0_median,
            }
        )

        # Plot diagnostics
        title = f"{row['track_id']} seg {int(row['segment_index'])}"
        out_png = fig_dir / f"{out_base}.png"
        plot_pitch(y, sr, f0, voiced_prob, out_png, title)

    out_csv = root / "data" / "processed" / "pitch_baseline_20.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print("\nDone.")
    print("Pitch summary CSV:", out_csv)
    print("Example plots folder:", fig_dir)


if __name__ == "__main__":
    main()
