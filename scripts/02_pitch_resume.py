"""
02_pitch_resume.py — Resume pitch extraction from where it stopped.
Only processes segments that don't already have a .npz file.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    root = repo_root()

    pitch_cfg_path = root / "configs" / "pitch.yaml"
    if pitch_cfg_path.exists():
        pcfg = load_cfg(str(pitch_cfg_path))
    else:
        pcfg = {"hop_length": 256, "fmin_note": "C2", "fmax_note": "C7"}

    hop_length = int(pcfg.get("hop_length", 256))
    fmin = librosa.note_to_hz(pcfg.get("fmin_note", "C2"))
    fmax = librosa.note_to_hz(pcfg.get("fmax_note", "C7"))

    segments_csv = root / "data" / "processed" / "segments_saraga_carnatic.csv"
    df = pd.read_csv(segments_csv)

    pitch_dir = root / "data" / "processed" / "pitch"
    sr_target = int(df["sample_rate"].iloc[0])

    # Find which segments are missing
    existing = {p.stem for p in pitch_dir.glob("*.npz")}
    
    missing = []
    for _, row in df.iterrows():
        out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
        if out_base not in existing:
            missing.append(row)

    print(f"Total segments: {len(df)}")
    print(f"Already done: {len(existing)}")
    print(f"Remaining: {len(missing)}")

    if len(missing) == 0:
        print("All done! Nothing to process.")
        # Just rebuild the CSV from all npz files
    else:
        import torchcrepe
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        for row in tqdm(missing, desc="Resuming pitch extraction"):
            seg_path = root / row["path"]
            y, sr = librosa.load(str(seg_path), sr=sr_target, mono=True)

            audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            pitch, periodicity = torchcrepe.predict(
                audio_tensor, sr, hop_length=hop_length,
                fmin=fmin, fmax=fmax, model="full",
                return_periodicity=True, batch_size=2048, device=device,
            )

            f0 = pitch.squeeze(0).cpu().numpy()
            confidence = periodicity.squeeze(0).cpu().numpy()
            voiced_flag = confidence >= 0.5
            f0_masked = f0.copy()
            f0_masked[~voiced_flag] = np.nan

            out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
            npy_path = pitch_dir / f"{out_base}.npz"
            np.savez_compressed(
                npy_path, f0=f0_masked, voiced_flag=voiced_flag,
                voiced_prob=confidence, sr=sr, hop_length=hop_length, method="crepe",
            )

    # Rebuild pitch_all.csv from ALL npz files
    print("\nRebuilding pitch_all.csv...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building CSV"):
        out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
        npy_path = pitch_dir / f"{out_base}.npz"

        if not npy_path.exists():
            continue

        data = np.load(npy_path)
        f0 = data["f0"]
        voiced_flag = data["voiced_flag"]

        valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]
        voiced_ratio = float(voiced_flag.mean())
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
            "method": "crepe",
        })

    out_csv = root / "data" / "processed" / "pitch_all.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Done. {len(results)} segments in {out_csv}")


if __name__ == "__main__":
    main()