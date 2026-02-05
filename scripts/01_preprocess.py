import os
import csv
import yaml
from pathlib import Path

import librosa
import soundfile as sf
import mirdata
from tqdm import tqdm


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main():
    root = repo_root()

    # --- Paths
    cfg = load_cfg(str(root / "configs" / "preprocessing.yaml"))

    dataset_name = cfg["dataset_name"]
    target_sr = int(cfg["sample_rate"])
    seg_s = float(cfg["segment_seconds"])
    hop_s = float(cfg["hop_seconds"])
    max_tracks = int(cfg["max_tracks"])
    trim_silence = bool(cfg.get("trim_silence", False))

    mirdata_home = root / "data" / "raw" / "mirdata"
    os.environ["MIRDATA_HOME"] = str(mirdata_home)

    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    ensure_dirs(interim_dir, processed_dir)

    out_csv = processed_dir / f"segments_{dataset_name}.csv"

    print("Dataset:", dataset_name)
    print("MIRDATA_HOME:", os.environ["MIRDATA_HOME"])
    print("Writing segments to:", interim_dir)
    print("Writing metadata to:", out_csv)

    # --- Load dataset
    ds = mirdata.initialize(dataset_name)
    track_ids = ds.track_ids[:max_tracks]

    rows = []
    seg_len = int(seg_s * target_sr)
    hop_len = int(hop_s * target_sr)

    for tid in tqdm(track_ids, desc="Tracks"):
        tr = ds.track(tid)

        # mirdata returns (audio, sr). We resample to target_sr for consistency.
        y, sr = tr.audio

        # Ensure float mono
        if y.ndim > 1:
            y = librosa.to_mono(y)
        y = y.astype("float32")

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=30)

        if len(y) < seg_len:
            continue  # skip too-short tracks

        # Segment with fixed windows
        seg_idx = 0
        for start in range(0, len(y) - seg_len + 1, hop_len):
            end = start + seg_len
            seg = y[start:end]

            # File name safe
            seg_name = f"{dataset_name}__{tid}__{seg_idx:05d}.wav"
            seg_path = interim_dir / seg_name

            # Write wav
            sf.write(seg_path, seg, target_sr)

            rows.append(
                {
                    "dataset": dataset_name,
                    "track_id": tid,
                    "segment_index": seg_idx,
                    "start_s": start / target_sr,
                    "end_s": end / target_sr,
                    "sample_rate": target_sr,
                    "path": str(seg_path.relative_to(root)).replace("\\", "/"),
                }
            )
            seg_idx += 1

    # --- Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "track_id",
                "segment_index",
                "start_s",
                "end_s",
                "sample_rate",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} segments.")
    print("Metadata CSV:", out_csv)


if __name__ == "__main__":
    main()
