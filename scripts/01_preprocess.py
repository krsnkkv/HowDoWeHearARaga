"""
01_preprocess.py — Stage 1: Preprocessing

Loads Saraga Carnatic tracks via mirdata, filters to tracks with
raga labels + tonic annotations, segments audio, and writes metadata CSV.

Usage:
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --min-tracks-per-raga 5   # stricter filter
"""
import os
import csv
import yaml
import argparse
from pathlib import Path
from collections import Counter

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


def get_usable_tracks(ds, min_tracks_per_raga: int = 3):
    """
    Filter to tracks that have:
      - raga label (non-empty)
      - tonic annotation
      - audio file
    Then filter to ragas with >= min_tracks_per_raga examples.
    Excludes Ragamalika (multi-raga compositions).
    """
    candidates = []

    for tid in ds.track_ids:
        tr = ds.track(tid)

        # Must have metadata with raga
        raaga = tr.metadata.get("raaga", []) if tr.metadata else []
        if not raaga or len(raaga) == 0:
            continue

        # Extract raga name
        r = raaga[0]
        raga_name = r.get("name", str(r)) if isinstance(r, dict) else str(r)

        # Skip Ragamalika (multi-raga)
        if "gam" in raga_name.lower() and "lika" in raga_name.lower():
            continue
        if raga_name == "Rāgamālika":
            continue

        # Must have tonic
        if tr.tonic is None:
            continue

        # Must have audio
        if not tr.audio_path or not Path(tr.audio_path).exists():
            continue

        candidates.append({
            "track_id": tid,
            "raga": raga_name,
            "tonic_hz": tr.tonic,
            "audio_path": tr.audio_path,
            "has_pitch_annotation": tr.pitch is not None,
        })

    # Filter to ragas with enough tracks
    raga_counts = Counter(c["raga"] for c in candidates)
    valid_ragas = {r for r, count in raga_counts.items() if count >= min_tracks_per_raga}

    usable = [c for c in candidates if c["raga"] in valid_ragas]

    return usable, raga_counts, valid_ragas


def main():
    parser = argparse.ArgumentParser(description="Preprocess Saraga Carnatic audio")
    parser.add_argument("--min-tracks-per-raga", type=int, default=3,
                        help="Minimum tracks per raga to include (default: 3)")
    args = parser.parse_args()

    root = repo_root()

    # --- Load config
    cfg = load_cfg(str(root / "configs" / "preprocessing.yaml"))
    target_sr = int(cfg["sample_rate"])
    seg_s = float(cfg["segment_seconds"])
    hop_s = float(cfg["hop_seconds"])
    trim_silence = bool(cfg.get("trim_silence", False))

    # --- mirdata setup
    mirdata_home = root / "data" / "raw" / "mirdata"
    os.environ["MIRDATA_HOME"] = str(mirdata_home)
    ds = mirdata.initialize("saraga_carnatic")

    # --- Filter to usable tracks
    usable, raga_counts, valid_ragas = get_usable_tracks(ds, args.min_tracks_per_raga)

    print(f"Total Saraga Carnatic tracks: {len(ds.track_ids)}")
    print(f"Usable tracks (raga + tonic + audio): {len(usable)}")
    print(f"Ragas with >= {args.min_tracks_per_raga} tracks: {len(valid_ragas)}")
    print()
    for raga in sorted(valid_ragas):
        print(f"  {raga_counts[raga]:3d}x  {raga}")
    print()

    # --- Output dirs
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    ensure_dirs(interim_dir, processed_dir)

    out_csv = processed_dir / "segments_saraga_carnatic.csv"

    # --- Segment audio
    seg_len = int(seg_s * target_sr)
    hop_len = int(hop_s * target_sr)
    rows = []

    for track_info in tqdm(usable, desc="Tracks"):
        tid = track_info["track_id"]
        raga = track_info["raga"]
        gt_tonic = track_info["tonic_hz"]
        audio_path = track_info["audio_path"]

        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        y = y.astype("float32")

        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=30)

        if len(y) < seg_len:
            continue

        # Segment with overlapping windows
        seg_idx = 0
        for start in range(0, len(y) - seg_len + 1, hop_len):
            end = start + seg_len
            seg = y[start:end]

            seg_name = f"saraga_carnatic__{tid}__{seg_idx:05d}.wav"
            seg_path = interim_dir / seg_name

            sf.write(seg_path, seg, target_sr)

            rows.append({
                "dataset": "saraga_carnatic",
                "track_id": tid,
                "raga": raga,
                "gt_tonic_hz": gt_tonic,
                "segment_index": seg_idx,
                "start_s": start / target_sr,
                "end_s": end / target_sr,
                "duration_s": seg_s,
                "sample_rate": target_sr,
                "path": str(seg_path.relative_to(root)).replace("\\", "/"),
            })
            seg_idx += 1

    # --- Write CSV
    fieldnames = [
        "dataset", "track_id", "raga", "gt_tonic_hz",
        "segment_index", "start_s", "end_s", "duration_s",
        "sample_rate", "path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary
    raga_seg_counts = Counter(r["raga"] for r in rows)
    print(f"\nDone. Wrote {len(rows)} segments from {len(usable)} tracks.")
    print(f"Metadata CSV: {out_csv}")
    print(f"\nSegments per raga:")
    for raga, count in raga_seg_counts.most_common():
        print(f"  {count:4d}  {raga}")


if __name__ == "__main__":
    main()
