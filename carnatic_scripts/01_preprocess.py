"""
01_preprocess.py — Stage 1: Preprocessing

SARAGA Audio + Metadata -> .wav segments + csv file with foundational information.

Loads Saraga Carnatic tracks via mirdata, filters to tracks with
raga labels + tonic annotations, segments audio, and writes metadata CSV.

Commands for runningg:
    python carnatic_scripts/01_preprocess.py
    python carnatic_scripts/01_preprocess.py --min-tracks-per-raga 5   # stricter filter


Loads the Saraga Carnatic Dataset
Filter tracks - Valid raga label + GT tonic annotation + Audio Files + Not Ragamalika + N Tracks
Preprocessing parameters loaded 
Resampling to fixed rate
Optionally trims leading/trailing silence
Splits each track into overlapping fixed length segments
Saves each segment as .wav 
MIN PER RAGA -> GROUP K FILTERING
Overlapping segmentation -> More training data, Augmentation
CSV File: Track ID, Raga label, GT Tonic, Segment index and timestamps, File path to segment

Imbalance
"""
import os #Used for setting environment variables
import csv #CSV Files writing
import yaml #Config files
import argparse #Command line flags like --min-tracks-per-raga
from pathlib import Path #Path finder handler
from collections import Counter # Counts 

import librosa #Loads audio and Silences
import soundfile as sf #.WAV file writer
import mirdata #Loads SARAGA
from tqdm import tqdm #Progress bar for loops (Console)

#Loads YAML file that contains preprocessing settings
# Returns a dict with config values: SR, Segment length, hopsilence, trim silen
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#Script finds repo root consistently
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

#Ensuresa directory exists - creates parent folders if folde rdoesnt exist
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

#Two stage track filtering:
def get_usable_tracks(ds, min_tracks_per_raga: int = 3):
    """

    Track Validity -> Raga label, tonic annotatin, audio file, not Ragamallika
    track id, raga, tonic hz, audio path, has pitch annot
    Class support filtering -> Counts how many tracks per raga, Keeps minimum number of tracks per raga 
    usable, raga counts, valid ragas
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

    # two stage filder: counting tracks per raga using counter, then keeping ones with min_tracks_per_raga or more
    # filter candidates down to only those with valid ragas
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

    # --- Load config: loads sr, seg len, hop, trim
    cfg = load_cfg(str(root / "configs" / "preprocessing.yaml"))
    target_sr = int(cfg["sample_rate"])
    seg_s = float(cfg["segment_seconds"])
    hop_s = float(cfg["hop_seconds"])
    trim_silence = bool(cfg.get("trim_silence", False))

    # --- mirdata setup
    mirdata_home = root / "data" / "raw" / "mirdata"
    os.environ["MIRDATA_HOME"] = str(mirdata_home)
    ds = mirdata.initialize("saraga_carnatic")

    # --- Filter to usable tracks (def get usable tracks)
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

    #Load audio at fixed sr, trim, skip if too short, slide a window across each track
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
