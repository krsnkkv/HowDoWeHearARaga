"""
demucs_tonic_estimation.py — Source Separation for Tonic Estimation

WHAT THIS SCRIPT DOES:
    Uses Meta's Demucs source separation model to isolate the vocal track from
    the polyphonic mix, then runs CREPE pitch extraction and harmonic tonic
    scoring on the clean vocal signal only. Outputs its own tonic_validation CSV
    for comparison against the baseline (which operates on the full mix).
    Standalone script — does not modify any existing pipeline files.

LOGIC AND DESIGN:
    The primary tonic estimation failure mode is CREPE locking onto the
    accompanying violin's Pa (perfect fifth above Sa), which is often louder
    than the vocalist's Sa in mixed Carnatic audio. Demucs separates the mix
    into vocals, drums, bass, and other stems using a deep U-Net architecture.
    By running pitch extraction on the vocal stem only, the violin Pa
    interference is removed. The rest of the tonic estimation pipeline
    (candidate search, harmonic scoring, octave correction) is identical to
    Stage 3.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    Human listeners perform a form of auditory source separation — attending to
    the vocalist's melody while backgrounding the tanpura drone and rhythmic
    accompaniment. This experiment tests whether computational source separation
    can replicate this attentional filtering to improve tonic estimation.
    Result: Demucs did not surpass 87% accuracy, because the vocal separation
    itself introduces artefacts that degrade pitch tracking. The polyphonic
    audio without vocal isolation represents a hard limit for the current
    pipeline, confirming that tonic estimation is the primary bottleneck.

INPUT:
    - data/processed/pitch_all.csv — segment metadata with track ids and audio paths
    - data/processed/tonic_validation.csv — baseline results for comparison (optional)
    - configs/tonic.yaml — tonic estimation parameters (optional, uses defaults)

OUTPUT:
    - data/processed/tonic_validation_demucs.csv — per-track demucs results

Usage:
    python carnatic_scripts/experiments/demucs_tonic_estimation.py
    python carnatic_scripts/experiments/demucs_tonic_estimation.py --tracks-only "157_,76_,91_"
"""
# cli argument parsing
import argparse
# environment variable access
import os
# filesystem path handling
from pathlib import Path

# numerical array operations
import numpy as np
# tabular data loading and manipulation
import pandas as pd
# pytorch tensor operations and gpu management
import torch
# audio loading and resampling
import torchaudio
# pitch extraction via pyin
import librosa
# non-interactive matplotlib backend for headless rendering
import matplotlib
matplotlib.use("Agg")
# plotting figures (unused but kept for potential extensions)
import matplotlib.pyplot as plt
# progress bar for track-level iteration
from tqdm import tqdm

# add experiments/ to sys.path for potential local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))


# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Tonic estimation (copied core functions to keep standalone) ───

# convert hz frequencies to cents relative to a reference pitch
def hz_to_cents(f_hz, ref_hz):
    return 1200.0 * np.log2(f_hz / ref_hz)


# build a normalised pitch-class histogram folded into one octave
def pitch_class_histogram(f0_hz, tonic_hz, bins_per_octave=120):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size == 0:
        return np.zeros(bins_per_octave, dtype=np.float32)
    cents = hz_to_cents(f0, tonic_hz)
    cents_mod = np.mod(cents, 1200.0)
    edges = np.linspace(0.0, 1200.0, bins_per_octave + 1)
    hist, _ = np.histogram(cents_mod, bins=edges)
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


# sum histogram mass within a window around a target cent value
def get_mass_around_bin(hist, center_cents, window_cents=30.0, bins_per_octave=120):
    bin_width = 1200.0 / bins_per_octave
    w = int(np.ceil(window_cents / bin_width))
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave
    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]
    return float(mass)


# score a candidate tonic using sa mass plus weighted pa mass
def tonic_score_harmonic(hist, bins_per_octave=120, pa_weight=0.5):
    if hist.sum() <= 0:
        return 0.0
    sa_mass = get_mass_around_bin(hist, 0.0, 30.0, bins_per_octave)
    pa_mass = get_mass_around_bin(hist, 702.0, 30.0, bins_per_octave)
    return float(sa_mass + pa_weight * pa_mass)


# search candidate frequencies and return the best tonic with confidence
def estimate_tonic(f0_hz, candidate_min=80.0, candidate_max=300.0,
                   step_cents=10.0, bins_per_octave=120, min_voiced_frames=200):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced_frames:
        return float("nan"), 0.0

    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)
    second = (-1.0, None)

    for k in range(n_steps):
        tonic_hz = candidate_min * (2 ** ((k * step_cents) / 1200.0))
        hist = pitch_class_histogram(f0, tonic_hz, bins_per_octave)
        score = tonic_score_harmonic(hist, bins_per_octave)
        if score > best[0]:
            second = best
            best = (score, tonic_hz)
        elif score > second[0]:
            second = (score, tonic_hz)

    best_score, best_hz = best
    second_score, _ = second

    # Octave correction
    for alt_hz in [best_hz * 2.0, best_hz / 2.0]:
        if 60.0 <= alt_hz <= 600.0:
            hist_alt = pitch_class_histogram(f0, alt_hz, bins_per_octave)
            score_alt = tonic_score_harmonic(hist_alt, bins_per_octave)
            if score_alt > best_score:
                second_score = best_score
                best_score = score_alt
                best_hz = alt_hz

    conf = 0.0 if best_score <= 0 else float(
        (best_score - second_score) / (best_score + 1e-9))
    return float(best_hz), conf


# compute absolute cents error between estimated and ground-truth tonic
def cents_error(est_hz, gt_hz):
    if np.isnan(est_hz) or np.isnan(gt_hz) or est_hz <= 0 or gt_hz <= 0:
        return float("nan")
    return abs(1200.0 * np.log2(est_hz / gt_hz))


# compute cents error folded into one octave to ignore octave mistakes
def octave_corrected_cents_error(est_hz, gt_hz):
    if np.isnan(est_hz) or np.isnan(gt_hz) or est_hz <= 0 or gt_hz <= 0:
        return float("nan")
    raw = 1200.0 * np.log2(est_hz / gt_hz)
    folded = raw % 1200.0
    if folded > 600:
        folded -= 1200.0
    return abs(folded)


# ── Demucs separation ────────────────────────────────────────────

# load the pretrained htdemucs model onto the specified device
def load_demucs_model(device="cuda"):
    """Load the Demucs htdemucs model."""
    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels

    model = get_model("htdemucs")
    if isinstance(model, BagOfModels):
        model = model.models[0]
    model = model.to(device)
    model.eval()
    return model


# run demucs source separation and return the mono vocals stem
def separate_vocals(model, audio_path, device="cuda", sr=44100):
    """
    Run Demucs on an audio file and return the vocals stem.
    Demucs operates at 44100 Hz internally.
    """
    from demucs.apply import apply_model

    # Load audio at Demucs' native sample rate
    wav, orig_sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    # Demucs expects (batch, channels, samples)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono → stereo (demucs expects stereo)

    wav = wav.unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=False)

    # sources shape: (batch, n_sources, channels, samples)
    # htdemucs sources: drums=0, bass=1, other=2, vocals=3
    vocals = sources[0, 3]  # (channels, samples)
    vocals_mono = vocals.mean(dim=0).cpu().numpy()  # mono

    return vocals_mono, sr


# extract f0 pitch contour from a raw audio array using pyin
def extract_pitch_from_audio(audio_array, sr=44100, hop_length=256,
                              fmin=65.0, fmax=2100.0):
    """Extract F0 from audio array using pyin."""
    if len(audio_array) < sr:
        return np.array([])
    f0, voiced_flag, _ = librosa.pyin(
        audio_array, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    if f0 is None:
        return np.array([])
    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    return f0_valid


# ── Main ─────────────────────────────────────────────────────────

# separate vocals with demucs, extract pitch, estimate tonic, and validate
def main():
    parser = argparse.ArgumentParser(
        description="Demucs vocal separation → tonic estimation")
    parser.add_argument("--tracks-only", type=str, default=None,
                        help="Comma-separated track_id prefixes to process "
                             "(default: all tracks)")
    parser.add_argument("--max-segments-per-track", type=int, default=3,
                        help="Max segments to separate per track (default: 3, "
                             "saves GPU time)")
    args = parser.parse_args()

    root = repo_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load segment metadata
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    df = pd.read_csv(pitch_csv)

    # Optionally filter to specific tracks
    if args.tracks_only:
        prefixes = [p.strip() for p in args.tracks_only.split(",")]
        mask = df["track_id"].apply(
            lambda tid: any(tid.startswith(p) for p in prefixes))
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} segments from {df['track_id'].nunique()} tracks")

    # Limit segments per track (separation is expensive)
    if args.max_segments_per_track:
        df = df.groupby("track_id").head(args.max_segments_per_track).reset_index(drop=True)

    print(f"Processing {len(df)} segments from {df['track_id'].nunique()} tracks")

    # Load Demucs model
    print("Loading Demucs htdemucs model...")
    model = load_demucs_model(device)
    print("Model loaded.")

    # Output dirs
    vocal_dir = root / "data" / "processed" / "demucs_vocals"
    fig_dir = root / "outputs" / "figures" / "demucs_tonic"
    ensure_dir(vocal_dir)
    ensure_dir(fig_dir)

    # Process each track
    track_results = {}

    for track_id, group in tqdm(df.groupby("track_id"), desc="Tracks"):
        all_f0 = []
        gt_tonic = group["gt_tonic_hz"].iloc[0]
        raga = group["raga"].iloc[0]

        for _, seg_row in group.iterrows():
            seg_path = root / seg_row["segment_path"]
            if not seg_path.exists():
                continue

            try:
                # Separate vocals
                vocals, sr = separate_vocals(model, seg_path, device)

                # Extract pitch from vocals
                f0_valid = extract_pitch_from_audio(vocals, sr=sr)

                if f0_valid.size > 0:
                    all_f0.append(f0_valid)

            except Exception as e:
                print(f"  Error on {seg_path.name}: {e}")
                continue

        if len(all_f0) == 0:
            print(f"  {track_id}: no valid pitch from vocals")
            track_results[track_id] = {
                "track_id": track_id, "raga": raga,
                "gt_tonic_hz": gt_tonic,
                "demucs_tonic_hz": float("nan"),
                "confidence": 0.0,
                "cents_error": float("nan"),
                "oc_cents_error": float("nan"),
                "n_voiced_frames": 0,
                "n_segments_used": len(group),
            }
            continue

        f0_concat = np.concatenate(all_f0)
        est_hz, conf = estimate_tonic(f0_concat)

        raw_err = cents_error(est_hz, gt_tonic)
        oc_err = octave_corrected_cents_error(est_hz, gt_tonic)

        track_results[track_id] = {
            "track_id": track_id, "raga": raga,
            "gt_tonic_hz": gt_tonic,
            "demucs_tonic_hz": est_hz,
            "confidence": conf,
            "cents_error": raw_err,
            "oc_cents_error": oc_err,
            "n_voiced_frames": int(f0_concat.size),
            "n_segments_used": len(group),
        }

        status = "OK" if oc_err <= 50 else f"ERROR ({oc_err:.0f}c)"
        print(f"  {track_id}: est={est_hz:.1f} Hz, gt={gt_tonic:.1f} Hz, "
              f"err={oc_err:.1f}c [{status}]")

    # Save results
    results_df = pd.DataFrame(list(track_results.values()))
    out_csv = root / "data" / "processed" / "tonic_validation_demucs.csv"
    results_df.to_csv(out_csv, index=False)

    # Summary
    valid = results_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int((valid["oc_cents_error"] <= 50).sum())

    print(f"\n{'='*60}")
    print(f"DEMUCS TONIC ESTIMATION RESULTS")
    print(f"{'='*60}")
    print(f"Tracks evaluated:     {n_total}")
    print(f"Within 50 cents (oc): {n_within_50}/{n_total} "
          f"({100*n_within_50/n_total:.1f}%)")
    print(f"Median error (oc):    {valid['oc_cents_error'].median():.1f} cents")

    # Compare with original estimation if available
    orig_csv = root / "data" / "processed" / "tonic_validation.csv"
    if orig_csv.exists():
        orig_df = pd.read_csv(orig_csv)
        orig_valid = orig_df.dropna(subset=["cents_error"])
        n_orig_50 = int((orig_valid["octave_corrected_cents_error"] <= 50).sum())
        print(f"\nOriginal (mix-based): {n_orig_50}/{len(orig_valid)} "
              f"({100*n_orig_50/len(orig_valid):.1f}%) within 50c")
        print(f"Demucs (vocal-based): {n_within_50}/{n_total} "
              f"({100*n_within_50/n_total:.1f}%) within 50c")

        # Per-track comparison for bad tracks
        print(f"\nPer-track comparison (worst original tracks):")
        orig_worst = orig_valid.nlargest(15, "octave_corrected_cents_error")
        for _, row in orig_worst.iterrows():
            tid = row["track_id"]
            if tid in track_results:
                dr = track_results[tid]
                print(f"  {tid}:")
                print(f"    Original: {row['octave_corrected_cents_error']:.1f}c "
                      f"(est={row['estimated_tonic_hz']:.1f} Hz)")
                print(f"    Demucs:   {dr['oc_cents_error']:.1f}c "
                      f"(est={dr['demucs_tonic_hz']:.1f} Hz)")

    # Worst cases
    print(f"\nWorst 5 estimates (demucs):")
    worst = valid.nlargest(5, "oc_cents_error")
    for _, r in worst.iterrows():
        print(f"  {r['track_id']}: est={r['demucs_tonic_hz']:.1f} Hz, "
              f"gt={r['gt_tonic_hz']:.1f} Hz, err={r['oc_cents_error']:.1f}c "
              f"({r['raga']})")

    print(f"\nResults saved: {out_csv}")


if __name__ == "__main__":
    main()
