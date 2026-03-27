"""
hpss_tonic_estimation.py — HPSS for Tonic Estimation

WHAT THIS SCRIPT DOES:
    Uses librosa's Harmonic-Percussive Source Separation (HPSS) to isolate the
    harmonic component of the audio before pitch extraction and tonic scoring.
    Removes percussive transients (mridangam, ghatam strikes) while preserving
    all harmonic content (voice, violin, tanpura). A lightweight, GPU-free
    alternative to Demucs that runs in seconds per segment.

LOGIC AND DESIGN:
    HPSS operates on the STFT magnitude spectrogram, applying median filtering
    along time (to extract harmonic content with sustained frequencies) and
    along frequency (to extract percussive content with broadband transients).
    The margin parameter controls the separation aggressiveness. After HPSS,
    the harmonic-only signal is fed through the same CREPE pitch extraction and
    harmonic tonic scoring as Stage 3. The hypothesis is that removing
    percussive energy improves CREPE's pitch confidence on frames near drum
    strikes.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    Carnatic percussion (mridangam, ghatam) produces broadband transients that
    can confuse pitch trackers but do not carry tonal information relevant to
    raga identity. Listeners naturally separate percussive rhythm from melodic
    content — a perceptual stream segregation process (Bregman, 1990). HPSS
    formalises this by removing the percussive stream before tonal analysis.
    Result: marginal improvement on a few tracks, but percussive interference
    was not the primary failure mode (violin Pa dominance is), so the overall
    87% accuracy ceiling is unchanged.

INPUT:
    - data/processed/pitch_all.csv — segment metadata with ground-truth tonics
    - audio segment files referenced by segment_path column

OUTPUT:
    - data/processed/tonic_validation_hpss.csv — per-track tonic estimates and errors

Usage:
    python carnatic_scripts/experiments/hpss_tonic_estimation.py
    python carnatic_scripts/experiments/hpss_tonic_estimation.py --tracks-only "157_,76_,91_"
    python carnatic_scripts/experiments/hpss_tonic_estimation.py --margin 3.0
"""
# cli argument parsing
import argparse
# filesystem path handling
from pathlib import Path

# numerical array operations
import numpy as np
# tabular data manipulation
import pandas as pd
# audio loading and harmonic-percussive source separation
import librosa
# non-interactive matplotlib backend for headless environments
import matplotlib
matplotlib.use("Agg")
# progress bar for iteration
from tqdm import tqdm


# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Tonic estimation (standalone copy) ────────────────────────────

# build a normalized pitch-class histogram relative to a candidate tonic
def pitch_class_histogram(f0_hz, tonic_hz, bins_per_octave=120):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size == 0:
        return np.zeros(bins_per_octave, dtype=np.float32)
    cents = 1200.0 * np.log2(f0 / tonic_hz)
    cents_mod = np.mod(cents, 1200.0)
    edges = np.linspace(0.0, 1200.0, bins_per_octave + 1)
    hist, _ = np.histogram(cents_mod, bins=edges)
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


# sum histogram mass in a window around a target cent position
def get_mass_around_bin(hist, center_cents, window_cents=30.0, bins_per_octave=120):
    bin_width = 1200.0 / bins_per_octave
    w = int(np.ceil(window_cents / bin_width))
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave
    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]
    return float(mass)


# score a candidate tonic using sa and pa histogram peaks
def tonic_score_harmonic(hist, bins_per_octave=120, pa_weight=0.5):
    if hist.sum() <= 0:
        return 0.0
    sa_mass = get_mass_around_bin(hist, 0.0, 30.0, bins_per_octave)
    pa_mass = get_mass_around_bin(hist, 702.0, 30.0, bins_per_octave)
    return float(sa_mass + pa_weight * pa_mass)


# sweep candidate tonics and return the best hz with a confidence score
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


# compute octave-folded cents error to ignore octave mismatches
def octave_corrected_cents_error(est_hz, gt_hz):
    if np.isnan(est_hz) or np.isnan(gt_hz) or est_hz <= 0 or gt_hz <= 0:
        return float("nan")
    raw = 1200.0 * np.log2(est_hz / gt_hz)
    folded = raw % 1200.0
    if folded > 600:
        folded -= 1200.0
    return abs(folded)


# ── HPSS + pitch extraction ──────────────────────────────────────

# apply hpss to isolate harmonics then extract pitch via pyin
def extract_harmonic_pitch(audio_path, sr=22050, hop_length=256,
                           fmin=65.0, fmax=2100.0, margin=3.0):
    """
    Load audio, apply HPSS to isolate harmonic component,
    then extract pitch via pyin.

    margin controls separation aggressiveness:
      1.0 = gentle separation
      3.0 = moderate (default)
      8.0 = aggressive (may lose some harmonic content)
    """
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # HPSS: separate harmonic and percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=margin)

    # Extract pitch from harmonic component only
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harmonic, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )

    if f0 is None:
        return np.array([])

    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    return f0_valid


# run hpss tonic estimation on all tracks and compare with baselines
def main():
    parser = argparse.ArgumentParser(
        description="HPSS harmonic separation → tonic estimation")
    parser.add_argument("--tracks-only", type=str, default=None,
                        help="Comma-separated track_id prefixes")
    parser.add_argument("--max-segments-per-track", type=int, default=5,
                        help="Max segments per track (default: 5)")
    parser.add_argument("--margin", type=float, default=3.0,
                        help="HPSS margin parameter (default: 3.0)")
    args = parser.parse_args()

    root = repo_root()

    # Load segment metadata
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    df = pd.read_csv(pitch_csv)

    # Filter to specific tracks if requested
    if args.tracks_only:
        prefixes = [p.strip() for p in args.tracks_only.split(",")]
        mask = df["track_id"].apply(
            lambda tid: any(tid.startswith(p) for p in prefixes))
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to {df['track_id'].nunique()} tracks")

    # Limit segments per track
    if args.max_segments_per_track:
        df = df.groupby("track_id").head(args.max_segments_per_track).reset_index(drop=True)

    print(f"Processing {len(df)} segments from {df['track_id'].nunique()} tracks")
    print(f"HPSS margin: {args.margin}")

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
                f0_valid = extract_harmonic_pitch(
                    seg_path, margin=args.margin)
                if f0_valid.size > 0:
                    all_f0.append(f0_valid)
            except Exception as e:
                print(f"  Error on {seg_path.name}: {e}")
                continue

        if len(all_f0) == 0:
            track_results[track_id] = {
                "track_id": track_id, "raga": raga,
                "gt_tonic_hz": gt_tonic,
                "hpss_tonic_hz": float("nan"),
                "confidence": 0.0,
                "cents_error": float("nan"),
                "oc_cents_error": float("nan"),
                "n_voiced_frames": 0,
            }
            continue

        f0_concat = np.concatenate(all_f0)
        est_hz, conf = estimate_tonic(f0_concat)

        raw_err = cents_error(est_hz, gt_tonic)
        oc_err = octave_corrected_cents_error(est_hz, gt_tonic)

        track_results[track_id] = {
            "track_id": track_id, "raga": raga,
            "gt_tonic_hz": gt_tonic,
            "hpss_tonic_hz": est_hz,
            "confidence": conf,
            "cents_error": raw_err,
            "oc_cents_error": oc_err,
            "n_voiced_frames": int(f0_concat.size),
        }

        status = "OK" if oc_err <= 50 else f"ERROR ({oc_err:.0f}c)"
        print(f"  {track_id}: est={est_hz:.1f} Hz, gt={gt_tonic:.1f} Hz, "
              f"err={oc_err:.1f}c [{status}]")

    # Save results
    results_df = pd.DataFrame(list(track_results.values()))
    out_csv = root / "data" / "processed" / "tonic_validation_hpss.csv"
    results_df.to_csv(out_csv, index=False)

    # Summary
    valid = results_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int((valid["oc_cents_error"] <= 50).sum())

    print(f"\n{'='*60}")
    print(f"HPSS TONIC ESTIMATION RESULTS (margin={args.margin})")
    print(f"{'='*60}")
    print(f"Tracks evaluated:     {n_total}")
    print(f"Within 50 cents (oc): {n_within_50}/{n_total} "
          f"({100*n_within_50/n_total:.1f}%)")
    print(f"Median error (oc):    {valid['oc_cents_error'].median():.1f} cents")

    # Compare with original + demucs
    orig_csv = root / "data" / "processed" / "tonic_validation.csv"
    demucs_csv = root / "data" / "processed" / "tonic_validation_demucs.csv"

    if orig_csv.exists():
        orig_df = pd.read_csv(orig_csv)
        orig_valid = orig_df.dropna(subset=["cents_error"])
        n_orig_50 = int((orig_valid["octave_corrected_cents_error"] <= 50).sum())
        print(f"\nComparison:")
        print(f"  Original (CREPE mix): {n_orig_50}/{len(orig_valid)} "
              f"({100*n_orig_50/len(orig_valid):.1f}%) within 50c")

    if demucs_csv.exists():
        demucs_df = pd.read_csv(demucs_csv)
        demucs_valid = demucs_df.dropna(subset=["cents_error"])
        n_demucs_50 = int((demucs_valid["oc_cents_error"] <= 50).sum())
        print(f"  Demucs (vocal sep):   {n_demucs_50}/{len(demucs_valid)} "
              f"({100*n_demucs_50/len(demucs_valid):.1f}%) within 50c")

    print(f"  HPSS (harmonic):      {n_within_50}/{n_total} "
          f"({100*n_within_50/n_total:.1f}%) within 50c")

    # Per-track detail for bad tracks
    if orig_csv.exists():
        print(f"\nPer-track comparison (originally worst tracks):")
        orig_df_indexed = orig_df.set_index("track_id")
        for _, r in valid.nlargest(min(15, len(valid)), "oc_cents_error").iterrows():
            tid = r["track_id"]
            orig_err = "N/A"
            if tid in orig_df_indexed.index:
                orig_err = f"{orig_df_indexed.loc[tid, 'octave_corrected_cents_error']:.1f}c"
            print(f"  {tid}: HPSS={r['oc_cents_error']:.1f}c, "
                  f"Original={orig_err}, gt={r['gt_tonic_hz']:.1f} Hz")

    print(f"\nResults saved: {out_csv}")


if __name__ == "__main__":
    main()
