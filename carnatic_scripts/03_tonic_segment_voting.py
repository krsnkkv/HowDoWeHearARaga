"""
03_tonic_segment_voting.py -- Stage 3 variant: Segment-Level Majority Voting

Instead of concatenating all F0 across segments and estimating once,
estimates tonic PER SEGMENT and takes majority vote across segments.

Rationale: violin/veena doesn't dominate equally in every segment.
Segments where the vocalist is more prominent will estimate correctly.
Majority vote lets the good segments outvote the bad ones.

Usage:
    python carnatic_scripts/03_tonic_segment_voting.py
"""
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -- Core functions (same as baseline) --

def hz_to_cents(f_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    return 1200.0 * np.log2(f_hz / ref_hz)


def pitch_class_histogram(f0_hz: np.ndarray, tonic_hz: float,
                          bins_per_octave: int = 120) -> np.ndarray:
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


def get_mass_around_bin(hist: np.ndarray, center_cents: float,
                        window_cents: float = 30.0,
                        bins_per_octave: int = 120) -> float:
    bin_width = 1200.0 / bins_per_octave
    w = int(np.ceil(window_cents / bin_width))
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave
    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]
    return float(mass)


def tonic_score_harmonic(hist: np.ndarray, bins_per_octave: int = 120,
                          pa_weight: float = 0.5) -> float:
    if hist.sum() <= 0:
        return 0.0
    sa_mass = get_mass_around_bin(hist, 0.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    pa_mass = get_mass_around_bin(hist, 702.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    return float(sa_mass + pa_weight * pa_mass)


def estimate_tonic_single(f0_hz: np.ndarray,
                           candidate_min: float = 80.0,
                           candidate_max: float = 300.0,
                           step_cents: float = 10.0,
                           bins_per_octave: int = 120,
                           min_voiced_frames: int = 50,
                           pa_weight: float = 0.5):
    """
    Estimate tonic from a single segment's F0 array.
    Lower min_voiced_frames than track-level (segments are shorter).
    Returns: (best_hz, confidence, best_score)
    """
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced_frames:
        return float("nan"), 0.0, 0.0

    score_fn = lambda h: tonic_score_harmonic(h, bins_per_octave, pa_weight)

    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)
    second = (-1.0, None)

    for k in range(n_steps):
        tonic_hz = candidate_min * (2 ** ((k * step_cents) / 1200.0))
        hist = pitch_class_histogram(f0, tonic_hz, bins_per_octave)
        score = score_fn(hist)
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
            score_alt = score_fn(hist_alt)
            if score_alt > best_score:
                second_score = best_score
                best_score = score_alt
                best_hz = alt_hz

    conf = 0.0 if best_score <= 0 else float(
        (best_score - second_score) / (best_score + 1e-9)
    )
    return float(best_hz), conf, float(best_score)


def quantize_hz(hz: float, step_cents: float = 10.0) -> float:
    """Quantize Hz to nearest step_cents grid for voting."""
    if np.isnan(hz) or hz <= 0:
        return float("nan")
    # Quantize in log space
    log_hz = 1200.0 * np.log2(hz)
    quantized = round(log_hz / step_cents) * step_cents
    return 2.0 ** (quantized / 1200.0)


def majority_vote_tonic(segment_estimates: list, step_cents: float = 50.0):
    """
    Majority vote across segment-level tonic estimates.

    Quantizes to 50-cent bins (half-semitone) so that small pitch
    differences between segments don't split the vote.

    Also considers octave equivalence: if two estimates differ by
    exactly 1200 cents, they vote for the same pitch class.

    Returns: (voted_hz, n_agreeing, n_total, vote_confidence)
    """
    valid = [(hz, conf) for hz, conf, _ in segment_estimates
             if not np.isnan(hz) and hz > 0]

    if len(valid) == 0:
        return float("nan"), 0, 0, 0.0

    # Quantize to pitch-class bins (fold octaves)
    # Use cent values mod 1200 for octave-invariant voting
    ref_hz = 1.0  # arbitrary reference for cent conversion
    vote_bins = []
    for hz, conf in valid:
        cents = 1200.0 * np.log2(hz)
        bin_cents = round(cents / step_cents) * step_cents
        vote_bins.append((bin_cents, hz, conf))

    # Vote on pitch class (mod 1200 cents) to handle octave differences
    pc_votes = Counter()
    pc_hz_map = {}  # pitch_class_bin -> list of (hz, conf)

    for bin_cents, hz, conf in vote_bins:
        pc_bin = round((bin_cents % 1200.0) / step_cents) * step_cents
        pc_votes[pc_bin] += 1
        if pc_bin not in pc_hz_map:
            pc_hz_map[pc_bin] = []
        pc_hz_map[pc_bin].append((hz, conf))

    # Winner
    winner_pc, n_agreeing = pc_votes.most_common(1)[0]
    n_total = len(valid)

    # Final Hz: confidence-weighted median of the winning bin's estimates
    winning_estimates = pc_hz_map[winner_pc]
    winning_hz_values = [hz for hz, _ in winning_estimates]
    winning_confs = [conf for _, conf in winning_estimates]

    # Use confidence-weighted average in log space
    log_hz = np.array([np.log2(h) for h in winning_hz_values])
    weights = np.array(winning_confs)
    if weights.sum() > 0:
        voted_log_hz = np.average(log_hz, weights=weights)
    else:
        voted_log_hz = np.median(log_hz)
    voted_hz = 2.0 ** voted_log_hz

    vote_confidence = n_agreeing / n_total if n_total > 0 else 0.0

    return voted_hz, n_agreeing, n_total, vote_confidence


# -- Validation helpers --

def cents_error(estimated_hz: float, gt_hz: float) -> float:
    if np.isnan(estimated_hz) or np.isnan(gt_hz) or estimated_hz <= 0 or gt_hz <= 0:
        return float("nan")
    return abs(1200.0 * np.log2(estimated_hz / gt_hz))


def octave_corrected_cents_error(estimated_hz: float, gt_hz: float) -> float:
    if np.isnan(estimated_hz) or np.isnan(gt_hz) or estimated_hz <= 0 or gt_hz <= 0:
        return float("nan")
    raw_cents = 1200.0 * np.log2(estimated_hz / gt_hz)
    folded = raw_cents % 1200.0
    if folded > 600:
        folded -= 1200.0
    return abs(folded)


def is_octave_error(estimated_hz: float, gt_hz: float,
                     threshold_cents: float = 50.0) -> bool:
    if np.isnan(estimated_hz) or np.isnan(gt_hz):
        return False
    raw_error = cents_error(estimated_hz, gt_hz)
    oc_error = octave_corrected_cents_error(estimated_hz, gt_hz)
    return raw_error > 600 and oc_error < threshold_cents


def plot_validation_summary(val_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    errors = val_df["cents_error"].dropna()
    axes[0].hist(errors, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(50, color="red", linestyle="--", label="50 cents threshold")
    axes[0].set_xlabel("Absolute error (cents)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tonic Error (raw) - Segment Voting")
    axes[0].legend()

    errors_oc = val_df["octave_corrected_cents_error"].dropna()
    axes[1].hist(errors_oc, bins=30, color="darkorange", edgecolor="white")
    axes[1].axvline(50, color="red", linestyle="--", label="50 cents threshold")
    axes[1].set_xlabel("Absolute error (cents)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Tonic Error (octave-corrected) - Segment Voting")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Tonic estimation via segment-level majority voting")
    parser.add_argument("--pa-weight", type=float, default=0.5)
    parser.add_argument("--min-voiced", type=int, default=50,
                        help="Min voiced frames per segment (default: 50)")
    args = parser.parse_args()

    root = repo_root()

    tonic_cfg_path = root / "configs" / "tonic.yaml"
    if tonic_cfg_path.exists():
        tcfg = load_cfg(str(tonic_cfg_path))
    else:
        tcfg = {}

    candidate_min = float(tcfg.get("candidate_min_hz", 80.0))
    candidate_max = float(tcfg.get("candidate_max_hz", 300.0))
    step_cents = float(tcfg.get("step_cents", 10.0))
    bins_per_octave = int(tcfg.get("bins_per_octave", 120))

    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    if not pitch_csv.exists():
        raise FileNotFoundError(f"Missing: {pitch_csv}")

    df = pd.read_csv(pitch_csv)
    print(f"Loaded {len(df)} segments, {df['track_id'].nunique()} tracks")

    out_dir = root / "data" / "processed"
    fig_dir = root / "outputs" / "figures" / "tonic_validation"
    ensure_dir(fig_dir)

    rows = []

    for track_id, group in tqdm(df.groupby("track_id"),
                                 desc="Tonic (segment voting)"):
        gt_tonic = group["gt_tonic_hz"].iloc[0]
        raga = group["raga"].iloc[0]

        # Estimate tonic per segment
        seg_estimates = []
        for _, seg_row in group.iterrows():
            npz_path = root / seg_row["pitch_npz"]
            data = np.load(npz_path)
            f0 = data["f0"]

            est_hz, conf, score = estimate_tonic_single(
                f0,
                candidate_min=candidate_min,
                candidate_max=candidate_max,
                step_cents=step_cents,
                bins_per_octave=bins_per_octave,
                min_voiced_frames=args.min_voiced,
                pa_weight=args.pa_weight,
            )
            seg_estimates.append((est_hz, conf, score))

        # Majority vote
        voted_hz, n_agree, n_total, vote_conf = majority_vote_tonic(seg_estimates)

        raw_error = cents_error(voted_hz, gt_tonic)
        oc_error = octave_corrected_cents_error(voted_hz, gt_tonic)

        rows.append({
            "track_id": track_id,
            "raga": raga,
            "gt_tonic_hz": gt_tonic,
            "estimated_tonic_hz": voted_hz,
            "confidence": vote_conf,
            "n_segments_voted": n_total,
            "n_segments_agreeing": n_agree,
            "vote_agreement": f"{n_agree}/{n_total}",
            "cents_error": raw_error,
            "octave_corrected_cents_error": oc_error,
            "within_50_cents": oc_error <= 50 if not np.isnan(oc_error) else False,
            "is_octave_error": is_octave_error(voted_hz, gt_tonic),
            "method": "segment_voting",
            "n_segments": len(group),
        })

    val_df = pd.DataFrame(rows)

    out_csv = out_dir / "tonic_validation_segment_voting.csv"
    val_df.to_csv(out_csv, index=False)

    plot_validation_summary(val_df, fig_dir / "tonic_error_histogram_segment_voting.png")

    # Print summary
    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave_err = int(valid["is_octave_error"].sum())
    n_raw_within_50 = int((valid["cents_error"] <= 50).sum())

    pct_within_50 = 100 * n_within_50 / n_total if n_total > 0 else 0
    pct_octave = 100 * n_octave_err / n_total if n_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION -- SEGMENT-LEVEL MAJORITY VOTING")
    print(f"{'='*60}")
    print(f"Tracks evaluated:              {n_total}")
    print(f"Within 50 cents (oc):          {n_within_50}/{n_total} ({pct_within_50:.1f}%)")
    print(f"Within 50 cents (raw):         {n_raw_within_50}/{n_total} ({100*n_raw_within_50/n_total:.1f}%)")
    print(f"Octave errors:                 {n_octave_err}/{n_total} ({pct_octave:.1f}%)")
    print(f"Median error (oc):             {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):               {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")

    print(f"\nWorst 5:")
    worst = valid.nlargest(5, "octave_corrected_cents_error")
    for _, r in worst.iterrows():
        print(f"  {r['track_id']}: est={r['estimated_tonic_hz']:.1f} Hz, "
              f"gt={r['gt_tonic_hz']:.1f} Hz, "
              f"error={r['octave_corrected_cents_error']:.1f}c, "
              f"vote={r['vote_agreement']}, raga={r['raga']}")

    # Compare vs baseline
    baseline_csv = out_dir / "tonic_validation.csv"
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv)
        bv = baseline.dropna(subset=["cents_error"])
        n_b50 = int((bv["octave_corrected_cents_error"] <= 50).sum())

        merged = bv[['track_id','octave_corrected_cents_error','within_50_cents']].merge(
            valid[['track_id','octave_corrected_cents_error','within_50_cents']],
            on='track_id', suffixes=('_base','_new'))

        got_better = merged[(merged['within_50_cents_base']==False) & (merged['within_50_cents_new']==True)]
        got_worse = merged[(merged['within_50_cents_base']==True) & (merged['within_50_cents_new']==False)]

        print(f"\n{'='*60}")
        print(f"vs BASELINE: {n_b50}/{len(bv)} ({100*n_b50/len(bv):.1f}%)")
        print(f"{'='*60}")

        if len(got_better) > 0:
            print(f"\nIMPROVED ({len(got_better)}):")
            for _, r in got_better.iterrows():
                print(f"  {r['track_id']}: {r['octave_corrected_cents_error_base']:.1f}c -> {r['octave_corrected_cents_error_new']:.1f}c")

        if len(got_worse) > 0:
            print(f"\nREGRESSED ({len(got_worse)}):")
            for _, r in got_worse.iterrows():
                print(f"  {r['track_id']}: {r['octave_corrected_cents_error_base']:.1f}c -> {r['octave_corrected_cents_error_new']:.1f}c")

        if len(got_better) == 0 and len(got_worse) == 0:
            print("\nNo tracks changed status.")

    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()
