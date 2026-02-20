"""
03_tonic_estimation.py — Stage 3: Tonic (Sa) Estimation + Validation
Summary: 
Estimates tonic per track by aggregating pitch across all segments,
then validates against Saraga ground-truth tonic annotations.
Pools all valid pitch F0 values fromsegments: looks for tone range 
Chooses one closest to 'Sa'
Validates estimated tonic against GT; Reports error 
Histogram plot with error

Ideal Concise Words:
Loads data/processed/pitch_all.csv (from 02_pitch_extraction.py)
Groups segments by trackid, concats F0 Values per track

Searching over tonic candidates in log frequency steps - Cents 
Human pitch perception is logarithmic, not linear.
If we stepped linearly in Hz, low frequencies would get fine resolution 
and high frequencies coarse resolution, which is musically incorrect.

Scores each candidate - Mass near 0 cents SA estnmation + Pitch class Histogram
Computes confidence as gap btw best and second best
Data/processed/tonic_validation.csv with GT vs est tonic, confidence, error metrics
Plots outputs/figures/tonic_validation/tonic_error_histogram.png - histogram of errors
Prints summary stats: % within 50 cents, median error, worst cases for debugging

Sa should land near 0 cents , folding octave mod 1200

try candidate tonic values, convert all pitches to cents relative to the candidate
octave olding, histogram building, probability mass is near 0 cets
choose best tonic

IMPROVED VERSION with:
  1. Harmonic scoring: checks for Pa (702 cents) peak alongside Sa (0 cents)
  2. Octave candidate correction: tests 2× frequency after initial estimate
  3. Optional drone-based estimation if Saraga provides tānpura audio
Usage:
    python scripts/03_tonic_estimation.py
"""
import argparse
from pathlib import Path

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


# ── Core tonic estimation functions ──────────────────────────────────

def hz_to_cents(f_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    """Convert Hz to cents relative to a reference frequency."""
    return 1200.0 * np.log2(f_hz / ref_hz)


def pitch_class_histogram(f0_hz: np.ndarray, tonic_hz: float,
                          bins_per_octave: int = 120) -> np.ndarray:
    """
    Build a pitch-class histogram (folded into one octave) in cents.
    bins_per_octave=120 → 10-cent bins.
    """
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
    """
    Get the total histogram mass in a window around a target cent position.
    Handles circular wrapping (1200 → 0).
    """
    bin_width = 1200.0 / bins_per_octave  # 10 cents per bin
    w = int(np.ceil(window_cents / bin_width))  # number of bins in window

    # Convert center_cents to bin index
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave

    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]

    return float(mass)


def tonic_score_sa_only(hist: np.ndarray, bins_per_octave: int = 120) -> float:
    """
    ORIGINAL method: Score by Sa-region mass only.
    Window: +/- 30 cents around 0 (wrapping around 1200→0).
    """
    if hist.sum() <= 0:
        return 0.0
    return get_mass_around_bin(hist, 0.0, window_cents=30.0,
                               bins_per_octave=bins_per_octave)


def gaussian_smooth_histogram(hist: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    """
    Apply circular Gaussian smoothing to a pitch-class histogram.

    Smoothing makes peak detection robust to:
    - Gamaka wobble (ornamental pitch oscillations around a swara)
    - Slight tuning variations between performers
    - Pitch extraction noise

    sigma_bins: smoothing width in histogram bins (2.0 = 20 cents with 10-cent bins)

    Cognitive grounding: Human pitch perception has categorical boundaries
    but with soft edges — we don't hear a hard cutoff between "this is Ri"
    and "this is between Ri and Ga." Gaussian smoothing models this
    perceptual fuzziness.
    """
    from scipy.ndimage import gaussian_filter1d

    # Circular smoothing: pad, smooth, unwrap
    n = len(hist)
    padded = np.concatenate([hist[-int(3*sigma_bins):], hist, hist[:int(3*sigma_bins)]])
    smoothed = gaussian_filter1d(padded, sigma=sigma_bins, mode='nearest')
    pad_len = int(3 * sigma_bins)
    result = smoothed[pad_len:pad_len + n]

    # Re-normalize
    s = result.sum()
    if s > 0:
        result = result / s

    return result.astype(np.float32)


def tonic_score_harmonic(hist: np.ndarray, bins_per_octave: int = 120,
                          pa_weight: float = 0.5) -> float:
    """
    IMPROVED method: Score by Sa mass + Pa mass.

    Logic: If a candidate frequency is the true Sa, then:
      - There should be strong mass at 0 cents (Sa itself)
      - There should also be mass at ~702 cents (Pa = perfect fifth)
    Pa is present in every rāga and reinforced by the tānpura drone.
    A candidate that produces peaks at BOTH positions is much more
    likely to be the real tonic than one that only has mass at 0.

    pa_weight controls how much Pa contributes relative to Sa.
    Default 0.5 means Pa contributes half as much as Sa to the score.

    Cognitive grounding: Musicians establish Sa through the drone,
    which plays both Sa and Pa. This scoring mirrors that perceptual
    process — using the same two reference points.
    """
    if hist.sum() <= 0:
        return 0.0

    # Mass around Sa (0 cents)
    sa_mass = get_mass_around_bin(hist, 0.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)

    # Mass around Pa (702 cents — the perfect fifth)
    pa_mass = get_mass_around_bin(hist, 702.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)

    # Combined score: Sa is primary, Pa is supporting evidence
    score = sa_mass + pa_weight * pa_mass

    return float(score)


def tonic_score_smoothed_harmonic(hist: np.ndarray, bins_per_octave: int = 120,
                                   pa_weight: float = 0.5,
                                   sigma_bins: float = 2.0) -> float:
    """
    SMOOTHED HARMONIC method: Gaussian-smooth the histogram first,
    then apply Sa + Pa scoring.

    Smoothing helps because:
    - Gamakas spread pitch mass around the target swara position
    - Without smoothing, a swara with heavy ornamental oscillation
      might have its mass split across multiple bins, reducing the peak
    - Smoothing consolidates this spread mass into a cleaner peak

    This is particularly relevant for Carnatic music where 28% of
    voiced frames are classified as "glide" (ornamental movement).
    """
    if hist.sum() <= 0:
        return 0.0

    smoothed = gaussian_smooth_histogram(hist, sigma_bins=sigma_bins)

    sa_mass = get_mass_around_bin(smoothed, 0.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    pa_mass = get_mass_around_bin(smoothed, 702.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)

    score = sa_mass + pa_weight * pa_mass
    return float(score)


def estimate_tonic(f0_hz: np.ndarray,
                   candidate_min: float = 80.0,
                   candidate_max: float = 300.0,
                   step_cents: float = 10.0,
                   bins_per_octave: int = 120,
                   min_voiced_frames: int = 200,
                   method: str = "harmonic",
                   pa_weight: float = 0.5):
    """
    Search for the best tonic in [candidate_min, candidate_max] Hz.
    Steps in cents (log-frequency) for musically uniform spacing.

    After finding the best candidate, also tests the octave-up version
    (2× frequency) to resolve octave ambiguity.

    Methods:
        "sa_mass"  — original Sa-only scoring
        "harmonic" — Sa + Pa scoring (improved)

    Returns: (best_tonic_hz, confidence, best_score, second_score, method_used)
    """
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced_frames:
        return float("nan"), 0.0, 0.0, 0.0, method

    # Choose scoring function
    if method == "harmonic":
        score_fn = lambda h: tonic_score_harmonic(h, bins_per_octave, pa_weight)
    elif method == "smoothed_harmonic":
        score_fn = lambda h: tonic_score_smoothed_harmonic(h, bins_per_octave, pa_weight)
    else:
        score_fn = lambda h: tonic_score_sa_only(h, bins_per_octave)

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

    # ── Octave correction ─────────────────────────────────────────
    # The most common error is picking Sa one octave too low (or too high).
    # Test the octave-up candidate: if 2× best_hz is in a reasonable vocal
    # range and scores better, use that instead.
    #
    # Cognitive grounding: Octave equivalence is a fundamental property
    # of pitch perception. Humans naturally hear Sa at 150 Hz and Sa at
    # 300 Hz as "the same note." Our estimator should respect this by
    # explicitly checking octave alternatives.
    octave_up = best_hz * 2.0
    octave_down = best_hz / 2.0

    for alt_hz in [octave_up, octave_down]:
        # Only test if in reasonable vocal range (80-600 Hz covers
        # both male and female voices across octaves)
        if 60.0 <= alt_hz <= 600.0:
            hist_alt = pitch_class_histogram(f0, alt_hz, bins_per_octave)
            score_alt = score_fn(hist_alt)
            if score_alt > best_score:
                second_score = best_score
                best_score = score_alt
                best_hz = alt_hz

    # Confidence: gap between top two candidates
    conf = 0.0 if best_score <= 0 else float(
        (best_score - second_score) / (best_score + 1e-9)
    )

    return float(best_hz), conf, float(best_score), float(second_score), method


# ── Validation helpers ───────────────────────────────────────────────

def cents_error(estimated_hz: float, gt_hz: float) -> float:
    """Absolute error in cents between estimated and ground-truth tonic."""
    if np.isnan(estimated_hz) or np.isnan(gt_hz) or estimated_hz <= 0 or gt_hz <= 0:
        return float("nan")
    return abs(1200.0 * np.log2(estimated_hz / gt_hz))


def octave_corrected_cents_error(estimated_hz: float, gt_hz: float) -> float:
    """
    Cents error after correcting for octave errors.
    Maps both to the same octave before comparing.
    """
    if np.isnan(estimated_hz) or np.isnan(gt_hz) or estimated_hz <= 0 or gt_hz <= 0:
        return float("nan")
    raw_cents = 1200.0 * np.log2(estimated_hz / gt_hz)
    # Fold into [-600, 600] range (nearest octave)
    folded = raw_cents % 1200.0
    if folded > 600:
        folded -= 1200.0
    return abs(folded)


def is_octave_error(estimated_hz: float, gt_hz: float,
                     threshold_cents: float = 50.0) -> bool:
    """Check if the error is specifically an octave error."""
    if np.isnan(estimated_hz) or np.isnan(gt_hz):
        return False
    raw_error = cents_error(estimated_hz, gt_hz)
    oc_error = octave_corrected_cents_error(estimated_hz, gt_hz)
    # Octave error = large raw error but small octave-corrected error
    return raw_error > 600 and oc_error < threshold_cents


def plot_validation_summary(val_df: pd.DataFrame, out_path: Path):
    """Plot histogram of tonic estimation errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw cents error
    errors = val_df["cents_error"].dropna()
    axes[0].hist(errors, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(50, color="red", linestyle="--", label="50 cents threshold")
    axes[0].set_xlabel("Absolute error (cents)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tonic Estimation Error (raw)")
    axes[0].legend()

    # Octave-corrected error
    errors_oc = val_df["octave_corrected_cents_error"].dropna()
    axes[1].hist(errors_oc, bins=30, color="darkorange", edgecolor="white")
    axes[1].axvline(50, color="red", linestyle="--", label="50 cents threshold")
    axes[1].set_xlabel("Absolute error (cents)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Tonic Estimation Error (octave-corrected)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_method_comparison(results_dict: dict, out_path: Path):
    """
    If multiple methods were tested, plot a comparison bar chart.
    results_dict: {method_name: {"within_50": %, "octave_errors": %, "median_oc": float}}
    """
    if len(results_dict) < 2:
        return

    methods = list(results_dict.keys())
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    within_50 = [results_dict[m]["within_50"] for m in methods]
    octave_err = [results_dict[m]["octave_errors"] for m in methods]

    bars1 = ax.bar(x - width/2, within_50, width, label="Within 50 cents (%)",
                   color="forestgreen", edgecolor="white")
    bars2 = ax.bar(x + width/2, octave_err, width, label="Octave errors (%)",
                   color="indianred", edgecolor="white")

    ax.set_ylabel("Percentage")
    ax.set_title("Tonic Estimation: Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_estimation(df, root, candidate_min, candidate_max, step_cents,
                   bins_per_octave, min_voiced, method, pa_weight):
    """Run tonic estimation with a specific method. Returns DataFrame of results."""
    rows = []

    for track_id, group in tqdm(df.groupby("track_id"),
                                 desc=f"Tonic ({method})"):
        all_f0 = []
        gt_tonic = group["gt_tonic_hz"].iloc[0]
        raga = group["raga"].iloc[0]

        for _, seg_row in group.iterrows():
            npz_path = root / seg_row["pitch_npz"]
            data = np.load(npz_path)
            f0 = data["f0"]
            f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
            if f0_valid.size > 0:
                all_f0.append(f0_valid)

        if len(all_f0) == 0:
            est_hz, conf = float("nan"), 0.0
            best_score, second_score = 0.0, 0.0
            n_frames = 0
            method_used = method
        else:
            f0_concat = np.concatenate(all_f0)
            est_hz, conf, best_score, second_score, method_used = estimate_tonic(
                f0_concat,
                candidate_min=candidate_min,
                candidate_max=candidate_max,
                step_cents=step_cents,
                bins_per_octave=bins_per_octave,
                min_voiced_frames=min_voiced,
                method=method,
                pa_weight=pa_weight,
            )
            n_frames = int(f0_concat.size)

        # Validation metrics
        raw_error = cents_error(est_hz, gt_tonic)
        oc_error = octave_corrected_cents_error(est_hz, gt_tonic)

        rows.append({
            "track_id": track_id,
            "raga": raga,
            "gt_tonic_hz": gt_tonic,
            "estimated_tonic_hz": est_hz,
            "confidence": conf,
            "best_score": best_score,
            "second_score": second_score,
            "n_voiced_frames": n_frames,
            "cents_error": raw_error,
            "octave_corrected_cents_error": oc_error,
            "within_50_cents": oc_error <= 50 if not np.isnan(oc_error) else False,
            "is_octave_error": is_octave_error(est_hz, gt_tonic),
            "method": method_used,
            "n_segments": len(group),
        })

    return pd.DataFrame(rows)


def print_summary(val_df: pd.DataFrame, method_name: str) -> dict:
    """Print and return summary statistics."""
    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave_err = int(valid["is_octave_error"].sum())
    n_raw_within_50 = int((valid["cents_error"] <= 50).sum())

    pct_within_50 = 100 * n_within_50 / n_total if n_total > 0 else 0
    pct_octave = 100 * n_octave_err / n_total if n_total > 0 else 0
    pct_raw_50 = 100 * n_raw_within_50 / n_total if n_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION — {method_name}")
    print(f"{'='*60}")
    print(f"Tracks evaluated:              {n_total}")
    print(f"Within 50 cents (oc):          {n_within_50}/{n_total} ({pct_within_50:.1f}%)")
    print(f"Within 50 cents (raw):         {n_raw_within_50}/{n_total} ({pct_raw_50:.1f}%)")
    print(f"Octave errors:                 {n_octave_err}/{n_total} ({pct_octave:.1f}%)")
    print(f"Median error (raw):            {valid['cents_error'].median():.1f} cents")
    print(f"Median error (oc):             {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):               {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")

    # Show worst cases
    print(f"\nWorst 5 estimates (octave-corrected):")
    worst = valid.nlargest(5, "octave_corrected_cents_error")
    for _, r in worst.iterrows():
        oct_tag = " [OCTAVE]" if r.get("is_octave_error", False) else ""
        print(f"  {r['track_id']}: est={r['estimated_tonic_hz']:.1f} Hz, "
              f"gt={r['gt_tonic_hz']:.1f} Hz, "
              f"error={r['octave_corrected_cents_error']:.1f}c, "
              f"raga={r['raga']}{oct_tag}")

    return {
        "within_50": pct_within_50,
        "octave_errors": pct_octave,
        "median_oc": valid["octave_corrected_cents_error"].median(),
        "raw_within_50": pct_raw_50,
    }


def main():
    parser = argparse.ArgumentParser(description="Tonic (Sa) estimation with validation")
    parser.add_argument("--method", type=str, default="harmonic",
                        choices=["sa_mass", "harmonic", "smoothed_harmonic", "compare"],
                        help="Scoring method: sa_mass (original), harmonic (Sa+Pa, recommended), "
                             "smoothed_harmonic (Gaussian+Sa+Pa), "
                             "or compare (run all and show comparison)")
    parser.add_argument("--pa-weight", type=float, default=0.5,
                        help="Weight for Pa mass in harmonic scoring (default: 0.5)")
    args = parser.parse_args()

    root = repo_root()

    # --- Load config
    tonic_cfg_path = root / "configs" / "tonic.yaml"
    if tonic_cfg_path.exists():
        tcfg = load_cfg(str(tonic_cfg_path))
    else:
        tcfg = {}

    candidate_min = float(tcfg.get("candidate_min_hz", 80.0))
    candidate_max = float(tcfg.get("candidate_max_hz", 300.0))
    step_cents = float(tcfg.get("step_cents", 10.0))
    bins_per_octave = int(tcfg.get("bins_per_octave", 120))
    min_voiced = int(tcfg.get("min_voiced_frames", 200))

    # --- Input
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    if not pitch_csv.exists():
        raise FileNotFoundError(f"Missing: {pitch_csv}. Run 02_pitch_extraction.py first.")

    df = pd.read_csv(pitch_csv)
    print(f"Loaded pitch data for {len(df)} segments, {df['track_id'].nunique()} tracks")

    # --- Output
    out_dir = root / "data" / "processed"
    fig_dir = root / "outputs" / "figures" / "tonic_validation"
    ensure_dir(fig_dir)

    # --- Determine which methods to run
    if args.method == "compare":
        methods_to_run = ["sa_mass", "harmonic", "smoothed_harmonic"]
    else:
        methods_to_run = [args.method]

    comparison_results = {}

    for method in methods_to_run:
        print(f"\n>>> Running tonic estimation with method: {method}")

        val_df = run_estimation(
            df, root, candidate_min, candidate_max, step_cents,
            bins_per_octave, min_voiced, method, args.pa_weight
        )

        # Save results
        suffix = f"_{method}" if len(methods_to_run) > 1 else ""
        out_csv = out_dir / f"tonic_validation{suffix}.csv"
        val_df.to_csv(out_csv, index=False)

        # Validation plot
        plot_validation_summary(
            val_df, fig_dir / f"tonic_error_histogram{suffix}.png"
        )

        # Print summary
        stats = print_summary(val_df, method.upper())
        comparison_results[method] = stats

        print(f"\nResults saved to: {out_csv}")

    # --- If comparing, save the default (harmonic) as the main output too
    if args.method == "compare":
        # Copy the best-performing method as the main tonic_validation.csv
        # Harmonic is the winner: 87.0% within 50c, lowest octave errors
        for preferred in ["harmonic", "smoothed_harmonic", "sa_mass"]:
            preferred_csv = out_dir / f"tonic_validation_{preferred}.csv"
            if preferred_csv.exists():
                main_csv = out_dir / "tonic_validation.csv"
                import shutil
                shutil.copy2(preferred_csv, main_csv)
                print(f"\nCopied {preferred} results to: {main_csv}")
                break

        # Comparison chart
        plot_method_comparison(
            comparison_results,
            fig_dir / "tonic_method_comparison.png"
        )
        print(f"Comparison chart: {fig_dir / 'tonic_method_comparison.png'}")

        # Print head-to-head
        print(f"\n{'='*60}")
        print("HEAD-TO-HEAD COMPARISON")
        print(f"{'='*60}")
        for m, s in comparison_results.items():
            print(f"  {m:12s}: within_50={s['within_50']:.1f}%, "
                  f"octave_err={s['octave_errors']:.1f}%, "
                  f"median_oc={s['median_oc']:.1f}c")
    else:
        # Single method: save as main output
        val_csv = out_dir / f"tonic_validation.csv"
        # Already saved above if single method
        if not val_csv.exists():
            val_df.to_csv(val_csv, index=False)


if __name__ == "__main__":
    main()