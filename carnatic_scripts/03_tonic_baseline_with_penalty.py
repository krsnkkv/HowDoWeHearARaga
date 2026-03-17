"""
03_tonic_baseline_with_penalty.py -- Stage 3 variant: Harmonic Triad Scoring

Replaces the Sa+Pa scoring with a Sa+Ma+Pa "Big Three" template matcher.
Instead of just looking for the highest peak, looks for the harmonic triad
shape that a real tonic produces: strong Sa (0c), Ma (498c), and Pa (702c).

Adds a ratio-based penalty: if Pa mass > Sa mass, the candidate is suspect
(likely Pa misidentified as Sa) and gets penalized proportionally.

Usage:
    python carnatic_scripts/03_tonic_baseline_with_penalty.py
    python carnatic_scripts/03_tonic_baseline_with_penalty.py --method compare
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


# -- Core functions --

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


# -- Scoring methods --

def tonic_score_sa_only(hist: np.ndarray, bins_per_octave: int = 120) -> float:
    """Original: Sa mass only."""
    if hist.sum() <= 0:
        return 0.0
    return get_mass_around_bin(hist, 0.0, window_cents=30.0,
                               bins_per_octave=bins_per_octave)


def tonic_score_harmonic(hist: np.ndarray, bins_per_octave: int = 120,
                          pa_weight: float = 0.5) -> float:
    """Baseline improved: Sa + Pa."""
    if hist.sum() <= 0:
        return 0.0
    sa_mass = get_mass_around_bin(hist, 0.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    pa_mass = get_mass_around_bin(hist, 702.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    return float(sa_mass + pa_weight * pa_mass)


def tonic_score_triad(hist: np.ndarray, bins_per_octave: int = 120,
                       pa_weight: float = 0.8, ma_weight: float = 0.6) -> float:
    """
    Pure Harmonic Triad scoring: Sa + Ma + Pa, no penalty.

    Instead of penalizing when Pa > Sa, we use Pa and Ma as positive
    evidence that this candidate is a real tonic. The question changes
    from "where is the loudest note?" to "where is the most stable
    harmonic house?"

    A tonic (Sa) is only valid if it has:
      - A roof: Pa at 702 cents (present in every raga, reinforced by drone)
      - A front door: Ma at 498 cents (present in most ragas)

    If the algorithm sits on the wrong note (e.g., Pa misidentified as Sa),
    it will look for a "roof" at +702c and a "door" at +498c from there.
    Those positions will be empty or weak in a real raga.

    If it sits on the true Sa, even if Sa is quiet, it will find the loud
    Pa (roof) and Ma (door) exactly where they belong. The combined mass
    naturally pulls the estimate to the correct Sa.
    """
    if hist.sum() <= 0:
        return 0.0

    sa_mass = get_mass_around_bin(hist, 0.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    ma_mass = get_mass_around_bin(hist, 498.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)
    pa_mass = get_mass_around_bin(hist, 702.0, window_cents=30.0,
                                  bins_per_octave=bins_per_octave)

    # Pure weighted support -- no penalty
    # Sa gets full weight, Pa and Ma provide harmonic evidence
    combined = sa_mass + (pa_weight * pa_mass) + (ma_weight * ma_mass)

    return float(combined)


def tonic_score_triad_smoothed(hist: np.ndarray, bins_per_octave: int = 120,
                                pa_weight: float = 0.8, ma_weight: float = 0.6,
                                sigma_bins: float = 2.0) -> float:
    """Pure triad scoring with Gaussian smoothing applied first."""
    from scipy.ndimage import gaussian_filter1d
    if hist.sum() <= 0:
        return 0.0
    n = len(hist)
    padded = np.concatenate([hist[-int(3*sigma_bins):], hist, hist[:int(3*sigma_bins)]])
    smoothed = gaussian_filter1d(padded, sigma=sigma_bins, mode='nearest')
    pad_len = int(3 * sigma_bins)
    result = smoothed[pad_len:pad_len + n]
    s = result.sum()
    if s > 0:
        result = (result / s).astype(np.float32)
    else:
        result = result.astype(np.float32)
    return tonic_score_triad(result, bins_per_octave, pa_weight, ma_weight)


def estimate_tonic(f0_hz: np.ndarray,
                   candidate_min: float = 80.0,
                   candidate_max: float = 300.0,
                   step_cents: float = 10.0,
                   bins_per_octave: int = 120,
                   min_voiced_frames: int = 200,
                   method: str = "triad",
                   pa_weight: float = 0.8,
                   ma_weight: float = 0.6):
    """
    Search for best tonic in [candidate_min, candidate_max] Hz.
    Steps in cents for musically uniform spacing.
    Tests octave alternatives after initial search.

    Returns: (best_tonic_hz, confidence, best_score, second_score, method_used)
    """
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced_frames:
        return float("nan"), 0.0, 0.0, 0.0, method

    # Choose scoring function
    if method == "triad":
        score_fn = lambda h: tonic_score_triad(h, bins_per_octave, pa_weight, ma_weight)
    elif method == "triad_smoothed":
        score_fn = lambda h: tonic_score_triad_smoothed(h, bins_per_octave, pa_weight, ma_weight)
    elif method == "harmonic":
        score_fn = lambda h: tonic_score_harmonic(h, bins_per_octave, 0.5)
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

    return float(best_hz), conf, float(best_score), float(second_score), method


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
    axes[0].set_title("Tonic Estimation Error (raw)")
    axes[0].legend()

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


def run_estimation(df, root, candidate_min, candidate_max, step_cents,
                   bins_per_octave, min_voiced, method, pa_weight, ma_weight):
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
                ma_weight=ma_weight,
            )
            n_frames = int(f0_concat.size)

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
    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave_err = int(valid["is_octave_error"].sum())
    n_raw_within_50 = int((valid["cents_error"] <= 50).sum())

    pct_within_50 = 100 * n_within_50 / n_total if n_total > 0 else 0
    pct_octave = 100 * n_octave_err / n_total if n_total > 0 else 0
    pct_raw_50 = 100 * n_raw_within_50 / n_total if n_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION -- {method_name}")
    print(f"{'='*60}")
    print(f"Tracks evaluated:              {n_total}")
    print(f"Within 50 cents (oc):          {n_within_50}/{n_total} ({pct_within_50:.1f}%)")
    print(f"Within 50 cents (raw):         {n_raw_within_50}/{n_total} ({pct_raw_50:.1f}%)")
    print(f"Octave errors:                 {n_octave_err}/{n_total} ({pct_octave:.1f}%)")
    print(f"Median error (raw):            {valid['cents_error'].median():.1f} cents")
    print(f"Median error (oc):             {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):               {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")

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
    parser = argparse.ArgumentParser(
        description="Tonic estimation with Sa+Ma+Pa triad scoring and ratio penalty")
    parser.add_argument("--method", type=str, default="triad",
                        choices=["sa_mass", "harmonic", "triad", "triad_smoothed", "compare"],
                        help="Scoring method (default: triad)")
    parser.add_argument("--pa-weight", type=float, default=0.8,
                        help="Weight for Pa mass in triad scoring (default: 0.8)")
    parser.add_argument("--ma-weight", type=float, default=0.6,
                        help="Weight for Ma mass in triad scoring (default: 0.6)")
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
    min_voiced = int(tcfg.get("min_voiced_frames", 200))

    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    if not pitch_csv.exists():
        raise FileNotFoundError(f"Missing: {pitch_csv}. Run 02_pitch_extraction.py first.")

    df = pd.read_csv(pitch_csv)
    print(f"Loaded pitch data for {len(df)} segments, {df['track_id'].nunique()} tracks")

    out_dir = root / "data" / "processed"
    fig_dir = root / "outputs" / "figures" / "tonic_validation"
    ensure_dir(fig_dir)

    if args.method == "compare":
        methods_to_run = ["sa_mass", "harmonic", "triad", "triad_smoothed"]
    else:
        methods_to_run = [args.method]

    comparison_results = {}

    for method in methods_to_run:
        print(f"\n>>> Running: {method}")

        val_df = run_estimation(
            df, root, candidate_min, candidate_max, step_cents,
            bins_per_octave, min_voiced, method, args.pa_weight, args.ma_weight,
        )

        suffix = f"_{method}_triad" if len(methods_to_run) > 1 else "_triad"
        out_csv = out_dir / f"tonic_validation{suffix}.csv"
        val_df.to_csv(out_csv, index=False)

        plot_validation_summary(
            val_df, fig_dir / f"tonic_error_histogram{suffix}.png"
        )

        stats = print_summary(val_df, method.upper())
        comparison_results[method] = stats
        print(f"Results saved to: {out_csv}")

    # Compare against baseline
    baseline_csv = out_dir / "tonic_validation.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        baseline_valid = baseline_df.dropna(subset=["cents_error"])
        n_baseline_50 = int((baseline_valid["octave_corrected_cents_error"] <= 50).sum())
        n_baseline = len(baseline_valid)

        print(f"\n{'='*60}")
        print("COMPARISON vs BASELINE (harmonic Sa+Pa, no penalty)")
        print(f"{'='*60}")
        print(f"  {'Baseline (Sa+Pa):':<28s} {n_baseline_50}/{n_baseline} within 50c "
              f"({100*n_baseline_50/n_baseline:.1f}%)")
        for method, stats in comparison_results.items():
            label = f"  {method}:"
            print(f"{label:<28s} within_50={stats['within_50']:.1f}%, "
                  f"octave_err={stats['octave_errors']:.1f}%, "
                  f"mean_oc={stats['median_oc']:.1f}c")

        # Track-level comparison for the primary method
        primary_method = methods_to_run[0] if len(methods_to_run) == 1 else "triad"
        primary_suffix = f"_{primary_method}_triad" if len(methods_to_run) > 1 else "_triad"
        primary_csv = out_dir / f"tonic_validation{primary_suffix}.csv"
        if primary_csv.exists():
            new_df = pd.read_csv(primary_csv)
            merged = baseline_valid[['track_id','octave_corrected_cents_error','within_50_cents']].merge(
                new_df[['track_id','octave_corrected_cents_error','within_50_cents']],
                on='track_id', suffixes=('_base','_new'))

            got_better = merged[(merged['within_50_cents_base']==False) & (merged['within_50_cents_new']==True)]
            got_worse = merged[(merged['within_50_cents_base']==True) & (merged['within_50_cents_new']==False)]

            if len(got_better) > 0:
                print(f"\n  IMPROVED tracks ({len(got_better)}):")
                for _, r in got_better.iterrows():
                    print(f"    {r['track_id']}: "
                          f"{r['octave_corrected_cents_error_base']:.1f}c -> "
                          f"{r['octave_corrected_cents_error_new']:.1f}c")

            if len(got_worse) > 0:
                print(f"\n  REGRESSED tracks ({len(got_worse)}):")
                for _, r in got_worse.iterrows():
                    print(f"    {r['track_id']}: "
                          f"{r['octave_corrected_cents_error_base']:.1f}c -> "
                          f"{r['octave_corrected_cents_error_new']:.1f}c")

            if len(got_better) == 0 and len(got_worse) == 0:
                print("\n  No tracks changed status.")


if __name__ == "__main__":
    main()
