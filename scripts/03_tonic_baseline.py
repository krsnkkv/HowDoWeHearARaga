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
Scores each candidate - Mass near 0 cents SA estnmation + Pitch class Histogram
Computes confidence as gap btw best and second best
Data/processed/tonic_validation.csv with GT vs est tonic, confidence, error metrics
Plots outputs/figures/tonic_validation/tonic_error_histogram.png - histogram of errors
Prints summary stats: % within 50 cents, median error, worst cases for debugging


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


# ── Core tonic estimation functions ──────────────────────────────

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


def tonic_score(hist: np.ndarray, bins_per_octave: int = 120) -> float:
    """
    Score a tonic candidate by measuring mass near Sa (0 cents).
    Pure Sa-mass scoring — no entropy bonus (simpler, more robust).
    Window: +/- 30 cents around 0 (wrapping around 1200→0).
    """
    if hist.sum() <= 0:
        return 0.0

    # With 10-cent bins: +/-30 cents = 3 bins on each side
    w = 3
    sa_mass = hist[0:w + 1].sum() + hist[-w:].sum()
    return float(sa_mass)


def estimate_tonic(f0_hz: np.ndarray,
                   candidate_min: float = 80.0,
                   candidate_max: float = 300.0,
                   step_cents: float = 10.0,
                   bins_per_octave: int = 120,
                   min_voiced_frames: int = 200):
    """
    Search for the best tonic in [candidate_min, candidate_max] Hz.
    Steps in cents (log-frequency) for musically uniform spacing.

    Returns: (best_tonic_hz, confidence, best_score, second_score)
    """
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced_frames:
        return float("nan"), 0.0, 0.0, 0.0

    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)
    second = (-1.0, None)

    for k in range(n_steps):
        tonic_hz = candidate_min * (2 ** ((k * step_cents) / 1200.0))
        hist = pitch_class_histogram(f0, tonic_hz, bins_per_octave)
        score = tonic_score(hist, bins_per_octave)

        if score > best[0]:
            second = best
            best = (score, tonic_hz)
        elif score > second[0]:
            second = (score, tonic_hz)

    best_score, best_hz = best
    second_score, _ = second

    # Confidence: gap between top two candidates
    conf = 0.0 if best_score <= 0 else float(
        (best_score - second_score) / (best_score + 1e-9)
    )

    return float(best_hz), conf, float(best_score), float(second_score)


# ── Validation helpers ───────────────────────────────────────────

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


def main():
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

    # --- Estimate tonic per track (aggregate all segments)
    rows = []

    for track_id, group in tqdm(df.groupby("track_id"), desc="Tonic estimation"):
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
        else:
            f0_concat = np.concatenate(all_f0)
            est_hz, conf, best_score, second_score = estimate_tonic(
                f0_concat,
                candidate_min=candidate_min,
                candidate_max=candidate_max,
                step_cents=step_cents,
                bins_per_octave=bins_per_octave,
                min_voiced_frames=min_voiced,
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
            "n_segments": len(group),
        })

    val_df = pd.DataFrame(rows)

    # --- Save results
    out_csv = out_dir / "tonic_validation.csv"
    val_df.to_csv(out_csv, index=False)

    # --- Validation plot
    plot_validation_summary(val_df, fig_dir / "tonic_error_histogram.png")

    # --- Print summary
    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave_err = int((valid["cents_error"] > 600).sum())

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Tracks evaluated:        {n_total}")
    print(f"Within 50 cents:         {n_within_50}/{n_total} "
          f"({100*n_within_50/n_total:.1f}%)")
    print(f"Octave errors (>600c):   {n_octave_err}/{n_total} "
          f"({100*n_octave_err/n_total:.1f}%)")
    print(f"Median error (raw):      {valid['cents_error'].median():.1f} cents")
    print(f"Median error (oc):       {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):         {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")
    print(f"\nResults saved to: {out_csv}")
    print(f"Error plot: {fig_dir / 'tonic_error_histogram.png'}")

    # Show worst cases for debugging
    print(f"\nWorst 5 estimates (octave-corrected):")
    worst = valid.nlargest(5, "octave_corrected_cents_error")
    for _, r in worst.iterrows():
        print(f"  {r['track_id']}: est={r['estimated_tonic_hz']:.1f} Hz, "
              f"gt={r['gt_tonic_hz']:.1f} Hz, "
              f"error={r['octave_corrected_cents_error']:.1f}c, "
              f"raga={r['raga']}")


if __name__ == "__main__":
    main()
