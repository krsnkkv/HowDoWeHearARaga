"""
04_swara_mapping.py — Stage 4: Pitch-to-Swara Mapping

WHAT THIS SCRIPT DOES:
    Converts continuous F0 pitch contours into discrete swara (note) labels
    relative to the tonic (Sa). Defines 12 swara bins in cents (Sa=0, Ri1=100,
    Ri2=200, ... Ni2=1100) and maps each voiced frame to the nearest bin using
    circular distance in cents (mod 1200). Frames within 40 cents of a bin are
    assigned that swara; frames outside tolerance are labelled "glide"
    (ornamental gamaka movement); unvoiced frames are labelled "silence". A
    minimum-duration filter converts swara runs shorter than 100 ms (~10 CREPE
    frames) to "glide", suppressing pitch-estimation noise. Outputs per-segment
    .npz swara traces, a summary CSV with glide ratio and top swara, and
    diagnostic plots showing the pitch contour with colour-coded swara labels.

LOGIC AND DESIGN:
    The 40-cent tolerance (slightly less than a half-semitone) was chosen to
    accommodate pitch drift in CREPE estimates while preventing adjacent swara
    bins from overlapping. Circular distance handles the octave boundary
    correctly (Ni2 at 1100 cents is 100 cents from Sa at 0/1200 cents). At this
    threshold, approximately 28% of voiced frames are classified as "glide",
    consistent with the heavy gamaka ornamentation in Carnatic vocal music. The
    minimum-duration filter uses run-length encoding to identify and relabel
    short runs, preventing rapid label flickering from being misinterpreted as
    meaningful swara transitions downstream.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    This stage directly implements Burns and Ward's (1978) categorical pitch
    perception: humans quantise continuous frequencies into discrete note
    categories. The tolerance threshold models the perceptual boundary between
    "a note" and "a glide between notes". The "glide" label explicitly
    acknowledges gamaka — the continuous pitch movements that Krishnaswamy (2003)
    showed are central to Carnatic expression. By labelling rather than
    discarding gamakas, the system preserves information about ornamental density
    (via auxiliary features downstream) while enabling discrete swara analysis.
    The 12 equal-tempered bins impose a Western tuning grid — a deliberate
    trade-off noted as a limitation, since Carnatic ragas use raga-specific
    intonation.

Usage:
    python carnatic_scripts/04_swara_mapping.py
    python carnatic_scripts/04_swara_mapping.py --use-estimated
    python carnatic_scripts/04_swara_mapping.py --plot-n 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Swara definitions ────────────────────────────────────────────

# Standard Carnatic 12-swara positions in cents relative to Sa
DEFAULT_SWARA_BINS = {
    "Sa":  0,
    "Ri1": 100,   # Shuddha Rishabham
    "Ri2": 200,   # Chatushruti Rishabham
    "Ga1": 300,   # Sadharana Gandharam
    "Ga2": 400,   # Antara Gandharam
    "Ma1": 500,   # Shuddha Madhyamam
    "Ma2": 600,   # Prati Madhyamam
    "Pa":  700,   # Panchamam (fixed)
    "Da1": 800,   # Shuddha Dhaivatam
    "Da2": 900,   # Chatushruti Dhaivatam
    "Ni1": 1000,  # Kaisiki Nishadam
    "Ni2": 1100,  # Kakali Nishadam
}


def hz_to_cents_relative(f0_hz: np.ndarray, sa_hz: float) -> np.ndarray:
    """Convert f0 in Hz to cents relative to Sa, folded into [0, 1200)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(f0_hz / sa_hz)
    # Fold into one octave [0, 1200)
    cents_mod = np.mod(cents, 1200.0)
    return cents_mod


def map_to_swara(cents_values: np.ndarray,
                 swara_bins: dict,
                 tolerance_cents: float = 40.0) -> tuple:
    """
    Map each cent value to the nearest swara bin.

    Returns:
        swara_labels: array of swara names (str), "glide" for ambiguous frames
        swara_indices: array of ints (0-11 for swaras, -1 for glide)
        distances: distance in cents to the nearest bin
    """
    bin_names = list(swara_bins.keys())
    bin_cents = np.array(list(swara_bins.values()), dtype=np.float64)

    n_frames = len(cents_values)
    swara_labels = np.empty(n_frames, dtype=object)
    swara_indices = np.full(n_frames, -1, dtype=np.int32)
    distances = np.full(n_frames, np.nan, dtype=np.float64)

    for i, c in enumerate(cents_values):
        if np.isnan(c):
            swara_labels[i] = "silence"
            swara_indices[i] = -2
            continue

        # Circular distance (wrapping around 1200)
        diffs = np.abs(c - bin_cents)
        diffs = np.minimum(diffs, 1200.0 - diffs)

        nearest_idx = np.argmin(diffs)
        min_dist = diffs[nearest_idx]
        distances[i] = min_dist

        if min_dist <= tolerance_cents:
            swara_labels[i] = bin_names[nearest_idx]
            swara_indices[i] = nearest_idx
        else:
            swara_labels[i] = "glide"
            swara_indices[i] = -1

    return swara_labels, swara_indices, distances


def apply_min_duration_filter(swara_labels: np.ndarray,
                              hop_length: int,
                              sr: int,
                              min_duration_ms: float = 100.0) -> np.ndarray:
    """
    Replace swara labels shorter than min_duration with 'glide'.
    Prevents rapid flickering between swaras.
    """
    min_frames = int((min_duration_ms / 1000.0) * sr / hop_length)
    if min_frames < 2:
        return swara_labels

    filtered = swara_labels.copy()
    n = len(filtered)

    # Find runs of the same label
    i = 0
    while i < n:
        label = filtered[i]
        j = i + 1
        while j < n and filtered[j] == label:
            j += 1
        run_length = j - i

        # If a swara run is too short, mark as glide
        if label not in ("silence", "glide") and run_length < min_frames:
            filtered[i:j] = "glide"

        i = j

    return filtered


def compute_swara_summary(swara_labels: np.ndarray) -> dict:
    """Compute basic statistics about the swara trace."""
    total = len(swara_labels)
    non_silence = swara_labels[swara_labels != "silence"]

    if len(non_silence) == 0:
        return {"n_frames": total, "voiced_frames": 0}

    unique, counts = np.unique(non_silence, return_counts=True)
    dist = dict(zip(unique, counts / len(non_silence)))

    return {
        "n_frames": total,
        "voiced_frames": len(non_silence),
        "glide_ratio": dist.get("glide", 0.0),
        "top_swara": max(
            [(k, v) for k, v in dist.items() if k != "glide"],
            key=lambda x: x[1],
            default=("none", 0.0)
        )[0],
        "unique_swaras": len([k for k in dist if k not in ("glide", "silence")]),
    }


def plot_swara_trace(f0, cents_mod, swara_labels, hop_length, sr,
                     title: str, out_path: Path, swara_bins: dict):
    """Overlay swara labels on the pitch contour."""
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    # Top: pitch contour in cents with swara bin lines
    ax1.plot(t, cents_mod, linewidth=0.5, color="steelblue", alpha=0.7)
    for sname, sval in swara_bins.items():
        ax1.axhline(sval, color="gray", linewidth=0.3, alpha=0.5)
        ax1.text(t[-1] * 1.01, sval, sname, fontsize=7, va="center")
    ax1.set_ylabel("Cents (mod 1200)")
    ax1.set_ylim(-50, 1250)
    ax1.set_title(title)

    # Bottom: swara labels as colored segments
    # Assign a color to each swara
    all_labels = list(swara_bins.keys()) + ["glide", "silence"]
    cmap = plt.cm.tab20(np.linspace(0, 1, len(all_labels)))
    color_map = {label: cmap[i] for i, label in enumerate(all_labels)}

    for i in range(len(t) - 1):
        label = swara_labels[i]
        color = color_map.get(label, (0.5, 0.5, 0.5, 1.0))
        ax2.axvspan(t[i], t[i + 1], color=color, alpha=0.7)

    ax2.set_ylabel("Swara")
    ax2.set_xlabel("Time (s)")
    ax2.set_yticks([])

    # Legend (only swaras that appear)
    present = set(swara_labels)
    patches = [Patch(color=color_map[l], label=l) for l in all_labels if l in present]
    ax2.legend(handles=patches, loc="upper right", fontsize=6, ncol=4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Map pitch to swaras")
    parser.add_argument("--use-gt", action="store_true",
                        help="Use ground-truth tonic instead of estimated (default: estimated)")
    # For backwards compatibility
    parser.add_argument("--use-estimated", action="store_true",
                        help="(deprecated, now default) Use estimated tonic")
    parser.add_argument("--plot-n", type=int, default=10,
                        help="Number of diagnostic plots to generate")
    args = parser.parse_args()

    root = repo_root()

    # --- Load swara config
    swara_cfg_path = root / "configs" / "swara.yaml"
    if swara_cfg_path.exists():
        scfg = load_cfg(str(swara_cfg_path))
        swara_bins = scfg.get("bins", DEFAULT_SWARA_BINS)
        tolerance = float(scfg.get("tolerance_cents", 40.0))
        min_dur_ms = float(scfg.get("min_duration_ms", 100.0))
    else:
        swara_bins = DEFAULT_SWARA_BINS
        tolerance = 40.0
        min_dur_ms = 100.0

    # --- Load pitch data
    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    if not pitch_csv.exists():
        raise FileNotFoundError(f"Missing: {pitch_csv}. Run 02_pitch_extraction.py first.")

    df = pd.read_csv(pitch_csv)
    print(f"Loaded {len(df)} segments")

    # --- Tonic source: estimated by default, GT only with --use-gt
    use_estimated = not args.use_gt  # Default: estimated tonic
    if use_estimated:
        tonic_csv = root / "data" / "processed" / "tonic_validation.csv"
        if not tonic_csv.exists():
            raise FileNotFoundError(f"Missing: {tonic_csv}. Run 03_tonic_estimation.py first.")
        tonic_df = pd.read_csv(tonic_csv).set_index("track_id")
        print("Using ESTIMATED tonic")
    else:
        tonic_df = None
        print("Using GROUND TRUTH tonic")

    # --- Output dirs
    swara_dir = root / "data" / "processed" / "swara"
    fig_dir = root / "outputs" / "figures" / "swara_debug"
    ensure_dir(swara_dir)
    ensure_dir(fig_dir)

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Swara mapping"):
        # Get tonic
        if use_estimated and row["track_id"] in tonic_df.index:
            sa_hz = tonic_df.loc[row["track_id"], "estimated_tonic_hz"]
        else:
            sa_hz = row["gt_tonic_hz"]

        if np.isnan(sa_hz) or sa_hz <= 0:
            continue

        # Load pitch
        npz_path = root / row["pitch_npz"]
        data = np.load(npz_path)
        f0 = data["f0"]
        hop_length = int(data["hop_length"])
        sr = int(data["sr"])

        # Convert to cents relative to Sa
        cents_mod = np.full_like(f0, np.nan)
        voiced_mask = np.isfinite(f0) & (f0 > 0)
        if voiced_mask.any():
            cents_mod[voiced_mask] = hz_to_cents_relative(f0[voiced_mask], sa_hz)

        # Map to swaras
        swara_labels, swara_indices, distances = map_to_swara(
            cents_mod, swara_bins, tolerance_cents=tolerance
        )

        # Apply duration filter
        swara_labels = apply_min_duration_filter(
            swara_labels, hop_length, sr, min_duration_ms=min_dur_ms
        )

        # Save swara trace
        out_base = f"{row['track_id']}__{int(row['segment_index']):05d}"
        trace_path = swara_dir / f"{out_base}_swara.npz"
        np.savez_compressed(
            trace_path,
            swara_labels=swara_labels,
            swara_indices=swara_indices,
            cents_mod=cents_mod,
            distances=distances,
            sa_hz=sa_hz,
        )

        # Summary
        summary = compute_swara_summary(swara_labels)
        results.append({
            "track_id": row["track_id"],
            "raga": row["raga"],
            "segment_index": int(row["segment_index"]),
            "sa_hz_used": sa_hz,
            "tonic_source": "estimated" if use_estimated else "ground_truth",
            "swara_npz": str(trace_path.relative_to(root)).replace("\\", "/"),
            "pitch_npz": row["pitch_npz"],
            "glide_ratio": summary.get("glide_ratio", 0.0),
            "top_swara": summary.get("top_swara", "none"),
            "unique_swaras": summary.get("unique_swaras", 0),
            "voiced_frames": summary.get("voiced_frames", 0),
        })

        # Diagnostic plots
        if i < args.plot_n:
            title = (f"{row['track_id']} seg {int(row['segment_index'])} | "
                     f"{row['raga']} | Sa={sa_hz:.1f} Hz")
            plot_swara_trace(
                f0, cents_mod, swara_labels, hop_length, sr,
                title, fig_dir / f"{out_base}_swara.png", swara_bins
            )

    # --- Save summary CSV
    out_csv = root / "data" / "processed" / "swara_all.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    # --- Print summary
    print(f"\nDone. Mapped {len(results)} segments.")
    print(f"Swara CSV: {out_csv}")
    print(f"Swara traces: {swara_dir}")
    print(f"\nGlide ratio — mean: {res_df['glide_ratio'].mean():.3f}, "
          f"median: {res_df['glide_ratio'].median():.3f}")
    print(f"Average unique swaras per segment: {res_df['unique_swaras'].mean():.1f}")

    # Top swaras across dataset
    print(f"\nMost common top swara per segment:")
    print(res_df["top_swara"].value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
