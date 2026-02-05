from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def hz_to_cents_ratio(f_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    return 1200.0 * np.log2(f_hz / tonic_hz)


def pitch_class_histogram(f0_hz: np.ndarray, tonic_hz: float, bins_per_octave: int = 120) -> np.ndarray:
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size == 0:
        return np.zeros(bins_per_octave, dtype=np.float32)

    cents = hz_to_cents_ratio(f0, tonic_hz)
    cents_mod = np.mod(cents, 1200.0)
    edges = np.linspace(0.0, 1200.0, bins_per_octave + 1)
    hist, _ = np.histogram(cents_mod, bins=edges)
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def tonic_score_from_hist(hist: np.ndarray, bins_per_octave: int = 120) -> float:
    if hist.sum() <= 0:
        return 0.0
    w = 3  # +/-30 cents with 10-cent bins
    sa_mass = hist[0 : w + 1].sum() + hist[-w:].sum()

    eps = 1e-12
    entropy = -np.sum(hist * np.log(hist + eps))
    peaky = np.log(bins_per_octave) - entropy
    return float(sa_mass + 0.15 * peaky)


def estimate_tonic_from_f0(f0_hz: np.ndarray, candidate_min=80.0, candidate_max=300.0, step_cents=10.0):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < 200:  # stricter since we aggregate
        return float("nan"), 0.0, 0.0, 0.0

    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)
    second = (-1.0, None)

    for k in range(n_steps):
        tonic_hz = candidate_min * (2 ** ((k * step_cents) / 1200.0))
        hist = pitch_class_histogram(f0, tonic_hz, bins_per_octave=120)
        score = tonic_score_from_hist(hist, bins_per_octave=120)

        if score > best[0]:
            second = best
            best = (score, tonic_hz)
        elif score > second[0]:
            second = (score, tonic_hz)

    best_score, best_hz = best
    second_score, _ = second

    conf = 0.0 if best_score <= 0 else float((best_score - second_score) / (best_score + 1e-9))
    return float(best_hz), conf, float(best_score), float(second_score)


def main():
    root = repo_root()

    pitch_csv = root / "data" / "processed" / "pitch_baseline_20.csv"
    if not pitch_csv.exists():
        alt = root / "data" / "processed" / "pitch_baseline_10.csv"
        if alt.exists():
            pitch_csv = alt
        else:
            raise FileNotFoundError("Run scripts/02_pitch_baseline.py first (pitch_baseline_*.csv missing).")

    df = pd.read_csv(pitch_csv)

    out_csv = root / "data" / "processed" / "tonic_per_track.csv"
    rows = []

    for track_id, g in tqdm(df.groupby("track_id"), desc="Tracks"):
        all_f0 = []
        voiced_ratios = []
        f0_medians = []

        for _, row in g.iterrows():
            npz_path = root / row["pitch_npz"]
            data = np.load(npz_path)
            f0 = data["f0"]
            voiced_flag = data["voiced_flag"]

            f0v = f0[np.isfinite(f0) & (f0 > 0)]
            if f0v.size > 0:
                all_f0.append(f0v)

            voiced_ratios.append(float(np.nanmean(voiced_flag.astype(float))))
            f0_medians.append(float(np.nanmedian(f0)) if np.any(~np.isnan(f0)) else float("nan"))

        if len(all_f0) == 0:
            sa_hz, conf, best_score, second_score = float("nan"), 0.0, 0.0, 0.0
            n_frames = 0
        else:
            f0_concat = np.concatenate(all_f0)
            sa_hz, conf, best_score, second_score = estimate_tonic_from_f0(f0_concat)
            n_frames = int(f0_concat.size)

        rows.append(
            {
                "track_id": track_id,
                "sa_hz": sa_hz,
                "confidence": conf,
                "best_score": best_score,
                "second_score": second_score,
                "n_f0_frames": n_frames,
                "avg_voiced_ratio": float(np.mean(voiced_ratios)) if voiced_ratios else float("nan"),
                "avg_f0_median_hz": float(np.nanmean(f0_medians)) if f0_medians else float("nan"),
                "n_segments_used": int(len(g)),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nDone.")
    print("Track-level tonic CSV:", out_csv)


if __name__ == "__main__":
    main()
