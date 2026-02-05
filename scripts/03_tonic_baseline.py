from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def hz_to_cents_ratio(f_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    # cents = 1200 * log2(f / tonic)
    return 1200.0 * np.log2(f_hz / tonic_hz)


def pitch_class_histogram(f0_hz: np.ndarray, tonic_hz: float, bins_per_octave: int = 120) -> np.ndarray:
    """
    Build a pitch-class (mod octave) histogram in cents.
    bins_per_octave=120 => 10 cent bins.
    """
    # remove nan and non-positive
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size == 0:
        return np.zeros(bins_per_octave, dtype=np.float32)

    cents = hz_to_cents_ratio(f0, tonic_hz)
    cents_mod = np.mod(cents, 1200.0)  # pitch class within octave

    # histogram
    edges = np.linspace(0.0, 1200.0, bins_per_octave + 1)
    hist, _ = np.histogram(cents_mod, bins=edges)
    hist = hist.astype(np.float32)

    # normalize to probability distribution
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def tonic_score_from_hist(hist: np.ndarray, bins_per_octave: int = 120) -> float:
    """
    Distribution-based score:
    - Sa should create a strong peak near 0 cents.
    - We score peak mass in a small window around 0 cents,
      plus a small bonus for overall 'peaky-ness' (lower entropy).
    """
    if hist.sum() <= 0:
        return 0.0

    # window around Sa: +/- 30 cents => 60 cents window
    # with 10-cent bins => +/-3 bins
    sa_bin = 0
    w = 3
    sa_mass = hist[sa_bin:sa_bin + w + 1].sum() + hist[-w:].sum()

    # entropy bonus (peaky distributions score higher)
    eps = 1e-12
    entropy = -np.sum(hist * np.log(hist + eps))
    # convert to "peaky" measure: max entropy is log(bins)
    peaky = np.log(bins_per_octave) - entropy

    return float(sa_mass + 0.15 * peaky)


def estimate_tonic(f0_hz: np.ndarray, sr: int, candidate_min: float = 80.0, candidate_max: float = 300.0,
                   step_cents: float = 10.0):
    """
    Search tonic candidates in [candidate_min, candidate_max] Hz.
    We step in cents (log-frequency) to cover musically uniform spacing.
    Returns: (best_tonic_hz, confidence, best_score, second_score)
    """
    # voiced-only f0
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < 20:
        return float("nan"), 0.0, 0.0, 0.0

    # candidates in cents grid relative to candidate_min
    # f = f_min * 2^(c/1200)
    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)      # (score, hz)
    second = (-1.0, None)

    for k in range(n_steps):
        c = k * step_cents
        tonic_hz = candidate_min * (2 ** (c / 1200.0))

        hist = pitch_class_histogram(f0, tonic_hz, bins_per_octave=120)
        score = tonic_score_from_hist(hist, bins_per_octave=120)

        if score > best[0]:
            second = best
            best = (score, tonic_hz)
        elif score > second[0]:
            second = (score, tonic_hz)

    best_score, best_hz = best
    second_score, _ = second

    # confidence: separation between top two
    if best_score <= 0:
        conf = 0.0
    else:
        conf = float((best_score - second_score) / (best_score + 1e-9))

    return float(best_hz), conf, float(best_score), float(second_score)


def main():
    root = repo_root()

    pitch_csv = root / "data" / "processed" / "pitch_baseline_20.csv"
    if not pitch_csv.exists():
        # if you ran the faster/10 version, adjust here
        alt = root / "data" / "processed" / "pitch_baseline_10.csv"
        if alt.exists():
            pitch_csv = alt
        else:
            raise FileNotFoundError("Could not find pitch_baseline_20.csv (or _10.csv). Run Week 2 first.")

    df = pd.read_csv(pitch_csv)

    out_csv = root / "data" / "processed" / "tonic_baseline.csv"

    results = []

    print("Reading pitch summary from:", pitch_csv)
    print("Writing tonic estimates to:", out_csv)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tonic"):
        npz_path = root / row["pitch_npz"]
        data = np.load(npz_path)

        f0 = data["f0"]
        sr = int(data["sr"])

        tonic_hz, conf, best_score, second_score = estimate_tonic(
            f0, sr,
            candidate_min=80.0,
            candidate_max=300.0,
            step_cents=10.0
        )

        results.append(
            {
                "track_id": row["track_id"],
                "segment_index": int(row["segment_index"]),
                "segment_path": row["segment_path"],
                "pitch_npz": row["pitch_npz"],
                "sa_hz": tonic_hz,
                "confidence": conf,
                "best_score": best_score,
                "second_score": second_score,
                "voiced_ratio": float(row["voiced_ratio"]),
                "f0_median_hz": float(row["f0_median_hz"]),
            }
        )

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("\nDone.")
    print("Tonic CSV:", out_csv)


if __name__ == "__main__":
    main()
