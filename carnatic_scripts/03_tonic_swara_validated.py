"""
03_tonic_swara_validated.py -- Stage 3 variant: Swara-Validated Ensemble

Runs sa_mass AND harmonic independently. For each estimate, computes
a lightweight "swara hit rate" -- what fraction of voiced frames land
within 40 cents of a swara bin (same tolerance as Stage 4).

Picks the estimate with the HIGHER swara hit rate. The correct tonic
will produce pitches that land on swara bins; the wrong tonic shifts
everything so frames fall between bins.

This uses the musical structure itself as validation.

Usage:
    python carnatic_scripts/03_tonic_swara_validated.py
"""
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

def hz_to_cents(f_hz, ref_hz):
    return 1200.0 * np.log2(f_hz / ref_hz)

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

def get_mass_around_bin(hist, center_cents, window_cents=30.0, bins_per_octave=120):
    bin_width = 1200.0 / bins_per_octave
    w = int(np.ceil(window_cents / bin_width))
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave
    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]
    return float(mass)

def tonic_score_sa_only(hist, bpo=120):
    if hist.sum() <= 0: return 0.0
    return get_mass_around_bin(hist, 0.0, 30.0, bpo)

def tonic_score_harmonic(hist, bpo=120, pa_w=0.5):
    if hist.sum() <= 0: return 0.0
    sa = get_mass_around_bin(hist, 0.0, 30.0, bpo)
    pa = get_mass_around_bin(hist, 702.0, 30.0, bpo)
    return float(sa + pa_w * pa)

def estimate_tonic(f0_hz, score_fn, candidate_min=80.0, candidate_max=300.0,
                   step_cents=10.0, bins_per_octave=120, min_voiced=200):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < min_voiced:
        return float("nan"), 0.0, 0.0
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
    for alt_hz in [best_hz * 2.0, best_hz / 2.0]:
        if 60.0 <= alt_hz <= 600.0:
            hist_alt = pitch_class_histogram(f0, alt_hz, bins_per_octave)
            score_alt = score_fn(hist_alt)
            if score_alt > best_score:
                second_score = best_score
                best_score = score_alt
                best_hz = alt_hz
    conf = 0.0 if best_score <= 0 else float(
        (best_score - second_score) / (best_score + 1e-9))
    return float(best_hz), conf, float(best_score)


def swara_hit_rate(f0_hz: np.ndarray, tonic_hz: float,
                   tolerance_cents: float = 40.0) -> float:
    """
    Compute the fraction of voiced frames that land within tolerance
    of a swara bin (0, 100, 200, ..., 1100 cents).

    This is the same criterion Stage 4 uses to distinguish swara vs glide.
    Higher hit rate = pitches land on swaras = tonic is likely correct.
    Lower hit rate = pitches fall between swaras = tonic is likely wrong.
    """
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size == 0:
        return 0.0

    cents = 1200.0 * np.log2(f0 / tonic_hz)
    cents_mod = np.mod(cents, 1200.0)

    # 12 swara bins at 100-cent intervals
    swara_bins = np.arange(0, 1200, 100, dtype=np.float64)

    # For each frame, find distance to nearest swara bin (circular)
    # Vectorized: compute distance to all bins, take min
    # Shape: (n_frames, 12)
    diffs = np.abs(cents_mod[:, None] - swara_bins[None, :])
    # Circular distance: min(diff, 1200 - diff)
    diffs = np.minimum(diffs, 1200.0 - diffs)
    min_dist = diffs.min(axis=1)

    # Count frames within tolerance
    n_hits = (min_dist <= tolerance_cents).sum()
    return float(n_hits / f0.size)


def cents_error(est, gt):
    if np.isnan(est) or np.isnan(gt) or est <= 0 or gt <= 0:
        return float("nan")
    return abs(1200.0 * np.log2(est / gt))

def octave_corrected_cents_error(est, gt):
    if np.isnan(est) or np.isnan(gt) or est <= 0 or gt <= 0:
        return float("nan")
    raw = 1200.0 * np.log2(est / gt)
    folded = raw % 1200.0
    if folded > 600: folded -= 1200.0
    return abs(folded)

def is_octave_error(est, gt, thresh=50.0):
    if np.isnan(est) or np.isnan(gt): return False
    return cents_error(est, gt) > 600 and octave_corrected_cents_error(est, gt) < thresh


def main():
    root = repo_root()
    tcfg = load_cfg(str(root / "configs" / "tonic.yaml")) if (root / "configs" / "tonic.yaml").exists() else {}

    candidate_min = float(tcfg.get("candidate_min_hz", 80.0))
    candidate_max = float(tcfg.get("candidate_max_hz", 300.0))
    step_cents = float(tcfg.get("step_cents", 10.0))
    bins_per_octave = int(tcfg.get("bins_per_octave", 120))
    min_voiced = int(tcfg.get("min_voiced_frames", 200))

    pitch_csv = root / "data" / "processed" / "pitch_all.csv"
    df = pd.read_csv(pitch_csv)
    print(f"Loaded {len(df)} segments, {df['track_id'].nunique()} tracks")

    out_dir = root / "data" / "processed"
    fig_dir = root / "outputs" / "figures" / "tonic_validation"
    ensure_dir(fig_dir)

    methods = {
        "sa_mass": lambda h: tonic_score_sa_only(h, bins_per_octave),
        "harmonic": lambda h: tonic_score_harmonic(h, bins_per_octave, 0.5),
    }

    rows = []
    for track_id, group in tqdm(df.groupby("track_id"), desc="Tonic (swara-validated)"):
        gt_tonic = group["gt_tonic_hz"].iloc[0]
        raga = group["raga"].iloc[0]

        all_f0 = []
        for _, seg_row in group.iterrows():
            data = np.load(root / seg_row["pitch_npz"])
            f0_valid = data["f0"][np.isfinite(data["f0"]) & (data["f0"] > 0)]
            if f0_valid.size > 0:
                all_f0.append(f0_valid)

        if len(all_f0) == 0:
            rows.append({"track_id": track_id, "raga": raga,
                         "gt_tonic_hz": gt_tonic, "estimated_tonic_hz": float("nan"),
                         "confidence": 0.0, "chosen_method": "none",
                         "swara_hit_rate": 0.0,
                         "cents_error": float("nan"),
                         "octave_corrected_cents_error": float("nan"),
                         "within_50_cents": False, "is_octave_error": False,
                         "n_segments": len(group)})
            continue

        f0_concat = np.concatenate(all_f0)

        # Run each method
        results = {}
        for name, score_fn in methods.items():
            hz, conf, score = estimate_tonic(
                f0_concat, score_fn, candidate_min, candidate_max,
                step_cents, bins_per_octave, min_voiced)
            # Compute swara hit rate for this estimate
            shr = swara_hit_rate(f0_concat, hz) if not np.isnan(hz) and hz > 0 else 0.0
            results[name] = {"hz": hz, "conf": conf, "score": score, "shr": shr}

        # Pick the estimate with HIGHER swara hit rate
        # If equal, prefer harmonic
        if results["sa_mass"]["shr"] > results["harmonic"]["shr"] + 0.01:
            chosen = "sa_mass"
        else:
            chosen = "harmonic"

        est_hz = results[chosen]["hz"]
        est_conf = results[chosen]["conf"]
        est_shr = results[chosen]["shr"]

        raw_err = cents_error(est_hz, gt_tonic)
        oc_err = octave_corrected_cents_error(est_hz, gt_tonic)

        rows.append({
            "track_id": track_id, "raga": raga,
            "gt_tonic_hz": gt_tonic, "estimated_tonic_hz": est_hz,
            "confidence": est_conf, "chosen_method": chosen,
            "swara_hit_rate": est_shr,
            "sa_mass_hz": results["sa_mass"]["hz"],
            "sa_mass_shr": results["sa_mass"]["shr"],
            "harmonic_hz": results["harmonic"]["hz"],
            "harmonic_shr": results["harmonic"]["shr"],
            "cents_error": raw_err,
            "octave_corrected_cents_error": oc_err,
            "within_50_cents": oc_err <= 50 if not np.isnan(oc_err) else False,
            "is_octave_error": is_octave_error(est_hz, gt_tonic),
            "n_segments": len(group),
        })

    val_df = pd.DataFrame(rows)
    out_csv = out_dir / "tonic_validation_swara_validated.csv"
    val_df.to_csv(out_csv, index=False)

    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave = int(valid["is_octave_error"].sum())
    chosen_counts = valid["chosen_method"].value_counts()

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION -- SWARA-VALIDATED ENSEMBLE")
    print(f"{'='*60}")
    print(f"Tracks evaluated:              {n_total}")
    print(f"Within 50 cents (oc):          {n_within_50}/{n_total} ({100*n_within_50/n_total:.1f}%)")
    print(f"Octave errors:                 {n_octave}/{n_total} ({100*n_octave/n_total:.1f}%)")
    print(f"Chosen method breakdown:       {dict(chosen_counts)}")
    print(f"Median error (oc):             {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):               {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")

    print(f"\nWorst 5:")
    for _, r in valid.nlargest(5, "octave_corrected_cents_error").iterrows():
        print(f"  {r['track_id']}: est={r['estimated_tonic_hz']:.1f}Hz, "
              f"gt={r['gt_tonic_hz']:.1f}Hz, err={r['octave_corrected_cents_error']:.1f}c, "
              f"chose={r['chosen_method']}, "
              f"shr_sa={r['sa_mass_shr']:.3f} shr_ha={r['harmonic_shr']:.3f}, "
              f"raga={r['raga']}")

    # Show disagreement tracks with SHR comparison
    disagree = valid[valid["sa_mass_hz"].apply(lambda x: not np.isnan(x))]
    disagree = disagree[disagree.apply(
        lambda r: octave_corrected_cents_error(r["sa_mass_hz"], r["harmonic_hz"]) > 50
        if not np.isnan(r["sa_mass_hz"]) and not np.isnan(r["harmonic_hz"])
        and r["sa_mass_hz"] > 0 and r["harmonic_hz"] > 0
        else False, axis=1)]

    if len(disagree) > 0:
        print(f"\nDisagreement tracks ({len(disagree)}) -- SHR comparison:")
        for _, r in disagree.iterrows():
            sa_err = octave_corrected_cents_error(r["sa_mass_hz"], r["gt_tonic_hz"])
            ha_err = octave_corrected_cents_error(r["harmonic_hz"], r["gt_tonic_hz"])
            correct_method = "sa_mass" if sa_err < ha_err else "harmonic"
            picked_right = r["chosen_method"] == correct_method
            print(f"  {r['track_id']}: "
                  f"sa_mass(shr={r['sa_mass_shr']:.3f}, err={sa_err:.0f}c) vs "
                  f"harmonic(shr={r['harmonic_shr']:.3f}, err={ha_err:.0f}c) "
                  f"-> chose {r['chosen_method']} "
                  f"{'CORRECT' if picked_right else 'WRONG'}")

    # Compare vs baseline
    baseline_csv = out_dir / "tonic_validation.csv"
    if baseline_csv.exists():
        bv = pd.read_csv(baseline_csv).dropna(subset=["cents_error"])
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
