"""
03_tonic_ensemble_v2.py — Stage 3 Variant: Harmonic-First Ensemble

WHAT THIS SCRIPT DOES:
    Runs both sa_mass and harmonic scoring independently. When the two methods
    disagree, prefers the harmonic estimate (which is correct 7 out of 10 times
    on disagreements in the dataset). Falls back to sa_mass only when harmonic
    confidence is very low. A refinement over the symmetric ensemble (v1).

LOGIC AND DESIGN:
    Analysis of disagreement cases showed harmonic scoring has a 70% win rate
    over sa_mass when they disagree. This asymmetric prior was hard-coded:
    harmonic is the default, sa_mass is the fallback only below a confidence
    threshold. The confidence threshold was tuned on the full dataset (not
    cross-validated), making this a post-hoc analysis rather than a deployable
    improvement.

COGNITIVE / MUSICOLOGICAL RATIONALE:
    The harmonic method's superiority reflects the acoustic reality that the
    tanpura drone (Sa+Pa) is the most reliable tonic cue in Carnatic
    performance. The sa_mass method captures only the Sa peak, missing the
    confirmatory Pa evidence. This experiment quantifies the advantage of
    multi-harmonic cues over single-peak detection, confirming that the
    tanpura's harmonic structure provides redundant tonic evidence. Result:
    marginal improvement insufficient to justify the added complexity.

INPUT:
    data/processed/pitch_all.csv       — per-segment pitch csv with track_id, gt_tonic_hz, raga, pitch_npz
    data/processed/pitch_all/*.npz     — per-segment f0 arrays
    configs/tonic.yaml                 — optional tuning parameters

OUTPUT:
    data/processed/tonic_validation_ensemble_v2.csv     — per-track tonic estimates and errors
    stdout                                              — summary statistics and baseline comparison

Usage:
    python carnatic_scripts/experiments/03_tonic_ensemble_v2.py
"""
# cli argument parsing
import argparse
# filesystem path handling
from pathlib import Path

# numerical array operations
import numpy as np
# tabular data manipulation
import pandas as pd
# yaml config file loading
import yaml
# non-interactive matplotlib backend for headless rendering
import matplotlib
matplotlib.use("Agg")
# plotting and figure creation
import matplotlib.pyplot as plt
# progress bar for loops
from tqdm import tqdm


# find the repo root (two levels up from experiments/)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

# create a directory and all parents if they don't exist
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# load a yaml config file and return it as a dict
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# convert a hz frequency array to cents relative to a reference pitch
def hz_to_cents(f_hz, ref_hz):
    return 1200.0 * np.log2(f_hz / ref_hz)

# build a normalised pitch-class histogram from f0 values given a candidate tonic
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

# sum histogram mass in a window around a target cent value
def get_mass_around_bin(hist, center_cents, window_cents=30.0, bins_per_octave=120):
    bin_width = 1200.0 / bins_per_octave
    w = int(np.ceil(window_cents / bin_width))
    center_bin = int(round(center_cents / bin_width)) % bins_per_octave
    mass = 0.0
    for offset in range(-w, w + 1):
        idx = (center_bin + offset) % bins_per_octave
        mass += hist[idx]
    return float(mass)

# score a candidate tonic using only the sa (0-cent) peak mass
def tonic_score_sa_only(hist, bpo=120):
    if hist.sum() <= 0: return 0.0
    return get_mass_around_bin(hist, 0.0, 30.0, bpo)

# score a candidate tonic using sa peak plus weighted pa (702-cent) peak
def tonic_score_harmonic(hist, bpo=120, pa_w=0.5):
    if hist.sum() <= 0: return 0.0
    sa = get_mass_around_bin(hist, 0.0, 30.0, bpo)
    pa = get_mass_around_bin(hist, 702.0, 30.0, bpo)
    return float(sa + pa_w * pa)

# sweep candidate tonics and return the best hz, confidence, and score using a given scoring function
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

# compute absolute cents error between estimated and ground-truth tonic
def cents_error(est, gt):
    if np.isnan(est) or np.isnan(gt) or est <= 0 or gt <= 0:
        return float("nan")
    return abs(1200.0 * np.log2(est / gt))

# compute octave-folded cents error to ignore octave mismatches
def octave_corrected_cents_error(est, gt):
    if np.isnan(est) or np.isnan(gt) or est <= 0 or gt <= 0:
        return float("nan")
    raw = 1200.0 * np.log2(est / gt)
    folded = raw % 1200.0
    if folded > 600: folded -= 1200.0
    return abs(folded)

# check whether an estimate is off by exactly one octave from ground truth
def is_octave_error(est, gt, thresh=50.0):
    if np.isnan(est) or np.isnan(gt): return False
    return cents_error(est, gt) > 600 and octave_corrected_cents_error(est, gt) < thresh


# run the harmonic-first ensemble tonic estimation and print validation results
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
    for track_id, group in tqdm(df.groupby("track_id"), desc="Tonic (ensemble_v2)"):
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
                         "cents_error": float("nan"),
                         "octave_corrected_cents_error": float("nan"),
                         "within_50_cents": False, "is_octave_error": False,
                         "n_segments": len(group)})
            continue

        f0_concat = np.concatenate(all_f0)

        results = {}
        for name, score_fn in methods.items():
            hz, conf, score = estimate_tonic(
                f0_concat, score_fn, candidate_min, candidate_max,
                step_cents, bins_per_octave, min_voiced)
            results[name] = {"hz": hz, "conf": conf, "score": score}

        # FLIPPED LOGIC: prefer harmonic, fall back to sa_mass
        # only when harmonic confidence is very low
        sa_hz = results["sa_mass"]["hz"]
        ha_hz = results["harmonic"]["hz"]
        sa_conf = results["sa_mass"]["conf"]
        ha_conf = results["harmonic"]["conf"]

        # Check if they agree (within 50 cents octave-corrected)
        if not np.isnan(sa_hz) and not np.isnan(ha_hz) and sa_hz > 0 and ha_hz > 0:
            agree = octave_corrected_cents_error(sa_hz, ha_hz) <= 50
        else:
            agree = False

        if agree:
            # Both agree -- use harmonic (slightly better overall)
            chosen = "harmonic"
        else:
            # They disagree -- use harmonic unless its confidence is very low
            # From the data: harmonic is right 7/10 times on disagreements
            chosen = "harmonic"
            # Only override if harmonic conf is near zero and sa_mass is strong
            if ha_conf < 0.01 and sa_conf > 0.1:
                chosen = "sa_mass"

        est_hz = results[chosen]["hz"]
        est_conf = results[chosen]["conf"]

        raw_err = cents_error(est_hz, gt_tonic)
        oc_err = octave_corrected_cents_error(est_hz, gt_tonic)

        rows.append({
            "track_id": track_id, "raga": raga,
            "gt_tonic_hz": gt_tonic, "estimated_tonic_hz": est_hz,
            "confidence": est_conf, "chosen_method": chosen,
            "sa_mass_hz": sa_hz, "sa_mass_conf": sa_conf,
            "harmonic_hz": ha_hz, "harmonic_conf": ha_conf,
            "methods_agree": agree,
            "cents_error": raw_err,
            "octave_corrected_cents_error": oc_err,
            "within_50_cents": oc_err <= 50 if not np.isnan(oc_err) else False,
            "is_octave_error": is_octave_error(est_hz, gt_tonic),
            "n_segments": len(group),
        })

    val_df = pd.DataFrame(rows)
    out_csv = out_dir / "tonic_validation_ensemble_v2.csv"
    val_df.to_csv(out_csv, index=False)

    valid = val_df.dropna(subset=["cents_error"])
    n_total = len(valid)
    n_within_50 = int(valid["within_50_cents"].sum())
    n_octave = int(valid["is_octave_error"].sum())

    print(f"\n{'='*60}")
    print(f"TONIC VALIDATION -- HARMONIC-FIRST ENSEMBLE")
    print(f"{'='*60}")
    print(f"Tracks evaluated:              {n_total}")
    print(f"Within 50 cents (oc):          {n_within_50}/{n_total} ({100*n_within_50/n_total:.1f}%)")
    print(f"Octave errors:                 {n_octave}/{n_total} ({100*n_octave/n_total:.1f}%)")
    print(f"Median error (oc):             {valid['octave_corrected_cents_error'].median():.1f} cents")
    print(f"Mean error (oc):               {valid['octave_corrected_cents_error'].mean():.1f} cents")
    print(f"{'='*60}")

    print(f"\nWorst 5:")
    for _, r in valid.nlargest(5, "octave_corrected_cents_error").iterrows():
        print(f"  {r['track_id']}: est={r['estimated_tonic_hz']:.1f}Hz, "
              f"gt={r['gt_tonic_hz']:.1f}Hz, err={r['octave_corrected_cents_error']:.1f}c, "
              f"chose={r['chosen_method']}, raga={r['raga']}")

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
