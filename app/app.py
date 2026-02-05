import sys
from pathlib import Path
import numpy as np
import streamlit as st
import librosa
import matplotlib.pyplot as plt


# ---------- helpers ----------
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

    # +/- 30 cents around Sa (0 cents) using 10-cent bins => +/-3 bins
    w = 3
    sa_mass = hist[0 : w + 1].sum() + hist[-w:].sum()

    eps = 1e-12
    entropy = -np.sum(hist * np.log(hist + eps))
    peaky = np.log(bins_per_octave) - entropy

    return float(sa_mass + 0.15 * peaky)


def estimate_tonic(f0_hz: np.ndarray, candidate_min: float = 80.0, candidate_max: float = 300.0, step_cents: float = 10.0):
    f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f0.size < 20:
        return float("nan"), 0.0, 0.0, 0.0

    cents_range = 1200.0 * np.log2(candidate_max / candidate_min)
    n_steps = int(np.floor(cents_range / step_cents)) + 1

    best = (-1.0, None)      # (score, hz)
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


def plot_pitch(y, sr, f0, voiced_prob, hop_length: int):
    t_audio = np.arange(len(y)) / sr
    t_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    fig1 = plt.figure(figsize=(10, 2.8))
    plt.plot(t_audio, y, linewidth=0.6)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()

    fig2 = plt.figure(figsize=(10, 2.8))
    plt.plot(t_f0, f0, linewidth=1.0)
    plt.title("Pitch contour F0 (Hz) — librosa.pyin")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.tight_layout()

    fig3 = plt.figure(figsize=(10, 2.4))
    plt.plot(t_f0, voiced_prob, linewidth=1.0)
    plt.title("Voiced probability")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1)
    plt.tight_layout()

    return fig1, fig2, fig3


# ---------- UI ----------
st.set_page_config(page_title="RagaLens — Pitch + Sa Demo", layout="wide")
st.title("RagaLens — Pitch + Sa (Tonic) Baseline Demo")
st.caption("Upload a clip or pick a pre-segmented file from data/interim, then run pitch + tonic estimation.")

root = repo_root()

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Input")
    mode = st.radio("Choose input source", ["Upload file", "Pick from data/interim"], horizontal=True)

    audio_bytes = None
    filename = None

    if mode == "Upload file":
        up = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
        if up is not None:
            audio_bytes = up.read()
            filename = up.name

    else:
        interim_dir = root / "data" / "interim"
        if not interim_dir.exists():
            st.warning("data/interim not found. Run preprocessing first.")
        else:
            wavs = sorted([p for p in interim_dir.glob("*.wav")])
            if not wavs:
                st.warning("No .wav files in data/interim yet.")
            else:
                chosen = st.selectbox("Select a segment", wavs, format_func=lambda p: p.name)
                filename = chosen.name
                audio_bytes = chosen.read_bytes()

    st.divider()
    st.subheader("Settings")
    target_sr = st.select_slider("Resample to", options=[16000, 22050, 44100], value=22050)
    hop_length = st.select_slider("Hop length (speed vs detail)", options=[256, 512, 1024], value=512)
    fmin_note = st.selectbox("Fmin (lowest pitch)", ["C2", "E2", "G2"], index=0)
    fmax_note = st.selectbox("Fmax (highest pitch)", ["C6", "C7"], index=1)

    run_btn = st.button("Analyze", type="primary", use_container_width=True)

with colB:
    st.subheader("Output")
    if audio_bytes is None:
        st.info("Choose a file (upload or pick a segment), then click Analyze.")
    else:
        st.audio(audio_bytes)

    if run_btn and audio_bytes is not None:
        # Save bytes to a temp file to let librosa load reliably
        tmp_path = root / "outputs" / "tmp_upload"
        tmp_path.mkdir(parents=True, exist_ok=True)
        temp_file = tmp_path / f"_tmp_{filename}"
        temp_file.write_bytes(audio_bytes)

        with st.spinner("Loading audio + extracting pitch (pyin)…"):
            y, sr = librosa.load(temp_file, sr=target_sr, mono=True)

            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz(fmin_note),
                fmax=librosa.note_to_hz(fmax_note),
                sr=sr,
                hop_length=hop_length,
            )

        with st.spinner("Estimating Sa (tonic)…"):
            sa_hz, conf, best_score, second_score = estimate_tonic(
                f0,
                candidate_min=80.0,
                candidate_max=300.0,
                step_cents=10.0,
            )

        st.success("Done.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Sa (Hz)", "—" if np.isnan(sa_hz) else f"{sa_hz:.2f}")
        m2.metric("Confidence", f"{conf:.3f}")
        m3.metric("Voiced ratio", f"{np.nanmean(voiced_flag.astype(float)):.3f}")

        st.caption(f"Scores: best={best_score:.4f}, second={second_score:.4f}")

        fig1, fig2, fig3 = plot_pitch(y, sr, f0, voiced_prob, hop_length)
        st.pyplot(fig1, clear_figure=True)
        st.pyplot(fig2, clear_figure=True)
        st.pyplot(fig3, clear_figure=True)
