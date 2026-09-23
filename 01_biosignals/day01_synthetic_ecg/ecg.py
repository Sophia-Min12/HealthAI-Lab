"""Day 1 — A synthetic ECG, and why synthetic is the honest choice.

Every later day in Level 1 detects something in this signal, so the signal
comes first — and it is **generated, not downloaded**.

That is not a shortcut around a licensing problem. It is the only way to
have exact ground truth. When Day 4's R-peak detector claims it found 74
beats, the question is whether those are *the* beats, and only a signal
whose beat positions were written down at generation time can answer it.
A public ECG database comes with expert annotations that are themselves
estimates, made by people who disagreed with each other.

So this module returns the sample positions of every R-peak it placed,
and every detector in this repo is scored against that list.

The waveform is a deliberate caricature: P wave, QRS complex, T wave, each
a Gaussian bump of roughly the right width and height. It is not a
physiological model, it will not fool a cardiologist, and it is entirely
sufficient for testing a peak finder — which is the honest description of
what most "synthetic ECG" code is for.
"""

from __future__ import annotations

import numpy as np

#: Typical adult resting heart rate, in beats per minute.
DEFAULT_BPM = 72.0

#: Sampling rate in Hz. 250 is a common clinical monitor rate and is
#: comfortably above the Nyquist limit for the ~40 Hz content of a QRS.
DEFAULT_FS = 250.0


def _gaussian(t: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    """One smooth bump, used for each of the P, Q, R, S and T deflections."""
    return height * np.exp(-0.5 * ((t - centre) / width) ** 2)


def beat_waveform(fs: float = DEFAULT_FS, duration: float = 0.8) -> np.ndarray:
    """One cardiac cycle, as a sum of five Gaussians.

    The parameters below are shaped to look like lead II: a small upright
    P, a sharp tall R flanked by small negative Q and S, and a broad
    low T. Amplitudes are in millivolts.

    The R peak sits at ``0.25 * duration`` deliberately, leaving room for
    the P wave before it. Day 4's detector must not assume that position.

    >>> wave = beat_waveform(fs=250.0, duration=0.8)
    >>> len(wave)
    200
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    samples = int(round(fs * duration))
    t = np.linspace(0.0, duration, samples, endpoint=False)
    r_time = 0.25 * duration

    wave = np.zeros(samples)
    wave += _gaussian(t, r_time - 0.16, 0.025, 0.12)   # P wave
    wave += _gaussian(t, r_time - 0.02, 0.008, -0.16)  # Q
    wave += _gaussian(t, r_time, 0.010, 1.20)          # R
    wave += _gaussian(t, r_time + 0.02, 0.008, -0.28)  # S
    wave += _gaussian(t, r_time + 0.18, 0.045, 0.30)   # T wave
    return wave


def synthesise_ecg(
    duration: float = 10.0,
    bpm: float = DEFAULT_BPM,
    fs: float = DEFAULT_FS,
    hrv: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(signal, r_peak_samples)``.

    ``hrv`` is the standard deviation of beat-to-beat interval jitter, in
    seconds. At 0 the rhythm is metronomic, which is unphysiological — a
    real heart never is — so Day 4 uses a non-zero value and Day 1 keeps
    both available to show the difference.

    The returned peak positions are **ground truth**, not an estimate.
    Everything downstream is scored against them.

    >>> signal, peaks = synthesise_ecg(duration=4.0, bpm=60.0, fs=100.0)
    >>> len(signal), len(peaks)
    (400, 4)
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if hrv < 0:
        raise ValueError(f"hrv must be non-negative, got {hrv}")

    rng = np.random.default_rng(seed)
    total_samples = int(round(duration * fs))
    signal = np.zeros(total_samples)
    peaks: list[int] = []

    mean_interval = 60.0 / bpm
    beat = beat_waveform(fs=fs, duration=mean_interval)
    r_offset = int(round(0.25 * len(beat)))

    time = 0.0
    while True:
        start = int(round(time * fs))
        if start >= total_samples:
            break
        end = min(start + len(beat), total_samples)
        signal[start:end] += beat[: end - start]
        r_position = start + r_offset
        if r_position < total_samples:
            peaks.append(r_position)

        interval = mean_interval
        if hrv > 0:
            interval = max(0.25, mean_interval + rng.normal(0.0, hrv))
        time += interval

    return signal, np.array(peaks, dtype=int)


def heart_rate(peaks: np.ndarray, fs: float = DEFAULT_FS) -> float:
    """Mean heart rate in beats per minute, from R-peak positions.

    Computed from the **intervals**, not from ``len(peaks) / duration``.
    The two differ whenever the recording does not start and end exactly
    on a beat, and the interval form is what a monitor actually reports.

    >>> round(heart_rate(np.array([0, 250, 500, 750]), fs=250.0), 3)
    60.0
    """
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        raise ValueError("need at least two peaks to measure a rate")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    intervals = np.diff(peaks) / fs
    return float(60.0 / intervals.mean())


def rr_intervals(peaks: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """Beat-to-beat intervals in seconds — the input to every HRV measure."""
    peaks = np.asarray(peaks)
    if peaks.size < 2:
        raise ValueError("need at least two peaks to measure intervals")
    return np.diff(peaks) / fs


def signal_to_noise(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in decibels, given the clean signal and its corrupted version.

    Only computable because the clean signal exists — which is the whole
    argument for synthesising it. On real data this number is unavailable
    and every filtering claim becomes a matter of opinion.
    """
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    if clean.shape != noisy.shape:
        raise ValueError(f"shapes differ: {clean.shape} and {noisy.shape}")
    noise = noisy - clean
    noise_power = float(np.mean(noise ** 2))
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(np.mean(clean ** 2) / noise_power))


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. Every signal here is generated by this file.\n")

    signal, peaks = synthesise_ecg(duration=10.0, bpm=72.0, fs=250.0)
    print(f"10 s at 250 Hz, 72 bpm: {len(signal)} samples, {len(peaks)} beats placed")
    print(f"  measured rate from the intervals: {heart_rate(peaks, 250.0):.2f} bpm")
    print(f"  amplitude range: {signal.min():+.2f} .. {signal.max():+.2f} mV")

    print("\none beat, sampled every 20 ms (the R peak is the tall one):")
    beat = beat_waveform(fs=250.0, duration=0.8)
    for i in range(0, len(beat), 5):
        bar = "#" * int(round((beat[i] + 0.3) * 20))
        print(f"  {i / 250.0:>5.2f}s {beat[i]:>+6.2f} {bar}")

    print("\nground truth is exact, which is the entire point:")
    print(f"  R peaks at samples: {peaks[:8].tolist()} ...")
    print(f"  spacing: {np.diff(peaks)[:7].tolist()} samples")
    print("  a detector is scored against this list, not against its own output")

    print("\nheart rate variability makes the rhythm physiological:")
    for hrv in (0.0, 0.02, 0.05):
        _, jittered = synthesise_ecg(duration=60.0, bpm=72.0, fs=250.0, hrv=hrv, seed=1)
        intervals = rr_intervals(jittered, 250.0)
        print(f"  hrv={hrv:.2f}s  mean RR {intervals.mean():.3f}s  "
              f"sd {intervals.std():.4f}s  rate {60 / intervals.mean():.1f} bpm")
    print("  a metronomic heart is a sign of illness, not of health")

    print("\nthe rate is computed from intervals, not from a count:")
    short, short_peaks = synthesise_ecg(duration=3.3, bpm=60.0, fs=250.0)
    naive = len(short_peaks) / 3.3 * 60
    print(f"  count / duration : {naive:.1f} bpm")
    print(f"  from intervals   : {heart_rate(short_peaks, 250.0):.1f} bpm  (correct)")
    print("  the two differ whenever the recording does not end on a beat")

    print("\nSNR is measurable only because the clean signal exists:")
    rng = np.random.default_rng(0)
    for amplitude in (0.01, 0.05, 0.2):
        noisy = signal + rng.normal(0.0, amplitude, size=signal.shape)
        print(f"  noise sd {amplitude:.2f} mV -> SNR {signal_to_noise(signal, noisy):>6.1f} dB")
    print("  on real data this number does not exist, and every filtering")
    print("  claim becomes a matter of opinion. Day 2 adds realistic noise.")
