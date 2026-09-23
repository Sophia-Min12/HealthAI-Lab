"""Day 1 tests — the generator, and the ground truth everything else is scored on.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from ecg import (
    DEFAULT_BPM,
    DEFAULT_FS,
    beat_waveform,
    heart_rate,
    rr_intervals,
    signal_to_noise,
    synthesise_ecg,
)


class TestBeatWaveform(unittest.TestCase):
    def test_length_follows_rate_and_duration(self):
        self.assertEqual(len(beat_waveform(fs=250.0, duration=0.8)), 200)
        self.assertEqual(len(beat_waveform(fs=100.0, duration=1.0)), 100)

    def test_r_peak_is_the_maximum(self):
        wave = beat_waveform()
        self.assertEqual(int(np.argmax(wave)), int(round(0.25 * len(wave))))

    def test_r_peak_dominates_every_other_deflection(self):
        wave = beat_waveform()
        peak = wave.max()
        # Mask out the QRS and check nothing else comes close.
        r_index = int(np.argmax(wave))
        elsewhere = np.concatenate([wave[: r_index - 12], wave[r_index + 12:]])
        self.assertGreater(peak, 3 * elsewhere.max())

    def test_has_negative_deflections(self):
        # Q and S are below baseline; a waveform that is all positive has
        # lost the QRS shape entirely.
        self.assertLess(beat_waveform().min(), -0.05)

    def test_starts_and_ends_near_baseline(self):
        wave = beat_waveform()
        self.assertLess(abs(wave[0]), 0.1)
        self.assertLess(abs(wave[-1]), 0.1)

    def test_bad_parameters_rejected(self):
        for kwargs in ({"fs": 0.0}, {"fs": -1.0}, {"duration": 0.0}):
            with self.assertRaises(ValueError):
                beat_waveform(**kwargs)


class TestSynthesiseEcg(unittest.TestCase):
    def test_length_matches_duration(self):
        signal, _ = synthesise_ecg(duration=10.0, fs=250.0)
        self.assertEqual(len(signal), 2500)

    def test_beat_count_is_about_right(self):
        _, peaks = synthesise_ecg(duration=60.0, bpm=60.0, fs=250.0)
        self.assertIn(len(peaks), (60, 61))

    def test_peaks_are_inside_the_signal(self):
        signal, peaks = synthesise_ecg(duration=10.0)
        self.assertTrue(np.all(peaks >= 0))
        self.assertTrue(np.all(peaks < len(signal)))

    def test_peaks_are_strictly_increasing(self):
        _, peaks = synthesise_ecg(duration=20.0, hrv=0.04, seed=3)
        self.assertTrue(np.all(np.diff(peaks) > 0))

    def test_the_signal_really_does_peak_at_the_reported_positions(self):
        # Ground truth has to be true, or every later day is scored
        # against a lie.
        signal, peaks = synthesise_ecg(duration=10.0, fs=250.0)
        for position in peaks[1:-1]:
            window = signal[position - 20:position + 20]
            self.assertEqual(int(np.argmax(window)), 20)

    def test_deterministic_for_a_seed(self):
        first = synthesise_ecg(duration=5.0, hrv=0.03, seed=7)
        second = synthesise_ecg(duration=5.0, hrv=0.03, seed=7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_different_seeds_differ_when_jittered(self):
        _, first = synthesise_ecg(duration=30.0, hrv=0.05, seed=1)
        _, second = synthesise_ecg(duration=30.0, hrv=0.05, seed=2)
        self.assertFalse(np.array_equal(first, second))

    def test_zero_hrv_is_metronomic(self):
        _, peaks = synthesise_ecg(duration=30.0, bpm=60.0, fs=250.0, hrv=0.0)
        spacing = np.diff(peaks)
        self.assertLessEqual(int(spacing.max() - spacing.min()), 1)  # rounding only

    def test_hrv_widens_the_spread(self):
        spreads = []
        for hrv in (0.0, 0.02, 0.05):
            _, peaks = synthesise_ecg(duration=120.0, hrv=hrv, seed=5)
            spreads.append(float(rr_intervals(peaks).std()))
        self.assertEqual(spreads, sorted(spreads))

    def test_faster_bpm_gives_more_beats(self):
        _, slow = synthesise_ecg(duration=30.0, bpm=50.0)
        _, fast = synthesise_ecg(duration=30.0, bpm=100.0)
        self.assertGreater(len(fast), len(slow))

    def test_bad_parameters_rejected(self):
        for kwargs in ({"duration": 0.0}, {"bpm": 0.0}, {"fs": -1.0}, {"hrv": -0.1}):
            with self.assertRaises(ValueError):
                synthesise_ecg(**kwargs)


class TestHeartRate(unittest.TestCase):
    def test_exact_case(self):
        self.assertAlmostEqual(heart_rate(np.array([0, 250, 500, 750]), fs=250.0), 60.0)

    def test_recovers_the_requested_rate(self):
        for bpm in (50.0, 72.0, 100.0, 140.0):
            _, peaks = synthesise_ecg(duration=60.0, bpm=bpm, fs=250.0)
            self.assertAlmostEqual(heart_rate(peaks, 250.0), bpm, delta=0.5)

    def test_uses_intervals_not_a_count(self):
        # A recording that does not end on a beat: the naive count is wrong.
        _, peaks = synthesise_ecg(duration=3.3, bpm=60.0, fs=250.0)
        naive = len(peaks) / 3.3 * 60
        self.assertAlmostEqual(heart_rate(peaks, 250.0), 60.0, delta=0.5)
        self.assertGreater(abs(naive - 60.0), 5.0)

    def test_needs_two_peaks(self):
        with self.assertRaises(ValueError):
            heart_rate(np.array([10]), fs=250.0)

    def test_bad_fs_rejected(self):
        with self.assertRaises(ValueError):
            heart_rate(np.array([0, 250]), fs=0.0)


class TestRrIntervals(unittest.TestCase):
    def test_length_is_one_less_than_peaks(self):
        _, peaks = synthesise_ecg(duration=20.0)
        self.assertEqual(len(rr_intervals(peaks)), len(peaks) - 1)

    def test_values_are_seconds(self):
        intervals = rr_intervals(np.array([0, 250, 500]), fs=250.0)
        np.testing.assert_allclose(intervals, [1.0, 1.0])

    def test_all_positive(self):
        _, peaks = synthesise_ecg(duration=30.0, hrv=0.05, seed=9)
        self.assertTrue(np.all(rr_intervals(peaks) > 0))

    def test_needs_two_peaks(self):
        with self.assertRaises(ValueError):
            rr_intervals(np.array([5]))


class TestSignalToNoise(unittest.TestCase):
    def test_identical_signals_are_infinite(self):
        signal, _ = synthesise_ecg(duration=2.0)
        self.assertEqual(signal_to_noise(signal, signal), float("inf"))

    def test_more_noise_lowers_snr(self):
        signal, _ = synthesise_ecg(duration=5.0)
        rng = np.random.default_rng(0)
        values = [
            signal_to_noise(signal, signal + rng.normal(0.0, sd, size=signal.shape))
            for sd in (0.01, 0.05, 0.2)
        ]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_is_finite_for_real_noise(self):
        signal, _ = synthesise_ecg(duration=5.0)
        rng = np.random.default_rng(1)
        noisy = signal + rng.normal(0.0, 0.05, size=signal.shape)
        self.assertTrue(math.isfinite(signal_to_noise(signal, noisy)))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            signal_to_noise(np.zeros(10), np.zeros(11))


class TestDefaults(unittest.TestCase):
    def test_defaults_are_physiological(self):
        self.assertGreater(DEFAULT_BPM, 40.0)
        self.assertLess(DEFAULT_BPM, 100.0)

    def test_sampling_rate_is_above_nyquist_for_a_qrs(self):
        # QRS content reaches ~40 Hz, so anything under 80 Hz aliases it.
        self.assertGreater(DEFAULT_FS, 80.0)


if __name__ == "__main__":
    unittest.main()
