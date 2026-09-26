"""Day 2 tests - the four artefacts, and what SNR does not tell you.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from noise import (
    DEFAULT_FS,
    amplitude_for_snr,
    band_energy,
    baseline_wander,
    dominant_frequency,
    mains_hum,
    match_peaks,
    motion_artefact,
    sensor_noise,
    signal_to_noise,
    synthesise_ecg,
    threshold_detect,
)

CLEAN, TRUTH = synthesise_ecg(duration=20.0, hrv=0.03, seed=0)


class TestBaselineWander(unittest.TestCase):
    def test_it_preserves_the_signal_length(self):
        self.assertEqual(len(baseline_wander(CLEAN)), len(CLEAN))

    def test_zero_amplitude_changes_nothing(self):
        np.testing.assert_allclose(baseline_wander(CLEAN, amplitude=0.0), CLEAN)

    def test_it_puts_its_energy_at_its_own_frequency(self):
        noisy = baseline_wander(CLEAN, amplitude=2.0, frequency=0.25)
        self.assertAlmostEqual(dominant_frequency(noisy), 0.25, delta=0.1)

    def test_it_does_not_add_high_frequency_energy(self):
        # The property that makes it filterable: it is entirely below the
        # QRS band. Stated as a fraction going DOWN rather than staying
        # equal, because band_energy is normalised by the total - adding
        # 1 mV of 0.25 Hz raises the denominator, so the high-frequency
        # share falls from 9.6e-5 to 7.3e-6 while the absolute energy up
        # there is untouched. The first version of this test asserted
        # equality and failed on exactly that.
        noisy = baseline_wander(CLEAN, amplitude=1.0)
        self.assertLess(band_energy(noisy, 40, DEFAULT_FS / 2),
                        band_energy(CLEAN, 40, DEFAULT_FS / 2))

    def test_the_absolute_high_frequency_energy_is_unchanged(self):
        # The claim the test above cannot make, made directly.
        def absolute(signal):
            spectrum = np.abs(np.fft.rfft(signal)) ** 2
            frequencies = np.fft.rfftfreq(len(signal), d=1.0 / DEFAULT_FS)
            return spectrum[frequencies >= 40].sum()

        noisy = baseline_wander(CLEAN, amplitude=1.0)
        self.assertAlmostEqual(absolute(noisy) / absolute(CLEAN), 1.0, places=6)

    def test_it_moves_the_signal_not_the_timing(self):
        # Every true beat still detected, in the right place.
        scores = match_peaks(threshold_detect(baseline_wander(CLEAN, amplitude=0.4)),
                             TRUTH)
        self.assertEqual(scores["sensitivity"], 1.0)

    def test_but_it_breaks_a_fixed_threshold(self):
        scores = match_peaks(threshold_detect(baseline_wander(CLEAN, amplitude=0.4)),
                             TRUTH)
        self.assertLess(scores["precision"], 1.0)

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            baseline_wander(CLEAN, amplitude=-1)
        with self.assertRaises(ValueError):
            baseline_wander(CLEAN, frequency=0)


class TestMainsHum(unittest.TestCase):
    def test_it_is_a_pure_tone_at_the_stated_frequency(self):
        noisy = mains_hum(CLEAN, amplitude=5.0, frequency=50.0)
        self.assertAlmostEqual(dominant_frequency(noisy), 50.0, delta=0.5)

    def test_sixty_hertz_works_too(self):
        noisy = mains_hum(CLEAN, amplitude=5.0, frequency=60.0)
        self.assertAlmostEqual(dominant_frequency(noisy), 60.0, delta=0.5)

    def test_it_adds_energy_above_the_qrs_band(self):
        self.assertGreater(band_energy(mains_hum(CLEAN, amplitude=0.2), 40, 125),
                           band_energy(CLEAN, 40, 125))

    def test_a_frequency_above_nyquist_is_refused_with_a_reason(self):
        # Silently aliasing to a different frequency would make the whole
        # demo a lie.
        with self.assertRaises(ValueError):
            mains_hum(CLEAN, fs=100.0, frequency=60.0)

    def test_exactly_nyquist_is_refused(self):
        with self.assertRaises(ValueError):
            mains_hum(CLEAN, fs=100.0, frequency=50.0)


class TestMotionArtefact(unittest.TestCase):
    def test_it_is_reproducible(self):
        first = motion_artefact(CLEAN, seed=3)
        second = motion_artefact(CLEAN, seed=3)
        np.testing.assert_allclose(first, second)

    def test_different_seeds_differ(self):
        self.assertFalse(np.allclose(motion_artefact(CLEAN, seed=1),
                                     motion_artefact(CLEAN, seed=2)))

    def test_zero_bursts_changes_nothing(self):
        np.testing.assert_allclose(motion_artefact(CLEAN, bursts=0), CLEAN)

    def test_it_is_broadband(self):
        # The property that makes it unfilterable: energy everywhere,
        # including above the QRS band where nothing else of ours lives.
        noisy = motion_artefact(CLEAN, amplitude=1.5)
        self.assertGreater(band_energy(noisy, 40, DEFAULT_FS / 2), 0.2)

    def test_it_is_localised_in_time(self):
        # Three bursts in twenty seconds: most of the trace is untouched.
        noisy = motion_artefact(CLEAN, bursts=3, amplitude=1.5)
        difference = np.abs(noisy - CLEAN)
        loud = difference > 0.1 * difference.max()
        self.assertLess(loud.mean(), 0.5)

    def test_it_can_exceed_the_r_peak(self):
        noisy = motion_artefact(CLEAN, amplitude=2.0)
        self.assertGreater(noisy.max(), CLEAN.max())


class TestSpectralHelpers(unittest.TestCase):
    def test_dominant_frequency_finds_a_known_tone(self):
        fs = 250.0
        time = np.arange(1000) / fs
        self.assertAlmostEqual(dominant_frequency(np.sin(2 * np.pi * 17.0 * time), fs),
                               17.0, delta=0.5)

    def test_band_energy_of_a_tone_is_concentrated(self):
        fs = 250.0
        time = np.arange(2000) / fs
        tone = np.sin(2 * np.pi * 10.0 * time)
        self.assertGreater(band_energy(tone, 5, 20, fs), 0.9)

    def test_band_energy_sums_towards_one_over_the_whole_range(self):
        self.assertAlmostEqual(band_energy(CLEAN, 0, DEFAULT_FS / 2 + 1), 1.0, places=6)

    def test_band_energy_of_silence_is_zero(self):
        self.assertEqual(band_energy(np.zeros(100), 5, 20), 0.0)

    def test_a_bad_band_is_refused(self):
        with self.assertRaises(ValueError):
            band_energy(CLEAN, 20, 5)


class TestThresholdDetect(unittest.TestCase):
    def test_it_finds_every_beat_in_a_clean_signal(self):
        scores = match_peaks(threshold_detect(CLEAN), TRUTH)
        self.assertEqual(scores["sensitivity"], 1.0)
        self.assertEqual(scores["precision"], 1.0)

    def test_the_refractory_period_prevents_double_counting(self):
        # Without it, every R peak is a dozen detections.
        detected = threshold_detect(CLEAN, refractory=0.2)
        self.assertEqual(len(detected), len(TRUTH))

    def test_a_threshold_above_everything_finds_nothing(self):
        self.assertEqual(len(threshold_detect(CLEAN, threshold=99.0)), 0)

    def test_detections_are_sorted_and_unique(self):
        detected = threshold_detect(motion_artefact(CLEAN))
        self.assertEqual(list(detected), sorted(set(detected.tolist())))

    def test_a_negative_refractory_is_refused(self):
        with self.assertRaises(ValueError):
            threshold_detect(CLEAN, refractory=-1)


class TestMatchPeaks(unittest.TestCase):
    def test_a_perfect_detection_scores_one(self):
        scores = match_peaks(TRUTH, TRUTH)
        self.assertEqual(scores["sensitivity"], 1.0)
        self.assertEqual(scores["precision"], 1.0)

    def test_finding_nothing_scores_zero_sensitivity(self):
        scores = match_peaks(np.array([], dtype=int), TRUTH)
        self.assertEqual(scores["sensitivity"], 0.0)
        self.assertEqual(scores["false_negative"], len(TRUTH))

    def test_each_true_peak_can_only_be_matched_once(self):
        # Otherwise a detector scores twice for finding one beat twice.
        doubled = np.repeat(TRUTH, 2)
        scores = match_peaks(doubled, TRUTH)
        self.assertEqual(scores["true_positive"], len(TRUTH))
        self.assertEqual(scores["false_positive"], len(TRUTH))

    def test_a_detection_just_inside_the_tolerance_counts(self):
        shifted = TRUTH + 20
        self.assertEqual(match_peaks(shifted, TRUTH, tolerance_samples=25)["true_positive"],
                         len(TRUTH))

    def test_a_detection_just_outside_it_does_not(self):
        shifted = TRUTH + 30
        self.assertEqual(match_peaks(shifted, TRUTH, tolerance_samples=25)["true_positive"], 0)


class TestSNRDoesNotPredictDamage(unittest.TestCase):
    """The day's finding."""

    @classmethod
    def setUpClass(cls):
        cls.makers = {
            "wander": lambda a: baseline_wander(CLEAN, amplitude=a),
            "hum": lambda a: mains_hum(CLEAN, amplitude=a),
            "sensor": lambda a: sensor_noise(CLEAN, amplitude=a),
            "motion": lambda a: motion_artefact(CLEAN, amplitude=a),
        }

    def precision_at_snr(self, name, target):
        make = self.makers[name]
        amplitude = amplitude_for_snr(CLEAN, make, target)
        return match_peaks(threshold_detect(make(amplitude)), TRUTH)["precision"]

    def test_the_tuner_hits_the_requested_snr(self):
        for name, make in self.makers.items():
            amplitude = amplitude_for_snr(CLEAN, make, 0.0)
            self.assertAlmostEqual(signal_to_noise(CLEAN, make(amplitude)), 0.0,
                                   delta=0.2, msg=name)

    def test_equal_snr_gives_unequal_damage(self):
        scores = {name: self.precision_at_snr(name, 0.0) for name in self.makers}
        self.assertGreater(max(scores.values()) - min(scores.values()), 0.2)

    def test_hum_damages_more_than_motion_at_equal_snr(self):
        # The counterintuitive half, pinned. Easiest to filter is not the
        # same as least damaging.
        self.assertLess(self.precision_at_snr("hum", 0.0),
                        self.precision_at_snr("motion", 0.0))

    def test_more_noise_never_helps(self):
        for name in self.makers:
            better = self.precision_at_snr(name, 10.0)
            worse = self.precision_at_snr(name, -5.0)
            self.assertGreaterEqual(better, worse, name)

    def test_sensitivity_survives_everything(self):
        # Noise adds energy, so the true peaks stay above the line. Every
        # failure is a false positive.
        for name, make in self.makers.items():
            amplitude = amplitude_for_snr(CLEAN, make, -5.0)
            scores = match_peaks(threshold_detect(make(amplitude)), TRUTH)
            self.assertGreaterEqual(scores["sensitivity"], 0.9, name)


if __name__ == "__main__":
    unittest.main()
