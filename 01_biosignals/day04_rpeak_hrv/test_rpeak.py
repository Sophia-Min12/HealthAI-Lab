"""Day 4 tests - the detector, and why HRV needs a better one than HR.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from rpeak import (
    DEFAULT_FS,
    add_false_beats,
    bandpass,
    baseline_wander,
    detect_r_peaks,
    detect_unrefined,
    drop_beats,
    heart_rate,
    integrate,
    jitter_peaks,
    mains_hum,
    match_peaks,
    motion_artefact,
    percent_error,
    pnn50,
    rmssd,
    rr_intervals,
    sdnn,
    sensor_noise,
    synthesise_ecg,
    timing_offsets,
)

CLEAN, TRUTH = synthesise_ecg(duration=60.0, hrv=0.04, seed=0)
TRUE_HR, TRUE_SDNN, TRUE_RMSSD = heart_rate(TRUTH), sdnn(TRUTH), rmssd(TRUTH)


class TestStages(unittest.TestCase):
    def test_bandpass_removes_the_baseline(self):
        wandering = baseline_wander(CLEAN, amplitude=0.5)
        self.assertLess(abs(bandpass(wandering).mean()), abs(wandering.mean()))

    def test_integration_is_non_negative(self):
        self.assertTrue(np.all(integrate(bandpass(CLEAN)) >= 0))

    def test_integration_produces_one_bump_per_beat(self):
        # Squaring and averaging turns a spike into a broad hump, which is
        # the point: the exact peak sample of a spike is noise-dependent.
        energy = integrate(bandpass(CLEAN))
        above = energy > 0.3 * energy.max()
        regions = np.sum(np.diff(above.astype(int)) == 1)
        self.assertAlmostEqual(regions, len(TRUTH), delta=3)

    def test_the_stages_preserve_length(self):
        self.assertEqual(len(integrate(bandpass(CLEAN))), len(CLEAN))


class TestDetector(unittest.TestCase):
    def test_it_is_perfect_on_a_clean_signal(self):
        scores = match_peaks(detect_r_peaks(CLEAN), TRUTH)
        self.assertEqual(scores["sensitivity"], 1.0)
        self.assertEqual(scores["precision"], 1.0)

    def test_it_survives_the_artefacts_that_broke_day_two(self):
        for name, signal in (("wander", baseline_wander(CLEAN, amplitude=0.4)),
                             ("hum", mains_hum(CLEAN, amplitude=0.3)),
                             ("sensor", sensor_noise(CLEAN, amplitude=0.05))):
            scores = match_peaks(detect_r_peaks(signal), TRUTH)
            self.assertEqual(scores["sensitivity"], 1.0, name)
            self.assertEqual(scores["precision"], 1.0, name)

    def test_motion_artefact_still_breaks_it(self):
        # Honest: the detector is much better and not good enough.
        scores = match_peaks(detect_r_peaks(motion_artefact(CLEAN, amplitude=1.5,
                                                            seed=1)), TRUTH)
        self.assertLess(scores["precision"], 1.0)

    def test_detections_are_sorted_and_unique(self):
        detected = detect_r_peaks(sensor_noise(CLEAN, amplitude=0.05))
        self.assertEqual(list(detected), sorted(set(detected.tolist())))

    def test_a_factor_outside_zero_to_one_is_refused(self):
        for factor in (0.0, 1.0, 2.0):
            with self.assertRaises(ValueError):
                detect_r_peaks(CLEAN, factor=factor)

    def test_a_flat_signal_finds_nothing(self):
        self.assertEqual(len(detect_r_peaks(np.zeros(2500))), 0)


class TestHRVMeasures(unittest.TestCase):
    def test_sdnn_of_a_metronome_is_zero(self):
        _, metronome = synthesise_ecg(duration=30.0, hrv=0.0, seed=0)
        self.assertLess(sdnn(metronome), 5.0)

    def test_sdnn_rises_with_injected_variability(self):
        values = []
        for hrv in (0.0, 0.02, 0.06):
            _, peaks = synthesise_ecg(duration=60.0, hrv=hrv, seed=0)
            values.append(sdnn(peaks))
        self.assertEqual(values, sorted(values))

    def test_sdnn_recovers_the_injected_value(self):
        # Independently-known answer: the generator was told 40 ms.
        _, peaks = synthesise_ecg(duration=120.0, hrv=0.04, seed=0)
        self.assertAlmostEqual(sdnn(peaks), 40.0, delta=8.0)

    def test_rmssd_is_zero_for_a_metronome(self):
        _, metronome = synthesise_ecg(duration=30.0, hrv=0.0, seed=0)
        self.assertLess(rmssd(metronome), 5.0)

    def test_pnn50_is_a_fraction(self):
        self.assertGreaterEqual(pnn50(TRUTH), 0.0)
        self.assertLessEqual(pnn50(TRUTH), 1.0)

    def test_too_few_beats_gives_zero_rather_than_an_error(self):
        # Day 1's rr_intervals raises below two peaks, which is right for
        # it and wrong here: an HRV measure of a one-beat recording is
        # zero variability, not an error. The guard has to sit in front
        # of the call, which is where the first version of this failed.
        for measure in (sdnn, rmssd, pnn50):
            self.assertEqual(measure(np.array([100])), 0.0)
            self.assertEqual(measure(np.array([100, 350])), 0.0)
            self.assertEqual(measure(np.array([], dtype=int)), 0.0)


class TestHRIsRobustAndHRVIsNot(unittest.TestCase):
    """The day's finding, as a controlled experiment."""

    def test_one_missed_beat_barely_moves_the_heart_rate(self):
        peaks = drop_beats(TRUTH, 1, seed=1)
        self.assertLess(percent_error(heart_rate(peaks), TRUE_HR), 3.0)

    def test_one_missed_beat_ruins_the_sdnn(self):
        peaks = drop_beats(TRUTH, 1, seed=1)
        self.assertGreater(percent_error(sdnn(peaks), TRUE_SDNN), 100.0)

    def test_the_amplification_is_about_a_hundredfold(self):
        peaks = drop_beats(TRUTH, 1, seed=1)
        hr_error = percent_error(heart_rate(peaks), TRUE_HR)
        sdnn_error = percent_error(sdnn(peaks), TRUE_SDNN)
        self.assertGreater(sdnn_error / hr_error, 50.0)

    def test_a_false_beat_does_the_same_in_the_other_direction(self):
        peaks = add_false_beats(TRUTH, 1, seed=1)
        self.assertLess(percent_error(heart_rate(peaks), TRUE_HR), 3.0)
        self.assertGreater(percent_error(sdnn(peaks), TRUE_SDNN), 50.0)

    def test_more_errors_are_worse(self):
        errors = [percent_error(sdnn(drop_beats(TRUTH, n, seed=1)), TRUE_SDNN)
                  for n in (1, 2, 4)]
        self.assertEqual(errors, sorted(errors))

    def test_a_missed_beat_creates_an_interval_of_double_length(self):
        # The mechanism, not just the symptom.
        peaks = drop_beats(TRUTH, 1, seed=1)
        intervals = rr_intervals(peaks)
        self.assertGreater(intervals.max(), 1.8 * np.median(rr_intervals(TRUTH)))


class TestTimingPrecision(unittest.TestCase):
    """Jitter matters; bias does not."""

    def test_a_constant_offset_changes_nothing(self):
        # HRV is built from differences, so a bias cancels exactly.
        shifted = TRUTH + 10
        self.assertAlmostEqual(sdnn(shifted), TRUE_SDNN, places=9)
        self.assertAlmostEqual(rmssd(shifted), TRUE_RMSSD, places=9)

    def test_jitter_leaves_the_heart_rate_alone(self):
        for amount in (1, 3, 6, 12):
            peaks = jitter_peaks(TRUTH, amount, seed=1)
            self.assertLess(percent_error(heart_rate(peaks), TRUE_HR), 2.0,
                            f"jitter {amount}")

    def test_jitter_inflates_rmssd(self):
        errors = [percent_error(rmssd(jitter_peaks(TRUTH, j, seed=1)), TRUE_RMSSD)
                  for j in (1, 3, 6, 12)]
        self.assertEqual(errors, sorted(errors))
        self.assertGreater(errors[-1], 20.0)

    def test_rmssd_is_more_jitter_sensitive_than_sdnn(self):
        # One bad peak corrupts two successive differences.
        peaks = jitter_peaks(TRUTH, 6, seed=1)
        self.assertGreater(percent_error(rmssd(peaks), TRUE_RMSSD),
                           percent_error(sdnn(peaks), TRUE_SDNN))


class TestRefinementIsNotFree(unittest.TestCase):
    """It helps under broadband noise and hurts under a tone."""

    def test_on_a_clean_signal_it_removes_a_constant_offset(self):
        rough = timing_offsets(detect_unrefined(CLEAN), TRUTH)
        refined = timing_offsets(detect_r_peaks(CLEAN), TRUTH)
        self.assertGreater(abs(rough.mean()), abs(refined.mean()))

    def test_and_therefore_changes_the_hrv_by_nothing(self):
        # Because a bias cancels, as the class above establishes.
        self.assertAlmostEqual(sdnn(detect_unrefined(CLEAN)),
                               sdnn(detect_r_peaks(CLEAN)), places=6)

    def test_under_sensor_noise_it_cuts_the_variance(self):
        noisy = sensor_noise(CLEAN, amplitude=0.08)
        rough = timing_offsets(detect_unrefined(noisy), TRUTH).std()
        refined = timing_offsets(detect_r_peaks(noisy), TRUTH).std()
        self.assertLess(refined, rough / 2)

    def test_and_that_is_what_improves_the_hrv(self):
        noisy = sensor_noise(CLEAN, amplitude=0.08)
        rough = percent_error(rmssd(detect_unrefined(noisy)), TRUE_RMSSD)
        refined = percent_error(rmssd(detect_r_peaks(noisy)), TRUE_RMSSD)
        self.assertLess(refined, rough)

    def test_under_mains_hum_it_makes_the_variance_worse(self):
        # The honest half. The integrated signal is smooth; the raw one
        # still carries a 50 Hz ripple that moves the local maximum.
        hummy = mains_hum(CLEAN, amplitude=0.3)
        rough = timing_offsets(detect_unrefined(hummy), TRUTH).std()
        refined = timing_offsets(detect_r_peaks(hummy), TRUTH).std()
        self.assertGreater(refined, rough)


class TestErrorInjection(unittest.TestCase):
    def test_dropping_beats_removes_exactly_that_many(self):
        self.assertEqual(len(drop_beats(TRUTH, 3, seed=0)), len(TRUTH) - 3)

    def test_dropping_never_removes_the_endpoints(self):
        # They anchor the heart-rate calculation.
        dropped = drop_beats(TRUTH, 5, seed=0)
        self.assertEqual(dropped[0], TRUTH[0])
        self.assertEqual(dropped[-1], TRUTH[-1])

    def test_adding_false_beats_increases_the_count(self):
        self.assertGreater(len(add_false_beats(TRUTH, 3, seed=0)), len(TRUTH))

    def test_jitter_keeps_the_count(self):
        self.assertEqual(len(jitter_peaks(TRUTH, 2, seed=0)), len(TRUTH))

    def test_all_three_are_reproducible(self):
        for maker in (lambda: drop_beats(TRUTH, 2, seed=7),
                      lambda: add_false_beats(TRUTH, 2, seed=7),
                      lambda: jitter_peaks(TRUTH, 2, seed=7)):
            np.testing.assert_array_equal(maker(), maker())

    def test_zero_errors_is_a_no_op(self):
        np.testing.assert_array_equal(drop_beats(TRUTH, 0), TRUTH)
        np.testing.assert_array_equal(add_false_beats(TRUTH, 0), TRUTH)
        np.testing.assert_array_equal(jitter_peaks(TRUTH, 0), TRUTH)


if __name__ == "__main__":
    unittest.main()
