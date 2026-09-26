"""Day 3 tests - filters, the delay they add, and the peak they cost.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from filtering import (
    DEFAULT_FS,
    adaptive_threshold_detect,
    band_energy,
    baseline_wander,
    mains_hum,
    match_peaks,
    motion_artefact,
    moving_average,
    notch_filter,
    peak_amplitude,
    peak_shift,
    remove_baseline,
    synthesise_ecg,
    threshold_detect,
)

CLEAN, TRUTH = synthesise_ecg(duration=20.0, hrv=0.03, seed=0)
BASE_AMPLITUDE = peak_amplitude(CLEAN, TRUTH)


class TestMovingAverage(unittest.TestCase):
    def test_a_window_of_one_is_the_identity(self):
        np.testing.assert_allclose(moving_average(CLEAN, 1), CLEAN)

    def test_it_preserves_length(self):
        for window in (3, 25, 101):
            self.assertEqual(len(moving_average(CLEAN, window)), len(CLEAN))

    def test_it_averages_a_known_case(self):
        np.testing.assert_allclose(moving_average(np.array([0.0, 3.0, 0.0]), 3),
                                   [1.0, 1.0, 1.0])

    def test_a_constant_signal_is_unchanged(self):
        constant = np.full(100, 2.5)
        np.testing.assert_allclose(moving_average(constant, 11), constant)

    def test_it_reduces_variance(self):
        noisy = CLEAN + np.random.default_rng(0).normal(0, 0.1, len(CLEAN))
        self.assertLess(moving_average(noisy, 11).var(), noisy.var())

    def test_a_window_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            moving_average(CLEAN, 0)


class TestCausalDelay(unittest.TestCase):
    """The delay a monitor cannot avoid, checked against (w-1)/2."""

    def test_a_centred_window_does_not_shift_a_surviving_peak(self):
        self.assertAlmostEqual(peak_shift(moving_average(CLEAN, 5), TRUTH), 0.0,
                               delta=1.0)

    def test_a_causal_window_shifts_by_exactly_half_the_window(self):
        # Exact, not approximate - while the peak survives. Measured:
        # w=3,5,7,9 give shifts of 1, 2, 3, 4 against predictions of the
        # same, with the R peak still at 94%, 83%, 70% and 57%.
        for window in (3, 5, 7, 9):
            shift = peak_shift(moving_average(CLEAN, window, causal=True), TRUTH)
            self.assertEqual(shift, (window - 1) / 2, msg=f"window {window}")

    def test_the_agreement_stops_when_the_peak_does(self):
        # Past w=9 the measured shift saturates at 4 while the prediction
        # keeps climbing, because the peak is under half its height and
        # argmax has moved to a neighbouring feature. The boundary is
        # where the peak stops surviving, not where the filter changes.
        for window in (13, 17, 21):
            shift = peak_shift(moving_average(CLEAN, window, causal=True), TRUTH)
            kept = peak_amplitude(moving_average(CLEAN, window), TRUTH) / BASE_AMPLITUDE
            self.assertLess(kept, 0.5, msg=f"window {window}")
            self.assertLess(shift, (window - 1) / 2, msg=f"window {window}")

    def test_a_wider_causal_window_delays_more(self):
        shifts = [peak_shift(moving_average(CLEAN, w, causal=True), TRUTH)
                  for w in (3, 5, 9)]
        self.assertEqual(shifts, sorted(shifts))

    def test_the_delay_measurement_fails_once_the_peak_is_gone(self):
        # Not a bug in peak_shift - a limit of it. By a 100 ms window the
        # R peak is a fifth of its height and the function reports the
        # distance to some other local maximum. Pinned so nobody reads
        # the large-window rows of the demo as a delay.
        window = int(round(0.1 * DEFAULT_FS))
        kept = peak_amplitude(moving_average(CLEAN, window), TRUTH) / BASE_AMPLITUDE
        self.assertLess(kept, 0.3)
        shift = peak_shift(moving_average(CLEAN, window, causal=True), TRUTH)
        self.assertGreater(abs(shift), 2 * (window - 1) / 2)


class TestSmoothingCostsThePeak(unittest.TestCase):
    def test_wider_windows_keep_less_of_the_r_peak(self):
        kept = [peak_amplitude(moving_average(CLEAN, w), TRUTH) / BASE_AMPLITUDE
                for w in (5, 12, 25, 50)]
        self.assertEqual(kept, sorted(kept, reverse=True))

    def test_a_window_near_the_qrs_width_destroys_it(self):
        # A QRS is about 80 ms; 100 ms of averaging leaves a fifth.
        window = int(round(0.1 * DEFAULT_FS))
        self.assertLess(peak_amplitude(moving_average(CLEAN, window), TRUTH)
                        / BASE_AMPLITUDE, 0.3)

    def test_a_short_window_leaves_most_of_it(self):
        window = int(round(0.02 * DEFAULT_FS))
        self.assertGreater(peak_amplitude(moving_average(CLEAN, window), TRUTH)
                           / BASE_AMPLITUDE, 0.75)


class TestRemoveBaseline(unittest.TestCase):
    def setUp(self):
        self.wandering = baseline_wander(CLEAN, amplitude=0.4)

    def test_it_removes_the_low_frequency_energy(self):
        before = band_energy(self.wandering, 0, 1)
        after = band_energy(remove_baseline(self.wandering, window_seconds=0.6), 0, 1)
        self.assertLess(after, before / 10)

    def test_a_too_short_window_eats_the_peak(self):
        short = peak_amplitude(remove_baseline(self.wandering, window_seconds=0.1),
                               TRUTH)
        long = peak_amplitude(remove_baseline(self.wandering, window_seconds=1.0),
                              TRUTH)
        self.assertLess(short, long)

    def test_a_too_long_window_leaves_the_wander(self):
        short = band_energy(remove_baseline(self.wandering, window_seconds=0.4), 0, 1)
        long = band_energy(remove_baseline(self.wandering, window_seconds=2.0), 0, 1)
        self.assertGreater(long, short)

    def test_the_middle_of_the_range_restores_detection(self):
        for seconds in (0.4, 0.6, 1.0):
            scores = match_peaks(
                threshold_detect(remove_baseline(self.wandering, window_seconds=seconds)),
                TRUTH)
            self.assertEqual(scores["precision"], 1.0, f"{seconds}s")

    def test_subtracting_a_causal_baseline_does_not_shift_the_peak(self):
        # Worth pinning, because it is the opposite of what section 2
        # shows for smoothing: the baseline estimate is delayed, but it
        # is subtracted from an undelayed signal, so the peak stays put.
        filtered = remove_baseline(self.wandering, window_seconds=0.6, causal=True)
        self.assertAlmostEqual(peak_shift(filtered, TRUTH), 0.0, delta=2.0)


class TestNotchFilter(unittest.TestCase):
    def setUp(self):
        self.hummy = mains_hum(CLEAN, amplitude=0.3)

    def test_it_removes_the_hum(self):
        self.assertLess(band_energy(notch_filter(self.hummy, frequency=50.0), 48, 52),
                        0.001)

    def test_it_restores_the_signal_to_noise_ratio(self):
        from filtering import signal_to_noise
        self.assertGreater(signal_to_noise(CLEAN, notch_filter(self.hummy, frequency=50.0)),
                           signal_to_noise(CLEAN, self.hummy) + 30)

    def test_it_costs_this_signal_nothing(self):
        filtered = notch_filter(self.hummy, frequency=50.0, width=5.0)
        self.assertGreater(peak_amplitude(filtered, TRUTH) / BASE_AMPLITUDE, 0.98)

    def test_and_the_reason_is_a_limitation_of_the_synthetic_waveform(self):
        # 0.0006% of the clean signal's energy is where the notch cuts,
        # because Day 1 built the QRS from smooth Gaussians. A real QRS
        # has a sharp upstroke and content past 40 Hz, so this result
        # would not transfer.
        self.assertLess(band_energy(CLEAN, 48, 52), 0.0001)
        self.assertLess(band_energy(CLEAN, 40, 125), 0.001)

    def test_it_preserves_length(self):
        self.assertEqual(len(notch_filter(self.hummy)), len(self.hummy))

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            notch_filter(self.hummy, width=0)
        with self.assertRaises(ValueError):
            notch_filter(self.hummy, frequency=200.0)


class TestMotionArtefactResists(unittest.TestCase):
    """Where filtering runs out."""

    def setUp(self):
        self.motion = motion_artefact(CLEAN, amplitude=1.5, seed=0)
        self.before = match_peaks(threshold_detect(self.motion), TRUTH)

    def test_baseline_removal_does_not_help(self):
        after = match_peaks(
            threshold_detect(remove_baseline(self.motion, window_seconds=0.6)), TRUTH)
        self.assertLessEqual(after["precision"], self.before["precision"] + 0.05)

    def test_a_notch_does_not_help(self):
        after = match_peaks(
            threshold_detect(notch_filter(self.motion, frequency=50.0, width=5.0)),
            TRUTH)
        self.assertLessEqual(after["precision"], self.before["precision"] + 0.05)

    def test_mild_smoothing_helps_a_little(self):
        after = match_peaks(threshold_detect(moving_average(self.motion, 5)), TRUTH)
        self.assertGreater(after["precision"], self.before["precision"])

    def test_more_smoothing_starts_deleting_beats(self):
        after = match_peaks(threshold_detect(moving_average(self.motion, 12)), TRUTH)
        self.assertLess(after["sensitivity"], 0.5)


class TestAdaptiveThreshold(unittest.TestCase):
    """The day's conclusion: the fix for wander was not a filter."""

    def test_it_matches_the_fixed_threshold_on_a_clean_signal(self):
        scores = match_peaks(adaptive_threshold_detect(CLEAN), TRUTH)
        self.assertEqual(scores["sensitivity"], 1.0)
        self.assertEqual(scores["precision"], 1.0)

    def test_it_survives_wander_that_breaks_the_fixed_one(self):
        wandering = baseline_wander(CLEAN, amplitude=0.4)
        fixed = match_peaks(threshold_detect(wandering), TRUTH)
        adaptive = match_peaks(adaptive_threshold_detect(wandering), TRUTH)
        self.assertLess(fixed["precision"], 1.0)
        self.assertEqual(adaptive["precision"], 1.0)

    def test_it_needs_no_filtering_at_all(self):
        # No window to tune, no delay, no filter in the path.
        wandering = baseline_wander(CLEAN, amplitude=0.6)
        fixed = match_peaks(threshold_detect(wandering), TRUTH)
        adaptive = match_peaks(adaptive_threshold_detect(wandering), TRUTH)
        self.assertGreater(adaptive["precision"], fixed["precision"])

    def test_it_is_only_better_within_a_range(self):
        # The qualification the day's conclusion needs. Up to 0.4 mV the
        # adaptive threshold is perfect where the fixed one has fallen to
        # 73% precision. Past 0.8 mV both collapse, and the adaptive
        # one's SENSITIVITY is the worse of the two - because the local
        # maximum it compares against is itself inflated by the wander.
        heavy = baseline_wander(CLEAN, amplitude=0.8)
        fixed = match_peaks(threshold_detect(heavy), TRUTH)
        adaptive = match_peaks(adaptive_threshold_detect(heavy), TRUTH)
        self.assertLess(adaptive["sensitivity"], fixed["sensitivity"])

    def test_and_it_is_clearly_better_inside_that_range(self):
        moderate = baseline_wander(CLEAN, amplitude=0.4)
        fixed = match_peaks(threshold_detect(moderate), TRUTH)
        adaptive = match_peaks(adaptive_threshold_detect(moderate), TRUTH)
        self.assertEqual(adaptive["sensitivity"], 1.0)
        self.assertEqual(adaptive["precision"], 1.0)
        self.assertLess(fixed["precision"], 0.8)

    def test_a_factor_outside_zero_to_one_is_refused(self):
        for factor in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                adaptive_threshold_detect(CLEAN, factor=factor)


if __name__ == "__main__":
    unittest.main()
