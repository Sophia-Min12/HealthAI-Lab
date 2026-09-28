"""Day 8 tests - calibration, and its independence from AUC.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from calibration import (
    apply_platt,
    auc,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    fit_logistic,
    make_cohort,
    miscalibrate,
    platt_scale,
    predict,
    reliability,
    report,
    threshold_for_cost_ratio,
)

COHORT = make_cohort(n=20000, seed=0, prevalence=0.12)
WEIGHTS, INTERCEPT = fit_logistic(COHORT.x, COHORT.y)
PROBABILITY = predict(COHORT.x, WEIGHTS, INTERCEPT)


class TestReliability(unittest.TestCase):
    def test_the_bins_cover_every_patient(self):
        rows = reliability(COHORT.y, PROBABILITY, bins=10)
        self.assertEqual(sum(row["count"] for row in rows), len(COHORT.y))

    def test_the_bins_are_ordered_and_contiguous(self):
        rows = reliability(COHORT.y, PROBABILITY, bins=10)
        for earlier, later in zip(rows, rows[1:]):
            self.assertAlmostEqual(earlier["high"], later["low"])

    def test_predicted_risk_lies_inside_its_bin(self):
        for row in reliability(COHORT.y, PROBABILITY, bins=10):
            if row["count"]:
                self.assertGreaterEqual(row["predicted"], row["low"] - 1e-9)
                self.assertLessEqual(row["predicted"], row["high"] + 1e-9)

    def test_the_fitted_model_is_well_calibrated(self):
        # It should be: the cohort was generated from a logistic model and
        # this is a logistic fit, so the family is exactly right. A
        # property of the experiment rather than an achievement.
        self.assertLess(expected_calibration_error(COHORT.y, PROBABILITY), 0.01)

    def test_a_constant_prediction_is_calibrated_only_at_the_base_rate(self):
        base = float(COHORT.y.mean())
        good = np.full(len(COHORT.y), base)
        bad = np.full(len(COHORT.y), base + 0.2)
        self.assertLess(expected_calibration_error(COHORT.y, good), 0.01)
        self.assertGreater(expected_calibration_error(COHORT.y, bad), 0.15)

    def test_zero_bins_is_refused(self):
        with self.assertRaises(ValueError):
            reliability(COHORT.y, PROBABILITY, bins=0)


class TestBrier(unittest.TestCase):
    def test_a_perfect_prediction_scores_zero(self):
        self.assertAlmostEqual(brier_score(np.array([1.0, 0.0]),
                                           np.array([1.0, 0.0])), 0.0)

    def test_the_worst_possible_prediction_scores_one(self):
        self.assertAlmostEqual(brier_score(np.array([1.0, 0.0]),
                                           np.array([0.0, 1.0])), 1.0)

    def test_predicting_the_base_rate_scores_its_variance(self):
        # Independently-known answer.
        y = np.array([1.0] * 300 + [0.0] * 700)
        self.assertAlmostEqual(brier_score(y, np.full(1000, 0.3)), 0.3 * 0.7,
                               places=9)

    def test_it_is_a_proper_scoring_rule(self):
        # Shading the truth in either direction makes it worse.
        y = np.array([1.0] * 300 + [0.0] * 700)
        honest = brier_score(y, np.full(1000, 0.3))
        for shaded in (0.2, 0.4, 0.5):
            self.assertGreater(brier_score(y, np.full(1000, shaded)), honest)


class TestCalibrationIsIndependentOfAUC(unittest.TestCase):
    """The day's central claim."""

    def test_a_monotone_distortion_preserves_the_order(self):
        distorted = miscalibrate(PROBABILITY, power=1.6, shift=0.9)
        self.assertTrue(np.all(np.diff(distorted[np.argsort(PROBABILITY)]) >= -1e-12))

    def test_and_therefore_leaves_the_auc_untouched(self):
        base = auc(COHORT.y, PROBABILITY)
        for power, shift in ((1.6, 0.0), (0.5, 0.0), (1.0, 1.1), (1.0, -1.1),
                             (0.4, 1.6)):
            distorted = miscalibrate(PROBABILITY, power=power, shift=shift)
            self.assertAlmostEqual(auc(COHORT.y, distorted), base, places=6)

    def test_while_destroying_the_calibration(self):
        base = expected_calibration_error(COHORT.y, PROBABILITY)
        distorted = miscalibrate(PROBABILITY, shift=1.1)
        self.assertGreater(expected_calibration_error(COHORT.y, distorted),
                           20 * base)

    def test_the_brier_score_notices_what_auc_cannot(self):
        base = brier_score(COHORT.y, PROBABILITY)
        distorted = miscalibrate(PROBABILITY, shift=1.1)
        self.assertGreater(brier_score(COHORT.y, distorted), base)

    def test_no_distortion_is_the_identity(self):
        np.testing.assert_allclose(miscalibrate(PROBABILITY), PROBABILITY,
                                   atol=1e-9)


class TestBrierDecomposition(unittest.TestCase):
    def test_it_reproduces_the_brier_score_up_to_binning(self):
        parts = brier_decomposition(COHORT.y, PROBABILITY, bins=20)
        self.assertAlmostEqual(parts["brier"],
                               brier_score(COHORT.y, PROBABILITY), delta=0.005)

    def test_uncertainty_is_the_base_rate_variance(self):
        parts = brier_decomposition(COHORT.y, PROBABILITY)
        base = float(COHORT.y.mean())
        self.assertAlmostEqual(parts["uncertainty"], base * (1 - base), places=9)

    def test_a_distortion_moves_reliability_and_leaves_resolution(self):
        # The decomposition earning its place: it separates the part a
        # monotone transform ruins from the part it cannot touch.
        original = brier_decomposition(COHORT.y, PROBABILITY)
        distorted = brier_decomposition(COHORT.y,
                                        miscalibrate(PROBABILITY, power=1.6,
                                                     shift=0.9))
        self.assertGreater(distorted["reliability"], 10 * original["reliability"])
        self.assertAlmostEqual(distorted["resolution"], original["resolution"],
                               delta=0.001)

    def test_uncertainty_is_untouched_by_any_model_change(self):
        original = brier_decomposition(COHORT.y, PROBABILITY)
        distorted = brier_decomposition(COHORT.y, miscalibrate(PROBABILITY, shift=2.0))
        self.assertAlmostEqual(original["uncertainty"], distorted["uncertainty"],
                               places=12)


class TestPlattScaling(unittest.TestCase):
    def setUp(self):
        self.broken = miscalibrate(PROBABILITY, power=1.6, shift=0.9)
        self.slope, self.offset = platt_scale(self.broken, COHORT.y)
        self.repaired = apply_platt(self.broken, self.slope, self.offset)

    def test_it_repairs_the_calibration(self):
        self.assertLess(expected_calibration_error(COHORT.y, self.repaired),
                        expected_calibration_error(COHORT.y, self.broken) / 5)

    def test_it_cannot_change_the_auc(self):
        self.assertAlmostEqual(auc(COHORT.y, self.repaired),
                               auc(COHORT.y, self.broken), places=6)

    def test_it_recovers_the_original_brier_score(self):
        self.assertAlmostEqual(brier_score(COHORT.y, self.repaired),
                               brier_score(COHORT.y, PROBABILITY), delta=0.002)

    def test_it_is_monotone(self):
        order = np.argsort(self.broken)
        self.assertTrue(np.all(np.diff(self.repaired[order]) >= -1e-12))

    def test_fitting_it_on_already_calibrated_scores_is_nearly_the_identity(self):
        slope, offset = platt_scale(PROBABILITY, COHORT.y)
        self.assertAlmostEqual(slope, 1.0, delta=0.15)
        self.assertAlmostEqual(offset, 0.0, delta=0.15)


class TestItBreaksDay7(unittest.TestCase):
    """Why this matters more than a wrong number on a report."""

    def test_the_same_cut_flags_wildly_different_numbers(self):
        cut = threshold_for_cost_ratio(5.0, 1.0)
        reference = report(COHORT.y, PROBABILITY >= cut)["flagged"]
        inflated = report(COHORT.y,
                          miscalibrate(PROBABILITY, shift=1.1) >= cut)["flagged"]
        deflated = report(COHORT.y,
                          miscalibrate(PROBABILITY, shift=-1.1) >= cut)["flagged"]
        self.assertGreater(inflated / reference, 2.0)
        self.assertLess(deflated / reference, 0.5)

    def test_some_miscalibrations_are_harmless_at_a_given_cut(self):
        # The honest half: a distortion with ECE 0.03 moved the workload
        # by 4%, and nothing in the summary statistics predicted that.
        cut = threshold_for_cost_ratio(5.0, 1.0)
        reference = report(COHORT.y, PROBABILITY >= cut)["flagged"]
        mild = report(COHORT.y,
                      miscalibrate(PROBABILITY, power=1.6, shift=0.9) >= cut)["flagged"]
        self.assertLess(abs(mild - reference) / reference, 0.2)

    def test_the_auc_is_identical_in_every_one_of_those_cases(self):
        base = auc(COHORT.y, PROBABILITY)
        for series in (miscalibrate(PROBABILITY, shift=1.1),
                       miscalibrate(PROBABILITY, shift=-1.1),
                       miscalibrate(PROBABILITY, power=1.6, shift=0.9)):
            self.assertAlmostEqual(auc(COHORT.y, series), base, places=6)


if __name__ == "__main__":
    unittest.main()
