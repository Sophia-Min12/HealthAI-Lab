"""Day 7 tests - threshold rules, and the cost ratio each one hides.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from thresholds import (
    cost_threshold,
    f1_threshold,
    fit_logistic,
    implied_cost_ratio,
    make_cohort,
    precision_recall_curve,
    predict,
    report,
    roc_curve,
    threshold_for_cost_ratio,
    total_cost,
    youden_threshold,
)

COHORT = make_cohort(n=8000, seed=0, prevalence=0.12)
WEIGHTS, INTERCEPT = fit_logistic(COHORT.x, COHORT.y)
PROBABILITY = predict(COHORT.x, WEIGHTS, INTERCEPT)


class TestROCCurve(unittest.TestCase):
    def test_both_rates_are_fractions(self):
        false_rate, true_rate, _ = roc_curve(COHORT.y, PROBABILITY)
        for series in (false_rate, true_rate):
            self.assertTrue(np.all(series >= 0))
            self.assertTrue(np.all(series <= 1))

    def test_both_rates_increase_with_a_lower_threshold(self):
        false_rate, true_rate, thresholds = roc_curve(COHORT.y, PROBABILITY)
        self.assertTrue(np.all(np.diff(thresholds) <= 0))
        self.assertTrue(np.all(np.diff(false_rate) >= -1e-12))
        self.assertTrue(np.all(np.diff(true_rate) >= -1e-12))

    def test_a_perfect_score_reaches_the_corner(self):
        y = np.array([0] * 50 + [1] * 50)
        score = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
        false_rate, true_rate, _ = roc_curve(y, score)
        self.assertTrue(np.any((true_rate > 0.99) & (false_rate < 0.01)))

    def test_one_class_only_is_refused(self):
        with self.assertRaises(ValueError):
            roc_curve(np.ones(10), np.linspace(0, 1, 10))


class TestPrecisionRecallCurve(unittest.TestCase):
    def test_recall_rises_as_the_threshold_falls(self):
        _, recall, thresholds = precision_recall_curve(COHORT.y, PROBABILITY)
        self.assertTrue(np.all(np.diff(thresholds) <= 0))
        self.assertTrue(np.all(np.diff(recall) >= -1e-12))

    def test_precision_starts_high_and_ends_at_prevalence(self):
        precision, _, _ = precision_recall_curve(COHORT.y, PROBABILITY)
        self.assertGreater(precision[0], COHORT.prevalence)
        self.assertAlmostEqual(precision[-1], COHORT.prevalence, delta=0.01)

    def test_all_values_are_fractions(self):
        precision, recall, _ = precision_recall_curve(COHORT.y, PROBABILITY)
        for series in (precision, recall):
            self.assertTrue(np.all(series >= 0))
            self.assertTrue(np.all(series <= 1))


class TestCostRatioRoundTrip(unittest.TestCase):
    """The identity the day rests on, and the bug it first had."""

    def test_the_two_functions_are_exact_inverses(self):
        # The first implied_cost_ratio returned the reciprocal, so this
        # round trip came back wrong while every other test passed.
        for miss in (1.0, 2.0, 5.0, 9.0, 20.0, 50.0):
            threshold = threshold_for_cost_ratio(miss, 1.0)
            self.assertAlmostEqual(implied_cost_ratio(threshold), miss, places=9)

    def test_half_means_equal_costs(self):
        self.assertAlmostEqual(implied_cost_ratio(0.5), 1.0, places=9)
        self.assertAlmostEqual(threshold_for_cost_ratio(1.0, 1.0), 0.5, places=9)

    def test_a_tenth_means_a_miss_costs_nine_times_as_much(self):
        self.assertAlmostEqual(implied_cost_ratio(0.1), 9.0, places=9)

    def test_a_lower_threshold_means_a_costlier_miss(self):
        ratios = [implied_cost_ratio(t) for t in (0.5, 0.2, 0.05, 0.01)]
        self.assertEqual(ratios, sorted(ratios))

    def test_doubling_both_costs_changes_nothing(self):
        self.assertAlmostEqual(threshold_for_cost_ratio(5.0, 1.0),
                               threshold_for_cost_ratio(10.0, 2.0), places=9)

    def test_bad_inputs_are_refused(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError):
                implied_cost_ratio(bad)
        with self.assertRaises(ValueError):
            threshold_for_cost_ratio(0.0, 1.0)


class TestThresholdRules(unittest.TestCase):
    def test_they_disagree_by_a_wide_margin(self):
        rules = [0.5, youden_threshold(COHORT.y, PROBABILITY),
                 f1_threshold(COHORT.y, PROBABILITY),
                 cost_threshold(COHORT.y, PROBABILITY, 20.0, 1.0)]
        self.assertGreater(max(rules) / min(rules), 5.0)

    def test_youden_finds_more_cases_than_the_default(self):
        default = report(COHORT.y, PROBABILITY >= 0.5)
        youden = report(COHORT.y, PROBABILITY >= youden_threshold(COHORT.y, PROBABILITY))
        self.assertGreater(youden["sensitivity"], default["sensitivity"])

    def test_and_flags_far_more_patients(self):
        default = report(COHORT.y, PROBABILITY >= 0.5)
        youden = report(COHORT.y, PROBABILITY >= youden_threshold(COHORT.y, PROBABILITY))
        self.assertGreater(youden["flagged"], 5 * default["flagged"])

    def test_a_costlier_miss_lowers_the_threshold(self):
        thresholds = [cost_threshold(COHORT.y, PROBABILITY, float(miss), 1.0)
                      for miss in (1, 5, 20, 50)]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_the_empirical_optimum_is_near_the_theoretical_one(self):
        # Not exact: the empirical search runs over the thresholds that
        # actually occur in a finite cohort.
        for miss in (2.0, 5.0, 20.0):
            empirical = cost_threshold(COHORT.y, PROBABILITY, miss, 1.0)
            theoretical = threshold_for_cost_ratio(miss, 1.0)
            self.assertAlmostEqual(empirical, theoretical, delta=0.08)

    def test_costs_must_be_positive(self):
        with self.assertRaises(ValueError):
            cost_threshold(COHORT.y, PROBABILITY, 0.0, 1.0)
        with self.assertRaises(ValueError):
            cost_threshold(COHORT.y, PROBABILITY, 1.0, -1.0)


class TestTotalCost(unittest.TestCase):
    def test_the_chosen_threshold_really_is_the_cheapest(self):
        for miss in (1.0, 5.0, 20.0):
            best = cost_threshold(COHORT.y, PROBABILITY, miss, 1.0)
            best_cost = total_cost(COHORT.y, PROBABILITY, best, miss, 1.0)
            for other in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
                self.assertLessEqual(
                    best_cost, total_cost(COHORT.y, PROBABILITY, other, miss, 1.0) + 1e-9)

    def test_the_best_threshold_moves_with_the_belief(self):
        # The day's conclusion: the cost ratio moves the cut further than
        # any model change in this repo.
        cheap = cost_threshold(COHORT.y, PROBABILITY, 1.0, 1.0)
        expensive = cost_threshold(COHORT.y, PROBABILITY, 20.0, 1.0)
        self.assertGreater(cheap / expensive, 5.0)

    def test_cost_is_zero_for_a_perfect_split(self):
        y = np.array([0] * 20 + [1] * 20)
        score = np.array([0.1] * 20 + [0.9] * 20)
        self.assertEqual(total_cost(y, score, 0.5, 3.0, 1.0), 0.0)

    def test_scaling_both_costs_scales_the_total(self):
        base = total_cost(COHORT.y, PROBABILITY, 0.2, 2.0, 1.0)
        doubled = total_cost(COHORT.y, PROBABILITY, 0.2, 4.0, 2.0)
        self.assertAlmostEqual(doubled, 2 * base, places=6)


if __name__ == "__main__":
    unittest.main()
