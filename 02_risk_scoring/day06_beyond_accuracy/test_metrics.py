"""Day 6 tests - the four rates, and which of them prevalence moves.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from metrics import (
    accuracy,
    always_negative_accuracy,
    auc,
    confusion,
    false_alarms_per_case,
    fit_logistic,
    make_cohort,
    npv,
    number_needed_to_screen,
    ppv,
    ppv_from_rates,
    predict,
    report,
    sensitivity,
    specificity,
)


class TestConfusion(unittest.TestCase):
    def test_it_counts_a_hand_worked_example(self):
        y = np.array([1, 1, 0, 0, 0])
        predicted = np.array([1, 0, 1, 0, 0])
        matrix = confusion(y, predicted)
        self.assertEqual(matrix.true_positive, 1)
        self.assertEqual(matrix.false_negative, 1)
        self.assertEqual(matrix.false_positive, 1)
        self.assertEqual(matrix.true_negative, 2)

    def test_the_counts_sum_to_the_cohort(self):
        y = np.random.default_rng(0).integers(0, 2, 500)
        predicted = np.random.default_rng(1).integers(0, 2, 500)
        self.assertEqual(confusion(y, predicted).total, 500)

    def test_prevalence_is_the_positive_rate(self):
        y = np.array([1] * 30 + [0] * 70)
        self.assertAlmostEqual(confusion(y, y).prevalence, 0.30)


class TestRatesOnHandWorkedCases(unittest.TestCase):
    def setUp(self):
        # 100 patients, 10 with the event, a test that finds 8 of them
        # and wrongly flags 9 of the 90 well ones.
        self.y = np.array([1] * 10 + [0] * 90)
        predicted = np.array([1] * 8 + [0] * 2 + [1] * 9 + [0] * 81)
        self.matrix = confusion(self.y, predicted)

    def test_sensitivity(self):
        self.assertAlmostEqual(sensitivity(self.matrix), 8 / 10)

    def test_specificity(self):
        self.assertAlmostEqual(specificity(self.matrix), 81 / 90)

    def test_ppv(self):
        self.assertAlmostEqual(ppv(self.matrix), 8 / 17)

    def test_npv(self):
        self.assertAlmostEqual(npv(self.matrix), 81 / 83)

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(self.matrix), 89 / 100)

    def test_ppv_matches_the_bayes_formula(self):
        # The same number two ways: counted, and from the rates.
        from_counts = ppv(self.matrix)
        from_rates = ppv_from_rates(sensitivity(self.matrix),
                                    specificity(self.matrix),
                                    self.matrix.prevalence)
        self.assertAlmostEqual(from_counts, from_rates, places=9)

    def test_empty_inputs_give_zero_rather_than_an_error(self):
        empty = confusion(np.array([]), np.array([]))
        for measure in (accuracy, sensitivity, specificity, ppv, npv):
            self.assertEqual(measure(empty), 0.0)


class TestAccuracyIsAboutPrevalence(unittest.TestCase):
    def test_always_saying_no_scores_one_minus_prevalence(self):
        for prevalence in (0.01, 0.1, 0.5):
            y = (np.arange(1000) < 1000 * prevalence).astype(int)
            matrix = confusion(y, np.zeros(1000))
            self.assertAlmostEqual(accuracy(matrix),
                                   always_negative_accuracy(prevalence), places=6)

    def test_a_model_finding_nothing_can_still_look_excellent(self):
        # The day's headline, as an assertion.
        cohort = make_cohort(n=40000, seed=0, prevalence=0.01)
        weights, intercept = fit_logistic(cohort.x, cohort.y)
        scores = report(cohort.y, predict(cohort.x, weights, intercept) > 0.5)
        self.assertGreater(scores["accuracy"], 0.98)
        self.assertLess(scores["sensitivity"], 0.05)

    def test_and_it_does_not_beat_the_trivial_baseline(self):
        cohort = make_cohort(n=40000, seed=0, prevalence=0.01)
        weights, intercept = fit_logistic(cohort.x, cohort.y)
        scores = report(cohort.y, predict(cohort.x, weights, intercept) > 0.5)
        baseline = always_negative_accuracy(cohort.prevalence)
        self.assertLess(abs(scores["accuracy"] - baseline), 0.01)


class TestPPVDependsOnPrevalence(unittest.TestCase):
    def test_a_known_textbook_case(self):
        # 95/95 test at 1% prevalence.
        self.assertAlmostEqual(ppv_from_rates(0.95, 0.95, 0.01), 0.1610, places=4)

    def test_it_rises_with_prevalence(self):
        values = [ppv_from_rates(0.95, 0.95, p)
                  for p in (0.001, 0.01, 0.05, 0.12, 0.30, 0.50)]
        self.assertEqual(values, sorted(values))

    def test_it_collapses_at_rare_disease(self):
        self.assertLess(ppv_from_rates(0.95, 0.95, 0.001), 0.03)

    def test_a_perfect_test_has_ppv_one_at_any_prevalence(self):
        for prevalence in (0.001, 0.5, 0.9):
            self.assertAlmostEqual(ppv_from_rates(1.0, 1.0, prevalence), 1.0)

    def test_false_alarms_per_case_is_the_same_fact(self):
        for prevalence in (0.01, 0.1, 0.4):
            value = ppv_from_rates(0.9, 0.9, prevalence)
            self.assertAlmostEqual(false_alarms_per_case(0.9, 0.9, prevalence),
                                   (1 - value) / value, places=9)

    def test_fifty_odd_false_alarms_at_one_in_a_thousand(self):
        self.assertGreater(false_alarms_per_case(0.95, 0.95, 0.001), 50.0)

    def test_number_needed_to_screen_falls_as_prevalence_rises(self):
        values = [number_needed_to_screen(0.95, p) for p in (0.001, 0.01, 0.1)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_bad_rates_are_refused(self):
        for bad in ((1.5, 0.9, 0.1), (0.9, -0.1, 0.1), (0.9, 0.9, 2.0)):
            with self.assertRaises(ValueError):
                ppv_from_rates(*bad)


class TestSensitivityAndSpecificityDoNot(unittest.TestCase):
    """They are measured within a group, so prevalence cannot move them."""

    def test_duplicating_the_well_patients_leaves_them_alone(self):
        y = np.array([1] * 10 + [0] * 90)
        predicted = np.array([1] * 8 + [0] * 2 + [1] * 9 + [0] * 81)
        base = confusion(y, predicted)
        # Same test, a population with three times as many well patients.
        diluted = confusion(np.concatenate([y, np.zeros(180)]),
                            np.concatenate([predicted, np.tile(
                                np.array([1] * 9 + [0] * 81), 2)]))
        self.assertAlmostEqual(sensitivity(base), sensitivity(diluted), places=9)
        self.assertAlmostEqual(specificity(base), specificity(diluted), places=9)

    def test_but_ppv_does_move(self):
        y = np.array([1] * 10 + [0] * 90)
        predicted = np.array([1] * 8 + [0] * 2 + [1] * 9 + [0] * 81)
        base = confusion(y, predicted)
        diluted = confusion(np.concatenate([y, np.zeros(180)]),
                            np.concatenate([predicted, np.tile(
                                np.array([1] * 9 + [0] * 81), 2)]))
        self.assertLess(ppv(diluted), ppv(base))


class TestAUC(unittest.TestCase):
    def test_a_perfect_ranking_scores_one(self):
        self.assertAlmostEqual(auc(np.array([0, 0, 1, 1]),
                                   np.array([0.1, 0.2, 0.3, 0.4])), 1.0)

    def test_a_reversed_ranking_scores_zero(self):
        self.assertAlmostEqual(auc(np.array([1, 1, 0, 0]),
                                   np.array([0.1, 0.2, 0.3, 0.4])), 0.0)

    def test_a_random_score_is_about_a_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 4000)
        self.assertAlmostEqual(auc(y, rng.random(4000)), 0.5, delta=0.03)

    def test_it_ignores_the_threshold(self):
        # Any monotone transform of the score leaves it unchanged.
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        score = rng.random(2000)
        self.assertAlmostEqual(auc(y, score), auc(y, score * 10 - 3), places=9)

    def test_it_barely_moves_with_prevalence(self):
        # The day's second point: the model is the same model.
        values = []
        for prevalence in (0.01, 0.05, 0.12, 0.26):
            cohort = make_cohort(n=40000, seed=0, prevalence=prevalence)
            weights, intercept = fit_logistic(cohort.x, cohort.y)
            values.append(auc(cohort.y, predict(cohort.x, weights, intercept)))
        self.assertLess(max(values) - min(values), 0.05)

    def test_while_sensitivity_at_a_fixed_threshold_collapses(self):
        values = []
        for prevalence in (0.01, 0.26):
            cohort = make_cohort(n=40000, seed=0, prevalence=prevalence)
            weights, intercept = fit_logistic(cohort.x, cohort.y)
            values.append(report(cohort.y,
                                 predict(cohort.x, weights, intercept) > 0.5)["sensitivity"])
        self.assertGreater(values[1] - values[0], 0.15)

    def test_a_single_class_gives_a_half_rather_than_an_error(self):
        self.assertEqual(auc(np.ones(10), np.random.default_rng(0).random(10)), 0.5)


if __name__ == "__main__":
    unittest.main()
