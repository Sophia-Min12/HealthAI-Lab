"""Day 5 tests - the fit, the coefficients, and what makes them unreliable.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from logistic import (
    FEATURE_NAMES,
    TRUE_COEFFICIENTS,
    TRUE_INTERCEPT,
    coefficient_spread,
    fit_logistic,
    intercept_for_prevalence,
    log_loss,
    make_cohort,
    odds_ratio,
    predict,
    recovery_error,
    risk_ratio,
    sigmoid,
)


class TestSigmoid(unittest.TestCase):
    def test_zero_is_a_half(self):
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5)

    def test_it_stays_within_zero_and_one(self):
        values = sigmoid(np.linspace(-50, 50, 500))
        self.assertTrue(np.all(values >= 0))
        self.assertTrue(np.all(values <= 1))

    def test_it_saturates_to_exactly_one(self):
        # Not a bug, and not something to assert away: 1 + 2.2e-16 is 1.0
        # at double precision, so the sigmoid reaches exactly 1.0 from
        # z = 37 upward. The first version of the test above asserted
        # strict inequality and failed here.
        self.assertLess(float(sigmoid(np.array([36.0]))[0]), 1.0)
        self.assertEqual(float(sigmoid(np.array([37.0]))[0]), 1.0)

    def test_and_that_is_why_log_loss_clips(self):
        # Saturation plus an unlucky patient is an infinite mean loss.
        saturated = sigmoid(np.array([40.0]))
        self.assertEqual(float(saturated[0]), 1.0)
        self.assertTrue(np.isfinite(log_loss(np.array([0.0]), saturated)))

    def test_it_does_not_overflow(self):
        # exp(800) is infinity in float64, and a confident model produces
        # linear predictors that large.
        values = sigmoid(np.array([-800.0, 800.0, -1e6, 1e6]))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_it_is_symmetric(self):
        for z in (0.3, 1.0, 4.0):
            self.assertAlmostEqual(float(sigmoid(np.array([z]))[0]),
                                   1 - float(sigmoid(np.array([-z]))[0]))

    def test_it_is_monotone(self):
        values = sigmoid(np.linspace(-6, 6, 100))
        self.assertTrue(np.all(np.diff(values) > 0))


class TestCohort(unittest.TestCase):
    def test_it_is_reproducible(self):
        first, second = make_cohort(n=200, seed=5), make_cohort(n=200, seed=5)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y, second.y)

    def test_the_outcome_is_sampled_not_thresholded(self):
        # The distinction Day 8 needs to exist: some high-risk patients
        # have no event and some low-risk ones do.
        cohort = make_cohort(n=4000, seed=0)
        high_without = ((cohort.risk > 0.5) & (cohort.y == 0)).sum()
        low_with = ((cohort.risk < 0.2) & (cohort.y == 1)).sum()
        self.assertGreater(high_without, 0)
        self.assertGreater(low_with, 0)

    def test_the_outcome_rate_tracks_the_true_risk(self):
        cohort = make_cohort(n=20000, seed=0)
        self.assertAlmostEqual(cohort.prevalence, float(cohort.risk.mean()),
                               delta=0.02)

    def test_features_are_centred_on_clinical_references(self):
        # Age in decades past 50, systolic in tens past 120, so a
        # coefficient reads per decade and the intercept describes a
        # reference patient rather than a newborn.
        cohort = make_cohort(n=5000, seed=0)
        self.assertAlmostEqual(float(cohort.x[:, 0].mean()), 0.8, delta=0.1)
        self.assertAlmostEqual(float(cohort.x[:, 1].mean()), 1.2, delta=0.1)

    def test_the_binary_features_are_binary(self):
        cohort = make_cohort(n=1000, seed=0)
        for column in (3, 4):
            self.assertEqual(set(np.unique(cohort.x[:, column])), {0.0, 1.0})

    def test_a_non_positive_size_is_refused(self):
        with self.assertRaises(ValueError):
            make_cohort(n=0)


class TestPrevalenceKnob(unittest.TestCase):
    def test_it_hits_the_requested_rate(self):
        for target in (0.05, 0.12, 0.26, 0.40):
            cohort = make_cohort(n=8000, seed=0, prevalence=target)
            self.assertAlmostEqual(cohort.prevalence, target, delta=0.02)

    def test_a_lower_target_gives_a_lower_intercept(self):
        low = intercept_for_prevalence(0.05)
        high = intercept_for_prevalence(0.40)
        self.assertLess(low, high)

    def test_the_default_intercept_gives_the_documented_rate(self):
        # The constant's comment claims about 26%; check it rather than
        # trusting it. The first version of that comment said 12%.
        self.assertAlmostEqual(make_cohort(n=20000, seed=0).prevalence, 0.26,
                               delta=0.02)

    def test_a_prevalence_outside_zero_to_one_is_refused(self):
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                intercept_for_prevalence(bad)


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = make_cohort(n=20000, seed=0)
        cls.weights, cls.intercept = fit_logistic(cls.cohort.x, cls.cohort.y)

    def test_it_recovers_every_coefficient(self):
        errors = recovery_error(self.weights)
        for name, error in errors.items():
            self.assertLess(error, 0.12, f"{name} off by {error:.3f}")

    def test_it_recovers_the_intercept(self):
        self.assertAlmostEqual(self.intercept, TRUE_INTERCEPT, delta=0.15)

    def test_it_gets_the_ordering_right(self):
        # Diabetic > smoker > age > systolic > cholesterol, as generated.
        order = [FEATURE_NAMES[i] for i in np.argsort(-self.weights)]
        self.assertEqual(order[0], "diabetic")
        self.assertEqual(order[1], "smoker")

    def test_more_data_reduces_the_error(self):
        small = np.mean(list(recovery_error(
            fit_logistic(*[getattr(make_cohort(n=400, seed=3), a)
                           for a in ("x", "y")])[0]).values()))
        large = np.mean(list(recovery_error(
            fit_logistic(*[getattr(make_cohort(n=20000, seed=3), a)
                           for a in ("x", "y")])[0]).values()))
        self.assertLess(large, small)

    def test_the_loss_decreases(self):
        early, _ = fit_logistic(self.cohort.x, self.cohort.y, iterations=5)
        late, late_b = fit_logistic(self.cohort.x, self.cohort.y, iterations=2000)
        early_loss = log_loss(self.cohort.y, predict(self.cohort.x, early, 0.0))
        late_loss = log_loss(self.cohort.y, predict(self.cohort.x, late, late_b))
        self.assertLess(late_loss, early_loss)

    def test_predictions_are_probabilities(self):
        probability = predict(self.cohort.x, self.weights, self.intercept)
        self.assertTrue(np.all(probability > 0))
        self.assertTrue(np.all(probability < 1))

    def test_l2_shrinks_the_weights(self):
        plain, _ = fit_logistic(self.cohort.x, self.cohort.y, l2=0.0)
        shrunk, _ = fit_logistic(self.cohort.x, self.cohort.y, l2=0.5)
        self.assertLess(np.abs(shrunk).sum(), np.abs(plain).sum())

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            fit_logistic(self.cohort.x, self.cohort.y, learning_rate=0)
        with self.assertRaises(ValueError):
            fit_logistic(self.cohort.x, self.cohort.y, l2=-1)


class TestLogLoss(unittest.TestCase):
    def test_a_perfect_prediction_scores_near_zero(self):
        y = np.array([1.0, 0.0, 1.0])
        self.assertLess(log_loss(y, np.array([0.999, 0.001, 0.999])), 0.01)

    def test_a_confident_wrong_prediction_is_large_but_finite(self):
        # Clipping is why. One such patient would otherwise make the
        # whole average infinite.
        value = log_loss(np.array([1.0]), np.array([0.0]))
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 10.0)

    def test_predicting_the_base_rate_gives_the_entropy(self):
        # Independently-known answer: guessing p for everyone scores the
        # binary entropy of p.
        y = np.array([1.0] * 300 + [0.0] * 700)
        expected = -(0.3 * np.log(0.3) + 0.7 * np.log(0.7))
        self.assertAlmostEqual(log_loss(y, np.full(1000, 0.3)), expected, places=6)


class TestOddsAndRisk(unittest.TestCase):
    def test_a_zero_coefficient_is_an_odds_ratio_of_one(self):
        self.assertAlmostEqual(odds_ratio(0.0), 1.0)

    def test_a_known_value(self):
        self.assertAlmostEqual(odds_ratio(0.8), 2.2255, places=3)

    def test_the_risk_ratio_approaches_the_odds_ratio_at_low_risk(self):
        self.assertAlmostEqual(risk_ratio(0.8, 0.001), odds_ratio(0.8), delta=0.01)

    def test_and_diverges_from_it_at_high_risk(self):
        self.assertLess(risk_ratio(0.8, 0.60), odds_ratio(0.8) * 0.7)

    def test_the_risk_ratio_falls_as_the_baseline_rises(self):
        ratios = [risk_ratio(0.8, base) for base in (0.01, 0.12, 0.30, 0.60)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))

    def test_a_baseline_outside_zero_to_one_is_refused(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                risk_ratio(0.8, bad)


class TestCollinearityCostsVarianceNotBias(unittest.TestCase):
    """The day's finding, and a correction of my own expectation."""

    def test_correlated_features_do_not_move_the_coefficient(self):
        # What I expected to happen, measured, and it does not: at twenty
        # thousand patients the estimates are essentially unchanged.
        independent, _ = fit_logistic(*[getattr(make_cohort(n=20000, seed=0), a)
                                        for a in ("x", "y")], iterations=6000)
        correlated_cohort = make_cohort(n=20000, seed=0, correlate_cholesterol=2.0)
        correlated, _ = fit_logistic(correlated_cohort.x, correlated_cohort.y,
                                     iterations=6000)
        self.assertAlmostEqual(independent[0], correlated[0], delta=0.1)

    def test_the_correlation_is_actually_present(self):
        cohort = make_cohort(n=20000, seed=0, correlate_cholesterol=2.0)
        r = float(np.corrcoef(cohort.x[:, 0], cohort.x[:, 2])[0, 1])
        self.assertGreater(r, 0.8)

    def test_but_it_doubles_the_spread(self):
        independent = coefficient_spread(300, 0.0, repeats=20)
        correlated = coefficient_spread(300, 2.0, repeats=20)
        self.assertGreater(correlated["sd"], 2 * independent["sd"])

    def test_the_mean_stays_unbiased_under_correlation(self):
        correlated = coefficient_spread(1000, 2.0, repeats=20)
        self.assertAlmostEqual(correlated["mean"],
                               TRUE_COEFFICIENTS["age_decades_over_50"], delta=0.12)

    def test_one_small_study_could_report_the_wrong_direction(self):
        # The consequence, as a number: forty honest studies of three
        # hundred patients, and the range of age odds ratios spans 1.0.
        spread = coefficient_spread(300, 2.0, repeats=40)
        self.assertLess(spread["min"], 0.0)
        self.assertGreater(spread["max"], 1.0)

    def test_independent_features_never_cross_zero(self):
        spread = coefficient_spread(300, 0.0, repeats=40)
        self.assertGreater(spread["min"], 0.0)


if __name__ == "__main__":
    unittest.main()
