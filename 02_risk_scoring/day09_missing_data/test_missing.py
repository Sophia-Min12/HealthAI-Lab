"""Day 9 tests - the three mechanisms, and where each imputation lies.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from missing import (
    CHOLESTEROL,
    TRUE_COEFFICIENTS,
    add_missing_indicator,
    complete_cases,
    drop_mar,
    drop_mcar,
    drop_mnar,
    fit_and_score,
    fit_logistic,
    impute_conditional,
    impute_mean,
    make_cohort,
    predict,
    subgroup_report,
)

COHORT = make_cohort(n=20000, seed=0, prevalence=0.12, correlate_cholesterol=1.5)
TRUTH = TRUE_COEFFICIENTS["cholesterol_over_5_per_1"]
DRIVERS = [0, 1, 3, 4]


class TestMechanisms(unittest.TestCase):
    def test_all_three_remove_about_the_requested_fraction(self):
        for damaged in (drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1),
                        drop_mar(COHORT.x, CHOLESTEROL, 0, 0.4, seed=1),
                        drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)):
            rate = float(np.isnan(damaged[:, CHOLESTEROL]).mean())
            self.assertAlmostEqual(rate, 0.4, delta=0.05)

    def test_they_only_touch_the_named_column(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        for column in range(COHORT.x.shape[1]):
            if column != CHOLESTEROL:
                self.assertFalse(np.isnan(damaged[:, column]).any())

    def test_mcar_leaves_the_observed_mean_unbiased(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        self.assertAlmostEqual(float(np.nanmean(damaged[:, CHOLESTEROL])),
                               float(COHORT.x[:, CHOLESTEROL].mean()), delta=0.05)

    def test_mnar_biases_the_observed_mean_upward(self):
        # The missing values are the low ones, so what remains is high.
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        self.assertGreater(float(np.nanmean(damaged[:, CHOLESTEROL])),
                           float(COHORT.x[:, CHOLESTEROL].mean()) + 0.3)

    def test_mar_missingness_is_predictable_from_the_driver(self):
        damaged = drop_mar(COHORT.x, CHOLESTEROL, 0, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertLess(float(COHORT.x[missing, 0].mean()),
                        float(COHORT.x[~missing, 0].mean()))

    def test_mnar_missingness_is_predictable_from_the_hidden_value(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertLess(float(COHORT.x[missing, CHOLESTEROL].mean()),
                        float(COHORT.x[~missing, CHOLESTEROL].mean()))

    def test_they_are_reproducible(self):
        np.testing.assert_array_equal(drop_mcar(COHORT.x, CHOLESTEROL, 0.3, seed=7),
                                      drop_mcar(COHORT.x, CHOLESTEROL, 0.3, seed=7))

    def test_a_rate_outside_zero_to_one_is_refused(self):
        with self.assertRaises(ValueError):
            drop_mcar(COHORT.x, CHOLESTEROL, 1.5)


class TestCompleteCases(unittest.TestCase):
    def test_it_keeps_only_whole_rows(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        kept_x, _ = complete_cases(damaged, COHORT.y)
        self.assertFalse(np.isnan(kept_x).any())

    def test_it_keeps_x_and_y_aligned(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        kept_x, kept_y = complete_cases(damaged, COHORT.y)
        self.assertEqual(len(kept_x), len(kept_y))

    def test_it_is_unbiased_under_mcar(self):
        # The one mechanism where discarding rows is safe.
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        kept_x, kept_y = complete_cases(damaged, COHORT.y)
        scores = fit_and_score(kept_x, kept_y)
        self.assertAlmostEqual(scores["weights"][CHOLESTEROL], TRUTH, delta=0.06)

    def test_it_throws_away_a_lot_of_data(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        kept_x, _ = complete_cases(damaged, COHORT.y)
        self.assertLess(len(kept_x), 0.7 * len(COHORT.x))


class TestMeanImputationAttenuates(unittest.TestCase):
    """The clearest failure mode, under the friendliest mechanism."""

    def test_it_fills_every_gap(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        self.assertFalse(np.isnan(impute_mean(damaged)).any())

    def test_it_puts_every_imputed_patient_at_the_same_value(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        filled = impute_mean(damaged)[missing, CHOLESTEROL]
        self.assertEqual(len(np.unique(filled)), 1)

    def test_it_shrinks_the_columns_variance(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        self.assertLess(impute_mean(damaged)[:, CHOLESTEROL].var(),
                        COHORT.x[:, CHOLESTEROL].var())

    def test_and_therefore_halves_the_coefficient(self):
        # Under MCAR, where nothing is informative and it is still wrong.
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        scores = fit_and_score(impute_mean(damaged), COHORT.y)
        self.assertLess(scores["weights"][CHOLESTEROL], 0.6 * TRUTH)

    def test_complete_cases_beat_it_under_mcar(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        kept_x, kept_y = complete_cases(damaged, COHORT.y)
        dropped = abs(fit_and_score(kept_x, kept_y)["weights"][CHOLESTEROL] - TRUTH)
        imputed = abs(fit_and_score(impute_mean(damaged),
                                    COHORT.y)["weights"][CHOLESTEROL] - TRUTH)
        self.assertLess(dropped, imputed)


class TestConditionalImputation(unittest.TestCase):
    def test_it_gives_each_patient_a_different_value(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        filled = impute_conditional(damaged, CHOLESTEROL, DRIVERS)[missing, CHOLESTEROL]
        self.assertGreater(len(np.unique(filled)), 100)

    def test_it_recovers_the_coefficient_under_all_three_mechanisms(self):
        for damaged in (drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1),
                        drop_mar(COHORT.x, CHOLESTEROL, 0, 0.4, seed=1),
                        drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)):
            scores = fit_and_score(
                impute_conditional(damaged, CHOLESTEROL, DRIVERS), COHORT.y)
            self.assertAlmostEqual(scores["weights"][CHOLESTEROL], TRUTH, delta=0.06)

    def test_it_needs_the_drivers_to_actually_predict(self):
        # On an uncorrelated cohort it is mean imputation with extra
        # steps, and that is the stated caveat rather than a surprise.
        plain = make_cohort(n=20000, seed=0, prevalence=0.12)
        damaged = drop_mcar(plain.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        mean_filled = impute_mean(damaged)[missing, CHOLESTEROL]
        conditional = impute_conditional(damaged, CHOLESTEROL,
                                         DRIVERS)[missing, CHOLESTEROL]
        self.assertLess(float(np.abs(mean_filled - conditional).max()), 0.2)

    def test_and_differs_substantially_when_they_do(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        mean_filled = impute_mean(damaged)[missing, CHOLESTEROL]
        conditional = impute_conditional(damaged, CHOLESTEROL,
                                         DRIVERS)[missing, CHOLESTEROL]
        self.assertGreater(float(np.abs(mean_filled - conditional).max()), 1.0)


class TestMNARDamageIsInTheSubgroup(unittest.TestCase):
    """It survives the coefficient and shows up per patient."""

    def test_mcar_imputation_recovers_the_hidden_mean(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        filled = impute_conditional(damaged, CHOLESTEROL, DRIVERS)
        self.assertAlmostEqual(float(filled[missing, CHOLESTEROL].mean()),
                               float(COHORT.x[missing, CHOLESTEROL].mean()),
                               delta=0.1)

    def test_mnar_imputation_does_not(self):
        # It fills too high, because the missing ones were the low ones.
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        filled = impute_conditional(damaged, CHOLESTEROL, DRIVERS)
        self.assertGreater(float(filled[missing, CHOLESTEROL].mean()),
                           float(COHORT.x[missing, CHOLESTEROL].mean()) + 0.4)

    def test_the_missing_group_is_over_predicted_under_mnar(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        rows = subgroup_report(COHORT,
                               impute_conditional(damaged, CHOLESTEROL, DRIVERS),
                               missing)
        by_label = {row["label"]: row for row in rows}
        self.assertGreater(by_label["value missing"]["predicted"],
                           by_label["value missing"]["true_risk"])

    def test_the_two_groups_have_genuinely_different_risk_under_mnar(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertLess(float(COHORT.risk[missing].mean()),
                        float(COHORT.risk[~missing].mean()) - 0.03)

    def test_and_not_under_mcar(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertAlmostEqual(float(COHORT.risk[missing].mean()),
                               float(COHORT.risk[~missing].mean()), delta=0.02)


class TestMissingnessIsInformative(unittest.TestCase):
    def test_the_indicator_must_be_added_before_imputing(self):
        # A silent failure otherwise: the column is there, all zeros, and
        # nothing errors. The first version of this day's demo did it in
        # the wrong order and three strategies produced identical numbers.
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        with self.assertRaises(ValueError):
            add_missing_indicator(impute_mean(damaged), CHOLESTEROL)

    def test_in_the_right_order_it_records_the_pattern(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        widened = add_missing_indicator(damaged, CHOLESTEROL)
        self.assertEqual(widened.shape[1], COHORT.x.shape[1] + 1)
        self.assertEqual(int(widened[:, -1].sum()),
                         int(np.isnan(damaged[:, CHOLESTEROL]).sum()))

    def test_under_mnar_missingness_separates_the_outcome(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertLess(float(COHORT.y[missing].mean()),
                        float(COHORT.y[~missing].mean()) - 0.04)

    def test_under_mcar_it_does_not(self):
        damaged = drop_mcar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        missing = np.isnan(damaged[:, CHOLESTEROL])
        self.assertAlmostEqual(float(COHORT.y[missing].mean()),
                               float(COHORT.y[~missing].mean()), delta=0.02)

    def test_the_indicator_gets_a_real_coefficient_under_mnar(self):
        damaged = drop_mnar(COHORT.x, CHOLESTEROL, 0.4, seed=1)
        scores = fit_and_score(
            impute_mean(add_missing_indicator(damaged, CHOLESTEROL)), COHORT.y)
        self.assertLess(scores["weights"][-1], -0.05)


if __name__ == "__main__":
    unittest.main()
