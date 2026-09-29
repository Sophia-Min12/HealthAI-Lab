"""Day 11 tests - evaluating de-identification.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import os
import re
import sys
import unittest

from evaluation import (
    GENERALISATIONS,
    SKEWED_CONDITIONS,
    anchored_gold,
    binomial_cdf,
    detect,
    detect_unicode,
    evaluate,
    k_anonymity,
    leak_rate_upper_bound,
    make_notes,
    name_span,
    partial_matches,
    quasi_identifier,
    relaxed_scores,
    rule_of_three,
    token_scores,
    uniqueness_by_condition,
)

NOTES = make_notes(n=300, seed=0)
STRICT = evaluate(NOTES)


class TestTheCorpusIsDay10s(unittest.TestCase):
    """Day 11 added attributes to Note. It must not have moved the corpus."""

    def test_attributes_cost_no_randomness(self):
        # Every number Day 10 reported has to survive the copy forward,
        # or the comparisons in this file are between two corpora.
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(here),
                                        "day10_deidentification"))
        try:
            import deid
        except ImportError:  # pragma: no cover - path-dependent
            self.skipTest("day10 module not importable from here")
        theirs = deid.make_notes(n=300, seed=0)
        key = lambda note: [(s.start, s.end, s.category, s.text)
                            for s in note.spans]
        self.assertEqual([n.text for n in NOTES], [n.text for n in theirs])
        self.assertEqual([key(n) for n in NOTES], [key(n) for n in theirs])

    def test_the_attributes_describe_the_note(self):
        pattern = re.compile(r"This (\d+)-year-old")
        for note in NOTES:
            self.assertEqual(int(pattern.search(note.text).group(1)),
                             note.attributes["age"])
            self.assertIn(note.attributes["condition"], note.text)
            self.assertIn(note.attributes["department"], note.text)

    def test_skewing_the_diagnoses_changes_the_corpus(self):
        skewed = make_notes(n=300, seed=0,
                            condition_weights=SKEWED_CONDITIONS)
        self.assertNotEqual([n.text for n in NOTES], [n.text for n in skewed])

    def test_the_skew_weights_are_a_distribution(self):
        self.assertAlmostEqual(sum(SKEWED_CONDITIONS), 1.0, places=9)


class TestThreeMetricsOneOutput(unittest.TestCase):
    def test_relaxed_can_never_be_below_strict(self):
        # True by construction: covered implies overlapping.
        self.assertGreaterEqual(relaxed_scores(NOTES)["recall"], STRICT.recall)

    def test_and_on_this_output_all_three_differ(self):
        token = token_scores(NOTES)["recall"]
        relaxed = relaxed_scores(NOTES)["recall"]
        self.assertNotAlmostEqual(token, relaxed, places=3)
        self.assertNotAlmostEqual(relaxed, STRICT.recall, places=3)

    def test_every_partial_match_leaves_text_behind(self):
        for _note_id, _truth, leftover in partial_matches(NOTES):
            self.assertTrue(leftover)


class TestTheUnicodeFix(unittest.TestCase):
    """The day's sharpest result."""

    def setUp(self):
        self.fixed = evaluate(NOTES, detect_unicode)

    def test_it_removes_every_partial_match(self):
        self.assertGreater(len(partial_matches(NOTES, detect)), 20)
        self.assertEqual(len(partial_matches(NOTES, detect_unicode)), 0)

    def test_strict_recall_rises(self):
        self.assertGreater(self.fixed.recall, STRICT.recall)
        self.assertGreater(self.fixed.clean_rate, STRICT.clean_rate)

    def test_relaxed_recall_does_not_move_at_all(self):
        # Not "barely moves" - does not move. Relaxed matching was
        # already scoring those spans as hits, so it cannot register
        # their repair.
        self.assertEqual(relaxed_scores(NOTES, detect)["recall"],
                         relaxed_scores(NOTES, detect_unicode)["recall"])

    def test_nor_does_token_recall(self):
        self.assertEqual(token_scores(NOTES, detect)["recall"],
                         token_scores(NOTES, detect_unicode)["recall"])

    def test_strict_after_the_fix_is_relaxed_before_it(self):
        # What relaxed matching was reporting all along: the score the
        # system would get once its leaks were fixed.
        self.assertAlmostEqual(self.fixed.recall,
                               relaxed_scores(NOTES, detect)["recall"],
                               places=12)

    def test_the_ascii_class_really_was_the_cause(self):
        text = "Patient: Astrid Bergström   MRN 1234567"
        self.assertEqual([s.text for s in detect(text) if s.category == "NAME"],
                         ["Astrid Bergstr"])
        self.assertEqual([s.text for s in detect_unicode(text)
                          if s.category == "NAME"], ["Astrid Bergström"])


class TestNameSpan(unittest.TestCase):
    def test_it_takes_two_capitalised_words(self):
        text = "Patient: Astrid Bergström   MRN"
        start = text.index("Astrid")
        self.assertEqual(name_span(text, start),
                         (start, start + len("Astrid Bergström")))

    def test_it_stops_at_a_lowercase_word(self):
        text = "Dr Abadi with atrial fibrillation"
        start = text.index("Abadi")
        self.assertEqual(name_span(text, start), (start, start + len("Abadi")))

    def test_it_does_not_cross_a_newline(self):
        text = "Haugen\nContact"
        self.assertEqual(name_span(text, 0), (0, len("Haugen")))

    def test_it_returns_none_where_there_is_no_name(self):
        self.assertIsNone(name_span("with atrial", 0))

    def test_it_respects_max_words(self):
        text = "Margaret Astrid Bergström"
        self.assertEqual(name_span(text, 0, max_words=1),
                         (0, len("Margaret")))
        self.assertEqual(name_span(text, 0, max_words=3), (0, len(text)))


class TestTheBinomialArithmetic(unittest.TestCase):
    def test_the_cdf_at_zero_is_the_closed_form(self):
        for n, p in ((10, 0.1), (300, 0.01), (50, 0.5)):
            self.assertAlmostEqual(binomial_cdf(0, n, p), (1 - p) ** n,
                                   places=12)

    def test_the_cdf_at_n_is_one(self):
        self.assertAlmostEqual(binomial_cdf(20, 20, 0.3), 1.0, places=12)

    def test_the_cdf_is_monotone_in_k(self):
        values = [binomial_cdf(k, 20, 0.3) for k in range(21)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later + 1e-15)

    def test_zero_events_matches_the_exact_closed_form(self):
        # P(X = 0) = alpha  =>  p = 1 - alpha ** (1/n)
        for n in (30, 100, 300, 1000):
            exact = 1 - 0.05 ** (1 / n)
            self.assertAlmostEqual(leak_rate_upper_bound(0, n), exact,
                                   places=9)

    def test_the_rule_of_three_is_within_a_few_percent(self):
        for n in (100, 300, 1000, 10000):
            exact = leak_rate_upper_bound(0, n)
            self.assertLess(abs(rule_of_three(n) - exact) / exact, 0.02)

    def test_the_approximation_is_worse_for_small_n(self):
        # 3/n overstates, and the overstatement is what makes it safe to
        # quote. It is still 5% out at n = 30.
        self.assertGreater(rule_of_three(30), leak_rate_upper_bound(0, 30))

    def test_the_bound_is_above_the_point_estimate(self):
        self.assertGreater(leak_rate_upper_bound(106, 300), 106 / 300)

    def test_the_bound_grows_with_the_leaks(self):
        bounds = [leak_rate_upper_bound(k, 300) for k in (0, 1, 10, 100)]
        for earlier, later in zip(bounds, bounds[1:]):
            self.assertLess(earlier, later)

    def test_all_leaks_bounds_at_one(self):
        self.assertEqual(leak_rate_upper_bound(300, 300), 1.0)

    def test_nonsense_is_refused(self):
        with self.assertRaises(ValueError):
            leak_rate_upper_bound(0, 0)
        with self.assertRaises(ValueError):
            leak_rate_upper_bound(5, 3)
        with self.assertRaises(ValueError):
            rule_of_three(0)


class TestTheAnchoredGoldStandard(unittest.TestCase):
    def test_a_fully_anchored_annotator_gives_a_perfect_score(self):
        # The annotator catches nothing the system missed, so the gold
        # standard is the system's own output and the system scores 1.0
        # against it. Not an approximation - exactly 1.0.
        gold = anchored_gold(NOTES, catch_probability=0.0, seed=1)
        scored = evaluate(gold)
        self.assertEqual(scored.recall, 1.0)
        self.assertEqual(scored.clean_rate, 1.0)

    def test_a_perfect_annotator_reproduces_the_truth(self):
        gold = anchored_gold(NOTES, catch_probability=1.0, seed=1)
        self.assertEqual([n.spans for n in gold], [n.spans for n in NOTES])

    def test_gold_is_always_a_subset_of_the_truth(self):
        gold = anchored_gold(NOTES, catch_probability=0.5, seed=1)
        for golden, true in zip(gold, NOTES):
            self.assertTrue(set(golden.spans) <= set(true.spans))

    def test_it_inflates_both_numbers(self):
        gold = evaluate(anchored_gold(NOTES, catch_probability=0.5, seed=1))
        self.assertGreater(gold.recall, STRICT.recall)
        self.assertGreater(gold.clean_rate, STRICT.clean_rate)

    def test_it_inflates_the_note_level_rate_far_more(self):
        # The day's claim: the metric that matters is the one this
        # distorts most.
        gold = evaluate(anchored_gold(NOTES, catch_probability=0.5, seed=1))
        span_inflation = gold.recall - STRICT.recall
        note_inflation = gold.clean_rate - STRICT.clean_rate
        self.assertGreater(note_inflation, 5 * span_inflation)

    def test_a_worse_annotator_inflates_more(self):
        careless = evaluate(anchored_gold(NOTES, catch_probability=0.25, seed=1))
        careful = evaluate(anchored_gold(NOTES, catch_probability=0.75, seed=1))
        self.assertGreater(careless.recall, careful.recall)
        self.assertGreater(careless.clean_rate, careful.clean_rate)


class TestWhatSurvivesDeIdentification(unittest.TestCase):
    def test_a_handmade_case_has_the_k_you_can_count(self):
        notes = make_notes(n=6, seed=3)
        for index, note in enumerate(notes):
            note.attributes["condition"] = "A" if index < 4 else "B"
        row = k_anonymity(notes, ("condition",))
        self.assertEqual(row["classes"], 2)
        self.assertEqual(row["k"], 2)
        self.assertEqual(row["unique"], 0)

    def test_a_singleton_class_is_unique(self):
        notes = make_notes(n=3, seed=3)
        for index, note in enumerate(notes):
            note.attributes["condition"] = f"C{index}"
        row = k_anonymity(notes, ("condition",))
        self.assertEqual(row["k"], 1)
        self.assertEqual(row["unique_rate"], 1.0)

    def test_almost_everyone_is_unique_on_the_retained_fields(self):
        row = k_anonymity(NOTES, GENERALISATIONS[0][1])
        self.assertGreater(row["unique_rate"], 0.95)

    def test_generalising_is_monotone(self):
        rows = [k_anonymity(NOTES, fields) for _label, fields in GENERALISATIONS]
        for earlier, later in zip(rows, rows[1:]):
            self.assertGreaterEqual(earlier["classes"], later["classes"])
            self.assertGreaterEqual(earlier["unique_rate"], later["unique_rate"])

    def test_the_age_band_is_the_decade(self):
        note = NOTES[0]
        note.attributes["age"] = 67
        self.assertEqual(quasi_identifier(note, ("band",)), (60,))

    def test_banding_never_reveals_more_than_the_exact_age(self):
        exact = k_anonymity(NOTES, ("age", "year", "condition"))
        banded = k_anonymity(NOTES, ("band", "year", "condition"))
        self.assertLessEqual(banded["unique_rate"], exact["unique_rate"])

    def test_the_rarest_diagnosis_is_the_most_identifiable(self):
        skewed = make_notes(n=300, seed=0,
                            condition_weights=SKEWED_CONDITIONS)
        rows = uniqueness_by_condition(skewed, ("band", "year", "condition"))
        self.assertEqual(sum(row["patients"] for row in rows), len(skewed))
        self.assertGreater(rows[0]["rate"], rows[-1]["rate"])

    def test_and_the_average_sits_between_them(self):
        skewed = make_notes(n=300, seed=0,
                            condition_weights=SKEWED_CONDITIONS)
        fields = ("band", "year", "condition")
        rows = uniqueness_by_condition(skewed, fields)
        overall = k_anonymity(skewed, fields)["unique_rate"]
        self.assertLess(overall, rows[0]["rate"])
        self.assertGreater(overall, rows[-1]["rate"])


if __name__ == "__main__":
    unittest.main()
