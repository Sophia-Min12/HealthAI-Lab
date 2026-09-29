"""Day 12 tests - negation, hedges, and abbreviations.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from clinical_text import (
    ABBREVIATIONS,
    ACTIONABLE,
    ASSERTIONS,
    HELDOUT_TEMPLATES,
    SENSE_FREQUENCY,
    TEMPLATES,
    AbbreviationUse,
    Mention,
    class_counts,
    classify,
    expand_by_department,
    expand_most_frequent,
    keyword_present,
    make_abbreviation_uses,
    make_combined,
    make_heldout_sentences,
    make_sentences,
    per_class_accuracy,
    score,
    score_expansion,
    score_pipeline,
    scope_prefix,
    scope_suffix,
)

SENTENCES = make_sentences(n=600, seed=0)
HELDOUT = make_heldout_sentences(n=600, seed=1)
FULL = score(SENTENCES, classify)


def one(text: str, concept: str, **kwargs) -> str:
    """Classify ``concept`` inside ``text``, by hand."""
    start = text.index(concept)
    return classify(text, Mention(concept, start, start + len(concept), ""),
                    **kwargs)


class TestTheGroundTruthIsActuallyTrue(unittest.TestCase):
    def test_every_mention_matches_the_text_at_its_offsets(self):
        for sentence in SENTENCES + HELDOUT:
            for mention in sentence.mentions:
                self.assertEqual(sentence.text[mention.start:mention.end],
                                 mention.concept)

    def test_every_assertion_is_declared(self):
        for sentence in SENTENCES + HELDOUT:
            for mention in sentence.mentions:
                self.assertIn(mention.assertion, ASSERTIONS)

    def test_the_corpus_is_deterministic(self):
        again = make_sentences(n=600, seed=0)
        self.assertEqual([s.text for s in SENTENCES], [s.text for s in again])

    def test_present_is_a_plurality_and_not_a_majority(self):
        # The single most important number in the file: it is the ceiling
        # on what a keyword search can achieve.
        counts = class_counts(SENTENCES)
        total = sum(counts.values())
        self.assertLess(counts["PRESENT"] / total, 0.5)
        self.assertGreater(counts["PRESENT"] / total, 0.25)

    def test_multi_concept_sentences_carry_different_assertions(self):
        mixed = [s for s in SENTENCES if len(s.mentions) > 1
                 and len({m.assertion for m in s.mentions}) > 1]
        self.assertGreater(len(mixed), 20)

    def test_the_heldout_forms_share_nothing_with_the_training_forms(self):
        for name, forms in HELDOUT_TEMPLATES.items():
            self.assertFalse(set(forms) & set(TEMPLATES[name]))


class TestTheKeywordBaseline(unittest.TestCase):
    def test_its_accuracy_is_exactly_the_present_share(self):
        # Closed form, not an approximation: it labels everything PRESENT.
        counts = class_counts(SENTENCES)
        total = sum(counts.values())
        naive = score(SENTENCES, keyword_present)
        self.assertAlmostEqual(naive.accuracy, counts["PRESENT"] / total,
                               places=12)

    def test_it_never_misses_an_actionable_mention(self):
        # It cannot: everything it returns is actionable. The error is
        # entirely in the other direction, which is the point.
        naive = score(SENTENCES, keyword_present)
        self.assertEqual(naive.missed_actionable, 0)
        self.assertGreater(naive.false_actionable, 300)


class TestNegation(unittest.TestCase):
    def test_the_plain_cases(self):
        self.assertEqual(one("No pneumonia.", "pneumonia"), "ABSENT")
        self.assertEqual(one("Denies pneumonia.", "pneumonia"), "ABSENT")
        self.assertEqual(one("No evidence of pneumonia.", "pneumonia"),
                         "ABSENT")
        self.assertEqual(one("Patient presents with pneumonia.", "pneumonia"),
                         "PRESENT")

    def test_a_negation_after_the_concept(self):
        # English puts it either side. A backwards-only rule set misses
        # every one of these.
        self.assertEqual(one("pneumonia was ruled out.", "pneumonia"),
                         "ABSENT")
        self.assertEqual(one("pneumonia has been excluded.", "pneumonia"),
                         "ABSENT")

    def test_a_historical_cue_after_the_concept(self):
        # Regression. The first run of the demo put all 18 of its errors
        # in this one template: the cue follows the concept, so a
        # backwards rule saw an empty prefix and called it current.
        text = "cough resolved, however atrial fibrillation persists."
        self.assertEqual(one(text, "cough"), "HISTORICAL")
        self.assertEqual(one(text, "atrial fibrillation"), "PRESENT")

    def test_scope_terminators_end_a_denial(self):
        text = "No fever but cough is present."
        self.assertEqual(one(text, "fever"), "ABSENT")
        self.assertEqual(one(text, "cough"), "PRESENT")

    def test_and_without_them_the_second_concept_inherits_the_negation(self):
        text = "No fever but cough is present."
        self.assertEqual(one(text, "cough", use_terminators=False), "ABSENT")

    def test_a_semicolon_also_terminates(self):
        text = "Denies chest pain; reports haemoptysis."
        self.assertEqual(one(text, "chest pain"), "ABSENT")
        self.assertEqual(one(text, "haemoptysis"), "PRESENT")

    def test_one_trigger_covers_a_list(self):
        text = "No fever, cough, or sepsis."
        for concept in ("fever", "cough", "sepsis"):
            self.assertEqual(one(text, concept), "ABSENT")

    def test_family_and_history_are_not_the_patient_now(self):
        self.assertEqual(one("Father had pneumonia.", "pneumonia"), "FAMILY")
        self.assertEqual(one("History of pneumonia, resolved.", "pneumonia"),
                         "HISTORICAL")

    def test_the_scope_helpers_agree_with_the_terminator_list(self):
        text = "No fever but cough is present."
        self.assertNotIn("No", scope_prefix(text, text.index("cough")))
        self.assertIn("No", scope_prefix(text, text.index("cough"),
                                         use_terminators=False))
        self.assertNotIn("however", scope_suffix("a however b", 1))


class TestHedges(unittest.TestCase):
    """Neither an assertion nor a denial, and the clinically loaded class."""

    def test_a_hedge_is_uncertain(self):
        self.assertEqual(one("Cannot rule out pneumonia.", "pneumonia"),
                         "UNCERTAIN")
        self.assertEqual(one("pneumonia cannot be excluded.", "pneumonia"),
                         "UNCERTAIN")
        self.assertEqual(one("Possible pneumonia.", "pneumonia"), "UNCERTAIN")

    def test_the_longest_trigger_wins_at_the_same_position(self):
        # "cannot rule out" and "rule out" end at the same character and
        # mean opposite things. Without the tie-breaker this is ABSENT.
        self.assertEqual(one("Cannot rule out sepsis.", "sepsis"), "UNCERTAIN")

    def test_a_two_class_detector_rounds_hedges_to_absent(self):
        self.assertEqual(one("Cannot rule out sepsis.", "sepsis",
                             use_hedges=False), "ABSENT")

    def test_and_that_is_where_its_error_lands(self):
        two_class = score(SENTENCES, lambda text, mention:
                          classify(text, mention, use_hedges=False))
        self.assertLess(two_class.accuracy, FULL.accuracy)
        self.assertGreater(two_class.missed_actionable,
                           FULL.missed_actionable)

    def test_uncertain_is_actionable(self):
        # "cannot rule out pulmonary embolism" is the sentence that
        # orders the scan.
        self.assertIn("UNCERTAIN", ACTIONABLE)
        self.assertIn("PRESENT", ACTIONABLE)
        self.assertNotIn("ABSENT", ACTIONABLE)


class TestTheCorpusItWasWrittenFor(unittest.TestCase):
    """The day's uncomfortable finding."""

    def test_it_is_perfect_on_the_sentences_the_rules_were_written_for(self):
        self.assertEqual(FULL.accuracy, 1.0)

    def test_and_much_worse_on_forms_it_has_not_seen(self):
        held = score(HELDOUT, classify)
        self.assertLess(held.accuracy, 0.85)
        self.assertGreater(held.accuracy, 0.6)

    def test_present_scores_one_on_held_out_text_by_defaulting(self):
        # Not recognition. PRESENT is what the classifier returns when no
        # rule fires, and on unfamiliar text no rule fires.
        rows = per_class_accuracy(HELDOUT, classify)
        self.assertEqual(rows["PRESENT"]["accuracy"], 1.0)
        for name in ("ABSENT", "UNCERTAIN", "FAMILY", "HISTORICAL"):
            self.assertLess(rows[name]["accuracy"], 0.8)

    def test_the_scope_rule_is_worth_a_real_amount(self):
        no_scope = score(SENTENCES, lambda text, mention:
                         classify(text, mention, use_terminators=False))
        self.assertGreater(FULL.accuracy - no_scope.accuracy, 0.05)


class TestAbbreviations(unittest.TestCase):
    def test_every_sense_frequency_is_a_distribution(self):
        for abbreviation, senses in SENSE_FREQUENCY.items():
            self.assertAlmostEqual(sum(senses.values()), 1.0, places=9,
                                   msg=abbreviation)

    def test_the_two_tables_describe_the_same_senses(self):
        for abbreviation, by_department in ABBREVIATIONS.items():
            self.assertEqual(set(by_department.values()),
                             set(SENSE_FREQUENCY[abbreviation]))

    def test_most_frequent_sense_ignores_the_department(self):
        cardiology = AbbreviationUse("", "MS", "Cardiology",
                                     "mitral stenosis", True)
        neurology = AbbreviationUse("", "MS", "Neurology",
                                    "multiple sclerosis", True)
        self.assertEqual(expand_most_frequent(cardiology),
                         expand_most_frequent(neurology))

    def test_the_department_rule_uses_it(self):
        cardiology = AbbreviationUse("", "MS", "Cardiology",
                                     "mitral stenosis", True)
        self.assertEqual(expand_by_department(cardiology), "mitral stenosis")

    def test_an_unknown_department_falls_back(self):
        stray = AbbreviationUse("", "MS", "Orthopaedics", "mitral stenosis",
                                False)
        self.assertEqual(expand_by_department(stray),
                         expand_most_frequent(stray))

    def test_the_department_rule_beats_the_baseline_overall(self):
        uses = make_abbreviation_uses(n=600, seed=0)
        baseline = score_expansion(uses, expand_most_frequent)
        conditioned = score_expansion(uses, expand_by_department)
        self.assertGreater(conditioned["accuracy"], baseline["accuracy"])

    def test_and_is_perfect_inside_the_department_and_useless_outside_it(self):
        # The headline is not a 94% system. It is a 100% system on most
        # notes and a 0% system on the rest, and the average hides which.
        uses = make_abbreviation_uses(n=600, seed=0)
        row = score_expansion(uses, expand_by_department)
        self.assertEqual(row["in_department"], 1.0)
        self.assertEqual(row["cross_department"], 0.0)
        self.assertGreater(row["cross_n"], 10)

    def test_with_no_cross_department_use_the_rule_looks_flawless(self):
        clean = make_abbreviation_uses(n=600, seed=0, cross_department=0.0)
        self.assertEqual(score_expansion(clean, expand_by_department)["accuracy"],
                         1.0)


class TestThePipeline(unittest.TestCase):
    def test_both_stages_must_be_right(self):
        row = score_pipeline(make_combined(n=600, seed=2))
        self.assertLessEqual(row["joint"], row["expansion"])
        self.assertLessEqual(row["joint"], row["assertion"])

    def test_the_product_is_close_to_the_measurement(self):
        row = score_pipeline(make_combined(n=600, seed=2))
        self.assertAlmostEqual(row["joint"], row["product"], delta=0.05)

    def test_and_well_below_either_stage(self):
        row = score_pipeline(make_combined(n=600, seed=2))
        self.assertLess(row["joint"], min(row["expansion"], row["assertion"])
                        - 0.03)

    def test_a_perfect_pipeline_is_the_product_of_ones(self):
        cases = make_combined(n=50, seed=2)
        row = score_pipeline(cases,
                             expander=lambda use: use.sense,
                             classifier=lambda text, mention: mention.assertion)
        self.assertEqual(row["joint"], 1.0)


if __name__ == "__main__":
    unittest.main()
