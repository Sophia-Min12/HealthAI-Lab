"""Day 10 tests - de-identification, and the two denominators.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import re
import unittest

from deid import (
    CATEGORIES,
    EPONYM_PHRASES,
    EPONYM_SURNAMES,
    GAZETTEER_SURNAMES,
    PATTERNS,
    Span,
    covered,
    detect,
    detect_aggressive,
    detect_without_eponyms,
    evaluate,
    independence_prediction,
    make_notes,
    over_redaction,
    partial_matches,
    redact,
)

NOTES = make_notes(n=300, seed=0)
RESULT = evaluate(NOTES)


class TestTheGroundTruthIsActuallyTrue(unittest.TestCase):
    """If the generator's record is wrong, every other number is decoration."""

    def test_every_span_matches_the_text_at_its_offsets(self):
        for note in NOTES:
            for span in note.spans:
                self.assertEqual(note.text[span.start:span.end], span.text)

    def test_no_two_truth_spans_overlap(self):
        for note in NOTES:
            ordered = sorted(note.spans, key=lambda s: s.start)
            for earlier, later in zip(ordered, ordered[1:]):
                self.assertLessEqual(earlier.end, later.start)

    def test_every_category_is_declared(self):
        for note in NOTES:
            for span in note.spans:
                self.assertIn(span.category, CATEGORIES)

    def test_the_corpus_is_deterministic(self):
        again = make_notes(n=300, seed=0)
        self.assertEqual([n.text for n in NOTES], [n.text for n in again])
        self.assertEqual([n.spans for n in NOTES], [n.spans for n in again])

    def test_a_different_seed_gives_a_different_corpus(self):
        other = make_notes(n=300, seed=1)
        self.assertNotEqual(NOTES[0].text, other[0].text)

    def test_ages_are_identifiers_only_above_89(self):
        # Safe Harbor's rule, and the reason it is a rule: the tail is
        # thin enough to single a person out.
        pattern = re.compile(r"This (\d+)-year-old")
        for note in NOTES:
            match = pattern.search(note.text)
            self.assertIsNotNone(match)
            age = int(match.group(1))
            marked = bool(note.by_category("AGE"))
            self.assertEqual(marked, age > 89,
                             f"age {age} marked={marked}")

    def test_clinical_eponym_phrases_are_not_identifiers(self):
        # The phrases exist in the corpus and none of them is truth. A
        # redactor that removes them is wrong, and section 5 measures it.
        occurrences = 0
        for note in NOTES:
            for phrase in EPONYM_PHRASES:
                for match in re.finditer(re.escape(phrase), note.text):
                    occurrences += 1
                    probe = Span(match.start(), match.end(), "CLINICAL", phrase)
                    for span in note.spans:
                        self.assertFalse(span.overlaps(probe))
        self.assertGreater(occurrences, 50)


class TestCoverage(unittest.TestCase):
    """``covered`` is the leak test, and overlap is not it."""

    def test_a_full_match_is_covered(self):
        truth = Span(10, 20, "NAME", "x" * 10)
        self.assertTrue(covered(truth, [Span(10, 20, "NAME", "x" * 10)]))

    def test_a_wider_prediction_is_covered(self):
        truth = Span(10, 20, "NAME", "x" * 10)
        self.assertTrue(covered(truth, [Span(5, 25, "NAME", "x" * 20)]))

    def test_a_partial_match_is_not_covered(self):
        # "Margaret Whitfield" caught as "Margaret" overlaps the truth
        # and leaves the surname on the page.
        truth = Span(10, 20, "NAME", "x" * 10)
        self.assertFalse(covered(truth, [Span(10, 15, "NAME", "x" * 5)]))

    def test_two_predictions_can_cover_between_them(self):
        truth = Span(10, 20, "NAME", "x" * 10)
        self.assertTrue(covered(truth, [Span(10, 15, "NAME", "x" * 5),
                                        Span(15, 20, "NAME", "x" * 5)]))

    def test_nothing_predicted_is_not_covered(self):
        self.assertFalse(covered(Span(0, 3, "NAME", "abc"), []))


class TestTheDetector(unittest.TestCase):
    def test_predicted_spans_do_not_overlap(self):
        for note in NOTES[:40]:
            predicted = detect(note.text)
            for earlier, later in zip(predicted, predicted[1:]):
                self.assertLessEqual(earlier.end, later.start)

    def test_span_recall_is_in_the_mid_nineties(self):
        # The figure a rule-based system reports, and the reason the
        # headline is reassuring.
        self.assertGreater(RESULT.recall, 0.93)
        self.assertLess(RESULT.recall, 0.97)

    def test_an_address_span_never_crosses_a_newline(self):
        # Regression. The first version used \s, which matches a newline,
        # so ", Larkfield" ran into the "Contact:" label on the next line.
        # It swallowed a field name, scored a clean hit on the address,
        # and was invisible to every recall figure in the demo.
        text = "Address on file: 153 Fairhaven Road, Larkfield\nContact: x"
        spans = [s for s in detect(text) if s.category == "ADDRESS"]
        self.assertEqual(len(spans), 1)
        self.assertNotIn("\n", spans[0].text)
        self.assertNotIn("Contact", spans[0].text)

    def test_the_address_pattern_itself_refuses_newlines(self):
        pattern = dict(PATTERNS)["ADDRESS"]
        match = pattern.search("1 Alder Road,\nAshford Hospital")
        self.assertEqual(match.group(), "1 Alder Road")

    def test_titles_catch_names_the_gazetteer_does_not_have(self):
        unknown = "Zlatanovic"
        self.assertNotIn(unknown, GAZETTEER_SURNAMES)
        spans = detect(f"seen by Dr {unknown} today")
        self.assertTrue(any(s.category == "NAME" and unknown in s.text
                            for s in spans))

    def test_an_ascii_character_class_truncates_a_name_with_a_diacritic(self):
        # Found by reading the miss list, not by reading the code. Every
        # class in the module is [a-z]; the pattern stops at the first
        # letter outside ASCII, so the rule fires, redacts most of the
        # name and leaves the distinctive part on the page.
        spans = detect("Patient: Astrid Bergström   MRN 1234567")
        names = [s for s in spans if s.category == "NAME"]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].text, "Astrid Bergstr")

    def test_and_that_leaves_the_tail_of_the_name_in_the_output(self):
        text = "Patient: Astrid Bergström   MRN 1234567"
        self.assertIn("öm", redact(text, detect(text)))

    def test_it_is_about_half_the_name_failures_in_the_corpus(self):
        partials = partial_matches(NOTES)
        truncated = {truth.text for _, truth, _ in partials}
        name_misses = [t for _, t in RESULT.misses if t.category == "NAME"]
        from_truncation = sum(1 for t in name_misses if t.text in truncated)
        self.assertGreater(from_truncation, 0.4 * len(name_misses))
        self.assertTrue(all(left == "öm" for _, _, left in partials))

    def test_an_uncued_unknown_surname_is_missed(self):
        # Not a defect to be fixed - the open-class problem. There is no
        # rule here that can fire, and no list that would contain it.
        unknown = "Zlatanovic"
        spans = detect(f"Case discussed with {unknown} at the MDT.")
        self.assertFalse(any(unknown in s.text for s in spans))

    def test_ages_at_or_below_89_are_left_alone(self):
        self.assertFalse([s for s in detect("This 89-year-old patient")
                          if s.category == "AGE"])
        self.assertTrue([s for s in detect("This 91-year-old patient")
                         if s.category == "AGE"])


class TestTheTwoDenominators(unittest.TestCase):
    """The day's central claim."""

    def test_the_note_level_rate_is_far_below_the_span_level_recall(self):
        self.assertLess(RESULT.clean_rate, RESULT.recall - 0.25)

    def test_roughly_a_third_of_notes_leak(self):
        self.assertGreater(RESULT.clean_rate, 0.60)
        self.assertLess(RESULT.clean_rate, 0.70)

    def test_the_independence_model_is_the_right_order_and_still_wrong(self):
        per_note = sum(len(n.spans) for n in NOTES) / len(NOTES)
        predicted = independence_prediction(RESULT.recall, per_note)
        self.assertAlmostEqual(predicted, RESULT.clean_rate, delta=0.05)
        self.assertNotAlmostEqual(predicted, RESULT.clean_rate, places=3)

    def test_misses_concentrate_in_some_notes(self):
        # If every miss landed in its own note, these would be equal.
        leaking = {note_id for note_id, _ in RESULT.misses}
        self.assertLess(len(leaking), RESULT.total_missed)

    def test_independence_prediction_is_the_closed_form(self):
        self.assertAlmostEqual(independence_prediction(0.9, 2.0), 0.81, places=9)
        self.assertAlmostEqual(independence_prediction(1.0, 9.0), 1.0, places=9)


class TestOverRedaction(unittest.TestCase):
    """The failure recall cannot see."""

    def test_the_eponym_gazetteer_destroys_diagnoses(self):
        damage = over_redaction(NOTES, detect)
        self.assertGreater(damage["phrases"], 50)
        self.assertGreater(damage["rate"], 0.5)

    def test_striking_them_off_destroys_none(self):
        damage = over_redaction(NOTES, detect_without_eponyms)
        self.assertEqual(damage["damaged"], 0)

    def test_and_that_is_the_whole_of_the_precision_loss(self):
        # Every false positive in the conservative run is an eponym.
        scored = evaluate(NOTES, detect_without_eponyms)
        self.assertAlmostEqual(scored.precision, 1.0, places=9)
        self.assertLess(RESULT.precision, 1.0)

    def test_but_it_costs_recall_and_clean_notes(self):
        # The trade, measured. Neither column is the safe one.
        scored = evaluate(NOTES, detect_without_eponyms)
        self.assertLess(scored.recall, RESULT.recall)
        self.assertLess(scored.clean_rate, RESULT.clean_rate)

    def test_the_eponyms_really_are_on_the_gazetteer(self):
        for surname in EPONYM_SURNAMES:
            if surname in GAZETTEER_SURNAMES:
                break
        else:
            self.fail("no eponym surname on the gazetteer - section 5 is vacuous")


class TestTheAggressiveRule(unittest.TestCase):
    def test_it_catches_the_bare_mrns(self):
        scored = evaluate(NOTES, detect_aggressive)
        self.assertGreater(scored.recall, RESULT.recall)
        self.assertGreater(scored.clean_rate, RESULT.clean_rate)

    def test_it_costs_nothing_on_this_corpus(self):
        scored = evaluate(NOTES, detect_aggressive)
        self.assertEqual(len(scored.false_positives),
                         len(RESULT.false_positives))

    def test_and_that_is_a_fact_about_the_corpus(self):
        # The reason it looks free: there is no six-to-eight-digit number
        # here that is not an MRN. Add one and the rule bills for it.
        text = "Accession 5512033 reported. Device serial 7742981 checked."
        spans = [s for s in detect_aggressive(text) if s.category == "MRN"]
        self.assertEqual(len(spans), 2)
        self.assertFalse([s for s in detect(text) if s.category == "MRN"])


class TestRedact(unittest.TestCase):
    def test_caught_identifiers_do_not_survive(self):
        for note in NOTES[:40]:
            predicted = detect(note.text)
            output = redact(note.text, predicted)
            for truth in note.spans:
                if covered(truth, predicted):
                    self.assertNotIn(truth.text, output)

    def test_missed_identifiers_do_survive(self):
        # The point of the exercise, stated as a test.
        leaked = 0
        for note in NOTES:
            predicted = detect(note.text)
            output = redact(note.text, predicted)
            for truth in note.spans:
                if not covered(truth, predicted) and truth.text in output:
                    leaked += 1
        self.assertGreater(leaked, 0)

    def test_no_spans_is_the_identity(self):
        self.assertEqual(redact("nothing here", []), "nothing here")

    def test_it_is_idempotent_on_its_own_output(self):
        # Placeholders must not themselves look like identifiers.
        once = redact(NOTES[0].text, detect(NOTES[0].text))
        twice = redact(once, detect(once))
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
