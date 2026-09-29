"""Day 15 tests - groundedness, and what it cannot mean.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from groundedness import (
    ANSWER_KINDS,
    HELD_OUT_KINDS,
    Answer,
    bump_numbers,
    by_kind,
    chunk_for,
    flip_polarity,
    is_supported,
    make_answers,
    make_guideline,
    make_held_out_answers,
    negated,
    overlap_groundedness,
    score_answer,
    section_chunks,
    separation_of,
    split_claims,
    strict_support,
    strip_negator,
)

GUIDELINE = make_guideline(seed=0)
CHUNKS = section_chunks(GUIDELINE)
ANSWERS = make_answers(GUIDELINE, CHUNKS)
SCORED = [score_answer(a, CHUNKS, GUIDELINE) for a in ANSWERS]
KINDS = by_kind(SCORED)


class TestTheHelpers(unittest.TestCase):
    def test_flip_polarity_is_its_own_inverse_on_a_plain_instruction(self):
        action = "offer a thiazide-like diuretic as first-line therapy"
        self.assertEqual(flip_polarity(flip_polarity(action)), action)

    def test_flipping_changes_the_polarity(self):
        action = "offer a supervised exercise programme"
        self.assertFalse(negated(action))
        self.assertTrue(negated(flip_polarity(action)))

    def test_and_flipping_a_negative_makes_it_positive(self):
        action = "do not offer routine dose escalation"
        self.assertTrue(negated(action))
        self.assertFalse(negated(flip_polarity(action)))

    def test_strip_negator_removes_only_a_leading_reversal(self):
        self.assertEqual(strip_negator("do not offer x"), "offer x")
        self.assertEqual(strip_negator("offer x"), "offer x")

    def test_bump_numbers_moves_every_number(self):
        self.assertEqual(bump_numbers("GFR below 30 and above 60"),
                         "GFR below 60 and above 120")

    def test_split_claims_splits_on_sentence_ends(self):
        claims = split_claims("One thing. Another thing; a third.")
        self.assertEqual(len(claims), 3)
        self.assertEqual([c.index for c in claims], [0, 1, 2])

    def test_an_empty_answer_has_no_claims(self):
        self.assertEqual(split_claims("   "), [])


class TestTheGroundTruth(unittest.TestCase):
    """Labelled from the generator's records, never from the checker."""

    def test_a_real_pairing_is_supported(self):
        recommendation = GUIDELINE.recommendations[0]
        self.assertTrue(is_supported(recommendation.population,
                                     recommendation.action,
                                     recommendation.section_path,
                                     GUIDELINE, CHUNKS))

    def test_an_invented_action_is_not(self):
        recommendation = GUIDELINE.recommendations[0]
        self.assertFalse(is_supported(recommendation.population,
                                      "offer intravenous magnesium",
                                      recommendation.section_path,
                                      GUIDELINE, CHUNKS))

    def test_an_unknown_section_is_not(self):
        recommendation = GUIDELINE.recommendations[0]
        self.assertFalse(is_supported(recommendation.population,
                                      recommendation.action,
                                      "99 Nowhere", GUIDELINE, CHUNKS))

    def test_every_faithful_answer_is_supported(self):
        for row in SCORED:
            if row["answer"].kind == "FAITHFUL":
                self.assertTrue(row["answer"].supported)

    def test_some_recombined_answers_are_genuinely_supported(self):
        # The eight eligibility clauses repeat, so recombining two
        # recommendations sometimes reproduces a pairing the document
        # contains. Calling those unsupported would have been wrong, and
        # an earlier version did.
        recombined = [r for r in SCORED if r["answer"].kind == "RECOMBINED"]
        supported = sum(1 for r in recombined if r["answer"].supported)
        self.assertGreater(supported, 0)
        self.assertLess(supported, len(recombined))

    def test_negated_and_invented_answers_never_are(self):
        for row in SCORED:
            if row["answer"].kind in ("NEGATED", "INVENTED", "MIXED"):
                self.assertFalse(row["answer"].supported)


class TestWordOverlapCannotSee(unittest.TestCase):
    """The day's central claim."""

    def test_a_negated_answer_scores_almost_the_same_as_a_faithful_one(self):
        self.assertGreater(KINDS["NEGATED"]["overlap"], 0.9)
        self.assertLess(KINDS["FAITHFUL"]["overlap"]
                        - KINDS["NEGATED"]["overlap"], 0.1)

    def test_a_changed_number_barely_moves_it(self):
        self.assertGreater(KINDS["NUMBER"]["overlap"], 0.85)

    def test_it_passes_every_negated_answer_at_a_usual_threshold(self):
        negated_rows = [r for r in SCORED if r["answer"].kind == "NEGATED"]
        passing = sum(1 for r in negated_rows if r["overlap_mean"] >= 0.75)
        self.assertEqual(passing, len(negated_rows))

    def test_even_a_perfect_threshold_lets_unsupported_answers_through(self):
        # Demanding that every word of the answer appear in the source
        # still passes them, because every word does.
        survivors = [r for r in SCORED if not r["answer"].supported
                     and r["overlap_min"] >= 1.0]
        self.assertGreater(len(survivors), 10)

    def test_faithful_answers_score_exactly_one(self):
        self.assertEqual(KINDS["FAITHFUL"]["overlap"], 1.0)

    def test_an_empty_claim_scores_zero(self):
        self.assertEqual(overlap_groundedness("", CHUNKS[0]), 0.0)

    def test_a_missing_chunk_scores_zero(self):
        self.assertEqual(overlap_groundedness("anything at all", None), 0.0)


class TestTheClaimChecks(unittest.TestCase):
    def test_it_agrees_with_the_ground_truth_everywhere(self):
        for row in SCORED:
            self.assertEqual(row["strict"], row["answer"].supported,
                             f"{row['answer'].kind}: {row['answer'].text[:60]}")

    def test_and_therefore_separates_perfectly(self):
        self.assertEqual(separation_of(SCORED, lambda r: float(r["strict"])),
                         1.0)

    def test_while_overlap_does_not(self):
        self.assertLess(separation_of(SCORED, lambda r: r["overlap_mean"]), 1.0)

    def test_polarity_is_checked_against_the_matching_recommendation(self):
        # Regression. The first version asked whether ANY recommendation
        # in the chunk had matching polarity, which a chunk containing
        # one "do not" always satisfies - it passed 45% of the flipped
        # claims.
        recommendation = next(r for r in GUIDELINE.recommendations
                              if not negated(r.action))
        chunk = chunk_for(CHUNKS, recommendation.section_path)
        claim = (f"{recommendation.population}, "
                 f"{flip_polarity(recommendation.action)}.")
        verdict = strict_support(claim, chunk, GUIDELINE)
        self.assertFalse(verdict.supported)
        self.assertEqual(verdict.reason, "polarity reversed")

    def test_recombination_is_checked_against_the_whole_chunk(self):
        # Regression. Narrowing the candidates by polarity before this
        # check discarded the recommendation the population came from,
        # so the check had nothing to compare and passed.
        pairs = [(a, b) for a in GUIDELINE.recommendations
                 for b in GUIDELINE.recommendations
                 if a.section_path == b.section_path and a.rec_id != b.rec_id
                 and a.population != b.population]
        self.assertTrue(pairs)
        first, second = pairs[0]
        chunk = chunk_for(CHUNKS, first.section_path)
        claim = f"{first.population}, {second.action}."
        if not is_supported(first.population, second.action,
                            first.section_path, GUIDELINE, CHUNKS):
            self.assertFalse(strict_support(claim, chunk, GUIDELINE).supported)

    def test_a_missing_section_is_refused(self):
        self.assertFalse(strict_support("anything", None, GUIDELINE).supported)

    def test_a_number_change_is_named_as_a_number(self):
        recommendation = next(r for r in GUIDELINE.recommendations
                              if any(ch.isdigit() for ch in r.population))
        chunk = chunk_for(CHUNKS, recommendation.section_path)
        claim = bump_numbers(f"{recommendation.population}, "
                             f"{recommendation.action}.")
        verdict = strict_support(claim, chunk, GUIDELINE)
        self.assertFalse(verdict.supported)
        self.assertTrue(verdict.reason.startswith("number not in source"))


class TestTheWorstClaimNotTheAverage(unittest.TestCase):
    def test_a_mixed_answer_has_one_good_claim_and_one_bad(self):
        mixed = next(r for r in SCORED if r["answer"].kind == "MIXED")
        self.assertEqual(len(mixed["claims"]), 2)
        self.assertTrue(mixed["verdicts"][0].supported)
        self.assertFalse(mixed["verdicts"][1].supported)

    def test_the_mean_hides_it_and_the_minimum_does_not(self):
        mixed = next(r for r in SCORED if r["answer"].kind == "MIXED")
        self.assertGreater(mixed["overlap_mean"], 0.4)
        self.assertLess(mixed["overlap_min"], 0.1)

    def test_an_answer_is_grounded_only_if_every_claim_is(self):
        for row in SCORED:
            if any(not v.supported for v in row["verdicts"]):
                self.assertFalse(row["strict"])


class TestWhatItStillCannotSee(unittest.TestCase):
    """The day's ending, and the reason the 1.000 above is not a score."""

    def setUp(self):
        self.held = [score_answer(a, CHUNKS, GUIDELINE)
                     for a in make_held_out_answers(GUIDELINE, CHUNKS)]

    def test_both_held_out_kinds_are_produced(self):
        self.assertEqual({r["answer"].kind for r in self.held},
                         set(HELD_OUT_KINDS))

    def test_they_are_all_marked_misleading(self):
        for row in self.held:
            self.assertTrue(row["answer"].misleading)

    def test_word_overlap_passes_every_one(self):
        for row in self.held:
            self.assertEqual(row["overlap_min"], 1.0)

    def test_and_so_does_the_strict_check(self):
        for row in self.held:
            self.assertTrue(row["strict"], row["answer"].text[:70])

    def test_dropping_the_scope_is_traceable_and_wrong(self):
        row = next(r for r in self.held
                   if r["answer"].kind == "SCOPE_DROPPED")
        self.assertTrue(row["strict"])
        recommendation = GUIDELINE.recommendation(row["answer"].rec_id)
        self.assertNotIn(recommendation.population, row["answer"].text)

    def test_ignoring_an_exception_is_traceable_and_wrong(self):
        row = next(r for r in self.held
                   if r["answer"].kind == "EXCEPTION_IGNORED")
        self.assertTrue(row["strict"])
        exception = next(e for e in GUIDELINE.exceptions
                         if e.rec_id == row["answer"].rec_id)
        self.assertNotEqual(exception.section_path,
                            row["answer"].cited_path)

    def test_traceable_and_correct_are_different_properties(self):
        # Stated as a test because it is the day's conclusion: every
        # answer here is supported by the line it cites, and none of
        # them should be given to a clinician as it stands.
        for row in self.held:
            self.assertTrue(row["answer"].supported)
            self.assertTrue(row["answer"].misleading)


class TestTheAnswerSet(unittest.TestCase):
    def test_every_kind_is_generated_for_every_recommendation(self):
        for kind in ANSWER_KINDS:
            rows = [a for a in ANSWERS if a.kind == kind]
            self.assertEqual(len(rows), len(GUIDELINE.recommendations))

    def test_a_miscited_answer_points_somewhere_else(self):
        for answer in ANSWERS:
            if answer.kind == "MISCITED":
                recommendation = GUIDELINE.recommendation(answer.rec_id)
                self.assertNotEqual(answer.cited_path,
                                    recommendation.section_path)

    def test_the_answers_are_deterministic(self):
        again = make_answers(GUIDELINE, CHUNKS)
        self.assertEqual([a.text for a in ANSWERS], [a.text for a in again])
        self.assertEqual([a.supported for a in ANSWERS],
                         [a.supported for a in again])

    def test_an_answer_carries_its_own_truth(self):
        answer = Answer("x", "FAITHFUL", "R1", "1 Scope and purpose", True)
        self.assertFalse(answer.misleading)


if __name__ == "__main__":
    unittest.main()
