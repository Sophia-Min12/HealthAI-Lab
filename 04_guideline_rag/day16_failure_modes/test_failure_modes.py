"""Day 16 tests - what the guards do not catch.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from failure_modes import (
    CURRENT_LABEL,
    PROBE_KINDS,
    SUPERSEDED_LABEL,
    SUPERSEDED_POPULATIONS,
    Probe,
    best_recommendation,
    chunk_for,
    evaluate_guards,
    is_harmful,
    make_library,
    make_probes,
    recommendations_in,
    run,
    split_claims,
    strict_support,
    tokenize,
)

LIBRARY = make_library(seed=0)
PROBES = make_probes(LIBRARY)
FLIPPED = make_library(seed=0, superseded_first=True)


class TestTheLibrary(unittest.TestCase):
    def test_both_editions_are_indexed(self):
        labels = {c.section_path.split(" 1")[0].split(" 2")[0].split(" 3")[0]
                  .split(" 4")[0].split(" 5")[0].strip()
                  for c in LIBRARY.chunks}
        self.assertIn(CURRENT_LABEL, labels)
        self.assertIn(SUPERSEDED_LABEL, labels)

    def test_they_are_the_same_size(self):
        current = [c for c in LIBRARY.chunks if not LIBRARY.is_stale(c.section_path)]
        stale = [c for c in LIBRARY.chunks if LIBRARY.is_stale(c.section_path)]
        self.assertEqual(len(current), len(stale))

    def test_the_editions_differ_only_in_their_thresholds(self):
        # Same recommendations, same actions, different eligibility
        # numbers. That is what a guideline revision usually is.
        current = LIBRARY.current.recommendations
        stale = LIBRARY.superseded.recommendations
        self.assertEqual([r.rec_id for r in current], [r.rec_id for r in stale])
        self.assertEqual([r.action for r in current], [r.action for r in stale])
        self.assertNotEqual([r.population for r in current],
                            [r.population for r in stale])

    def test_the_superseded_populations_are_a_parallel_set(self):
        self.assertEqual(len(SUPERSEDED_POPULATIONS), 8)

    def test_staleness_is_read_off_the_path(self):
        self.assertTrue(LIBRARY.is_stale(f"{SUPERSEDED_LABEL} 3.2.1 x"))
        self.assertFalse(LIBRARY.is_stale(f"{CURRENT_LABEL} 3.2.1 x"))

    def test_the_right_guideline_is_looked_up_for_a_path(self):
        self.assertIs(LIBRARY.guideline_for(f"{SUPERSEDED_LABEL} 1 x"),
                      LIBRARY.superseded)
        self.assertIs(LIBRARY.guideline_for(f"{CURRENT_LABEL} 1 x"),
                      LIBRARY.current)

    def test_insertion_order_changes_only_the_order(self):
        self.assertEqual(sorted(c.section_path for c in LIBRARY.chunks),
                         sorted(c.section_path for c in FLIPPED.chunks))


class TestTheGenerator(unittest.TestCase):
    def test_every_word_it_emits_is_in_the_chunk_it_cites(self):
        # It has no model in it and cannot hallucinate a word. Everything
        # this day measures survives that restriction.
        for probe in PROBES:
            answer = run(probe, LIBRARY)
            if answer.refused or not answer.text:
                continue
            chunk = chunk_for(LIBRARY.chunks, answer.cited_path)
            source = set(tokenize(chunk.text))
            for term in tokenize(answer.text):
                self.assertIn(term, source, answer.text[:60])

    def test_it_cites_a_real_section(self):
        for probe in PROBES:
            answer = run(probe, LIBRARY)
            if answer.refused:
                continue
            self.assertIsNotNone(chunk_for(LIBRARY.chunks, answer.cited_path))

    def test_it_names_a_recommendation_that_is_in_that_section(self):
        for probe in PROBES:
            answer = run(probe, LIBRARY)
            if answer.refused or answer.rec_id is None:
                continue
            chunk = chunk_for(LIBRARY.chunks, answer.cited_path)
            guideline = LIBRARY.guideline_for(answer.cited_path)
            ids = {r.rec_id for r in recommendations_in(chunk, guideline)}
            self.assertIn(answer.rec_id, ids)

    def test_a_chunk_with_no_recommendation_yields_nothing(self):
        empty = next(c for c in LIBRARY.chunks
                     if not recommendations_in(
                         c, LIBRARY.guideline_for(c.section_path)))
        self.assertIsNone(best_recommendation(
            empty, LIBRARY.guideline_for(empty.section_path), "anything"))


class TestTheGuards(unittest.TestCase):
    def test_with_no_guard_the_only_refusals_are_structural(self):
        # Not a guard: the top chunk held no recommendation, so there was
        # nothing to quote. An earlier version emitted a sentence of its
        # own here and a test caught it.
        report = evaluate_guards(PROBES, LIBRARY, 0.0, False, "none")
        self.assertGreater(report.refused, 0)
        for probe in PROBES:
            answer = run(probe, LIBRARY, 0.0, False)
            if answer.refused:
                self.assertEqual(answer.refusal_reason,
                                 "cited section states no recommendation")

    def test_the_score_guard_reduces_harm(self):
        none = evaluate_guards(PROBES, LIBRARY, 0.0, False, "none")
        refusal = evaluate_guards(PROBES, LIBRARY, 0.9, False, "refusal")
        self.assertLess(refusal.harmful, none.harmful)

    def test_the_groundedness_guard_catches_nothing_here(self):
        # Not a defect in Day 15's checker. This generator only quotes,
        # so it cannot produce an ungrounded answer, and a guard against
        # a failure the system cannot commit fires zero times. The check
        # earns its keep against a generator that paraphrases.
        none = evaluate_guards(PROBES, LIBRARY, 0.0, False, "none")
        grounded = evaluate_guards(PROBES, LIBRARY, 0.0, True, "grounded")
        self.assertEqual(grounded.delivered, none.delivered)
        self.assertEqual(grounded.harmful, none.harmful)

    def test_together_they_are_not_additive(self):
        # They fire on the same items: a question the document cannot
        # answer usually retrieves a chunk the answer cannot be grounded
        # in.
        refusal = evaluate_guards(PROBES, LIBRARY, 0.9, False, "refusal")
        both = evaluate_guards(PROBES, LIBRARY, 0.9, True, "both")
        self.assertEqual(both.harmful, refusal.harmful)

    def test_no_in_scope_question_is_refused_at_this_threshold(self):
        report = evaluate_guards(PROBES, LIBRARY, 0.9, True, "both")
        self.assertEqual(report.refused_in_scope, 0)

    def test_and_no_in_scope_answer_is_harmful(self):
        report = evaluate_guards(PROBES, LIBRARY, 0.9, True, "both")
        self.assertEqual(report.per_kind["IN_SCOPE"]["harmful"], 0)

    def test_the_counts_add_up(self):
        report = evaluate_guards(PROBES, LIBRARY, 0.9, True, "both")
        self.assertEqual(report.delivered + report.refused, len(PROBES))
        self.assertEqual(sum(row["n"] for row in report.per_kind.values()),
                         len(PROBES))

    def test_every_probe_kind_is_produced(self):
        self.assertEqual({p.kind for p in PROBES}, set(PROBE_KINDS))


class TestInsertionOrder(unittest.TestCase):
    """The day's finding, and nothing about the system changed."""

    def test_current_first_never_cites_the_withdrawn_edition(self):
        report = evaluate_guards(PROBES, LIBRARY, 0.0, False, "none")
        self.assertEqual(report.stale, 0)

    def test_superseded_first_cites_it_almost_every_time(self):
        report = evaluate_guards(make_probes(FLIPPED), FLIPPED, 0.0, False,
                                 "none")
        self.assertGreater(report.stale, 0.8 * report.delivered)

    def test_the_guards_do_not_move_it(self):
        loose = evaluate_guards(make_probes(FLIPPED), FLIPPED, 0.0, False, "a")
        strict = evaluate_guards(make_probes(FLIPPED), FLIPPED, 0.9, True, "b")
        self.assertGreater(strict.stale, 0.8 * strict.delivered)
        self.assertLess(strict.stale, loose.stale)  # only because fewer answer

    def test_a_stale_answer_is_perfectly_grounded(self):
        # This is why no text-level guard catches it. The chunk is real,
        # the quotation is exact, and the document is withdrawn.
        stale = [run(p, FLIPPED, 0.9, True) for p in make_probes(FLIPPED)]
        delivered = [a for a in stale
                     if not a.refused and FLIPPED.is_stale(a.cited_path)]
        self.assertTrue(delivered)
        for answer in delivered[:10]:
            chunk = chunk_for(FLIPPED.chunks, answer.cited_path)
            guideline = FLIPPED.guideline_for(answer.cited_path)
            for claim in split_claims(answer.text):
                self.assertTrue(
                    strict_support(claim.text, chunk, guideline).supported)

    def test_delivery_counts_are_identical_in_both_orders(self):
        # The only thing that changed is which edition was cited.
        for threshold, grounded in ((0.0, False), (0.9, True)):
            first = evaluate_guards(PROBES, LIBRARY, threshold, grounded, "a")
            second = evaluate_guards(make_probes(FLIPPED), FLIPPED, threshold,
                                     grounded, "b")
            self.assertEqual(first.delivered, second.delivered)
            self.assertEqual(first.refused, second.refused)


class TestHarm(unittest.TestCase):
    def test_a_refusal_is_never_harmful(self):
        refused = run(Probe("xyzzy plugh", "OUT_OF_SCOPE"), LIBRARY,
                      refuse_below=1e9)
        self.assertTrue(refused.refused)
        self.assertFalse(is_harmful(refused, LIBRARY))

    def test_any_out_of_scope_answer_is(self):
        for probe in PROBES:
            if probe.kind != "OUT_OF_SCOPE":
                continue
            answer = run(probe, LIBRARY)
            if not answer.refused:
                self.assertTrue(is_harmful(answer, LIBRARY))

    def test_a_stale_citation_is_harmful_whatever_was_asked(self):
        # The first version only counted staleness for questions written
        # to provoke it, which measured the questions and not the index.
        answers = [run(p, FLIPPED) for p in make_probes(FLIPPED)]
        stale = [a for a in answers
                 if not a.refused and FLIPPED.is_stale(a.cited_path)]
        self.assertTrue(any(a.question.kind == "IN_SCOPE" for a in stale))
        for answer in stale:
            self.assertTrue(is_harmful(answer, FLIPPED))

    def test_an_exception_answer_that_misses_section_4_is_harmful(self):
        for probe in PROBES:
            if probe.kind != "EXCEPTION":
                continue
            answer = run(probe, LIBRARY, 0.9, True)
            self.assertFalse(answer.refused)
            self.assertNotIn("4.", answer.cited_path)
            self.assertTrue(is_harmful(answer, LIBRARY))


if __name__ == "__main__":
    unittest.main()
