"""Day 17 tests - the whole pipeline.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from capstone import (
    ACTIONABLE,
    CASE_ASSERTIONS,
    CASE_FRAMES,
    HELD_OUT_CASE_FRAMES,
    OUT_OF_SCOPE_FINDINGS,
    POPULATIONS,
    SUPERSEDED_POPULATIONS,
    covered,
    detect_unicode,
    finding_of,
    main,
    make_cases,
    make_library,
    redact,
    run_all,
    run_case,
)

LIBRARY = make_library(seed=0)
CASES = make_cases(LIBRARY, n=300, seed=0)
OUTCOMES, REPORT = run_all(CASES, LIBRARY)


class TestTheCases(unittest.TestCase):
    def test_every_phi_span_matches_the_note(self):
        for case in CASES:
            for span in case.phi:
                self.assertEqual(case.text[span.start:span.end], span.text)

    def test_the_finding_occurs_exactly_once(self):
        # It is located by searching, which is only safe because of this.
        # Day 10's generator went out of its way to avoid searching and
        # had good reason to.
        for case in CASES:
            self.assertEqual(case.text.count(case.finding), 1, case.finding)

    def test_the_finding_offsets_are_right(self):
        for case in CASES:
            self.assertEqual(
                case.text[case.finding_start:case.finding_end], case.finding)

    def test_the_finding_is_not_an_identifier(self):
        # If redaction ate it, nothing downstream could recover.
        for case in CASES:
            for span in case.phi:
                self.assertFalse(span.start < case.finding_end
                                 and case.finding_start < span.end)

    def test_both_frame_sets_cover_every_assertion(self):
        self.assertEqual(set(CASE_FRAMES), set(CASE_ASSERTIONS))
        self.assertEqual(set(HELD_OUT_CASE_FRAMES), set(CASE_ASSERTIONS))

    def test_the_two_frame_sets_share_no_phrasing(self):
        self.assertFalse(set(CASE_FRAMES.values())
                         & set(HELD_OUT_CASE_FRAMES.values()))

    def test_both_kinds_of_case_are_generated(self):
        self.assertGreater(sum(c.held_out for c in CASES), 50)
        self.assertGreater(sum(not c.held_out for c in CASES), 50)
        self.assertGreater(sum(not c.in_guideline for c in CASES), 20)

    def test_out_of_scope_findings_are_out_of_scope(self):
        for case in CASES:
            if not case.in_guideline:
                self.assertIn(case.finding.replace(case.finding[0],
                                                   case.finding[0].lower(), 1),
                              OUT_OF_SCOPE_FINDINGS)
                self.assertEqual(case.population, "")

    def test_actionability_follows_the_assertion(self):
        for case in CASES:
            self.assertEqual(case.actionable, case.assertion in ACTIONABLE)

    def test_finding_of_strips_the_guideline_preamble(self):
        self.assertEqual(finding_of("In adults aged 75 and over"),
                         "aged 75 and over")
        self.assertEqual(finding_of("aged 75 and over"), "aged 75 and over")

    def test_the_cases_are_deterministic(self):
        again = make_cases(LIBRARY, n=300, seed=0)
        self.assertEqual([c.text for c in CASES], [c.text for c in again])

    def test_turning_both_knobs_off_makes_every_case_easy(self):
        # And that is exactly why they default to something else: with
        # no held-out phrasings and nothing out of scope, three of the
        # five stages report 1.000 and the table says nothing.
        easy = make_cases(LIBRARY, n=100, seed=0, held_out_rate=0.0,
                          out_of_scope_rate=0.0)
        _outcomes, report = run_all(easy, LIBRARY)
        self.assertEqual(report.rate("assertion"), 1.0)
        self.assertEqual(report.rate("retrieval"), 1.0)


class TestTheStages(unittest.TestCase):
    def test_the_pipeline_works_on_the_redacted_text(self):
        # The only order it is allowed to run in, and it means a
        # de-identification bug is also a retrieval bug.
        for outcome in OUTCOMES[:50]:
            for span in outcome.case.phi:
                if covered(span, detect_unicode(outcome.case.text)):
                    self.assertNotIn(span.text, outcome.redacted)

    def test_redaction_never_eats_the_finding(self):
        for outcome in OUTCOMES:
            self.assertNotEqual(outcome.predicted_assertion, "LOST")

    def test_every_stage_is_between_a_half_and_one(self):
        for stage in ("deid", "assertion", "query", "retrieval", "verified"):
            self.assertGreater(REPORT.rate(stage), 0.5, stage)
            self.assertLessEqual(REPORT.rate(stage), 1.0, stage)

    def test_the_de_identification_rate_matches_day_10s_finding(self):
        # A note is not partly safe, and about a quarter of them leak.
        self.assertGreater(REPORT.rate("deid"), 0.65)
        self.assertLess(REPORT.rate("deid"), 0.85)

    def test_the_assertion_stage_sits_between_day_12s_two_numbers(self):
        # Half the notes use phrasings the rules were built for (1.000)
        # and half use phrasings they were not (0.730).
        self.assertGreater(REPORT.rate("assertion"), 0.55)
        self.assertLess(REPORT.rate("assertion"), 0.95)

    def test_every_assertion_error_defaults_to_present(self):
        # PRESENT is what the classifier returns when no rule fires, and
        # on unfamiliar phrasing no rule fires. Day 12 measured this.
        for outcome in OUTCOMES:
            if not outcome.assertion_ok:
                self.assertEqual(outcome.predicted_assertion, "PRESENT")

    def test_a_non_actionable_case_asks_nothing(self):
        for outcome in OUTCOMES:
            if outcome.predicted_assertion not in ACTIONABLE:
                self.assertFalse(outcome.queried)
                self.assertIsNone(outcome.answer)


class TestTheCompounding(unittest.TestCase):
    """The number the capstone exists to produce."""

    def test_end_to_end_is_far_below_every_stage(self):
        for stage in ("deid", "assertion", "query", "retrieval", "verified"):
            self.assertLess(REPORT.rate("end_to_end"),
                            REPORT.rate(stage), stage)

    def test_it_is_below_the_worst_stage_too(self):
        worst = min(REPORT.rate(s) for s in
                    ("deid", "assertion", "query", "retrieval", "verified"))
        self.assertLess(REPORT.rate("end_to_end"), worst)

    def test_the_independence_model_understates_it(self):
        # The stages are correlated: a misread assertion usually takes
        # the query decision down with it, so the errors overlap and
        # fewer notes are affected than independence predicts.
        self.assertLess(REPORT.product, REPORT.rate("end_to_end"))

    def test_but_it_is_the_right_order_of_magnitude(self):
        self.assertAlmostEqual(REPORT.product, REPORT.rate("end_to_end"),
                               delta=0.2)

    def test_end_to_end_requires_every_stage(self):
        for outcome in OUTCOMES:
            if outcome.end_to_end:
                self.assertTrue(outcome.deid_ok)
                self.assertTrue(outcome.assertion_ok)
                self.assertTrue(outcome.query_ok)
                self.assertTrue(outcome.retrieval_ok)
                self.assertTrue(outcome.verified_ok)

    def test_the_counts_are_consistent(self):
        self.assertEqual(REPORT.n, len(CASES))
        self.assertEqual(REPORT.end_to_end,
                         sum(1 for o in OUTCOMES if o.end_to_end))


class TestTheWithdrawnEdition(unittest.TestCase):
    def test_the_correct_index_order_still_cites_it_sometimes(self):
        # Not a tie broken badly: the 2019 edition is the better lexical
        # match for a note recording the 2019 threshold.
        stale = [o for o in OUTCOMES
                 if o.answer is not None and not o.answer.refused
                 and LIBRARY.is_stale(o.answer.cited_path)]
        self.assertTrue(stale)
        for outcome in stale:
            self.assertIn("15", outcome.case.finding)

    def test_flipping_the_index_order_moves_answers_onto_it(self):
        flipped = make_library(seed=0, superseded_first=True)
        cases = make_cases(flipped, n=300, seed=0)
        outcomes, _report = run_all(cases, flipped)
        answered = [o for o in outcomes
                    if o.answer is not None and not o.answer.refused]
        stale = [o for o in answered if flipped.is_stale(o.answer.cited_path)]
        self.assertGreater(len(stale), 0.3 * len(answered))

    def test_and_only_where_the_revision_changed_nothing(self):
        # The sharper version of Day 16's finding. Where the note names a
        # threshold the revision moved, the current edition is the better
        # lexical match and wins on merit. Where the criterion is word
        # for word the same in both editions, there is nothing to choose
        # between them and insertion order decides.
        shared = {finding_of(p).lower()
                  for p in set(POPULATIONS) & set(SUPERSEDED_POPULATIONS)}
        self.assertTrue(shared)
        flipped = make_library(seed=0, superseded_first=True)
        cases = make_cases(flipped, n=300, seed=0)
        outcomes, _report = run_all(cases, flipped)
        for outcome in outcomes:
            answer = outcome.answer
            if answer is None or answer.refused:
                continue
            if not flipped.is_stale(answer.cited_path):
                continue
            self.assertTrue(
                outcome.case.finding.lower() in shared
                or not outcome.case.in_guideline,
                outcome.case.finding)

    def test_and_those_answers_are_still_verified(self):
        flipped = make_library(seed=0, superseded_first=True)
        cases = make_cases(flipped, n=100, seed=0)
        outcomes, _report = run_all(cases, flipped)
        stale = [o for o in outcomes
                 if o.answer is not None and not o.answer.refused
                 and flipped.is_stale(o.answer.cited_path)]
        self.assertTrue(stale)
        # Grounded, quoted correctly, and withdrawn.
        self.assertTrue(any(o.verified_ok for o in stale))


class TestTheCLI(unittest.TestCase):
    def test_every_command_runs(self):
        for argv in (["capstone.py", "note"],
                     ["capstone.py", "deid", "Patient: Margaret Whitfield"],
                     ["capstone.py", "ask", "newly diagnosed stage 1 disease"],
                     ["capstone.py", "writeup"]):
            self.assertEqual(main(argv), 0, argv[1])

    def test_an_unknown_command_is_refused(self):
        self.assertEqual(main(["capstone.py", "frobnicate"]), 2)

    def test_deid_removes_what_it_reports(self):
        case = CASES[0]
        output = redact(case.text, detect_unicode(case.text))
        self.assertIn("[NAME]", output)
        self.assertNotIn("Patient: " + case.text.split("Patient: ")[1][:5],
                         output)


if __name__ == "__main__":
    unittest.main()
