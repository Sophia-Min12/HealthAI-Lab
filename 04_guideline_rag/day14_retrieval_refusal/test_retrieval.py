"""Day 14 tests - retrieval, and the decision to refuse.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import subprocess
import sys
import unittest

from retrieval import (
    CRITERIA,
    OUT_OF_SCOPE,
    BM25,
    Chunk,
    Question,
    apply_threshold,
    by_style,
    citation,
    criterion_value,
    make_guideline,
    make_questions,
    retrieve_all,
    section_chunks,
    separation,
    tokenize,
)

GUIDELINE = make_guideline(seed=0)
CHUNKS = section_chunks(GUIDELINE)
INDEX = BM25(CHUNKS)
QUESTIONS = make_questions(GUIDELINE, INDEX)
RETRIEVALS = retrieve_all(INDEX, QUESTIONS, k=5)
ANSWERABLE = [r for r in RETRIEVALS if r.question.answerable]
UNANSWERABLE = [r for r in RETRIEVALS if not r.question.answerable]


class TestTokenizing(unittest.TestCase):
    def test_it_lowercases_and_splits_on_punctuation(self):
        self.assertEqual(tokenize("Offer an ACE-inhibitor, daily.",
                                  drop_stopwords=False),
                         ["offer", "an", "ace", "inhibitor", "daily"])

    def test_stopwords_go_by_default(self):
        self.assertNotIn("the", tokenize("the treatment of the patient"))

    def test_numbers_survive(self):
        self.assertIn("30", tokenize("GFR below 30 mL/min"))


class TestBM25(unittest.TestCase):
    def test_a_term_in_every_chunk_carries_almost_no_weight(self):
        # IDF is the whole reason a common word cannot dominate a score.
        common = max(INDEX.idf, key=lambda t: sum(
            1 for d in INDEX.documents if t in d))
        rare = min(INDEX.idf, key=lambda t: sum(
            1 for d in INDEX.documents if t in d))
        self.assertLess(INDEX.idf[common], INDEX.idf[rare])

    def test_idf_is_never_negative(self):
        # The +1 inside the log is what guarantees it, and without it a
        # term in more than half the chunks would subtract from a score.
        for value in INDEX.idf.values():
            self.assertGreaterEqual(value, 0.0)

    def test_a_query_of_unknown_terms_scores_zero_everywhere(self):
        ranked = INDEX.rank("xyzzy plugh frobnicate", k=5)
        for _chunk, score in ranked:
            self.assertEqual(score, 0.0)

    def test_term_frequency_saturates(self):
        # Four occurrences in the DOCUMENT must not score four times one
        # occurrence: that is what k1 is for. Both chunks are the same
        # length, so b changes nothing between them.
        index = BM25([Chunk("disease filler filler filler", 0, 28),
                      Chunk("disease disease disease disease", 0, 31)])
        single = index.score(["disease"], 0)
        repeated = index.score(["disease"], 1)
        self.assertGreater(repeated, single)
        self.assertLess(repeated, 4 * single)

    def test_a_longer_document_is_penalised(self):
        # And that is what b is for. Same single occurrence, more padding.
        index = BM25([Chunk("disease filler", 0, 14),
                      Chunk("disease " + "filler " * 40, 0, 288)])
        self.assertGreater(index.score(["disease"], 0),
                           index.score(["disease"], 1))

    def test_ranking_is_ordered_and_capped(self):
        ranked = INDEX.rank("first-line therapy", k=3)
        self.assertEqual(len(ranked), 3)
        scores = [score for _chunk, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_nonsense_parameters_are_refused(self):
        for k1, b in ((-1.0, 0.75), (1.5, -0.1), (1.5, 1.1)):
            with self.assertRaises(ValueError):
                BM25(CHUNKS, k1=k1, b=b)

    def test_it_always_returns_something(self):
        # The day's premise: a ranker has no "nothing" option.
        for _label, text in OUT_OF_SCOPE:
            self.assertEqual(len(INDEX.rank(text, k=3)), 3)


class TestTheQuestions(unittest.TestCase):
    def test_they_are_deterministic_across_processes(self):
        # Regression. The first version deduped the action's terms with
        # set(), whose iteration order changes between processes, so the
        # short questions - and every number in the demo - moved from
        # run to run under the default hash randomisation.
        script = (
            "import retrieval as r;"
            "g = r.make_guideline(seed=0);"
            "i = r.BM25(r.section_chunks(g));"
            "print([q.text for q in r.make_questions(g, i)])"
        )
        runs = {subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True,
                               cwd=__file__.rsplit("test_retrieval.py")[0] or ".",
                               check=True).stdout
                for _ in range(3)}
        self.assertEqual(len(runs), 1)

    def test_each_recommendation_gets_a_short_and_a_long_form(self):
        for recommendation in GUIDELINE.recommendations:
            forms = {q.style for q in QUESTIONS
                     if q.rec_id == recommendation.rec_id}
            self.assertEqual(forms, {"short", "long"})

    def test_the_long_form_really_is_longer(self):
        for recommendation in GUIDELINE.recommendations:
            pair = {q.style: q for q in QUESTIONS
                    if q.rec_id == recommendation.rec_id}
            self.assertGreater(len(tokenize(pair["long"].text)),
                               len(tokenize(pair["short"].text)))

    def test_short_questions_carry_content_words(self):
        # An earlier version took the first three tokens of the action,
        # which produced "do not offer?" - a question with nothing in it.
        for question in QUESTIONS:
            if question.style != "short" or not question.answerable:
                continue
            self.assertTrue(set(tokenize(question.text)) & set(INDEX.idf),
                            question.text)

    def test_the_out_of_scope_questions_are_out_of_scope(self):
        self.assertEqual(len(UNANSWERABLE), len(OUT_OF_SCOPE))
        for retrieval in UNANSWERABLE:
            self.assertIsNone(retrieval.question.rec_id)


class TestRetrievalWorks(unittest.TestCase):
    def test_hit_at_1_is_high(self):
        hits = sum(r.hits(GUIDELINE) for r in ANSWERABLE)
        self.assertGreater(hits / len(ANSWERABLE), 0.85)

    def test_and_hit_at_k_is_monotone(self):
        rates = [sum(r.hits_at(GUIDELINE, k) for r in ANSWERABLE)
                 for k in (1, 2, 3, 5)]
        for earlier, later in zip(rates, rates[1:]):
            self.assertLessEqual(earlier, later)

    def test_an_unanswerable_question_never_hits(self):
        for retrieval in UNANSWERABLE:
            self.assertFalse(retrieval.hits(GUIDELINE))
            self.assertFalse(retrieval.hits_at(GUIDELINE, 5))


class TestTheScoreIsALength(unittest.TestCase):
    """The day's first finding."""

    def test_the_raw_score_runs_far_higher_on_long_questions(self):
        styles = by_style(RETRIEVALS, "top_score")
        self.assertGreater(styles["long"], 2 * styles["short"])

    def test_the_margin_does_not_fix_it(self):
        # The criterion that ought to be immune to query length, and is
        # not. This file predicted otherwise before the table was run.
        styles = by_style(RETRIEVALS, "margin")
        self.assertGreater(styles["long"], 1.5 * styles["short"])

    def test_nor_does_it_separate_better_than_the_raw_score(self):
        self.assertLess(separation(RETRIEVALS, "margin"),
                        separation(RETRIEVALS, "top_score"))

    def test_dividing_by_query_length_flattens_it_and_overshoots(self):
        styles = by_style(RETRIEVALS, "score_per_term")
        self.assertGreater(styles["short"], styles["long"])

    def test_and_separates_best_of_the_three(self):
        best = max(CRITERIA, key=lambda c: separation(RETRIEVALS, c))
        self.assertEqual(best, "score_per_term")

    def test_an_unanswerable_long_question_outscores_an_answerable_short_one(self):
        worst = max(UNANSWERABLE, key=lambda r: r.top_score)
        shorts = [r for r in ANSWERABLE
                  if r.question.style == "short" and r.hits(GUIDELINE)]
        self.assertTrue(any(r.top_score < worst.top_score for r in shorts))

    def test_every_criterion_beats_a_coin(self):
        for criterion in CRITERIA:
            self.assertGreater(separation(RETRIEVALS, criterion), 0.5)


class TestRefusal(unittest.TestCase):
    def test_a_zero_threshold_answers_everything(self):
        report = apply_threshold(RETRIEVALS, GUIDELINE, 0.0, "score_per_term")
        self.assertEqual(report.refused_answerable, 0)
        self.assertEqual(report.answer_rate_on_unanswerable, 1.0)

    def test_a_huge_threshold_refuses_everything(self):
        report = apply_threshold(RETRIEVALS, GUIDELINE, 1e9, "score_per_term")
        self.assertEqual(report.refusal_rate_on_answerable, 1.0)
        self.assertEqual(report.answer_rate_on_unanswerable, 0.0)

    def test_refusal_on_answerable_only_grows(self):
        rates = [apply_threshold(RETRIEVALS, GUIDELINE, t, "score_per_term")
                 .refusal_rate_on_answerable for t in (0.0, 0.5, 1.0, 2.0, 3.0)]
        for earlier, later in zip(rates, rates[1:]):
            self.assertLessEqual(earlier, later)

    def test_and_answers_on_unanswerable_only_shrink(self):
        rates = [apply_threshold(RETRIEVALS, GUIDELINE, t, "score_per_term")
                 .answer_rate_on_unanswerable for t in (0.0, 0.5, 1.0, 2.0)]
        for earlier, later in zip(rates, rates[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_no_threshold_makes_both_errors_zero(self):
        # Searched over every value any question actually produces, not
        # over a grid - so this is a statement about the data.
        candidates = sorted({criterion_value(r, "score_per_term")
                             for r in RETRIEVALS})
        for threshold in candidates:
            report = apply_threshold(RETRIEVALS, GUIDELINE, threshold,
                                     "score_per_term")
            self.assertGreater(report.refusal_rate_on_answerable
                               + report.answer_rate_on_unanswerable, 0.0)

    def test_the_counts_always_add_up(self):
        report = apply_threshold(RETRIEVALS, GUIDELINE, 1.0, "score_per_term")
        self.assertEqual(report.answered_answerable + report.refused_answerable,
                         len(ANSWERABLE))
        self.assertEqual(report.answered_unanswerable
                         + report.refused_unanswerable, len(UNANSWERABLE))
        self.assertLessEqual(report.correct_answers, report.answered_answerable)


class TestCitations(unittest.TestCase):
    def test_a_citation_names_a_section_and_its_recommendations(self):
        for retrieval in ANSWERABLE[:10]:
            text = citation(retrieval, GUIDELINE)
            self.assertIn("[", text)
            self.assertTrue(text.split(" [")[0].strip())

    def test_it_names_the_recommendation_that_answers_the_question(self):
        for retrieval in ANSWERABLE:
            if retrieval.hits(GUIDELINE):
                self.assertIn(retrieval.question.rec_id,
                              citation(retrieval, GUIDELINE))

    def test_with_nothing_retrieved_there_is_no_source(self):
        empty = type(RETRIEVALS[0])(Question("x", None, "short"), (), ())
        self.assertEqual(citation(empty, GUIDELINE), "no source")


if __name__ == "__main__":
    unittest.main()
