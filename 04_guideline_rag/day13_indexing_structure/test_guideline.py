"""Day 13 tests - indexing a guideline with its structure.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from guideline import (
    ACTIONS_BY_SECTION,
    DISCLAIMER,
    OUTLINE,
    POPULATIONS,
    Chunk,
    actions_for,
    distance_between,
    exceptions_colocated,
    fixed_chunks,
    index_report,
    make_guideline,
    section_chunks,
)

GUIDELINE = make_guideline(seed=0)
SECTIONED = section_chunks(GUIDELINE)


class TestTheDocumentIsWhatItSaysItIs(unittest.TestCase):
    def test_it_says_it_is_synthetic_before_anything_else(self):
        self.assertTrue(GUIDELINE.text.startswith(DISCLAIMER))

    def test_every_recommendation_matches_the_text_at_its_offsets(self):
        for recommendation in GUIDELINE.recommendations:
            span = GUIDELINE.text[recommendation.start:recommendation.end]
            self.assertTrue(span.startswith(recommendation.rec_id + "."))
            self.assertIn(recommendation.population, span)
            self.assertIn(recommendation.action, span)

    def test_the_population_and_grade_offsets_are_right(self):
        for recommendation in GUIDELINE.recommendations:
            self.assertEqual(
                GUIDELINE.text[recommendation.population_start:
                               recommendation.population_end],
                recommendation.population)
            grade = GUIDELINE.text[recommendation.grade_start:
                                   recommendation.grade_end]
            self.assertTrue(grade.startswith("[Strength:"))
            self.assertIn(recommendation.strength, grade)
            self.assertIn(recommendation.evidence, grade)

    def test_the_grade_sits_after_the_action(self):
        for recommendation in GUIDELINE.recommendations:
            self.assertLess(recommendation.population_end,
                            recommendation.grade_start)

    def test_recommendation_ids_are_unique_and_sequential(self):
        ids = [r.rec_id for r in GUIDELINE.recommendations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [f"R{i + 1}" for i in range(len(ids))])

    def test_sections_tile_the_document_without_gaps(self):
        top = sorted((s for s in GUIDELINE.sections), key=lambda s: s.start)
        for earlier, later in zip(top, top[1:]):
            self.assertEqual(earlier.end, later.start)

    def test_every_recommendation_sits_in_the_section_it_claims(self):
        for recommendation in GUIDELINE.recommendations:
            section = GUIDELINE.section_at(recommendation.start)
            self.assertIsNotNone(section)
            self.assertEqual(section.path, recommendation.section_path)

    def test_the_actions_belong_to_their_section(self):
        # A diagnosis chapter that recommends a diuretic is not a
        # guideline, and a document that does not hang together cannot
        # teach anything about indexing one.
        for recommendation in GUIDELINE.recommendations:
            number = recommendation.section_path.split()[0]
            self.assertIn(recommendation.action, actions_for(number))

    def test_actions_for_takes_the_longest_matching_prefix(self):
        self.assertEqual(actions_for("3.2.1"), ACTIONS_BY_SECTION["3.2"])
        self.assertEqual(actions_for("3.1"), ACTIONS_BY_SECTION["3.1"])
        self.assertEqual(actions_for("2.2"), ACTIONS_BY_SECTION["2"])

    def test_the_document_is_deterministic(self):
        self.assertEqual(make_guideline(seed=0).text, GUIDELINE.text)

    def test_the_outline_and_the_sections_agree(self):
        self.assertEqual({s.number for s in GUIDELINE.sections},
                         {number for number, _t, _n in OUTLINE})

    def test_some_populations_differ_only_by_a_negation(self):
        # Day 12's problem, carried into retrieval: these share every
        # content word and mean opposite things.
        with_damage = "In adults with stage 2 disease and established end-organ damage"
        without = "In adults with stage 2 disease and no evidence of end-organ damage"
        self.assertIn(with_damage, POPULATIONS)
        self.assertIn(without, POPULATIONS)


class TestFixedChunking(unittest.TestCase):
    def test_the_chunks_reconstruct_the_document(self):
        chunks = fixed_chunks(GUIDELINE, size=400, overlap=0)
        self.assertEqual("".join(c.text for c in chunks), GUIDELINE.text)

    def test_offsets_match_the_text(self):
        for chunk in fixed_chunks(GUIDELINE, size=400, overlap=50):
            self.assertEqual(GUIDELINE.text[chunk.start:chunk.end], chunk.text)

    def test_overlap_makes_consecutive_chunks_share_text(self):
        chunks = fixed_chunks(GUIDELINE, size=400, overlap=50)
        self.assertEqual(chunks[0].end - chunks[1].start, 50)

    def test_nonsense_sizes_are_refused(self):
        for size, overlap in ((0, 0), (-1, 0), (100, 100), (100, 200),
                              (100, -1)):
            with self.assertRaises(ValueError):
                fixed_chunks(GUIDELINE, size=size, overlap=overlap)

    def test_it_severs_recommendations_at_small_sizes(self):
        report = index_report(GUIDELINE, fixed_chunks(GUIDELINE, 200, 25))
        self.assertGreater(report.severed_population, 5)
        self.assertLess(report.whole_rate, 0.6)

    def test_and_stops_severing_once_the_chunks_are_large(self):
        report = index_report(GUIDELINE, fixed_chunks(GUIDELINE, 1600, 200))
        self.assertEqual(report.severed_population, 0)
        self.assertEqual(report.whole_rate, 1.0)

    def test_but_large_chunks_lose_the_section_entirely(self):
        # The cost the intact-rate column cannot show: a chunk that size
        # spans four sections, so none of it is attributable to one.
        report = index_report(GUIDELINE, fixed_chunks(GUIDELINE, 1600, 200))
        self.assertEqual(report.attributable_rate, 0.0)

    def test_and_dilute_what_is_retrieved(self):
        small = index_report(GUIDELINE, fixed_chunks(GUIDELINE, 200, 25))
        large = index_report(GUIDELINE, fixed_chunks(GUIDELINE, 1600, 200))
        self.assertGreater(large.recommendations_per_chunk,
                           3 * small.recommendations_per_chunk)


class TestSectionChunking(unittest.TestCase):
    def test_every_chunk_carries_a_path(self):
        for chunk in SECTIONED:
            self.assertTrue(chunk.section_path)

    def test_every_chunk_lies_in_exactly_one_section(self):
        report = index_report(GUIDELINE, SECTIONED)
        self.assertEqual(report.attributable_rate, 1.0)

    def test_no_recommendation_is_severed(self):
        report = index_report(GUIDELINE, SECTIONED)
        self.assertEqual(report.whole_rate, 1.0)
        self.assertEqual(report.severed_population, 0)
        self.assertEqual(report.severed_grade, 0)

    def test_no_grade_is_orphaned(self):
        self.assertEqual(index_report(GUIDELINE, SECTIONED).orphan_grades, 0)

    def test_it_beats_every_fixed_size_on_both_axes_at_once(self):
        # The claim the table is making, as an assertion.
        structured = index_report(GUIDELINE, SECTIONED)
        for size in (200, 400, 800, 1600):
            fixed = index_report(GUIDELINE,
                                 fixed_chunks(GUIDELINE, size, size // 8))
            better_or_equal = (structured.whole_rate >= fixed.whole_rate
                               and structured.attributable_rate
                               >= fixed.attributable_rate)
            self.assertTrue(better_or_equal, f"lost to fixed {size}")
            self.assertTrue(structured.whole_rate > fixed.whole_rate
                            or structured.attributable_rate
                            > fixed.attributable_rate,
                            f"only tied fixed {size}")

    def test_a_long_section_is_split_and_keeps_its_path(self):
        chunks = section_chunks(GUIDELINE, max_size=120)
        self.assertGreater(len(chunks), len(SECTIONED))
        for chunk in chunks:
            self.assertTrue(chunk.section_path)

    def test_splitting_a_section_loses_nothing_from_the_text(self):
        chunks = section_chunks(GUIDELINE, max_size=120)
        for chunk in chunks:
            self.assertEqual(GUIDELINE.text[chunk.start:chunk.end], chunk.text)


class TestTheCrossReferences(unittest.TestCase):
    """The part chunking cannot fix."""

    def test_the_exceptions_target_real_recommendations(self):
        self.assertTrue(GUIDELINE.exceptions)
        for exception in GUIDELINE.exceptions:
            self.assertIsNotNone(GUIDELINE.recommendation(exception.rec_id))
            self.assertIn(exception.rec_id, exception.text)

    def test_they_target_recommendations_in_the_management_chapter(self):
        for exception in GUIDELINE.exceptions:
            recommendation = GUIDELINE.recommendation(exception.rec_id)
            self.assertTrue(recommendation.section_path.startswith("3"))
            self.assertTrue(exception.section_path.startswith("4"))

    def test_they_target_different_recommendations(self):
        targets = [e.rec_id for e in GUIDELINE.exceptions]
        self.assertEqual(len(targets), len(set(targets)))

    def test_no_chunking_puts_a_recommendation_with_its_exception(self):
        for chunks in (fixed_chunks(GUIDELINE, 400, 50),
                       fixed_chunks(GUIDELINE, 1600, 200),
                       SECTIONED):
            row = exceptions_colocated(GUIDELINE, chunks)
            self.assertEqual(row["together"], 0)
            self.assertEqual(row["apart"], len(GUIDELINE.exceptions))

    def test_and_the_reason_is_the_distance_not_the_chunker(self):
        for _rec_id, gap in distance_between(GUIDELINE):
            self.assertGreater(gap, 1500)

    def test_a_chunk_spanning_the_whole_document_would_hold_both(self):
        # Proof that the chunker is not the obstacle: the only chunking
        # that keeps them together is the one that gives up on chunking.
        whole = [Chunk(GUIDELINE.text, 0, len(GUIDELINE.text), "all")]
        row = exceptions_colocated(GUIDELINE, whole)
        self.assertEqual(row["together"], len(GUIDELINE.exceptions))


if __name__ == "__main__":
    unittest.main()
