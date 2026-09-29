"""Day 13 - Indexing a clinical guideline with its section structure.

A guideline is not prose. It is a numbered tree of sections in which a
recommendation carries three things that are written in different
places: **what to do**, **who to do it to**, and **how sure anyone is**.
Chunk it as though it were prose and those three come apart.

The condition here is invented. ``Synthetic Cardiometabolic Syndrome``
does not exist, and neither do these recommendations - the drug classes
are real, the indication is fiction, and nothing in this file should be
read as clinical guidance. What is real is the *shape*: numbered
sections, eligibility clauses, strength and evidence grades, and a
separate chapter of exceptions that contradicts the general advice for
particular patients.

Three things measured here rather than asserted:

* **Fixed-size chunking severs recommendations from their eligibility
  clause and their grade.** "Offer an ACE inhibitor" and "in adults
  without renal impairment" are different instructions, and a retriever
  that returns the first without the second has changed the guidance.

* **Structure-aware chunking fixes that completely**, which is the easy
  half of the day.

* **And it cannot fix cross-references at all.** The general
  recommendation lives in section 3 and the exception that reverses it
  for a subpopulation lives in section 4. No chunking of a tree can put
  them in the same chunk, because the document itself separates them.
  This is not a tuning problem; it is the structure of the genre.

NOT A MEDICAL DEVICE. The condition, the recommendations and the
evidence grades in this file are all invented by it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

DISCLAIMER = (
    "SYNTHETIC DOCUMENT. 'Synthetic Cardiometabolic Syndrome' is not a "
    "real condition and none of the recommendations below is real "
    "clinical guidance. This document exists to be indexed."
)


@dataclass(frozen=True)
class Recommendation:
    """One numbered recommendation, and its three separable parts."""

    rec_id: str
    section_path: str
    population: str
    action: str
    strength: str
    evidence: str
    start: int
    end: int
    #: Character range of the eligibility clause alone.
    population_start: int = 0
    population_end: int = 0
    #: Character range of the ``[Strength: ... Evidence: ...]`` tag.
    grade_start: int = 0
    grade_end: int = 0

    @property
    def text_range(self) -> tuple:
        return (self.start, self.end)


@dataclass(frozen=True)
class Exception_:
    """A statement elsewhere in the document that modifies a recommendation.

    Named with a trailing underscore because ``Exception`` is taken, and
    the collision is worth the accuracy: in a guideline this is exactly
    what these are called.
    """

    rec_id: str
    section_path: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    start: int
    end: int

    @property
    def path(self) -> str:
        return f"{self.number} {self.title}"

    @property
    def depth(self) -> int:
        return self.number.count(".") + 1


@dataclass
class Guideline:
    text: str
    sections: tuple = ()
    recommendations: tuple = ()
    exceptions: tuple = ()

    def section_at(self, offset: int):
        """The deepest section containing ``offset``."""
        best = None
        for section in self.sections:
            if section.start <= offset < section.end:
                if best is None or section.depth > best.depth:
                    best = section
        return best

    def recommendation(self, rec_id: str):
        for recommendation in self.recommendations:
            if recommendation.rec_id == rec_id:
                return recommendation
        return None


# --- the document -----------------------------------------------------------

#: Eligibility clauses. Several contain a negation, which is not
#: decoration: Day 12 spent a day on the fact that "without renal
#: impairment" and "with renal impairment" share every content word, and
#: a retriever that scores on word overlap cannot tell them apart.
POPULATIONS = (
    "In adults aged 18 and over with newly diagnosed stage 1 disease",
    "In adults with stage 2 disease and no evidence of end-organ damage",
    "In adults with stage 2 disease and established end-organ damage",
    "In adults aged 75 and over",
    "In adults with an estimated GFR above 60 mL/min",
    "In adults with an estimated GFR below 30 mL/min",
    "In adults who have not tolerated first-line therapy",
    "In adults with concurrent stage 3 disease",
)

#: Actions by the section they belong in. Keyed by section number
#: prefix, longest match wins. A guideline whose diagnosis chapter
#: recommends a diuretic is not a guideline, and a document that does
#: not hang together cannot teach anything about indexing one.
ACTIONS_BY_SECTION = {
    "2": (
        "measure serum electrolytes before starting treatment",
        "offer ambulatory monitoring to confirm the diagnosis",
        "do not routinely offer screening in asymptomatic people",
        "record an estimated GFR at the point of diagnosis",
    ),
    "3.1": (
        "offer structured lifestyle support",
        "offer a supervised exercise programme",
        "consider referral to a structured education programme",
    ),
    "3.2": (
        "offer a thiazide-like diuretic as first-line therapy",
        "offer an ACE inhibitor as first-line therapy",
        "consider a calcium channel blocker in addition to first-line therapy",
        "do not offer combination therapy without specialist review",
    ),
    "4": (
        "seek specialist advice before starting pharmacological therapy",
        "consider referral to a specialist service",
        "do not offer routine dose escalation",
    ),
    "5": (
        "review treatment at 4 weeks and then at 6 months",
        "offer annual review of treatment burden",
        "measure serum electrolytes at each review",
    ),
}


def actions_for(number: str) -> tuple:
    """The action pool for a section number, by longest matching prefix."""
    best = ""
    for prefix in ACTIONS_BY_SECTION:
        if number.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return ACTIONS_BY_SECTION[best] if best else ACTIONS_BY_SECTION["5"]

STRENGTHS = ("strong", "strong", "conditional", "conditional")
EVIDENCE = ("A", "B", "B", "C", "C", "D")

#: The section tree. Sections 3 and 4 are the pair the day turns on: the
#: general pharmacological advice lives in 3, and 4 is where it is
#: reversed for particular patients.
OUTLINE = (
    ("1", "Scope and purpose", 0),
    ("2", "Diagnosis", 0),
    ("2.1", "Initial assessment", 2),
    ("2.2", "Investigations", 2),
    ("3", "Management", 0),
    ("3.1", "Non-pharmacological management", 2),
    ("3.2", "Pharmacological management", 0),
    ("3.2.1", "First-line therapy", 3),
    ("3.2.2", "Second-line therapy", 3),
    ("4", "Special populations", 0),
    ("4.1", "Pregnancy", 2),
    ("4.2", "Renal impairment", 2),
    ("4.3", "Frailty and older people", 2),
    ("5", "Monitoring and review", 2),
)

FILLER = (
    "The evidence base for this section is drawn from the synthetic "
    "literature review described in appendix A.",
    "The committee noted considerable variation in current practice and "
    "agreed that a recommendation was warranted.",
    "Health economic modelling was not undertaken for this question.",
    "This section should be read alongside the accompanying patient "
    "decision aid.",
    "Where the evidence was judged insufficient, the committee has made "
    "a research recommendation instead.",
)

EXCEPTION_TEMPLATES = (
    "Recommendation {rec_id} does not apply in this population; "
    "withhold pharmacological therapy and seek specialist advice.",
    "Do not follow recommendation {rec_id} in this group. The committee "
    "judged the risk of harm to outweigh the expected benefit.",
    "Recommendation {rec_id} should be modified in this population: "
    "halve the starting dose and review at 2 weeks.",
)


class _Builder:
    """Assembles the document while recording where everything landed."""

    def __init__(self) -> None:
        self._parts: list = []
        self.length = 0

    def add(self, text: str) -> int:
        """Append ``text``; return the offset it starts at."""
        start = self.length
        self._parts.append(text)
        self.length += len(text)
        return start

    def build(self) -> str:
        return "".join(self._parts)


def make_guideline(seed: int = 0, scale: int = 1) -> Guideline:
    """Build the synthetic guideline and record its structure exactly.

    Offsets are recorded as the document is written. Recovering them
    afterwards by searching would be the same mistake Day 10 avoided,
    and here it would be worse: recommendation text repeats.
    """
    rng = np.random.default_rng(seed)
    builder = _Builder()
    sections: list = []
    recommendations: list = []
    exceptions: list = []
    section_starts: dict = {}

    builder.add(DISCLAIMER + "\n\n")
    builder.add("Synthetic Guideline SG-1: Management of Synthetic "
                "Cardiometabolic Syndrome\n\n")

    counter = 1
    for number, title, rec_count in OUTLINE:
        start = builder.add(f"{number} {title}\n")
        section_starts[number] = start
        for _ in range(max(1, 3 - len(number))):
            builder.add(str(rng.choice(FILLER)) + "\n")

        for _ in range(rec_count * scale):
            rec_id = f"R{counter}"
            counter += 1
            population = str(rng.choice(POPULATIONS))
            action = str(rng.choice(actions_for(number)))
            strength = str(rng.choice(STRENGTHS))
            evidence = str(rng.choice(EVIDENCE))

            rec_start = builder.add(f"{rec_id}. ")
            population_start = builder.add(population)
            population_end = builder.length
            builder.add(", ")
            builder.add(action)
            builder.add(". ")
            grade_start = builder.add(
                f"[Strength: {strength} - Evidence: {evidence}]")
            grade_end = builder.length
            builder.add("\n")

            recommendations.append(Recommendation(
                rec_id, f"{number} {title}", population, action, strength,
                evidence, rec_start, builder.length - 1,
                population_start, population_end, grade_start, grade_end))
        builder.add("\n")

    # Section 4's subsections each reverse a recommendation made earlier.
    # This is where a real guideline puts them, and it is the reason the
    # day ends where it does.
    special = [number for number, _title, _n in OUTLINE
               if number.startswith("4.")]
    targets = [r for r in recommendations if r.section_path.startswith("3")]
    for index, number in enumerate(special):
        if index >= len(targets):
            break
        # Evenly spaced through section 3, so the exceptions do not all
        # land on the same recommendation.
        target = targets[(index * len(targets)) // len(special)]
        title = dict((n, t) for n, t, _ in OUTLINE)[number]
        template = EXCEPTION_TEMPLATES[index % len(EXCEPTION_TEMPLATES)]
        text = template.format(rec_id=target.rec_id)
        # Append into the document at the end, under a restated heading,
        # because inserting mid-string would invalidate every offset
        # already recorded.
        builder.add(f"{number} {title} (continued)\n")
        start = builder.add(text + "\n\n")
        exceptions.append(Exception_(target.rec_id, f"{number} {title}",
                                     text, start, start + len(text)))

    text = builder.build()

    ordered = sorted(section_starts.items(), key=lambda item: item[1])
    titles = dict((n, t) for n, t, _ in OUTLINE)
    for index, (number, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        sections.append(Section(number, titles[number], start, end))

    return Guideline(text, tuple(sections), tuple(recommendations),
                     tuple(exceptions))


# --- chunking ---------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int
    #: What the indexer believes this chunk came from. Empty for fixed
    #: chunking, which knows nothing about the document's structure -
    #: that is the whole difference between the two strategies.
    section_path: str = ""

    def contains(self, start: int, end: int) -> bool:
        return self.start <= start and end <= self.end

    def touches(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


def fixed_chunks(guideline: Guideline, size: int = 400,
                 overlap: int = 50) -> list:
    """Sliding character windows. The default in most RAG tutorials."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")
    text = guideline.text
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def section_chunks(guideline: Guideline, max_size: int = 1200) -> list:
    """One chunk per leaf section, carrying its path.

    A section longer than ``max_size`` is split at line boundaries, and
    every piece keeps the path - so the structure survives even when the
    section does not fit.
    """
    leaves = []
    for section in guideline.sections:
        deeper = [s for s in guideline.sections
                  if s.number.startswith(section.number + ".")]
        body_end = min([s.start for s in deeper], default=section.end)
        if body_end > section.start:
            leaves.append((section, section.start, body_end))

    chunks = []
    for section, start, end in leaves:
        body = guideline.text[start:end]
        if len(body) <= max_size:
            chunks.append(Chunk(body, start, end, section.path))
            continue
        cursor = start
        for line in body.splitlines(keepends=True):
            if cursor + len(line) > start + max_size and cursor > start:
                chunks.append(Chunk(guideline.text[start:cursor], start,
                                    cursor, section.path))
                start = cursor
            cursor += len(line)
        chunks.append(Chunk(guideline.text[start:end], start, end,
                            section.path))
    return chunks


# --- what chunking costs ----------------------------------------------------


GRADE_PATTERN = re.compile(r"\[Strength:")
REC_PATTERN = re.compile(r"\bR\d+\.")


@dataclass
class IndexReport:
    """How much of each recommendation survives the chunking."""

    chunks: int = 0
    mean_size: float = 0.0
    whole: int = 0
    severed_population: int = 0
    severed_grade: int = 0
    orphan_grades: int = 0
    single_section: int = 0
    total: int = 0
    recommendations_per_chunk: float = 0.0
    examples: list = field(default_factory=list)

    @property
    def whole_rate(self) -> float:
        return self.whole / self.total if self.total else 0.0

    @property
    def attributable_rate(self) -> float:
        """Fraction of chunks lying inside exactly one leaf section.

        The honest version of "can you tell where this came from". A
        chunk straddling a heading has no single section path. An
        earlier draft counted a chunk as attributable whenever it
        contained a heading, which marked a 1600-character chunk
        spanning four sections as perfectly attributable.
        """
        return self.single_section / self.chunks if self.chunks else 0.0


def index_report(guideline: Guideline, chunks) -> IndexReport:
    """Score a chunking against the structure the generator recorded.

    A recommendation is intact if some single chunk holds all of it. It
    is severed at the population if a chunk holds the action and not the
    eligibility clause - which is the case that changes the guidance
    rather than merely losing some of it.
    """
    report = IndexReport(chunks=len(chunks), total=len(guideline.recommendations))
    report.mean_size = (sum(len(c.text) for c in chunks) / len(chunks)
                        if chunks else 0.0)

    for recommendation in guideline.recommendations:
        start, end = recommendation.text_range
        if any(chunk.contains(start, end) for chunk in chunks):
            report.whole += 1
            continue
        action_start = recommendation.population_end
        holders = [c for c in chunks if c.touches(action_start, end)]
        if any(not c.contains(recommendation.population_start,
                              recommendation.population_end)
               for c in holders):
            report.severed_population += 1
            if len(report.examples) < 3:
                holder = holders[0]
                report.examples.append((recommendation.rec_id,
                                        holder.text.strip()[:110]))
        if any(not c.contains(recommendation.grade_start,
                              recommendation.grade_end) for c in holders):
            report.severed_grade += 1

    held = 0
    for chunk in chunks:
        opening = guideline.section_at(chunk.start)
        closing = guideline.section_at(max(chunk.start, chunk.end - 1))
        if opening is not None and opening is closing:
            report.single_section += 1
        if GRADE_PATTERN.search(chunk.text) and not REC_PATTERN.search(chunk.text):
            report.orphan_grades += 1
        held += sum(1 for r in guideline.recommendations
                    if chunk.touches(*r.text_range))
    report.recommendations_per_chunk = held / len(chunks) if chunks else 0.0

    return report


def exceptions_colocated(guideline: Guideline, chunks) -> dict:
    """Can any chunk hold a recommendation and the exception that reverses it?

    The answer is no, for every chunking, and the reason is not the
    chunker. The document puts them in different chapters.
    """
    together = 0
    for exception in guideline.exceptions:
        recommendation = guideline.recommendation(exception.rec_id)
        if recommendation is None:
            continue
        for chunk in chunks:
            if (chunk.contains(*recommendation.text_range)
                    and chunk.contains(exception.start, exception.end)):
                together += 1
                break
    return {
        "exceptions": len(guideline.exceptions),
        "together": together,
        "apart": len(guideline.exceptions) - together,
    }


def distance_between(guideline: Guideline) -> list:
    """Characters between each recommendation and the text that reverses it."""
    gaps = []
    for exception in guideline.exceptions:
        recommendation = guideline.recommendation(exception.rec_id)
        if recommendation is not None:
            gaps.append((exception.rec_id,
                         exception.start - recommendation.end))
    return gaps


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. The condition and every recommendation")
    print("in this document are invented by this file.")

    guideline = make_guideline(seed=0)
    print()
    print(f"document {len(guideline.text)} characters, "
          f"{len(guideline.sections)} sections, "
          f"{len(guideline.recommendations)} recommendations, "
          f"{len(guideline.exceptions)} exceptions")

    print()
    print("=" * 74)
    print("1. what a recommendation is made of")
    print("=" * 74)
    sample = guideline.recommendations[0]
    print(f"  {guideline.text[sample.start:sample.end]}")
    print()
    print(f"  {'section':<14}{sample.section_path}")
    print(f"  {'who':<14}{sample.population}")
    print(f"  {'what':<14}{sample.action}")
    print(f"  {'how sure':<14}strength {sample.strength}, "
          f"evidence {sample.evidence}")
    print()
    print("  three separable statements in one sentence. 'Offer an ACE")
    print("  inhibitor' and 'in adults with an estimated GFR below 30' are")
    print("  different instructions, and a retriever that returns one")
    print("  without the other has not lost detail - it has changed the")
    print("  guidance.")

    print()
    print("=" * 74)
    print("2. fixed-size chunking, at the sizes people use")
    print("=" * 74)
    print(f"  {'strategy':<18}{'chunks':>7}{'mean':>6}{'intact':>8}"
          f"{'severed':>9}{'orphan':>8}{'one section':>13}{'recs/chunk':>12}")
    rows = [(f"fixed {size}", fixed_chunks(guideline, size=size,
                                           overlap=size // 8))
            for size in (200, 400, 800, 1600)]
    structured = section_chunks(guideline)
    rows.append(("by section", structured))
    for label, chunks in rows:
        report = index_report(guideline, chunks)
        print(f"  {label:<18}{report.chunks:>7}{report.mean_size:>6.0f}"
              f"{report.whole_rate:>8.0%}{report.severed_population:>9}"
              f"{report.orphan_grades:>8}{report.attributable_rate:>13.0%}"
              f"{report.recommendations_per_chunk:>12.1f}")

    print()
    broken = index_report(guideline, fixed_chunks(guideline, size=400,
                                                  overlap=50))
    for rec_id, snippet in broken.examples:
        print(f"    {rec_id} arrives as: ...{snippet}...")
    print()
    print("  the fix is not subtle and not clever: chunk on the headings")
    print("  the document already has. A guideline is a tree and the tree")
    print("  is written down in the text.")
    print()
    print("  and read the last two columns before concluding that a large")
    print("  enough fixed chunk solves it. At 1600 characters nothing is")
    print("  severed - because a chunk that size spans four sections, so")
    print("  none of them can be attributed to one section, and retrieving")
    print("  any chunk hands back six recommendations to answer a question")
    print("  about one.")
    print()
    print("  chunking by section is the only row that is good in both")
    print("  directions at once, and it is good in both by not choosing a")
    print("  size at all - the document chose it.")

    print()
    print("=" * 74)
    print("3. and the part chunking cannot fix")
    print("=" * 74)
    print(f"  {'strategy':<26}{'exceptions':>12}{'together':>10}{'apart':>8}")
    for label, chunks in (("fixed 400", fixed_chunks(guideline, 400, 50)),
                          ("fixed 1600", fixed_chunks(guideline, 1600, 200)),
                          ("by section", structured)):
        row = exceptions_colocated(guideline, chunks)
        print(f"  {label:<26}{row['exceptions']:>12}{row['together']:>10}"
              f"{row['apart']:>8}")
    print()
    for rec_id, gap in distance_between(guideline):
        recommendation = guideline.recommendation(rec_id)
        exception = next(e for e in guideline.exceptions if e.rec_id == rec_id)
        print(f"    {rec_id} in {recommendation.section_path}")
        print(f"       reversed in {exception.section_path}, "
              f"{gap} characters later")
    print()
    print("  no chunking gets those into one chunk, and the chunker is not")
    print("  the reason. Section 3 says what to do and section 4 says when")
    print("  not to, because that is how guidelines are written - the")
    print("  general case first and the exceptions in their own chapter.")
    print()
    gaps = [gap for _rec_id, gap in distance_between(guideline)]
    print(f"  a retriever asked 'what is first-line therapy' will rank")
    print("  section 3 above section 4 on every lexical measure there is,")
    print(f"  and return a recommendation the document contradicts between")
    print(f"  {min(gaps)} and {max(gaps)} characters further down.")

    print()
    print("=" * 74)
    print("4. what the index has to carry")
    print("=" * 74)
    print("  it is not enough to store the text. Each chunk needs:")
    print()
    print("    the section path      so 'first-line' is distinguishable")
    print("                          from 'second-line' when both chunks")
    print("                          say 'offer an ACE inhibitor'")
    print("    the recommendation    so a retrieved chunk can be checked")
    print("    ids it contains       for whether it is whole")
    print("    the grades            so a strong recommendation from")
    print("                          evidence A is not presented")
    print("                          identically to a conditional one")
    print("                          from evidence D")
    print("    cross-references      so section 4 can be pulled in when")
    print("                          section 3 is retrieved")
    print()
    counts = {}
    for recommendation in guideline.recommendations:
        counts[(recommendation.strength, recommendation.evidence)] = \
            counts.get((recommendation.strength, recommendation.evidence), 0) + 1
    print(f"  {'strength':<14}{'evidence':<10}{'count':>6}")
    for (strength, evidence), count in sorted(counts.items()):
        print(f"  {strength:<14}{evidence:<10}{count:>6}")
    print()
    weakest = counts.get(("conditional", "D"), 0)
    strongest = counts.get(("strong", "A"), 0)
    print("  a chunk that arrives without its grade reads as advice, and")
    print(f"  the {weakest} conditional-from-D recommendations then read")
    print(f"  exactly like the {strongest} strong-from-A ones. The tag is")
    print("  the only thing that separated them, and it is one character")
    print("  range away from being in a different chunk.")

    print()
    print("  Day 14 retrieves against this index and has to decide what to")
    print("  do when the best chunk is not good enough.")
