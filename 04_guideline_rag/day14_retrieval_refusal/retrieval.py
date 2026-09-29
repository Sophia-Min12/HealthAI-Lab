"""Day 14 - Retrieval that must cite, and refuse when it cannot.

BM25 over Day 13's section chunks, and a rule about when to say nothing.

A retriever is a ranker. It has no "nothing" option: asked about
paediatric dosing, a document with no paediatric chapter still returns
its best five chunks, in order, with scores. The refusal has to be built
on top, and building it is where the day goes.

Three things measured rather than asserted:

* **BM25 scores are not comparable across queries.** The score grows
  with the number of query terms that match anything at all, so a long
  question about nothing outscores a short question about something. A
  single global threshold is therefore not a quality threshold - it is
  mostly a length threshold, and the demo measures how much of it is
  length.

* **The obvious fix for that is the wrong one.** The margin between
  the best chunk and the second best looks like it should be immune to
  query length, and it is not - it still runs 1.8x higher on long
  questions, and it separates answerable from unanswerable questions
  *worse* than the raw score does. Both claims are the opposite of what
  this file said before the table was printed. Dividing the score by
  the number of query terms is the crude fix and the one that works,
  and it over-corrects: short questions now score roughly twice what
  long ones do. It is the best of the three and it is not principled.

* **And no criterion separates them completely**, because some
  out-of-scope questions really do share vocabulary with the document.
  Day 7's arithmetic returns: the threshold is a choice about which
  error to make, and here one error is a refusal and the other is a
  confident citation of a chunk that does not answer the question.

NOT A MEDICAL DEVICE. The condition, the recommendations and the
evidence grades in this file are all invented by it.
"""

# --- reused from day13 ------------------------------------------------------


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


# --- Day 14 -----------------------------------------------------------------

import math
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")

#: Words carrying no discriminative weight in a document about one
#: condition. Kept short on purpose: an aggressive stop list is a way of
#: hiding a retrieval problem rather than fixing it.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "should", "the", "to", "what", "when", "which",
    "who", "with", "how",
})


def tokenize(text: str, drop_stopwords: bool = True) -> list:
    """Lowercase alphanumeric runs. Deliberately plain."""
    terms = TOKEN.findall(text.lower())
    if drop_stopwords:
        terms = [t for t in terms if t not in STOPWORDS]
    return terms


class BM25:
    """Okapi BM25, written out so the length terms are visible.

    ``k1`` controls how fast term frequency saturates; ``b`` how much a
    long document is penalised. The reason both appear here rather than
    in a library call is section 3: the length normalisation applies to
    documents, and nothing anywhere normalises for the length of the
    *query*.
    """

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)
                               if self.lengths else 0.0)
        frequency: Counter = Counter()
        for document in self.documents:
            frequency.update(document.keys())
        total = len(self.documents)
        self.idf = {
            term: math.log((total - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in frequency.items()
        }

    def score(self, terms, index: int) -> float:
        document = self.documents[index]
        length = self.lengths[index]
        total = 0.0
        for term in terms:
            count = document.get(term, 0)
            if not count:
                continue
            denominator = count + self.k1 * (
                1 - self.b + self.b * length / (self.average_length or 1.0))
            total += self.idf[term] * count * (self.k1 + 1) / denominator
        return total

    def rank(self, query: str, k: int = 5) -> list:
        """The top ``k`` chunks, best first. Always returns something."""
        terms = tokenize(query)
        scored = [(self.score(terms, index), index)
                  for index in range(len(self.chunks))]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self.chunks[index], score) for score, index in scored[:k]]


# --- questions --------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """A question and the recommendation that answers it, if any."""

    text: str
    rec_id: str | None
    style: str  # "short" or "long"

    @property
    def answerable(self) -> bool:
        return self.rec_id is not None


#: Out-of-scope questions. Every one is a reasonable thing to ask a
#: clinical guideline and none of it is in this one. Several share
#: vocabulary with the document on purpose - that is what makes the
#: separation in section 4 incomplete rather than merely imperfect.
OUT_OF_SCOPE = (
    ("children", "What is the recommended dose in children?"),
    ("neonates", "How should neonates be managed?"),
    ("surgery", "What is the surgical approach?"),
    ("meningitis", "Which antibiotic is first-line for meningitis?"),
    ("vaccination", "Is vaccination recommended?"),
    ("imaging", "Which imaging modality should be used for staging?"),
    ("cost", "What is the cost-effectiveness threshold used?"),
    ("adults dose", "In adults with stage 2 disease, what dose of "
                    "amoxicillin should be offered?"),
    ("adults review", "In adults with an estimated GFR below 30 mL/min, "
                      "what is the recommended dialysis schedule?"),
    ("pregnancy dose", "In pregnancy, what is the recommended insulin "
                       "regimen for this condition?"),
)


def make_questions(guideline: Guideline, index: "BM25 | None" = None) -> list:
    """One short and one long question per recommendation, plus the rest.

    The pair is the instrument for section 3: both forms ask for the
    same chunk, and differ only in how many words they use to do it.

    The short form is the three rarest terms of the action, by IDF,
    which is what a person typing a short query actually does - they
    type the distinctive words. Taking the first three tokens instead,
    as the first version did, produced "do not offer?" for one
    recommendation in four, and a question with no content words in it
    measures nothing.
    """
    questions = []
    for recommendation in guideline.recommendations:
        terms = tokenize(recommendation.action)
        if index is not None:
            # dict.fromkeys dedupes in first-appearance order and the
            # term itself breaks IDF ties. An earlier version used
            # set(), whose iteration order changes between processes,
            # so the questions - and every number in this file - moved
            # from run to run.
            terms = sorted(dict.fromkeys(terms),
                           key=lambda t: (-index.idf.get(t, 0.0), t))
        action_terms = terms[:3]
        short = " ".join(action_terms) + "?"
        long = (f"{recommendation.population}, what does the guideline "
                f"recommend about {recommendation.action}?")
        questions.append(Question(short, recommendation.rec_id, "short"))
        questions.append(Question(long, recommendation.rec_id, "long"))
    for _label, text in OUT_OF_SCOPE:
        questions.append(Question(text, None,
                                  "long" if len(tokenize(text)) > 5 else "short"))
    return questions


# --- retrieval, and the decision to answer ----------------------------------


@dataclass
class Retrieval:
    """What came back, and the three numbers a refusal rule can use."""

    question: Question
    chunks: tuple
    scores: tuple

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def margin(self) -> float:
        """Best minus second best. Does not depend on the query's length."""
        if len(self.scores) < 2:
            return self.top_score
        return self.scores[0] - self.scores[1]

    @property
    def score_per_term(self) -> float:
        terms = len(tokenize(self.question.text)) or 1
        return self.top_score / terms

    def hits(self, guideline: Guideline) -> bool:
        """Does the top chunk contain the recommendation that answers it?"""
        if not self.question.answerable or not self.chunks:
            return False
        recommendation = guideline.recommendation(self.question.rec_id)
        return self.chunks[0].touches(*recommendation.text_range)

    def hits_at(self, guideline: Guideline, k: int) -> bool:
        if not self.question.answerable:
            return False
        recommendation = guideline.recommendation(self.question.rec_id)
        return any(chunk.touches(*recommendation.text_range)
                   for chunk in self.chunks[:k])


def retrieve_all(index: BM25, questions, k: int = 5) -> list:
    out = []
    for question in questions:
        ranked = index.rank(question.text, k=k)
        out.append(Retrieval(question,
                             tuple(chunk for chunk, _ in ranked),
                             tuple(score for _, score in ranked)))
    return out


def citation(retrieval: Retrieval, guideline: Guideline) -> str:
    """What an answer must carry: a section path and the ids it used.

    An answer without one of these cannot be checked, and Day 15 is
    entirely about checking them.
    """
    if not retrieval.chunks:
        return "no source"
    chunk = retrieval.chunks[0]
    ids = [r.rec_id for r in guideline.recommendations
           if chunk.touches(*r.text_range)]
    path = chunk.section_path or "unknown section"
    return f"{path} [{', '.join(ids) if ids else 'no recommendation'}]"


CRITERIA = ("top_score", "margin", "score_per_term")


def criterion_value(retrieval: Retrieval, criterion: str) -> float:
    return getattr(retrieval, criterion)


@dataclass
class RefusalReport:
    """Two error rates that cannot both be driven to zero."""

    threshold: float = 0.0
    answered_answerable: int = 0
    refused_answerable: int = 0
    answered_unanswerable: int = 0
    refused_unanswerable: int = 0
    correct_answers: int = 0

    @property
    def refusal_rate_on_answerable(self) -> float:
        total = self.answered_answerable + self.refused_answerable
        return self.refused_answerable / total if total else 0.0

    @property
    def answer_rate_on_unanswerable(self) -> float:
        total = self.answered_unanswerable + self.refused_unanswerable
        return self.answered_unanswerable / total if total else 0.0


def apply_threshold(retrievals, guideline: Guideline, threshold: float,
                    criterion: str = "top_score") -> RefusalReport:
    """Answer when the criterion clears the threshold, refuse otherwise."""
    report = RefusalReport(threshold=threshold)
    for retrieval in retrievals:
        answering = criterion_value(retrieval, criterion) >= threshold
        if retrieval.question.answerable:
            if answering:
                report.answered_answerable += 1
                report.correct_answers += retrieval.hits(guideline)
            else:
                report.refused_answerable += 1
        else:
            if answering:
                report.answered_unanswerable += 1
            else:
                report.refused_unanswerable += 1
    return report


def separation(retrievals, criterion: str) -> float:
    """Probability a random answerable question outranks a random one that is not.

    The rank-based AUC, computed directly. 0.5 is no separation at all
    and 1.0 is perfect, and a criterion that cannot beat 0.5 is not
    measuring answerability.
    """
    positives = [criterion_value(r, criterion) for r in retrievals
                 if r.question.answerable]
    negatives = [criterion_value(r, criterion) for r in retrievals
                 if not r.question.answerable]
    if not positives or not negatives:
        return float("nan")
    wins = ties = 0
    for high in positives:
        for low in negatives:
            if high > low:
                wins += 1
            elif high == low:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def by_style(retrievals, criterion: str) -> dict:
    """The same criterion, split by how long the question was."""
    groups: dict = {}
    for retrieval in retrievals:
        if not retrieval.question.answerable:
            continue
        values = groups.setdefault(retrieval.question.style, [])
        values.append(criterion_value(retrieval, criterion))
    return {style: sum(values) / len(values)
            for style, values in groups.items()}


def sweep(retrievals, guideline, criterion: str, steps: int = 9) -> list:
    """Threshold sweep across the observed range of a criterion."""
    values = sorted(criterion_value(r, criterion) for r in retrievals)
    low, high = values[0], values[-1]
    return [apply_threshold(retrievals, guideline,
                            low + (high - low) * step / (steps - 1), criterion)
            for step in range(steps)]


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. The condition and every recommendation")
    print("in this document are invented by this file.")

    guideline = make_guideline(seed=0)
    chunks = section_chunks(guideline)
    index = BM25(chunks)
    questions = make_questions(guideline, index)
    retrievals = retrieve_all(index, questions, k=5)
    answerable = [r for r in retrievals if r.question.answerable]
    unanswerable = [r for r in retrievals if not r.question.answerable]

    print()
    print(f"index {len(chunks)} section chunks, {len(questions)} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} not)")

    print()
    print("=" * 74)
    print("1. a retriever has no 'nothing' option")
    print("=" * 74)
    easy = next(r for r in unanswerable
                if "neonates" in r.question.text.lower())
    print(f"  the easy case: {easy.question.text}")
    print(f"  no term in it appears anywhere in the document, so every")
    print(f"  chunk scores {easy.top_score:.2f} and the ranking is just the")
    print("  chunks in document order. A score of zero is a usable signal.")
    print()
    probe = max(unanswerable, key=lambda r: r.top_score)
    print(f"  the hard case: {probe.question.text}")
    print("  the document has no dialysis chapter. It answers anyway, and")
    print("  it answers confidently:")
    print()
    for chunk, score in zip(probe.chunks[:3], probe.scores[:3]):
        print(f"    {score:6.2f}  {chunk.section_path}")
    print()
    print("  that is not a bug. Ranking is what a retriever does, and a")
    print("  ranked list of the least-bad chunks is what it returns when")
    print("  none of them is any good. Somebody downstream has to decide")
    print("  the best of them is not good enough, and nothing in the")
    print("  ranking says so.")

    print()
    print("=" * 74)
    print("2. when the answer is in there, retrieval finds it")
    print("=" * 74)
    print(f"  {'k':<5}{'hit@k':>9}")
    for k in (1, 2, 3, 5):
        hits = sum(r.hits_at(guideline, k) for r in answerable)
        print(f"  {k:<5}{hits / len(answerable):>9.3f}")
    print()
    print("  so the retrieval half is not the problem. Everything below is")
    print("  about the other half: deciding whether to use what came back.")

    print()
    print("=" * 74)
    print("3. the score is mostly a measurement of the question's length")
    print("=" * 74)
    print("  every answerable question was asked twice - a short form and a")
    print("  long one, both wanting the same chunk:")
    print()
    pair = [r for r in answerable if r.question.rec_id == "R1"]
    for retrieval in sorted(pair, key=lambda r: r.question.style == "long"):
        print(f"    [{retrieval.question.style:<5}] {retrieval.question.text[:58]}")
        print(f"             top score {retrieval.top_score:.2f}")
    print()
    print(f"  {'criterion':<18}{'short':>9}{'long':>9}{'ratio':>9}")
    for criterion in CRITERIA:
        styles = by_style(retrievals, criterion)
        ratio = styles["long"] / styles["short"] if styles["short"] else 0.0
        print(f"  {criterion:<18}{styles['short']:>9.2f}{styles['long']:>9.2f}"
              f"{ratio:>9.2f}")
    print()
    print("  BM25 normalises for the length of the DOCUMENT - that is what")
    print("  b is for. Nothing anywhere normalises for the length of the")
    print("  QUERY, because a longer query genuinely does carry more")
    print("  evidence. The trouble is that it carries more of it whether or")
    print("  not the document can answer:")
    print()
    short_hit = max((r for r in answerable if r.question.style == "short"
                     and r.hits(guideline)), key=lambda r: -r.top_score)
    worst = max(unanswerable, key=lambda r: r.top_score)
    print(f"    answerable, short   {short_hit.top_score:6.2f}  "
          f"{short_hit.question.text[:44]}")
    print(f"    unanswerable, long  {worst.top_score:6.2f}  "
          f"{worst.question.text[:44]}")
    print()
    print("  a single absolute threshold set between those two numbers")
    print("  refuses the question the document answers and accepts the one")
    print("  it does not.")

    print()
    print("=" * 74)
    print("4. so refuse on something that is not a length")
    print("=" * 74)
    print(f"  {'criterion':<18}{'separation':>12}{'long/short':>12}")
    for criterion in CRITERIA:
        styles = by_style(retrievals, criterion)
        ratio = styles["long"] / styles["short"] if styles["short"] else 0.0
        print(f"  {criterion:<18}{separation(retrievals, criterion):>12.3f}"
              f"{ratio:>12.2f}")
    print()
    print("  separation is the probability that a random answerable")
    print("  question scores above a random unanswerable one. 0.5 is no")
    print("  information at all.")
    print()
    margin_styles = by_style(retrievals, "margin")
    margin_ratio = margin_styles["long"] / margin_styles["short"]
    print("  the margin between the best and second-best chunk is the")
    print("  criterion that ought to be immune to query length. It is not.")
    print(f"  It still runs {margin_ratio:.1f}x higher on the long questions, and it")
    print(f"  separates {separation(retrievals, 'margin'):.3f} against the raw "
          f"score's {separation(retrievals, 'top_score'):.3f} -")
    print("  worse, not better. Both halves of that are the opposite of")
    print("  what this file predicted before the table was printed.")
    print()
    per_term = by_style(retrievals, "score_per_term")
    print("  dividing by the number of query terms is the crude fix and")
    print(f"  the one that works, separating {separation(retrievals, 'score_per_term'):.3f}.")
    print(f"  It also over-corrects: short questions now score "
          f"{per_term['short'] / per_term['long']:.1f}x what")
    print("  long ones do. It is the best available criterion here and it")
    print("  is not a principled one.")

    print()
    print("=" * 74)
    print("5. and now it is Day 7 again")
    print("=" * 74)
    print(f"  {'threshold':>10}{'refused, answerable':>21}"
          f"{'answered, unanswerable':>24}{'correct':>9}")
    for report in sweep(retrievals, guideline, "score_per_term", steps=9):
        print(f"  {report.threshold:>10.2f}"
              f"{report.refusal_rate_on_answerable:>21.1%}"
              f"{report.answer_rate_on_unanswerable:>24.1%}"
              f"{report.correct_answers:>9}")
    print()
    candidates = sorted({criterion_value(r, "score_per_term")
                         for r in retrievals})
    best = min(
        (apply_threshold(retrievals, guideline, t, "score_per_term")
         for t in candidates),
        key=lambda rep: (rep.refusal_rate_on_answerable
                         + rep.answer_rate_on_unanswerable))
    print("  no row has both middle columns at zero, and that is not an")
    print("  artefact of the grid. Searching every threshold any question")
    print("  actually produces, the best either error can be made together")
    print(f"  is {best.refusal_rate_on_answerable:.1%} refused and "
          f"{best.answer_rate_on_unanswerable:.1%} answered, at "
          f"{best.threshold:.2f}.")
    print()
    print("  every threshold is a statement about which error is worse, and")
    print("  the statement is being made whether or not anybody writes it")
    print("  down.")
    print()
    print("  a refusal costs a clinician a lookup. A confident citation of")
    print("  a chunk that does not answer the question costs them the")
    print("  belief that the system checked. Those are not the same price,")
    print("  and the ratio between them is not 1.")

    print()
    print("=" * 74)
    print("6. what an answer has to carry")
    print("=" * 74)
    best = apply_threshold(retrievals, guideline, 0.9, "score_per_term")
    print(f"  at threshold 0.90 on score per term: "
          f"{best.answered_answerable} answered, "
          f"{best.refused_answerable} refused,")
    print(f"  {best.answered_unanswerable} of {len(unanswerable)} "
          f"out-of-scope questions answered anyway.")
    print()
    for retrieval in answerable[:3]:
        print(f"    Q {retrieval.question.text[:56]}")
        print(f"      -> {citation(retrieval, guideline)}")
    print()
    print("  a section path and the recommendation ids the chunk contains.")
    print("  Without both, an answer cannot be checked at all - and Day 15")
    print("  is entirely about checking them.")

    print()
    print("=" * 74)
    print("7. the ones that get through anything")
    print("=" * 74)
    survivors = sorted(unanswerable, key=lambda r: -r.score_per_term)[:3]
    for retrieval in survivors:
        print(f"  {retrieval.score_per_term:5.2f}  {retrieval.question.text}")
        print(f"         retrieved: {retrieval.chunks[0].section_path}")
    print()
    print("  these are out of scope and they are not nonsense. They ask")
    print("  about adults with this condition, using this document's own")
    print("  vocabulary, about something it does not cover. No threshold on")
    print("  a lexical score separates them, because lexically they belong.")
    print()
    print("  Day 16 is about what happens to these when a generator is put")
    print("  on the end of the pipeline. Day 15 first has to establish what")
    print("  it would mean for an answer to be supported at all.")
