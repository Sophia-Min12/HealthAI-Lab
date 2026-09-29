"""Day 15 - Measuring groundedness: every claim traced to a line.

Day 14 made the system cite. This day asks whether the citation means
anything, and finds that the measure everybody reaches for cannot tell.

**Word overlap is a statement about vocabulary.** Every failure mode
here keeps the vocabulary and changes the meaning:

* **NEGATED** - "do not offer" where the source says "offer". Two words
  added, both of them stopwords in most pipelines.
* **NUMBER** - "GFR below 60" where the source says 30. One token.
* **RECOMBINED** - this recommendation's action with that one's
  eligibility clause. Every word is in the chunk; the pairing is an
  invention.
* **INVENTED** - a treatment the document never mentions, in a sentence
  otherwise copied from it.
* **MISCITED** - a correct claim pointing at the wrong section.
* **MIXED** - two quoted sentences and one fabricated one, which is
  what an averaged groundedness score is least able to see.

The alternative is not a cleverer similarity. It is a list of checks
that a similarity skips - polarity, numbers, and whether the population
and the action came from the *same* recommendation - applied per claim,
with an answer counted grounded only if every claim in it is.

NOT A MEDICAL DEVICE. The condition, the recommendations and the
evidence grades in this file are all invented by it.
"""

# --- reused from day13 and day14 --------------------------------------------


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



# --- Day 15 -----------------------------------------------------------------

NUMBER = re.compile(r"\d+")

#: Phrases that reverse an instruction. Carried forward in spirit from
#: Day 12: the polarity of a sentence is not in its content words, and
#: every overlap measure in this file is built out of content words.
NEGATORS = ("do not", "should not", "must not", "avoid", "never",
            "is not recommended", "no longer")


@dataclass(frozen=True)
class Claim:
    """One checkable assertion out of an answer."""

    text: str
    #: Where in the answer it came from, for pointing at it later.
    index: int


@dataclass(frozen=True)
class Answer:
    """A generated answer and the truth about whether the source supports it."""

    text: str
    kind: str
    rec_id: str
    cited_path: str
    #: Is every claim in it traceable to the cited chunk?
    supported: bool
    #: Traceable and still wrong for the patient. The two are not the
    #: same property, and the last section of the demo is about the gap.
    misleading: bool = False


#: The ways an answer can be wrong while still looking like the source.
#: Each is a real failure mode of a generator given a correct chunk, and
#: each is chosen because it survives a word-overlap check.
ANSWER_KINDS = ("FAITHFUL", "NEGATED", "NUMBER", "RECOMBINED",
                "INVENTED", "MISCITED", "MIXED")


def split_claims(text: str) -> list:
    """Split an answer into sentences. Crude, and the crudeness is visible.

    A real claim decomposition is harder than this, and the harder
    version would not change the day's finding: the failure is in how
    each claim is *checked*, not in how the answer is cut up.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", text) if p.strip()]
    return [Claim(part, index) for index, part in enumerate(parts)]


def negated(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in NEGATORS)


def flip_polarity(action: str) -> str:
    """Turn an instruction into its opposite, changing as little as possible."""
    if action.startswith("do not "):
        return action[len("do not "):]
    for verb in ("offer", "consider", "measure", "review", "record", "seek"):
        if action.startswith(verb):
            return "do not " + action
    return "do not " + action


def bump_numbers(text: str) -> str:
    """Double every number. One token changes; the guidance reverses."""
    return NUMBER.sub(lambda m: str(int(m.group()) * 2), text)


def is_supported(population: str, action: str, path: str, guideline: Guideline,
                 chunks) -> bool:
    """Does the chunk at ``path`` hold a recommendation with exactly these parts?

    The ground truth for every answer below, and it is deliberately not
    a text matcher: it compares the generator's own recorded fields for
    equality. Labelling the answers with the same logic that scores them
    would be the circularity Day 12 spent a section on.

    It also catches a real labelling error. The populations are drawn
    from eight options, so an answer built by recombining two
    recommendations sometimes reproduces a pairing the document
    genuinely contains, and calling that unsupported would be wrong.
    """
    chunk = chunk_for(chunks, path)
    if chunk is None:
        return False
    return any(r.population == population and r.action == action
               for r in recommendations_in(chunk, guideline))


def make_answers(guideline: Guideline, chunks) -> list:
    """One answer of each kind per recommendation, with its truth recorded."""
    answers = []
    recommendations = list(guideline.recommendations)

    def add(text, kind, recommendation, path, population, action):
        answers.append(Answer(
            text, kind, recommendation.rec_id, path,
            is_supported(population, action, path, guideline, chunks)))

    for index, recommendation in enumerate(recommendations):
        other = recommendations[(index + 7) % len(recommendations)]
        elsewhere = recommendations[(index + 3) % len(recommendations)]
        path = recommendation.section_path
        population = recommendation.population
        action = recommendation.action
        faithful = f"{population}, {action}."

        add(faithful, "FAITHFUL", recommendation, path, population, action)
        add(f"{population}, {flip_polarity(action)}.", "NEGATED",
            recommendation, path, population, flip_polarity(action))
        if NUMBER.search(faithful):
            add(bump_numbers(faithful), "NUMBER", recommendation, path,
                bump_numbers(population), bump_numbers(action))
        else:
            add(f"{population} aged 40, {action}.", "NUMBER", recommendation,
                path, f"{population} aged 40", action)
        add(f"{other.population}, {action}.", "RECOMBINED", recommendation,
            path, other.population, action)
        add(f"{population}, offer intravenous magnesium.", "INVENTED",
            recommendation, path, population, "offer intravenous magnesium")
        add(faithful, "MISCITED", recommendation, elsewhere.section_path,
            population, action)
        # The second sentence is not in the document under any citation,
        # so the answer is unsupported however the first one scores.
        answers.append(Answer(
            f"{faithful} The guideline also advises routine genetic testing.",
            "MIXED", recommendation.rec_id, path, False))
    return answers


def chunk_for(chunks, path: str):
    for chunk in chunks:
        if chunk.section_path == path:
            return chunk
    return None


# --- the check everybody writes first ---------------------------------------


def overlap_groundedness(text: str, chunk) -> float:
    """Fraction of the claim's content terms that appear in the source.

    The measure a groundedness dashboard reports, and the reason this
    day exists. It is a statement about vocabulary. Every failure mode
    below keeps the vocabulary and changes the meaning.
    """
    terms = tokenize(text)
    if not terms:
        return 0.0
    source = set(tokenize(chunk.text)) if chunk is not None else set()
    return sum(1 for term in terms if term in source) / len(terms)


# --- the check that survives the failure modes ------------------------------


@dataclass(frozen=True)
class Verdict:
    supported: bool
    reason: str


def recommendations_in(chunk, guideline: Guideline) -> list:
    if chunk is None:
        return []
    return [r for r in guideline.recommendations
            if chunk.touches(*r.text_range)]


def strip_negator(text: str) -> str:
    """The instruction without its leading reversal, if it has one."""
    lowered = text.lower()
    for phrase in NEGATORS:
        if lowered.startswith(phrase + " "):
            return text[len(phrase) + 1:]
    return text


def strict_support(text: str, chunk, guideline: Guideline) -> Verdict:
    """Check a claim against the cited chunk, in the order errors bite.

    Nothing here is clever. It is the set of checks a word-overlap score
    skips, written out one at a time so the demo can attribute each
    caught error to the check that caught it - which is also how the
    first version's polarity bug was found.
    """
    if chunk is None:
        return Verdict(False, "no such section")

    claim_terms = tokenize(text)
    claim_set = set(claim_terms)
    candidates = recommendations_in(chunk, guideline)
    if not candidates:
        return Verdict(False, "cited section holds no recommendation")

    # Polarity, against the recommendation the claim actually restates.
    #
    # The first version asked whether ANY recommendation in the chunk
    # had matching polarity, which a chunk containing one "do not"
    # always satisfies - so it passed 45% of the flipped claims. The
    # question is not whether the chunk contains a negative sentence.
    # It is whether THIS instruction is negative in the source.
    core = [r for r in candidates
            if set(tokenize(strip_negator(r.action))) <= claim_set]
    if core:
        polarity = negated(text)
        agreeing = [r for r in core if negated(r.action) == polarity]
        if not agreeing:
            return Verdict(False, "polarity reversed")

    # Numbers before unfamiliar terms, so a changed threshold is
    # attributed to the number and not to the digits being unfamiliar.
    missing = set(NUMBER.findall(text)) - set(NUMBER.findall(chunk.text))
    if missing:
        return Verdict(False, f"number not in source: {sorted(missing)[0]}")

    source_terms = set(tokenize(chunk.text))
    unknown = [t for t in claim_terms if t not in source_terms]
    if unknown:
        return Verdict(False, f"term not in source: {unknown[0]}")

    # Recombination: the eligibility clause and the action have to come
    # from the SAME recommendation. Every word can be in the chunk and
    # the pairing still be an invention.
    # Against every recommendation in the chunk, not the polarity-
    # narrowed set. Narrowing first discarded the recommendation the
    # population had been lifted from, so the check had nothing left to
    # compare and passed two recombined claims.
    holds_population = [r for r in candidates
                        if set(tokenize(r.population)) <= claim_set]
    holds_action = [r for r in candidates
                    if set(tokenize(r.action)) <= claim_set]
    if holds_population and holds_action:
        shared = {r.rec_id for r in holds_population} & {
            r.rec_id for r in holds_action}
        if not shared:
            return Verdict(False, "population and action from different "
                                  "recommendations")
    return Verdict(True, "supported")


def score_answer(answer: Answer, chunks, guideline: Guideline) -> dict:
    """Both checks, claim by claim.

    An answer is grounded only if every claim in it is. Averaging the
    claims would let one invented sentence hide behind two quoted ones,
    which is the MIXED case below.
    """
    chunk = chunk_for(chunks, answer.cited_path)
    claims = split_claims(answer.text)
    overlaps = [overlap_groundedness(claim.text, chunk) for claim in claims]
    verdicts = [strict_support(claim.text, chunk, guideline) for claim in claims]
    return {
        "answer": answer,
        "claims": claims,
        "overlap_mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "overlap_min": min(overlaps) if overlaps else 0.0,
        "verdicts": verdicts,
        "strict": all(v.supported for v in verdicts),
        "reason": next((v.reason for v in verdicts if not v.supported),
                       "supported"),
    }


def by_kind(scored) -> dict:
    """Mean overlap and strict pass rate, per failure mode."""
    groups: dict = {}
    for row in scored:
        bucket = groups.setdefault(row["answer"].kind,
                                   {"n": 0, "overlap": 0.0, "min": 0.0,
                                    "strict": 0})
        bucket["n"] += 1
        bucket["overlap"] += row["overlap_mean"]
        bucket["min"] += row["overlap_min"]
        bucket["strict"] += row["strict"]
    return {
        kind: {
            "n": data["n"],
            "overlap": data["overlap"] / data["n"],
            "overlap_min": data["min"] / data["n"],
            "strict_pass": data["strict"] / data["n"],
        }
        for kind, data in groups.items()
    }


def separation_of(scored, key) -> float:
    """Rank AUC between supported and unsupported answers, for one measure."""
    positives = [key(r) for r in scored if r["answer"].supported]
    negatives = [key(r) for r in scored if not r["answer"].supported]
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


# --- the failures the checks above were not written for ---------------------

#: Both of these are *traceable*. Every word is in the cited chunk,
#: nothing is negated, no number moved, and the population and action
#: come from the same recommendation. Both scorers pass them, correctly,
#: and both answers are wrong for the patient in front of the clinician.
#:
#: They are here because groundedness is not the property anybody
#: actually wants. It is the property that is easy to define.
HELD_OUT_KINDS = ("SCOPE_DROPPED", "EXCEPTION_IGNORED")


def make_held_out_answers(guideline: Guideline, chunks) -> list:
    """Answers that are supported by the source and still misleading."""
    answers = []
    for recommendation in guideline.recommendations:
        # The eligibility clause simply removed. A recommendation for
        # one group of patients, restated as advice for everybody -
        # which is the thing Day 13 spent a day keeping in one chunk.
        action = recommendation.action
        answers.append(Answer(
            action[0].upper() + action[1:] + ".",
            "SCOPE_DROPPED", recommendation.rec_id,
            recommendation.section_path, True, misleading=True))

    for exception in guideline.exceptions:
        recommendation = guideline.recommendation(exception.rec_id)
        if recommendation is None:
            continue
        # Correct in the cited chunk, and reversed for this patient two
        # chapters further down. Day 13 measured that no chunking puts
        # the two together; this measures that no checker notices.
        answers.append(Answer(
            f"{recommendation.population}, {recommendation.action}.",
            "EXCEPTION_IGNORED", recommendation.rec_id,
            recommendation.section_path, True, misleading=True))
    return answers


if __name__ == "__main__":
    import sys
    from collections import Counter

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. The condition and every recommendation")
    print("in this document are invented by this file.")

    guideline = make_guideline(seed=0)
    chunks = section_chunks(guideline)
    answers = make_answers(guideline, chunks)
    scored = [score_answer(answer, chunks, guideline) for answer in answers]
    kinds = by_kind(scored)

    print()
    print(f"{len(answers)} answers over {len(guideline.recommendations)} "
          f"recommendations, "
          f"{sum(1 for a in answers if a.supported)} of them supported")

    print()
    print("=" * 74)
    print("1. six ways to be wrong while quoting the source")
    print("=" * 74)
    sample = {row["answer"].kind: row for row in scored
              if row["answer"].rec_id == "R7"}
    source = chunk_for(chunks, guideline.recommendation("R7").section_path)
    print(f"  source, {guideline.recommendation('R7').section_path}:")
    print(f"    {guideline.text[guideline.recommendation('R7').start:guideline.recommendation('R7').end]}")
    print()
    for kind in ANSWER_KINDS:
        if kind not in sample:
            continue
        print(f"  {kind}")
        print(f"    {sample[kind]['answer'].text[:96]}")
    del source

    print()
    print("=" * 74)
    print("2. what word overlap says about them")
    print("=" * 74)
    print(f"  {'kind':<12}{'n':>4}{'supported':>11}{'overlap':>9}"
          f"{'passes overlap >= 0.75':>24}")
    for kind in ANSWER_KINDS:
        row = kinds[kind]
        supported = sum(1 for r in scored
                        if r["answer"].kind == kind and r["answer"].supported)
        passing = sum(1 for r in scored
                      if r["answer"].kind == kind and r["overlap_mean"] >= 0.75)
        print(f"  {kind:<12}{row['n']:>4}{supported:>11}{row['overlap']:>9.3f}"
              f"{passing:>24}")
    print()
    print("  a negated answer scores 0.959. It reverses the instruction and")
    print("  it adds two words, both of which most pipelines would drop as")
    print("  stopwords. A changed threshold scores 0.905, and the one token")
    print("  that moved is the only thing the recommendation was about.")
    print()
    print("  overlap is a statement about vocabulary. Every failure mode")
    print("  above keeps the vocabulary.")

    print()
    print("=" * 74)
    print("3. the checks a similarity skips")
    print("=" * 74)
    print(f"  {'kind':<12}{'n':>4}{'supported':>11}{'strict pass':>13}")
    for kind in ANSWER_KINDS:
        row = kinds[kind]
        supported = sum(1 for r in scored
                        if r["answer"].kind == kind and r["answer"].supported)
        print(f"  {kind:<12}{row['n']:>4}{supported:>11}"
              f"{row['strict_pass'] * row['n']:>13.0f}")
    print()
    print("  the supported column is not always zero, and that is a")
    print("  correction rather than a subtlety. The eight eligibility")
    print("  clauses repeat, so four of the twenty recombined answers")
    print("  reproduce a pairing the document genuinely contains. Labelling")
    print("  those unsupported would have been wrong, and the ground truth")
    print("  is computed by comparing the generator's recorded fields for")
    print("  equality - never by running the checker being tested.")
    print()
    print(f"  {'which check caught it':<50}{'count':>7}")
    reasons = Counter(row["reason"].split(":")[0]
                      for row in scored if not row["strict"])
    for reason, count in reasons.most_common():
        print(f"  {reason:<50}{count:>7}")

    print()
    print("=" * 74)
    print("4. the two measures, side by side")
    print("=" * 74)
    print(f"  {'measure':<34}{'separation':>12}")
    for label, key in (("word overlap, mean over claims",
                        lambda r: r["overlap_mean"]),
                       ("word overlap, worst claim",
                        lambda r: r["overlap_min"]),
                       ("claim checks, all must pass",
                        lambda r: float(r["strict"]))):
        print(f"  {label:<34}{separation_of(scored, key):>12.3f}")
    print()
    print("  and a threshold has to be chosen for the overlap measures,")
    print("  which the strict check does not need:")
    print()
    print(f"  {'threshold':>10}{'supported refused':>19}"
          f"{'unsupported passed':>20}")
    for threshold in (0.70, 0.80, 0.90, 0.95, 1.00):
        refused = sum(1 for r in scored if r["answer"].supported
                      and r["overlap_min"] < threshold)
        passed = sum(1 for r in scored if not r["answer"].supported
                     and r["overlap_min"] >= threshold)
        print(f"  {threshold:>10.2f}{refused:>19}{passed:>20}")
    survivors = sum(1 for r in scored if not r["answer"].supported
                    and r["overlap_min"] >= 1.0)
    print()
    print(f"  at 1.00 - demanding that every single word of the answer")
    print(f"  appear in the source - {survivors} unsupported answers still pass,")
    print("  because every word of a negated answer IS in the source. The")
    print("  measure has no headroom left and the errors are still there.")

    print()
    print("=" * 74)
    print("5. why the worst claim and not the average")
    print("=" * 74)
    mixed = next(r for r in scored if r["answer"].kind == "MIXED")
    print(f"  {mixed['answer'].text[:110]}")
    print()
    for claim, verdict in zip(mixed["claims"], mixed["verdicts"]):
        overlap = overlap_groundedness(
            claim.text, chunk_for(chunks, mixed["answer"].cited_path))
        print(f"    claim {claim.index}: overlap {overlap:.3f}  "
              f"{'ok' if verdict.supported else verdict.reason}")
    print(f"    answer: mean {mixed['overlap_mean']:.3f}, "
          f"worst {mixed['overlap_min']:.3f}")
    print()
    print("  one fabricated sentence hiding behind one quoted sentence. The")
    print("  mean puts the answer at 0.500 and the worst claim puts it at")
    print("  0.000.")
    print()
    print("  the two columns in section 4 are equal because almost every")
    print("  answer here carries one claim, so the mean and the minimum are")
    print("  the same number - MIXED is the only kind where they differ.")
    print("  That is a fact about this answer set and not a reason to")
    print("  average: a real answer is several sentences, and averaging is")
    print("  how one fabrication among four quotations gets a pass.")

    print()
    print("=" * 74)
    print("6. and now the part that should be uncomfortable")
    print("=" * 74)
    held = [score_answer(answer, chunks, guideline)
            for answer in make_held_out_answers(guideline, chunks)]
    held_kinds = by_kind(held)
    print(f"  {'kind':<20}{'n':>4}{'overlap':>9}{'strict pass':>13}")
    for kind in HELD_OUT_KINDS:
        row = held_kinds[kind]
        print(f"  {kind:<20}{row['n']:>4}{row['overlap']:>9.3f}"
              f"{row['strict_pass'] * row['n']:>13.0f}")
    print()
    for kind in HELD_OUT_KINDS:
        example = next(r for r in held if r["answer"].kind == kind)
        print(f"  {kind}")
        print(f"    {example['answer'].text[:94]}")
    print()
    print("  both scorers pass every one of those, and both are right to.")
    print("  Every word is in the cited chunk, nothing is negated, no")
    print("  number moved, and the population and the action come from the")
    print("  same recommendation.")
    print()
    print("  the first drops the eligibility clause, so a recommendation")
    print("  for some patients is restated as advice for all of them -")
    print("  which is the thing Day 13 spent a day keeping inside one")
    print("  chunk. The second is correct in the chunk it cites and")
    print("  reversed for this patient in section 4, which Day 13 measured")
    print("  as unreachable by any chunking.")
    print()
    print("  so the strict checker scores 1.000 on the six failure modes it")
    print("  was written for and passes both of the two it was not. That")
    print("  ratio is not a property of the checker. It is a property of")
    print("  the list of failure modes, and I wrote both lists.")
    print()
    print("  groundedness is not the property anybody wants. It is the")
    print("  property that is easy to define. An answer can be traceable to")
    print("  the line it cites and still be the wrong answer for the")
    print("  patient in front of the clinician, and no amount of tracing")
    print("  closes that gap.")

    print()
    print("  Day 16 stops adding failure modes by hand and asks what a")
    print("  generator does with the questions Day 14 could not refuse.")
