"""Day 17 - Capstone: the full pipeline, a CLI, and the honest writeup.

Seventeen days wired into one path: a synthetic clinical note goes in,
and either a cited recommendation or a refusal comes out.

    python capstone.py note           one case, every stage shown
    python capstone.py run            the stage table and the compounding
    python capstone.py deid "<text>"  de-identify a note
    python capstone.py ask "<text>"   one question against the guideline
    python capstone.py writeup        what seventeen days established

Four stages, each of which this repo spent days measuring:

1. **De-identify** the note - Day 11's letter-aware detector, because
   Day 10's ASCII one left the tail of every name with a diacritic on
   the page.
2. **Read the assertion** - Day 12's negation rules, because "no
   evidence of stage 2 disease" contains "stage 2 disease" and a
   pipeline that acts on the phrase treats the two alike.
3. **Retrieve and refuse** - Day 14's score-per-term threshold, because
   the raw BM25 score is mostly a measurement of how long the question
   was.
4. **Compose and check** - Day 15's claim checks over Day 16's
   extractive generator.

The number this file exists to produce is the one at the bottom of the
stage table. Every stage is respectable on its own and the product of
them is not, and no stage's report contains it.

``writeup`` prints what the seventeen days established, including the
parts that do not flatter the system: the note-level de-identification
rate is far below the span-level one, the negation rules score 1.000 on
the sentences they were written for and 0.730 on sentences they were
not, and flipping the order two editions were added to the index moved
26 of 28 answers onto a withdrawn guideline without changing a single
score.

NOT A MEDICAL DEVICE. Every note, every guideline and every evidence
grade in this repo is invented by it. No patient data, real or derived,
appears anywhere.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


# --- reused from day10 and day11 - de-identification ---------------------------------


# --- reused from day10 ------------------------------------------------------





# --- what counts as an identifier ------------------------------------------

#: The Safe Harbor categories this day implements. HIPAA lists eighteen;
#: these are the seven that appear in free text often enough to carry the
#: lesson. The omissions are not an oversight - see ``SAFE_HARBOR_GAPS``.
CATEGORIES = ("NAME", "MRN", "DATE", "PHONE", "AGE", "ADDRESS", "EMAIL")

#: The categories a reader should know are missing, and why. Kept in the
#: module because a de-identification tool that does not state its scope
#: is making a claim it has not earned.
SAFE_HARBOR_GAPS = {
    "device identifiers": "serial numbers of implanted devices",
    "vehicle identifiers": "licence plates, in accident narratives",
    "biometric identifiers": "fingerprints, voiceprints - not text at all",
    "full-face photographs": "not text",
    "URLs and IP addresses": "common in pasted referral text",
    "any other unique number": (
        "the catch-all clause, and the reason a rule list "
        "can never be declared complete"
    ),
}


@dataclass(frozen=True)
class Span:
    """A character range in a note, and what kind of identifier it is."""

    start: int
    end: int
    category: str
    text: str

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Note:
    """A synthetic note and the ground truth of what was put in it.

    ``attributes`` is new in Day 11: the structured facts the note was
    built from, which is what a re-identification attacker joins on.
    Recording them draws no extra randomness, so the corpus text is
    byte-identical to Day 10's and every number carries over.
    """

    text: str
    spans: tuple = ()
    note_id: int = 0
    attributes: dict = field(default_factory=dict)

    def by_category(self, category: str) -> list:
        return [s for s in self.spans if s.category == category]


class _NoteBuilder:
    """Assembles a note while recording the offsets of what it inserts.

    Offsets are computed as the text is built, never recovered afterwards
    by searching for the string. Searching would find the wrong
    occurrence the first time a surname also appears as a disease name,
    which is exactly the case this day is about.
    """

    def __init__(self) -> None:
        self._parts: list = []
        self._length = 0
        self._spans: list = []

    def add(self, text: str, category: str | None = None) -> "_NoteBuilder":
        if category is not None:
            self._spans.append(Span(self._length, self._length + len(text),
                                    category, text))
        self._parts.append(text)
        self._length += len(text)
        return self

    def build(self, note_id: int = 0, attributes: dict | None = None) -> Note:
        return Note("".join(self._parts), tuple(self._spans), note_id,
                    dict(attributes or {}))


# --- the generator's pools --------------------------------------------------

#: Given names and surnames the generator draws from. Deliberately wider
#: than the detector's gazetteer below, because that is the real
#: situation: a name list is a snapshot of a population and the next
#: patient through the door is under no obligation to be in it.
GIVEN_NAMES = (
    "Margaret", "Thomas", "Aisha", "Daniel", "Priya", "Robert", "Ingrid",
    "Samuel", "Yuki", "Helen", "Marcus", "Fatima", "Oliver", "Beatriz",
    "Kwame", "Linnea", "Dmitri", "Sofia", "Tobias", "Nadia", "Emeka",
    "Astrid", "Rashid", "Colette", "Mateo", "Siobhan", "Hiroshi", "Lucia",
)

#: Surnames that are also medical eponyms. Real surnames - Bell and
#: Graves and Parkinson are ordinary names carried by ordinary people -
#: and the reason a name gazetteer is dangerous rather than merely
#: incomplete.
EPONYM_SURNAMES = ("Parkinson", "Crohn", "Graves", "Hodgkin", "Paget",
                   "Bell", "Cushing")

SURNAMES = (
    "Whitfield", "Okonkwo", "Lindqvist", "Ferreira", "Nakamura", "Abadi",
    "Castellanos", "Bergström", "Adeyemi", "Kowalski", "Petrossian",
    "Oyelaran", "Vasquez", "Thornbury", "Mbeki", "Haugen", "Delacroix",
    "Szabo", "Rasmussen", "Chatterjee", "Balogun", "Kaminski", "Novotny",
) + EPONYM_SURNAMES

#: Clinical phrases in which those words are *not* identifiers. A
#: redactor that removes the eponym from any of these has damaged the
#: note without leaking anything, and no recall figure will say so.
EPONYM_PHRASES = (
    "Parkinson's disease", "Crohn's disease", "Graves' disease",
    "Hodgkin lymphoma", "Paget's disease of bone", "Bell's palsy",
    "Cushing's syndrome",
)

STREETS = ("Alder", "Kingsway", "Rosemount", "Fairhaven", "Blackthorn",
           "Windermere", "Sycamore", "Caldwell")
STREET_TYPES = ("Street", "Avenue", "Road", "Lane", "Crescent")
TOWNS = ("Ashford", "Milburn", "Redhill", "Kestrel Bay", "Northgate",
         "Larkfield")

CONDITIONS = (
    "type 2 diabetes mellitus", "chronic kidney disease stage 3",
    "atrial fibrillation", "community-acquired pneumonia",
    "congestive heart failure", "essential hypertension",
    "chronic obstructive pulmonary disease", "iron deficiency anaemia",
)

MEDICATIONS = (
    "metformin 500 mg BD", "amlodipine 5 mg OD", "furosemide 40 mg OD",
    "apixaban 5 mg BD", "atorvastatin 20 mg nocte", "ramipril 2.5 mg OD",
    "salbutamol 100 mcg PRN", "ferrous sulfate 200 mg BD",
)

NOTE_DEPARTMENTS = ("Cardiology", "Respiratory Medicine", "Nephrology",
               "General Medicine", "Endocrinology")

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
MONTH_ABBR = tuple(m[:3] for m in MONTHS)

#: Seven date renderings and how often each is used. The detector covers
#: six. The two-digit-year form is rare here for the same reason it is
#: rare in a development sample: it is the one a rule author does not
#: happen to see, and therefore the one the rules do not carry.
DATE_STYLE_WEIGHTS = (0.22, 0.20, 0.14, 0.14, 0.08, 0.12, 0.10)

#: Three MRN renderings. The unlabelled one is the interesting case: a
#: bare seven-digit number in running text is indistinguishable from a
#: lab value or an accession number without reading the sentence.
MRN_STYLE_WEIGHTS = (0.40, 0.28, 0.16, 0.16)


def _render_date(rng, day: int, month: int, year: int) -> str:
    """One date, in one of seven renderings, chosen by ``rng``."""
    style = int(rng.choice(len(DATE_STYLE_WEIGHTS), p=DATE_STYLE_WEIGHTS))
    if style == 0:
        return f"{month:02d}/{day:02d}/{year}"
    if style == 1:
        return f"{year}-{month:02d}-{day:02d}"
    if style == 2:
        return f"{day:02d}-{MONTH_ABBR[month - 1]}-{year}"
    if style == 3:
        return f"{MONTHS[month - 1]} {day}, {year}"
    if style == 4:  # two-digit year, no leading zeros - uncovered
        return f"{month}/{day}/{year % 100}"
    if style == 5:  # day first, dotted
        return f"{day:02d}.{month:02d}.{year}"
    return f"{day} {MONTHS[month - 1]} {year}"  # day first, no comma


def _render_mrn(rng) -> str:
    """An MRN, labelled or bare."""
    number = int(rng.integers(1000000, 9999999))
    style = int(rng.choice(len(MRN_STYLE_WEIGHTS), p=MRN_STYLE_WEIGHTS))
    if style == 0:
        return f"MRN {number}"
    if style == 1:
        return f"MRN: {number}"
    if style == 2:
        return f"#{number}"
    return str(number)  # bare - uncovered by the conservative rules


def _render_phone(rng) -> str:
    area = int(rng.integers(200, 999))
    mid = int(rng.integers(200, 999))
    last = int(rng.integers(1000, 9999))
    style = int(rng.integers(0, 3))
    if style == 0:
        return f"({area}) {mid}-{last}"
    if style == 1:
        return f"{area}-{mid}-{last}"
    return f"{area}.{mid}.{last}"


def make_notes(n: int = 300, seed: int = 0, condition_weights=None) -> list:
    """Generate ``n`` synthetic notes, each carrying its own ground truth.

    Every identifier is recorded as it is written, with exact offsets.
    Nothing here is derived from a real record; the templates read like
    discharge summaries and contain the shapes a de-identifier has to
    cope with.

    ``condition_weights`` skews the diagnosis distribution. Left as
    ``None`` the draw is the uniform one Day 10 used, down to the random
    numbers consumed, so the default corpus is unchanged. Section 6
    passes weights, because a uniform distribution of diagnoses is the
    one case in which no diagnosis is rare - and rarity is what makes a
    record identifiable.
    """
    rng = np.random.default_rng(seed)
    notes = []
    for note_id in range(n):
        builder = _NoteBuilder()
        given = str(rng.choice(GIVEN_NAMES))
        surname = str(rng.choice(SURNAMES))
        clinician = str(rng.choice(SURNAMES))
        age = int(rng.integers(24, 96))
        condition = str(rng.choice(CONDITIONS) if condition_weights is None
                        else rng.choice(CONDITIONS, p=condition_weights))
        medication = str(rng.choice(MEDICATIONS))
        department = str(rng.choice(NOTE_DEPARTMENTS))

        year = int(rng.integers(2019, 2025))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))

        builder.add(f"{department} - Discharge Summary\n\n")
        builder.add("Patient: ")
        builder.add(f"{given} {surname}", "NAME")
        builder.add("   ")
        builder.add(_render_mrn(rng), "MRN")
        builder.add("\nDate of admission: ")
        builder.add(_render_date(rng, day, month, year), "DATE")
        builder.add("\n\n")

        # Age is an identifier only over 89 under Safe Harbor, because
        # that tail is small enough to single a person out. Below it the
        # age is clinical information and redacting it is a loss.
        builder.add("This ")
        builder.add(f"{age}-year-old", "AGE" if age > 89 else None)
        builder.add(" patient was admitted under the care of Dr ")
        builder.add(clinician, "NAME")
        builder.add(f" with {condition}.\n")

        if int(rng.integers(0, 3)) == 0:
            builder.add("Background includes ")
            builder.add(str(rng.choice(EPONYM_PHRASES)))
            builder.add(", stable.\n")

        builder.add(f"Treated with {medication}. ")
        builder.add("Reviewed on ")
        builder.add(_render_date(rng, min(day, 28), month % 12 + 1, year), "DATE")
        builder.add(" and discharged home.\n")

        # A bare surname with no title and no field label in front of it.
        # The gazetteer is the only thing that can catch this one, which
        # is what makes the eponym trade-off in section 5 a real trade.
        if int(rng.integers(0, 5)) < 2:
            builder.add("Case discussed with ")
            builder.add(str(rng.choice(SURNAMES)), "NAME")
            builder.add(" at the MDT.\n")

        builder.add("\nAddress on file: ")
        street_no = int(rng.integers(1, 200))
        builder.add(
            f"{street_no} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}, "
            f"{rng.choice(TOWNS)}",
            "ADDRESS",
        )
        builder.add("\nContact: ")
        builder.add(_render_phone(rng), "PHONE")

        if int(rng.integers(0, 2)) == 0:
            builder.add("   ")
            builder.add(f"{given.lower()}.{surname.lower()}@example-mail.org",
                        "EMAIL")

        builder.add("\n\nFollow-up with ")
        builder.add(f"{str(rng.choice(GIVEN_NAMES))} {clinician}", "NAME")
        builder.add(f" in {department} clinic.\n")

        notes.append(builder.build(note_id, {
            "age": age, "condition": condition, "department": department,
            "year": year, "month": month,
        }))
    return notes


# --- the detector -----------------------------------------------------------

#: The surnames the detector knows. Not a slice: written out, because
#: which names are on the list is the subject of section 5 and hiding it
#: behind an index would hide the point. The eponyms are included
#: because they genuinely are common surnames - that is the problem, not
#: an error in the list.
GAZETTEER_SURNAMES = frozenset({
    "Whitfield", "Okonkwo", "Lindqvist", "Ferreira", "Nakamura", "Abadi",
    "Castellanos", "Adeyemi", "Kowalski", "Petrossian", "Vasquez",
    "Thornbury", "Haugen", "Delacroix", "Rasmussen", "Chatterjee",
    "Kaminski",
    "Parkinson", "Graves", "Bell", "Paget", "Cushing",
})

GAZETTEER_GIVEN = frozenset({
    "Margaret", "Thomas", "Daniel", "Robert", "Samuel", "Helen", "Marcus",
    "Oliver", "Linnea", "Sofia", "Tobias", "Nadia", "Astrid", "Colette",
    "Mateo", "Lucia", "Aisha", "Priya",
})

#: Ordered, because the first pattern to claim a character wins. MRN
#: before DATE matters: a labelled MRN and an ISO date are both runs of
#: digits and punctuation, and the more specific pattern goes first.
PATTERNS = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}"
                         r"|\b\d{3}-\d{3}-\d{4}\b"
                         r"|\b\d{3}\.\d{3}\.\d{4}\b")),
    ("MRN", re.compile(r"\bMRN:?\s*\d{6,8}\b|#\d{6,8}\b")),
    ("DATE", re.compile(
        r"\b\d{1,2}/\d{1,2}/\d{4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b"
        r"|\b\d{1,2}\.\d{1,2}\.\d{4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},\s*\d{4}\b"
        r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+\d{4}\b")),
    # Safe Harbor: ages over 89 only. 90-129 covers the plausible range.
    ("AGE", re.compile(r"\b(?:9\d|1[01]\d|12\d)[-\s]?year[-\s]?old\b")),
    # The whitespace classes are [ \t] and not \s on purpose. \s matches
    # a newline, and the first version of this pattern ran ", Larkfield"
    # straight into the "Contact:" label on the line below - swallowing a
    # field name, scoring a clean hit on the address, and never showing up
    # in any recall figure. Over-capture is invisible to recall by
    # construction, which is section 5's point arriving early and uninvited.
    ("ADDRESS", re.compile(
        r"\b\d{1,4}[ \t]+[A-Z][a-z]+[ \t]+"
        r"(?:Street|Avenue|Road|Lane|Crescent|Drive|Way)\b"
        r"(?:,[ \t]*[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?)?")),
)

#: The rule the conservative detector leaves out: any bare six-to-eight
#: digit run. Section 6 turns it on and measures what it costs.
BARE_NUMBER_PATTERN = re.compile(r"\b\d{6,8}\b")

#: Titles that make the next capitalised token a name regardless of any
#: list. This rule rescues most of the gazetteer's misses and is why the
#: detector's recall is respectable rather than poor.
TITLE_PATTERN = re.compile(
    r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")

#: Structural cues that a name is about to appear. Worth having, and
#: worth noticing that they are template-specific: they work on this
#: corpus and would need rewriting for the next hospital's note format.
FIELD_PATTERN = re.compile(
    r"(?:Patient|Name|Seen by|Follow-up with|Referred by):?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")

WORD_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")


def detect(text: str, gazetteer=None, bare_numbers: bool = False,
           unicode_names: bool = False) -> list:
    """Find identifiers in ``text``. A good-faith rule system, not a strawman.

    Patterns first, then title and field context, then the gazetteer. A
    later match overlapping an earlier one is dropped, so the order above
    is the precedence.

    ``gazetteer`` swaps the surname list, ``bare_numbers`` turns on the
    aggressive MRN rule, and ``unicode_names`` replaces the ASCII-only
    name patterns with the letter-aware ones below. All three exist so
    the demo can measure a change instead of describing one.
    """
    surnames = GAZETTEER_SURNAMES if gazetteer is None else gazetteer
    found: list = []

    def claim(start: int, end: int, category: str) -> None:
        candidate = Span(start, end, category, text[start:end])
        if any(candidate.overlaps(existing) for existing in found):
            return
        found.append(candidate)

    for category, pattern in PATTERNS:
        for match in pattern.finditer(text):
            claim(match.start(), match.end(), category)

    if bare_numbers:
        for match in BARE_NUMBER_PATTERN.finditer(text):
            claim(match.start(), match.end(), "MRN")

    if unicode_names:
        for cue in (TITLE_CUE, FIELD_CUE):
            for match in cue.finditer(text):
                bounds = name_span(text, match.end())
                if bounds is not None:
                    claim(bounds[0], bounds[1], "NAME")
        token_pattern = NAME_TOKEN
    else:
        for pattern in (TITLE_PATTERN, FIELD_PATTERN):
            for match in pattern.finditer(text):
                claim(match.start(1), match.end(1), "NAME")
        token_pattern = WORD_PATTERN

    for match in token_pattern.finditer(text):
        word = match.group()
        if word[:1].isupper() and (word in surnames or word in GAZETTEER_GIVEN):
            claim(match.start(), match.end(), "NAME")

    found.sort(key=lambda s: s.start)
    return found


def detect_without_eponyms(text: str) -> list:
    """The same rules with the eponym surnames struck off the list."""
    return detect(text, gazetteer=GAZETTEER_SURNAMES - set(EPONYM_SURNAMES))


def detect_aggressive(text: str) -> list:
    """The same rules plus ``\\b\\d{6,8}\\b``."""
    return detect(text, bare_numbers=True)


def redact(text: str, spans) -> str:
    """Replace each span with ``[CATEGORY]``. What actually leaves the building."""
    out: list = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:  # overlapping spans: keep the first
            continue
        out.append(text[cursor:span.start])
        out.append(f"[{span.category}]")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


# --- scoring ----------------------------------------------------------------


def covered(truth: Span, predicted) -> bool:
    """Is every character of ``truth`` inside some predicted span?

    Overlap is the wrong test for a leak. ``Margaret Whitfield`` caught
    as ``Margaret`` overlaps the truth and leaves the surname on the
    page. Day 11 takes this apart properly; here the strict rule is used
    because the question is only whether anything leaked.
    """
    remaining = set(range(truth.start, truth.end))
    for span in predicted:
        remaining -= set(range(span.start, span.end))
    return not remaining


def partial_matches(notes, detector=detect) -> list:
    """Truths a prediction touches without covering, and what survives.

    Written to answer a question about the miss list and immediately
    useful for a different reason: half the NAME failures in this corpus
    are not misses at all. A rule fires, redacts most of a name, and
    leaves the rest on the page. Returns ``(note_id, truth, leftover)``.
    """
    out = []
    for note in notes:
        predicted = detector(note.text)
        for truth in note.spans:
            if covered(truth, predicted):
                continue
            if not any(truth.overlaps(span) for span in predicted):
                continue  # missed outright, which is a different failure
            remaining = set(range(truth.start, truth.end))
            for span in predicted:
                remaining -= set(range(span.start, span.end))
            out.append((note.note_id, truth,
                        "".join(note.text[i] for i in sorted(remaining))))
    return out


@dataclass
class Evaluation:
    """Span-level and note-level results, which do not agree."""

    per_category: dict = field(default_factory=dict)
    recall: float = 0.0
    precision: float = 0.0
    total_truth: int = 0
    total_predicted: int = 0
    total_missed: int = 0
    false_positives: list = field(default_factory=list)
    clean_notes: int = 0
    n_notes: int = 0
    misses: list = field(default_factory=list)

    @property
    def clean_rate(self) -> float:
        return self.clean_notes / self.n_notes if self.n_notes else 0.0


def evaluate(notes, detector=detect) -> Evaluation:
    """Score a detector against the spans the generator recorded."""
    counts = {c: {"truth": 0, "found": 0} for c in CATEGORIES}
    result = Evaluation(n_notes=len(notes))
    matched_predictions = 0

    for note in notes:
        predicted = detector(note.text)
        result.total_predicted += len(predicted)
        note_is_clean = True
        for truth in note.spans:
            counts[truth.category]["truth"] += 1
            result.total_truth += 1
            if covered(truth, predicted):
                counts[truth.category]["found"] += 1
            else:
                note_is_clean = False
                result.total_missed += 1
                result.misses.append((note.note_id, truth))
        for span in predicted:
            if any(span.overlaps(truth) for truth in note.spans):
                matched_predictions += 1
            else:
                result.false_positives.append((note.note_id, span))
        if note_is_clean:
            result.clean_notes += 1

    result.per_category = {
        category: {
            "truth": data["truth"],
            "found": data["found"],
            "recall": data["found"] / data["truth"] if data["truth"] else float("nan"),
        }
        for category, data in counts.items()
    }
    result.recall = ((result.total_truth - result.total_missed)
                     / result.total_truth) if result.total_truth else 0.0
    result.precision = (matched_predictions / result.total_predicted
                        if result.total_predicted else 0.0)
    return result


def over_redaction(notes, detector=detect) -> dict:
    """Count clinical eponyms destroyed by the redactor.

    A note can pass de-identification with nothing leaked and be
    clinically useless, and no recall figure will say so.
    """
    damaged = []
    total_phrases = 0
    for note in notes:
        predicted = detector(note.text)
        for phrase in EPONYM_PHRASES:
            for match in re.finditer(re.escape(phrase), note.text):
                total_phrases += 1
                # Only a hit if the redactor touches the eponym itself,
                # not merely the words after it.
                eponym_length = len(phrase.split("'")[0].split()[0])
                eponym = Span(match.start(), match.start() + eponym_length,
                              "CLINICAL", phrase)
                if any(span.overlaps(eponym) for span in predicted):
                    damaged.append((note.note_id, phrase))
    return {
        "phrases": total_phrases,
        "damaged": len(damaged),
        "rate": len(damaged) / total_phrases if total_phrases else 0.0,
        "examples": damaged[:4],
    }


def independence_prediction(recall: float, identifiers_per_note: float) -> float:
    """``recall ** k`` - the clean rate if misses were independent.

    Named, because the comparison with the measured rate is the day's
    quietest finding: the two disagree, and the direction of the
    disagreement says the errors are systematic.
    """
    return recall ** identifiers_per_note



# --- the fix Day 10's miss list asked for ----------------------------------

#: A run of letters in any script. ``[^\W\d_]`` is a word character that
#: is neither a digit nor an underscore, and Python's ``re`` is
#: Unicode-aware, so that is exactly the set of letters. ``[a-z]`` is the
#: set of letters in one alphabet, and Day 10 shipped it.
NAME_TOKEN = re.compile(r"[^\W\d_]+")

#: The same cues as Day 10's TITLE_PATTERN and FIELD_PATTERN with the
#: capture group removed - the name itself is found by ``name_span``,
#: which can count letters that a character class cannot.
TITLE_CUE = re.compile(
    r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor)\.?[ \t]+")
FIELD_CUE = re.compile(
    r"(?:Patient|Name|Seen by|Follow-up with|Referred by):?[ \t]+")


def name_span(text: str, start: int, max_words: int = 2):
    """Extend a name of up to ``max_words`` capitalised tokens from ``start``.

    The capitalisation test is ``str.isupper`` on the first character
    rather than a character class, because a character class is where
    Day 10's bug lived. ``str`` methods know about every script; ``[A-Z]``
    knows about one, and silently truncates at the first letter outside
    it.
    """
    end = start
    for index in range(max_words):
        probe = end
        if index:
            gap = re.match(r"[ \t]+", text[end:])
            if not gap:
                break
            probe = end + gap.end()
        match = NAME_TOKEN.match(text, probe)
        if not match or not match.group()[0].isupper():
            break
        end = match.end()
    return (start, end) if end > start else None


def detect_unicode(text: str) -> list:
    """Day 10's rules with the ASCII assumption taken out of the name half."""
    return detect(text, unicode_names=True)


# --- Day 11 -----------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\S+")


def token_scores(notes, detector=detect) -> dict:
    """Token-level recall: the metric that gives partial credit.

    A PHI token is a whitespace-delimited token overlapping a truth span.
    It counts as recovered if any prediction touches it. ``Astrid
    Bergström`` caught as ``Astrid`` scores one of two.
    """
    total = found = 0
    for note in notes:
        predicted = detector(note.text)
        for match in TOKEN_PATTERN.finditer(note.text):
            token = Span(match.start(), match.end(), "TOKEN", match.group())
            if not any(token.overlaps(truth) for truth in note.spans):
                continue
            total += 1
            if any(token.overlaps(span) for span in predicted):
                found += 1
    return {"tokens": total, "found": found,
            "recall": found / total if total else 0.0}


def relaxed_scores(notes, detector=detect) -> dict:
    """Span recall where *any* overlap counts as a hit.

    Reported in the literature as "relaxed" or "partial" match, and for
    this task it is not a lenient metric - it is a wrong one. Every
    relaxed hit that is not a strict hit is a span with text still on the
    page.
    """
    total = found = 0
    for note in notes:
        predicted = detector(note.text)
        for truth in note.spans:
            total += 1
            if any(truth.overlaps(span) for span in predicted):
                found += 1
    return {"spans": total, "found": found,
            "recall": found / total if total else 0.0}


# --- how sure can you be that nothing leaked? -------------------------------


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, summed directly."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k + 1))


def leak_rate_upper_bound(leaks: int, n: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson upper bound on the leak rate, by bisection.

    The honest way to report "we found no leaks". Zero leaks in ``n``
    notes does not mean the rate is zero; it means the rate is below
    whatever this returns.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= leaks <= n:
        raise ValueError(f"leaks must be in [0, {n}], got {leaks}")
    if leaks >= n:
        return 1.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if binomial_cdf(leaks, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def rule_of_three(n: int) -> float:
    """``3 / n`` - the approximation worth carrying in your head.

    Exact for zero events is ``1 - alpha ** (1 / n)``; this is within a
    percent of it for any n worth running, which a test pins.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


# --- the evaluation set is annotated by people holding the rules ------------


def anchored_gold(notes, detector=detect, catch_probability: float = 0.5,
                  seed: int = 0) -> list:
    """A gold standard built by correcting the system's own output.

    Standard practice, and it saves real annotator time: run the rules,
    hand a human the pre-annotated text, ask them to fix it. The human
    finds each genuine miss with probability ``catch_probability`` -
    anchoring, fatigue, and the fact that an unmarked span in a marked
    document does not draw the eye.

    What comes back is not ground truth. It is the detector's output plus
    some of its errors, and scoring the detector against it is scoring it
    against a blurred copy of itself.
    """
    rng = np.random.default_rng(seed)
    gold = []
    for note in notes:
        predicted = detector(note.text)
        kept = tuple(truth for truth in note.spans
                     if covered(truth, predicted)
                     or rng.random() < catch_probability)
        gold.append(Note(note.text, kept, note.note_id))
    return gold


# --- what survives de-identification ----------------------------------------

#: Age is redacted only above 89, the year of an event may be retained,
#: and the department and the diagnosis are the clinical content - the
#: whole reason the note was released. Every one of these is legitimately
#: still there after a perfect Safe Harbor de-identification.
GENERALISATIONS = (
    ("age, year, dept, diagnosis", ("age", "year", "department", "condition")),
    ("age band, year, dept, diagnosis", ("band", "year", "department", "condition")),
    ("age band, year, diagnosis", ("band", "year", "condition")),
    ("age band, diagnosis", ("band", "condition")),
    ("diagnosis only", ("condition",)),
)


def quasi_identifier(note, fields) -> tuple:
    """The tuple an attacker joins on, from a note's retained attributes."""
    attributes = dict(note.attributes)
    attributes["band"] = (attributes["age"] // 10) * 10
    return tuple(attributes[field] for field in fields)


def k_anonymity(notes, fields) -> dict:
    """Equivalence-class sizes for one choice of retained fields.

    ``k`` is the size of the smallest class: the number of people an
    attacker cannot tell apart. ``k = 1`` means a record is alone in its
    class and the join is unambiguous.
    """
    classes = Counter(quasi_identifier(note, fields) for note in notes)
    sizes = sorted(classes.values())
    unique = sum(1 for note in notes
                 if classes[quasi_identifier(note, fields)] == 1)
    return {
        "classes": len(classes),
        "k": sizes[0],
        "median_class": sizes[len(sizes) // 2],
        "unique": unique,
        "unique_rate": unique / len(notes) if notes else 0.0,
    }


def uniqueness_by_condition(notes, fields) -> list:
    """Who is identifiable, broken down by how common their diagnosis is.

    The averaged uniqueness rate is the wrong summary for the same reason
    Day 6's accuracy was: it is dominated by the common cases, and the
    rare ones are the ones at risk.
    """
    classes = Counter(quasi_identifier(note, fields) for note in notes)
    frequency = Counter(note.attributes["condition"] for note in notes)
    per_condition = {}
    for note in notes:
        condition = note.attributes["condition"]
        bucket = per_condition.setdefault(condition, [0, 0])
        bucket[0] += 1
        if classes[quasi_identifier(note, fields)] == 1:
            bucket[1] += 1
    return sorted(
        ({"condition": condition, "patients": frequency[condition],
          "unique": found, "rate": found / total}
         for condition, (total, found) in per_condition.items()),
        key=lambda row: row["patients"],
    )


#: A diagnosis distribution with a tail, for section 6. Real cohorts are
#: never uniform, and the tail is the part that matters.
SKEWED_CONDITIONS = (0.30, 0.22, 0.16, 0.12, 0.09, 0.06, 0.035, 0.015)


# --- reused from day12 - negation and abbreviations ----------------------------------





# --- what a mention can mean ------------------------------------------------

#: The assertion classes. Two of them - PRESENT and ABSENT - are what a
#: negation detector reports. The other three are the reason a negation
#: detector is not enough: each describes a mention that is neither a
#: denial nor a statement that this patient has this thing now.
ASSERTIONS = ("PRESENT", "ABSENT", "UNCERTAIN", "FAMILY", "HISTORICAL")

#: Which assertion classes should put a patient on a diagnostic pathway.
#: UNCERTAIN belongs here: "cannot rule out pulmonary embolism" is the
#: sentence that orders the scan.
ACTIONABLE = frozenset({"PRESENT", "UNCERTAIN"})


@dataclass(frozen=True)
class Mention:
    """A concept in a sentence, and what the sentence says about it."""

    concept: str
    start: int
    end: int
    assertion: str


@dataclass
class Sentence:
    """A synthetic clinical sentence and the ground truth of its mentions."""

    text: str
    mentions: tuple = ()
    department: str = ""
    sentence_id: int = 0


CONCEPTS = (
    "pneumonia", "pulmonary embolism", "sepsis", "myocardial infarction",
    "deep vein thrombosis", "fever", "chest pain", "cough",
    "shortness of breath", "haemoptysis", "atrial fibrillation",
    "pleural effusion",
)

#: Templates, by what the sentence actually asserts. ``{c}`` is the
#: concept. The variety is the point: a rule set that handles "no" and
#: stops there handles a quarter of the ways a clinician writes a denial.
TEMPLATES = {
    "PRESENT": (
        "Patient presents with {c}.",
        "Examination confirms {c}.",
        "{c} noted on admission.",
        "Findings consistent with {c}.",
        "Treated for {c} on the ward.",
        "{c} is present and improving.",
    ),
    "ABSENT": (
        "No {c}.",
        "Denies {c}.",
        "No evidence of {c}.",
        "Patient is without {c}.",
        "Negative for {c}.",
        "{c} was ruled out.",
        "{c} has been excluded.",
        "No signs of {c} on examination.",
    ),
    "UNCERTAIN": (
        "Cannot rule out {c}.",
        "{c} cannot be excluded.",
        "Possible {c}.",
        "Query {c}.",
        "Suspicion of {c} remains.",
        "Findings may represent {c}.",
        "Concerning for {c}.",
        "Differential includes {c}.",
    ),
    "FAMILY": (
        "Father had {c}.",
        "Mother treated for {c}.",
        "Family history of {c}.",
        "Brother diagnosed with {c} at 50.",
    ),
    "HISTORICAL": (
        "History of {c}, resolved.",
        "Past medical history includes {c}.",
        "Previous episode of {c}.",
        "Prior admission for {c}.",
    ),
}

#: Sentences carrying more than one concept, with the assertion of each.
#: These are where scope goes wrong, and they are ordinary clinical
#: prose rather than adversarial constructions.
MULTI_TEMPLATES = (
    ("No {a}, {b}, or {c}.", ("ABSENT", "ABSENT", "ABSENT")),
    ("No {a} but {b} is present.", ("ABSENT", "PRESENT")),
    ("Denies {a}; reports {b}.", ("ABSENT", "PRESENT")),
    ("{a} resolved, however {b} persists.", ("HISTORICAL", "PRESENT")),
    ("No evidence of {a}, although {b} cannot be excluded.",
     ("ABSENT", "UNCERTAIN")),
    ("Family history of {a}; patient denies {b}.", ("FAMILY", "ABSENT")),
)

WARD_DEPARTMENTS = ("Cardiology", "Respiratory Medicine", "Neurology",
               "Rheumatology", "Palliative Care", "Haematology")


class _SentenceBuilder:
    """Assembles a sentence while recording where each concept landed."""

    def __init__(self) -> None:
        self._parts: list = []
        self._length = 0
        self._mentions: list = []

    def add(self, text: str, concept: str | None = None,
            assertion: str | None = None) -> "_SentenceBuilder":
        if concept is not None:
            self._mentions.append(Mention(concept, self._length,
                                          self._length + len(text), assertion))
        self._parts.append(text)
        self._length += len(text)
        return self

    def build(self, department: str, sentence_id: int) -> Sentence:
        return Sentence("".join(self._parts), tuple(self._mentions),
                        department, sentence_id)


def _fill(builder, template: str, concepts, assertions) -> None:
    """Write a template, recording the offset of each concept it inserts."""
    cursor = 0
    for index, match in enumerate(re.finditer(r"\{(\w)\}", template)):
        builder.add(template[cursor:match.start()])
        builder.add(concepts[index], concepts[index], assertions[index])
        cursor = match.end()
    builder.add(template[cursor:])


def make_sentences(n: int = 600, seed: int = 0, templates=None,
                   multi_templates=None, multi_rate: float = 0.2) -> list:
    """Generate sentences whose assertion status is recorded, not inferred.

    The class mix is roughly what chart-review studies report: a little
    over half of concept mentions in clinical text are something other
    than a plain assertion that the patient has the thing.

    ``templates`` swaps the sentence forms. ``HELDOUT_TEMPLATES`` below
    is the reason the parameter exists.
    """
    rng = np.random.default_rng(seed)
    #: PRESENT is the plurality and not the majority. This ratio is the
    #: single most important number in the file: it sets the ceiling on
    #: what a keyword search can achieve.
    weights = {"PRESENT": 0.42, "ABSENT": 0.30, "UNCERTAIN": 0.13,
               "FAMILY": 0.07, "HISTORICAL": 0.08}
    templates = TEMPLATES if templates is None else templates
    multi_templates = (MULTI_TEMPLATES if multi_templates is None
                       else multi_templates)
    classes = tuple(name for name in weights if templates.get(name))
    total = sum(weights[name] for name in classes)
    probabilities = tuple(weights[name] / total for name in classes)

    sentences = []
    for sentence_id in range(n):
        builder = _SentenceBuilder()
        department = str(rng.choice(WARD_DEPARTMENTS))
        if multi_templates and rng.random() < multi_rate:
            template, assertions = multi_templates[
                int(rng.integers(0, len(multi_templates)))]
            picked = [str(c) for c in
                      rng.choice(CONCEPTS, size=len(assertions), replace=False)]
            _fill(builder, template, picked, assertions)
        else:
            assertion = str(rng.choice(classes, p=probabilities))
            options = templates[assertion]
            template = options[int(rng.integers(0, len(options)))]
            _fill(builder, template, [str(rng.choice(CONCEPTS))], [assertion])
        sentences.append(builder.build(department, sentence_id))
    return sentences


# --- the detector -----------------------------------------------------------

#: Hedges. Neither an assertion nor a denial, and the clinically loaded
#: class: "cannot rule out pulmonary embolism" is the sentence that
#: orders the scan. Checked before negation because "cannot rule out"
#: contains "rule out", which is on the negation list.
HEDGES = (
    "cannot rule out", "can not rule out", "cannot be excluded",
    "cannot be ruled out", "possible", "probable", "query",
    "suspicion of", "suspicious for", "may represent", "concerning for",
    "differential includes", "consider",
)

#: Pre-concept negation triggers.
NEGATIONS = (
    "no evidence of", "no signs of", "no sign of", "not", "no", "denies",
    "denied", "without", "negative for", "free of", "rule out", "ruled out",
    "absence of", "resolved without",
)

#: Post-concept negation triggers. English puts the negation after the
#: concept about as often as before it, and a system that only looks
#: backwards misses every one of these.
POST_NEGATIONS = (
    "was ruled out", "is ruled out", "has been excluded", "was excluded",
    "not seen", "was negative", "is negative",
)

#: Post-concept historical cues. Added after the first run of the demo
#: put every one of its 18 errors in the same template: "cough resolved,
#: however atrial fibrillation persists". The concept comes before its
#: cue, so a backwards-looking rule sees an empty prefix and reports the
#: resolved problem as current. English marks the past after the noun as
#: readily as before it, and a rule set that only looks one way will
#: always have a class of failures shaped like this one.
POST_HISTORICAL = ("resolved", "has resolved", "now resolved",
                   "in remission", "since resolved")

FAMILY_CUES = (
    "family history of", "father", "mother", "brother", "sister",
    "sibling", "parents", "aunt", "uncle", "grandmother", "grandfather",
)

HISTORY_CUES = (
    "history of", "past medical history", "previous episode of",
    "previous", "prior admission for", "prior", "formerly", "resolved",
)

#: Tokens that end a trigger's scope. Without these, "No fever but cough
#: is present" reports the cough as denied.
TERMINATORS = (r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b",
               r"\bnevertheless\b", r"\byet\b", r";")

_TERMINATOR_PATTERN = re.compile("|".join(TERMINATORS), re.IGNORECASE)


def scope_prefix(text: str, start: int, use_terminators: bool = True) -> str:
    """The text a backwards-looking trigger is allowed to be found in.

    Everything from the last scope terminator before ``start``. Getting
    this wrong is how "No fever but cough is present" reports a denied
    cough - and ``use_terminators=False`` exists so the demo can measure
    that rather than claim it.
    """
    window = text[:start]
    if not use_terminators:
        return window
    cut = 0
    for match in _TERMINATOR_PATTERN.finditer(window):
        cut = match.end()
    return window[cut:]


def scope_suffix(text: str, end: int) -> str:
    """The text a forwards-looking trigger is allowed to be found in."""
    window = text[end:]
    match = _TERMINATOR_PATTERN.search(window)
    return window[:match.start()] if match else window


def _closest_trigger(prefix: str, groups) -> str | None:
    """The trigger nearest the concept wins; on a tie, the longest phrase.

    The tie-breaker is not decoration. ``cannot rule out`` and ``rule
    out`` end at the same character, and they mean opposite things.
    """
    lowered = prefix.lower()
    best = None
    for label, phrases in groups:
        for phrase in phrases:
            for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)",
                                     lowered):
                key = (match.end(), len(phrase))
                if best is None or key > best[0]:
                    best = (key, label)
    return best[1] if best else None


PRE_GROUPS = (("UNCERTAIN", HEDGES), ("FAMILY", FAMILY_CUES),
              ("HISTORICAL", HISTORY_CUES), ("ABSENT", NEGATIONS))


def classify(text: str, mention: Mention, use_hedges: bool = True,
             use_terminators: bool = True) -> str:
    """Assign an assertion class to one mention. NegEx, with the extras.

    ``use_hedges=False`` drops the hedge class, which is what a
    two-class negation detector does. Section 3 measures what that
    costs and where the cost lands.
    """
    groups = PRE_GROUPS if use_hedges else tuple(
        (label, phrases) for label, phrases in PRE_GROUPS
        if label != "UNCERTAIN")
    if not use_hedges:
        # Without a hedge class, "cannot rule out" falls through to the
        # negation list, which contains "rule out". The sentence that
        # orders the scan is read as the sentence that cancels it.
        groups = groups + (("ABSENT", HEDGES),)

    prefix = scope_prefix(text, mention.start, use_terminators)
    label = _closest_trigger(prefix, groups)
    if label is not None:
        return label

    # Nothing before the concept. Look after it, most specific first.
    suffix = scope_suffix(text, mention.end).lower()
    post_groups = [("ABSENT", POST_NEGATIONS)]
    if use_hedges:
        post_groups.append(("UNCERTAIN", ("cannot be excluded",
                                          "cannot be ruled out")))
    post_groups.append(("HISTORICAL", POST_HISTORICAL))
    for label, phrases in post_groups:
        for phrase in phrases:
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", suffix):
                return label
    return "PRESENT"


def keyword_present(text: str, mention: Mention) -> str:
    """The baseline: the word is there, so the patient has it."""
    del text, mention
    return "PRESENT"


# --- scoring ----------------------------------------------------------------


@dataclass
class Report:
    """Accuracy, and the two error directions that are not the same size."""

    total: int = 0
    correct: int = 0
    confusion: dict = field(default_factory=dict)
    missed_actionable: int = 0
    false_actionable: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def score(sentences, classifier) -> Report:
    """Score a classifier against the assertions the generator recorded.

    ``missed_actionable`` counts mentions that should have put a patient
    on a pathway and did not. ``false_actionable`` counts the reverse.
    They are reported separately because in this domain they are not
    interchangeable, and an accuracy figure adds them together.
    """
    report = Report()
    confusion = Counter()
    for sentence in sentences:
        for mention in sentence.mentions:
            predicted = classifier(sentence.text, mention)
            report.total += 1
            confusion[(mention.assertion, predicted)] += 1
            if predicted == mention.assertion:
                report.correct += 1
            elif mention.assertion in ACTIONABLE and predicted not in ACTIONABLE:
                report.missed_actionable += 1
            elif mention.assertion not in ACTIONABLE and predicted in ACTIONABLE:
                report.false_actionable += 1
    report.confusion = dict(confusion)
    return report


def class_counts(sentences) -> Counter:
    return Counter(mention.assertion
                   for sentence in sentences for mention in sentence.mentions)


# --- abbreviations ----------------------------------------------------------

#: Each abbreviation, the department that usually means each sense, and
#: the expansion. Every one of these collisions is real.
ABBREVIATIONS = {
    "MS": {"Cardiology": "mitral stenosis",
           "Neurology": "multiple sclerosis",
           "Palliative Care": "morphine sulfate"},
    "PT": {"Haematology": "prothrombin time",
           "Rheumatology": "physical therapy"},
    "RA": {"Cardiology": "right atrium",
           "Rheumatology": "rheumatoid arthritis",
           "Respiratory Medicine": "room air"},
    "CVA": {"Neurology": "cerebrovascular accident",
            "Haematology": "costovertebral angle"},
    "DM": {"Cardiology": "diabetes mellitus",
           "Rheumatology": "dermatomyositis"},
    "SOB": {"Respiratory Medicine": "shortness of breath"},
}

#: How often each expansion is written across the whole corpus,
#: irrespective of department. A most-frequent-sense baseline needs this
#: and nothing else, which is both its appeal and its problem.
SENSE_FREQUENCY = {
    "MS": {"multiple sclerosis": 0.45, "mitral stenosis": 0.35,
           "morphine sulfate": 0.20},
    "PT": {"prothrombin time": 0.55, "physical therapy": 0.45},
    "RA": {"rheumatoid arthritis": 0.50, "room air": 0.30,
           "right atrium": 0.20},
    "CVA": {"cerebrovascular accident": 0.80, "costovertebral angle": 0.20},
    "DM": {"diabetes mellitus": 0.90, "dermatomyositis": 0.10},
    "SOB": {"shortness of breath": 1.0},
}

ABBREVIATION_TEMPLATES = (
    "Reviewed for {a} today.",
    "{a} documented in the notes.",
    "Impression: {a}.",
    "Plan addresses {a}.",
)


@dataclass
class AbbreviationUse:
    text: str
    abbreviation: str
    department: str
    sense: str
    in_department: bool


def make_abbreviation_uses(n: int = 600, seed: int = 0,
                           cross_department: float = 0.08) -> list:
    """Sentences using an abbreviation, with the intended sense recorded.

    ``cross_department`` is the fraction written in a department that
    does not usually mean that sense - a cardiology note using MS for
    morphine sulfate at the end of life. It is small, it is real, and
    section 5 is about what it does to a department-conditioned rule.
    """
    rng = np.random.default_rng(seed)
    abbreviations = tuple(ABBREVIATIONS)
    uses = []
    for _ in range(n):
        abbreviation = str(rng.choice(abbreviations))
        senses = ABBREVIATIONS[abbreviation]
        home_departments = tuple(senses)
        department = str(rng.choice(home_departments))
        in_department = True
        if len(home_departments) > 1 and rng.random() < cross_department:
            # Same sense, written on a different ward.
            others = [d for d in home_departments if d != department]
            sense = senses[department]
            department = str(rng.choice(others))
            in_department = False
        else:
            sense = senses[department]
        template = ABBREVIATION_TEMPLATES[
            int(rng.integers(0, len(ABBREVIATION_TEMPLATES)))]
        uses.append(AbbreviationUse(template.format(a=abbreviation),
                                    abbreviation, department, sense,
                                    in_department))
    return uses


def expand_most_frequent(use: AbbreviationUse) -> str:
    """Always the commonest sense. The baseline every paper compares to."""
    senses = SENSE_FREQUENCY[use.abbreviation]
    return max(senses, key=senses.get)


def expand_by_department(use: AbbreviationUse) -> str:
    """Use the department. Works, for the notes the department describes."""
    senses = ABBREVIATIONS[use.abbreviation]
    if use.department in senses:
        return senses[use.department]
    return expand_most_frequent(use)


def score_expansion(uses, expander) -> dict:
    """Overall accuracy, split by whether the department was the usual one."""
    groups = {True: [0, 0], False: [0, 0]}
    for use in uses:
        bucket = groups[use.in_department]
        bucket[0] += 1
        if expander(use) == use.sense:
            bucket[1] += 1
    total = sum(g[0] for g in groups.values())
    correct = sum(g[1] for g in groups.values())
    return {
        "accuracy": correct / total if total else 0.0,
        "in_department": groups[True][1] / groups[True][0] if groups[True][0] else 0.0,
        "cross_department": (groups[False][1] / groups[False][0]
                             if groups[False][0] else float("nan")),
        "cross_n": groups[False][0],
        "n": total,
    }


# --- the sentences the rules were not written for ---------------------------

#: Every form here is ordinary clinical shorthand and none of it appears
#: in TEMPLATES. That is the whole construction: the rule lists above
#: were written while looking at TEMPLATES, so scoring them on TEMPLATES
#: measures the overlap between two things written by the same person on
#: the same afternoon.
#:
#: Choosing these was not neutral - a different set would give a
#: different number, and one could be chosen to give almost any number.
#: That is the finding rather than a caveat on it. Section 4 says so in
#: the demo instead of burying it here.
HELDOUT_TEMPLATES = {
    "PRESENT": (
        "{c} ongoing.",
        "Ward round: {c}.",
        "Impression is {c}.",
    ),
    "ABSENT": (
        # Forms the listed cues do reach, in sentence shapes they have
        # not seen. Real unseen text is mostly like this.
        "There is no {c} today.",
        "Denies any {c}.",
        "Negative for {c} on review.",
        "No further {c}.",
        # Forms no cue on any list reaches.
        "Nil {c}.",
        "{c}: none.",
        "Absent {c}.",
        "{c} not demonstrated.",
    ),
    "UNCERTAIN": (
        "Possible early {c}.",
        "Query underlying {c}.",
        "Concerning for {c} on imaging.",
        "{c} unlikely.",
        "Doubt {c}.",
        "?{c}",
    ),
    "FAMILY": (
        "Mother had {c} in her sixties.",
        "Family history of {c} on both sides.",
        "FHx {c}.",
        "Paternal {c}.",
    ),
    "HISTORICAL": (
        "History of {c} in childhood.",
        "Previous {c}, now stable.",
        "Longstanding {c}.",
        "{c} in 2019.",
    ),
}


def make_heldout_sentences(n: int = 600, seed: int = 1) -> list:
    """Sentences in forms the rule lists do not carry. Single concept only."""
    return make_sentences(n, seed=seed, templates=HELDOUT_TEMPLATES,
                          multi_templates=(), multi_rate=0.0)


def per_class_accuracy(sentences, classifier) -> dict:
    """Accuracy split by the true class, because the average hides which."""
    counts = {}
    for sentence in sentences:
        for mention in sentence.mentions:
            bucket = counts.setdefault(mention.assertion, [0, 0])
            bucket[0] += 1
            if classifier(sentence.text, mention) == mention.assertion:
                bucket[1] += 1
    return {name: {"n": total, "correct": right, "accuracy": right / total}
            for name, (total, right) in counts.items()}


# --- and the two halves are not independent of each other -------------------


@dataclass
class CombinedCase:
    """A sentence where the concept is written as an abbreviation.

    Both stages have to be right. The abbreviation has to expand to the
    sense the clinician meant, and the sentence has to be read for
    whether the patient has it.
    """

    text: str
    abbreviation: str
    department: str
    sense: str
    assertion: str
    mention: Mention


def make_combined(n: int = 600, seed: int = 2,
                  cross_department: float = 0.08) -> list:
    """Held-out sentence forms with an abbreviation in the concept slot."""
    rng = np.random.default_rng(seed)
    abbreviations = tuple(ABBREVIATIONS)
    classes = tuple(name for name in HELDOUT_TEMPLATES if HELDOUT_TEMPLATES[name])
    cases = []
    for _ in range(n):
        abbreviation = str(rng.choice(abbreviations))
        senses = ABBREVIATIONS[abbreviation]
        home = tuple(senses)
        department = str(rng.choice(home))
        if len(home) > 1 and rng.random() < cross_department:
            sense = senses[department]
            department = str(rng.choice([d for d in home if d != department]))
        else:
            sense = senses[department]

        assertion = str(rng.choice(classes))
        options = HELDOUT_TEMPLATES[assertion]
        template = options[int(rng.integers(0, len(options)))]
        builder = _SentenceBuilder()
        _fill(builder, template, [abbreviation], [assertion])
        sentence = builder.build(department, 0)
        cases.append(CombinedCase(sentence.text, abbreviation, department,
                                  sense, assertion, sentence.mentions[0]))
    return cases


def score_pipeline(cases, expander=None, classifier=None) -> dict:
    """Each stage on its own, and both of them together.

    The joint figure is the one a clinician would care about, and it is
    the one neither stage reports.
    """
    expander = expand_by_department if expander is None else expander
    classifier = classify if classifier is None else classifier
    expansion_right = assertion_right = both_right = 0
    for case in cases:
        use = AbbreviationUse(case.text, case.abbreviation, case.department,
                              case.sense, True)
        expanded = expander(use) == case.sense
        asserted = classifier(case.text, case.mention) == case.assertion
        expansion_right += expanded
        assertion_right += asserted
        both_right += expanded and asserted
    n = len(cases)
    expansion = expansion_right / n
    assertion = assertion_right / n
    return {
        "n": n,
        "expansion": expansion,
        "assertion": assertion,
        "product": expansion * assertion,
        "joint": both_right / n,
    }


# --- reused from day13 to day16 - indexing, retrieval, groundedness, failure modes ---


# --- reused from day13, day14 and day15 -------------------------------------


# --- reused from day13 and day14 --------------------------------------------


# --- reused from day13 ------------------------------------------------------





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


def make_guideline(seed: int = 0, scale: int = 1, populations=None
                   ) -> Guideline:
    """Build the synthetic guideline and record its structure exactly.

    Offsets are recorded as the document is written. Recovering them
    afterwards by searching would be the same mistake Day 10 avoided,
    and here it would be worse: recommendation text repeats.

    populations swaps the eligibility clauses, which is how Day 16
    builds the superseded edition: the same document with the numeric
    thresholds moved, because that is what a guideline revision usually
    is.
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
            population = str(rng.choice(POPULATIONS if populations is None
                                        else populations))
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



# --- Day 16 -----------------------------------------------------------------

#: The 2019 edition's thresholds. Same clauses, different numbers - which
#: is what a guideline revision usually is. Both editions are in the
#: index, because both are in the hospital's document store, because
#: nobody deletes the old one.
SUPERSEDED_POPULATIONS = (
    "In adults aged 21 and over with newly diagnosed stage 1 disease",
    "In adults with stage 2 disease and no evidence of end-organ damage",
    "In adults with stage 2 disease and established end-organ damage",
    "In adults aged 80 and over",
    "In adults with an estimated GFR above 45 mL/min",
    "In adults with an estimated GFR below 15 mL/min",
    "In adults who have not tolerated first-line therapy",
    "In adults with concurrent stage 3 disease",
)

CURRENT_LABEL = "SG-1 (2024)"
SUPERSEDED_LABEL = "SG-1 (2019, superseded)"


def tag_edition(chunks, label: str) -> list:
    """Prefix each chunk's section path with the edition it came from."""
    return [Chunk(c.text, c.start, c.end, f"{label} {c.section_path}")
            for c in chunks]


@dataclass
class Library:
    """Two editions of one guideline, indexed together.

    Nothing about this is contrived. A document store contains what has
    been put in it, and withdrawing a superseded guideline is an
    administrative act that happens later than the revision does.
    """

    current: Guideline
    superseded: Guideline
    chunks: tuple = ()
    index: object = None

    def guideline_for(self, path: str):
        if path.startswith(SUPERSEDED_LABEL):
            return self.superseded
        return self.current

    def is_stale(self, path: str) -> bool:
        return path.startswith(SUPERSEDED_LABEL)


def make_library(seed: int = 0, superseded_first: bool = False) -> Library:
    """Both editions in one index.

    ``superseded_first`` changes nothing except the order the chunks are
    added in. Section 3 is about what that does, which should be
    nothing and is not.
    """
    current = make_guideline(seed=seed)
    superseded = make_guideline(seed=seed, populations=SUPERSEDED_POPULATIONS)
    new = tag_edition(section_chunks(current), CURRENT_LABEL)
    old = tag_edition(section_chunks(superseded), SUPERSEDED_LABEL)
    chunks = old + new if superseded_first else new + old
    library = Library(current, superseded, tuple(chunks))
    library.index = BM25(chunks)
    return library


# --- a generator, with no model in it ---------------------------------------


@dataclass(frozen=True)
class Generated:
    """What the pipeline hands back."""

    question: "Probe"
    text: str
    cited_path: str
    rec_id: str | None
    refused: bool = False
    refusal_reason: str = ""


def best_recommendation(chunk, guideline: Guideline, question: str):
    """The recommendation in the chunk sharing most terms with the question."""
    terms = set(tokenize(question))
    candidates = recommendations_in(chunk, guideline)
    if not candidates:
        return None
    return max(candidates, key=lambda r: (
        len(terms & set(tokenize(r.population + " " + r.action))), r.rec_id))


def compose(question: "Probe", chunk, library: Library) -> str:
    """Extractive, template-driven, and deliberately not a language model.

    Every word of the output is lifted from the cited chunk. That is the
    point: this generator cannot hallucinate a word, and the day is
    about the failures that survive anyway. A fluent model would add
    failures to these, not replace them.
    """
    guideline = library.guideline_for(chunk.section_path)
    recommendation = best_recommendation(chunk, guideline, question.text)
    if recommendation is None:
        # A chunk with no recommendation in it. The first version
        # emitted "The guideline does not state." here, which is a
        # sentence this generator wrote rather than one it quoted - and
        # a test that every emitted word comes from the source caught
        # it. Semantically it was always a refusal, so it is one now.
        return None
    return (f"{recommendation.population}, {recommendation.action}. "
            f"[Strength: {recommendation.strength} - "
            f"Evidence: {recommendation.evidence}]")


# --- the questions, by what is wrong with answering them --------------------

PROBE_KINDS = ("IN_SCOPE", "OUT_OF_SCOPE", "EXCEPTION")


@dataclass(frozen=True)
class Probe:
    """A question, and what a correct system would have to do with it."""

    text: str
    kind: str
    rec_id: str | None = None


def make_probes(library: Library) -> list:
    """One probe per failure mode, over the recommendations that carry them."""
    probes = []
    current = library.current

    for recommendation in current.recommendations:
        terms = sorted(dict.fromkeys(tokenize(recommendation.action)),
                       key=lambda t: (-library.index.idf.get(t, 0.0), t))[:4]
        probes.append(Probe(" ".join(terms) + "?", "IN_SCOPE",
                            recommendation.rec_id))

    for _label, text in OUT_OF_SCOPE:
        probes.append(Probe(text, "OUT_OF_SCOPE"))

    # Questions whose answer is reversed two chapters later.
    for exception in current.exceptions:
        recommendation = current.recommendation(exception.rec_id)
        if recommendation is None:
            continue
        probes.append(Probe(
            f"{recommendation.population}, what does the guideline "
            f"recommend about {recommendation.action}?",
            "EXCEPTION", recommendation.rec_id))
    return probes


# --- guards -----------------------------------------------------------------


def run(probe: Probe, library: Library, refuse_below: float = 0.0,
        require_grounded: bool = False) -> Generated:
    """Retrieve, decide whether to answer, compose, and check the answer."""
    ranked = library.index.rank(probe.text, k=5)
    if not ranked:
        return Generated(probe, "", "", None, True, "nothing retrieved")
    chunks = tuple(chunk for chunk, _ in ranked)
    scores = tuple(score for _, score in ranked)
    retrieval = Retrieval(Question(probe.text, probe.rec_id, "long"),
                          chunks, scores)

    if retrieval.score_per_term < refuse_below:
        return Generated(probe, "", "", None, True, "below the score threshold")

    chunk = chunks[0]
    guideline = library.guideline_for(chunk.section_path)
    text = compose(probe, chunk, library)
    if text is None:
        return Generated(probe, "", chunk.section_path, None, True,
                         "cited section states no recommendation")
    recommendation = best_recommendation(chunk, guideline, probe.text)
    rec_id = recommendation.rec_id if recommendation else None

    if require_grounded:
        for claim in split_claims(text):
            if not strict_support(claim.text, chunk, guideline).supported:
                return Generated(probe, "", chunk.section_path, rec_id, True,
                                 "answer not grounded in the cited chunk")
    return Generated(probe, text, chunk.section_path, rec_id)


def is_harmful(answer: Generated, library: Library) -> bool:
    """Would delivering this answer mislead the clinician who asked?

    Refusals are never harmful here, only costly. The categories are
    judged on what the answer does, not on how it scored.
    """
    if answer.refused:
        return False
    kind = answer.question.kind
    # A citation of the withdrawn edition is harmful whatever was
    # asked. The first version only counted it for questions written to
    # provoke it, which measured the questions rather than the index.
    if library.is_stale(answer.cited_path):
        return True
    if kind == "OUT_OF_SCOPE":
        return True
    if kind == "EXCEPTION":
        # The exception lives in section 4 and the answer cites section
        # 3. Delivering it without the reversal is the harm.
        return "4." not in answer.cited_path
    return False


@dataclass
class GuardReport:
    label: str = ""
    delivered: int = 0
    refused: int = 0
    harmful: int = 0
    refused_in_scope: int = 0
    stale: int = 0
    per_kind: dict = field(default_factory=dict)


def evaluate_guards(probes, library: Library, refuse_below: float,
                    require_grounded: bool, label: str) -> GuardReport:
    report = GuardReport(label=label)
    for probe in probes:
        answer = run(probe, library, refuse_below, require_grounded)
        bucket = report.per_kind.setdefault(
            probe.kind, {"n": 0, "delivered": 0, "harmful": 0})
        bucket["n"] += 1
        if answer.refused:
            report.refused += 1
            if probe.kind == "IN_SCOPE":
                report.refused_in_scope += 1
        else:
            report.delivered += 1
            bucket["delivered"] += 1
            report.stale += library.is_stale(answer.cited_path)
        if is_harmful(answer, library):
            report.harmful += 1
            bucket["harmful"] += 1
    return report


# --- Day 17: the pipeline ---------------------------------------------------

#: Frames that turn a guideline eligibility clause into a line of a
#: clinical note, one per assertion class. The wording of the finding is
#: the guideline's own, because the point being measured here is the
#: pipeline and not a paraphrase problem - and a real note would phrase
#: it differently, which is one more thing this corpus cannot tell you.
CASE_FRAMES = {
    "PRESENT": "Criteria met: {finding}.",
    "ABSENT": "Criteria not met: {finding}.",
    "UNCERTAIN": "Possible criteria: {finding}.",
    "HISTORICAL": "Previous assessment: {finding}.",
}

#: The same four assertions in phrasings the Day 12 rules were not
#: written for. Half the notes use these, because a corpus written
#: entirely in the forms the rules already handle measures the overlap
#: between two things I wrote rather than the pipeline.
HELD_OUT_CASE_FRAMES = {
    "PRESENT": "{finding}, confirmed on review.",
    "ABSENT": "Nil {finding}.",
    "UNCERTAIN": "{finding} unlikely.",
    "HISTORICAL": "Longstanding {finding}.",
}

CASE_ASSERTIONS = tuple(CASE_FRAMES)

#: Criteria a clinician might reasonably record and this guideline does
#: not cover. The pipeline's correct response to one of these is a
#: refusal, which is the only stage where refusing is the right answer.
OUT_OF_SCOPE_FINDINGS = (
    # Day 14's easy case: no term appears in the document, every chunk
    # scores zero, and refusing is trivial.
    "severe hepatic impairment",
    "active malignancy under treatment",
    "documented penicillin anaphylaxis",
    # Day 14's hard case: the document's own vocabulary, describing
    # something it does not cover. "stage 4" is outside a guideline that
    # stages 1 to 3, and a GFR below 15 is the threshold the 2019
    # edition used and the 2024 one does not.
    "stage 4 disease with established end-organ damage",
    "an estimated GFR below 15 mL/min",
)


def finding_of(population: str) -> str:
    """The eligibility clause as a clinician would write it in a note."""
    return population[len("In adults "):] if population.startswith("In adults ") \
        else population


@dataclass(frozen=True)
class Case:
    """A note, the identifiers in it, and what the guideline should be asked."""

    text: str
    phi: tuple
    population: str
    finding: str
    assertion: str
    finding_start: int
    finding_end: int
    #: Does the guideline cover this criterion at all? When it does not,
    #: the right outcome is a refusal rather than an answer.
    in_guideline: bool = True
    #: Was the sentence written in a form Day 12's rules were built
    #: against, or one they were not?
    held_out: bool = False

    @property
    def actionable(self) -> bool:
        """Should this note produce a question at all?

        ABSENT and HISTORICAL findings must not. A pipeline that asks
        the guideline what to do about a criterion the note says is not
        met has already made the mistake Day 12 was about.
        """
        return self.assertion in ACTIONABLE


def make_cases(library: Library, n: int = 200, seed: int = 0,
               held_out_rate: float = 0.5,
               out_of_scope_rate: float = 0.2) -> list:
    """Notes carrying both identifiers and one clinical finding.

    ``held_out_rate`` is the fraction written in phrasings Day 12's
    rules were not built against, and ``out_of_scope_rate`` the fraction
    whose finding this guideline does not cover. Both default to
    something other than zero on purpose: with both at zero every stage
    below scores 1.000 and the table says nothing.
    """
    rng = np.random.default_rng(seed)
    populations = sorted({r.population for r in library.current.recommendations})
    cases = []
    for _ in range(n):
        builder = _NoteBuilder()
        given = str(rng.choice(GIVEN_NAMES))
        surname = str(rng.choice(SURNAMES))
        clinician = str(rng.choice(SURNAMES))
        age = int(rng.integers(24, 96))
        department = str(rng.choice(NOTE_DEPARTMENTS))
        assertion = str(rng.choice(CASE_ASSERTIONS))
        in_guideline = rng.random() >= out_of_scope_rate
        if in_guideline:
            population = str(rng.choice(populations))
            finding = finding_of(population)
        else:
            population = ""
            finding = str(rng.choice(OUT_OF_SCOPE_FINDINGS))
        held_out = rng.random() < held_out_rate

        builder.add(f"{department} - Clinic Letter\n\n")
        builder.add("Patient: ")
        builder.add(f"{given} {surname}", "NAME")
        builder.add("   ")
        builder.add(_render_mrn(rng), "MRN")
        builder.add("\nDate of review: ")
        builder.add(_render_date(rng, int(rng.integers(1, 29)),
                                 int(rng.integers(1, 13)),
                                 int(rng.integers(2019, 2025))), "DATE")
        builder.add("\n\nThis ")
        builder.add(f"{age}-year-old", "AGE" if age > 89 else None)
        builder.add(" patient was reviewed by Dr ")
        builder.add(clinician, "NAME")
        builder.add(".\n")

        frames = HELD_OUT_CASE_FRAMES if held_out else CASE_FRAMES
        prefix, suffix = frames[assertion].split("{finding}")
        if not prefix:
            # The finding opens the sentence, so it is capitalised - and
            # the stored finding has to be the string that is actually in
            # the note, or every offset downstream is looking for text
            # that is not there.
            finding = finding[0].upper() + finding[1:]
        builder.add(prefix)
        builder.add(finding)
        builder.add(suffix + "\n")

        builder.add("\nContact: ")
        builder.add(_render_phone(rng), "PHONE")
        builder.add("\n")

        note = builder.build(len(cases))
        # Located by searching, which is safe here and only here: the
        # finding is a long distinctive clause occurring once in the
        # note. A test asserts that, because Day 10's generator went out
        # of its way to avoid exactly this and had good reason to.
        start = note.text.index(finding)
        cases.append(Case(note.text, note.spans, population, finding,
                          assertion, start, start + len(finding),
                          in_guideline, held_out))
    return cases


@dataclass
class Outcome:
    """What each stage did with one case, and whether it was right."""

    case: Case
    redacted: str = ""
    deid_ok: bool = False
    predicted_assertion: str = ""
    assertion_ok: bool = False
    queried: bool = False
    query_ok: bool = False
    answer: object = None
    retrieval_ok: bool = False
    verified_ok: bool = False

    @property
    def end_to_end(self) -> bool:
        return (self.deid_ok and self.assertion_ok and self.query_ok
                and self.retrieval_ok and self.verified_ok)


def run_case(case: Case, library: Library, refuse_below: float = 0.9) -> Outcome:
    """One note through all four stages.

    Every stage after the first works on the **de-identified** text, not
    the original. That is the only order the pipeline is allowed to run
    in, and it means a de-identification bug is also a retrieval bug.
    """
    outcome = Outcome(case=case)

    # 1. De-identify.
    predicted = detect_unicode(case.text)
    outcome.deid_ok = all(covered(span, predicted) for span in case.phi)
    outcome.redacted = redact(case.text, predicted)

    # 2. Read the assertion, in the redacted text.
    start = outcome.redacted.find(case.finding)
    if start < 0:
        # Redaction ate the finding. Nothing downstream can recover.
        outcome.predicted_assertion = "LOST"
        return outcome
    mention = Mention(case.finding, start, start + len(case.finding), "")
    outcome.predicted_assertion = classify(outcome.redacted, mention)
    outcome.assertion_ok = outcome.predicted_assertion == case.assertion

    # 3. Ask, but only if the note says the criterion applies.
    outcome.queried = outcome.predicted_assertion in ACTIONABLE
    outcome.query_ok = outcome.queried == case.actionable
    if not outcome.queried:
        # A refusal to ask is correct when the criterion is not met, and
        # there is nothing left to get wrong.
        outcome.retrieval_ok = outcome.verified_ok = not case.actionable
        return outcome

    probe = Probe(case.finding, "IN_SCOPE" if case.in_guideline
                  else "OUT_OF_SCOPE")
    answer = run(probe, library, refuse_below=refuse_below,
                 require_grounded=True)
    outcome.answer = answer
    if answer.refused:
        # Refusing is right when the guideline does not cover the
        # criterion, and wrong when it does.
        outcome.retrieval_ok = outcome.verified_ok = not case.in_guideline
        return outcome
    if not case.in_guideline:
        return outcome

    # 4. Did it cite a current-edition recommendation for this patient?
    chunk = chunk_for(library.chunks, answer.cited_path)
    guideline = library.guideline_for(answer.cited_path)
    outcome.retrieval_ok = (
        not library.is_stale(answer.cited_path)
        and any(r.population == case.population
                for r in recommendations_in(chunk, guideline)))
    outcome.verified_ok = all(
        strict_support(claim.text, chunk, guideline).supported
        for claim in split_claims(answer.text))
    return outcome


@dataclass
class StageReport:
    """Per-stage accuracy, and the product nobody reports."""

    n: int = 0
    deid: int = 0
    assertion: int = 0
    query: int = 0
    retrieval: int = 0
    verified: int = 0
    end_to_end: int = 0

    def rate(self, stage: str) -> float:
        return getattr(self, stage) / self.n if self.n else 0.0

    @property
    def product(self) -> float:
        return (self.rate("deid") * self.rate("assertion")
                * self.rate("query") * self.rate("retrieval")
                * self.rate("verified"))


def run_all(cases, library: Library, refuse_below: float = 0.9):
    outcomes = [run_case(case, library, refuse_below) for case in cases]
    report = StageReport(n=len(outcomes))
    for outcome in outcomes:
        report.deid += outcome.deid_ok
        report.assertion += outcome.assertion_ok
        report.query += outcome.query_ok
        report.retrieval += outcome.retrieval_ok
        report.verified += outcome.verified_ok
        report.end_to_end += outcome.end_to_end
    return outcomes, report


# --- the CLI ----------------------------------------------------------------

BANNER = ("NOT A MEDICAL DEVICE. Every note, every guideline and every "
          "evidence\ngrade in this repo is invented by it.")

STAGES = (
    ("de-identify the note", "deid",
     "Day 11's letter-aware detector, note-level"),
    ("read the assertion", "assertion", "Day 12's negation rules"),
    ("ask, or decline to ask", "query", "actionable findings only"),
    ("retrieve and refuse", "retrieval", "Day 14's score per term"),
    ("check the answer", "verified", "Day 15's claim checks"),
)


def command_note(library: Library) -> None:
    """One case through every stage, printed."""
    cases = make_cases(library, n=40, seed=3)
    case = next(c for c in cases if c.in_guideline and c.actionable)
    outcome = run_case(case, library)
    print("the note as written")
    for line in case.text.rstrip().splitlines():
        print(f"  | {line}")
    print()
    print("1. de-identified")
    for line in outcome.redacted.rstrip().splitlines():
        print(f"  | {line}")
    print(f"     every identifier caught: {outcome.deid_ok}")
    print()
    print("2. the assertion, read from the redacted text")
    print(f"     finding   {case.finding}")
    print(f"     recorded  {case.assertion}")
    print(f"     read as   {outcome.predicted_assertion}")
    print()
    print("3. ask the guideline?")
    print(f"     the note says the criterion applies: {case.actionable}")
    print(f"     the pipeline asked:                  {outcome.queried}")
    print()
    if outcome.answer is not None and not outcome.answer.refused:
        print("4. the answer")
        print(f"     {outcome.answer.text}")
        print(f"     {outcome.answer.cited_path} [{outcome.answer.rec_id}]")
        print(f"     every claim traced to that chunk: {outcome.verified_ok}")
    elif outcome.answer is not None:
        print("4. refused")
        print(f"     {outcome.answer.refusal_reason}")
    print()
    print(f"end to end correct: {outcome.end_to_end}")


def command_run(library: Library, n: int = 300) -> None:
    """The stage table, and the number nobody reports."""
    cases = make_cases(library, n=n, seed=0)
    _outcomes, report = run_all(cases, library)
    print(f"{report.n} notes, {sum(c.held_out for c in cases)} written in")
    print(f"phrasings Day 12's rules were not built against, "
          f"{sum(not c.in_guideline for c in cases)} with a")
    print("finding this guideline does not cover.")
    print()
    print(f"  {'stage':<26}{'correct':>9}   what it is")
    for label, key, note in STAGES:
        print(f"  {label:<26}{report.rate(key):>9.3f}   {note}")
    print()
    print(f"  {'product of the five':<26}{report.product:>9.3f}")
    print(f"  {'measured end to end':<26}"
          f"{report.rate('end_to_end'):>9.3f}")
    print()
    print("  the product is the number you get by assuming the stages fail")
    print("  independently, and they do not: a note whose identifiers leak")
    print("  can still have its assertion read correctly, and a misread")
    print("  assertion usually takes the query decision down with it. So")
    print("  the measurement is better than the model, and both are far")
    print("  below every stage in the table above them.")
    print()
    print("  no stage's report contains the bottom line. Five teams could")
    print("  each publish their column and every one of them would be")
    print("  reporting honestly.")


def command_editions(library: Library, n: int = 300) -> None:
    """Two ways to end up quoting a withdrawn guideline."""
    print(f"  {'insertion order':<26}{'end to end':>12}{'stale citations':>18}")
    rows = {}
    for superseded_first in (False, True):
        shelf = make_library(seed=0, superseded_first=superseded_first)
        cases = make_cases(shelf, n=n, seed=0)
        outcomes, report = run_all(cases, shelf)
        stale = [o for o in outcomes
                 if o.answer is not None and not o.answer.refused
                 and shelf.is_stale(o.answer.cited_path)]
        order = ("superseded added first" if superseded_first
                 else "current added first")
        rows[superseded_first] = stale
        print(f"  {order:<26}{report.rate('end_to_end'):>12.3f}{len(stale):>18}")
    print()
    findings = Counter(o.case.finding for o in rows[False])
    print("  the top row is the correct configuration and it is not zero.")
    for finding, count in findings.most_common():
        print(f"    {count} notes recording {finding!r}")
    print()
    print("  that is not a tie broken badly. The 2019 edition is the better")
    print("  lexical match for those notes, because the threshold written")
    print("  in them is the one the 2019 edition used. A clinician who")
    print("  records the old criterion gets the old guideline back, and the")
    print("  retriever is working exactly as specified while it happens.")
    print()
    shared = {finding_of(p).lower()
              for p in set(POPULATIONS) & set(SUPERSEDED_POPULATIONS)}
    moved = [o for o in rows[True] if o.case.finding.lower() in shared]
    print("  the bottom row is Day 16's finding arriving end to end: the")
    print("  same notes, the same guards, the same scores, and the chunks")
    print("  added to the index in the other order.")
    print()
    print(f"  it is sharper here than it was on Day 16. {len(moved)} of the")
    print(f"  {len(rows[True])} answers that moved are notes recording one of "
          f"the {len(shared)}")
    print("  criteria the revision did not change - word for word the same")
    print("  in both editions, so there is nothing to choose between them")
    print("  and insertion order decides.")
    print()
    print("  where the note names a threshold the revision DID move, the")
    print("  current edition is the better lexical match and wins on merit.")
    print("  The retriever gets the right answer for a reason that has")
    print("  nothing to do with the date, on the questions where the date")
    print("  happens to be encoded in the words.")
    print()
    print("  a pipeline that de-identifies correctly, reads the negation")
    print("  correctly, retrieves correctly and verifies every claim -")
    print("  quoting a guideline that was withdrawn.")


def command_deid(text: str) -> None:
    spans = detect_unicode(text)
    print(redact(text, spans))
    print()
    print(f"  {len(spans)} identifiers removed")
    for span in spans:
        print(f"    {span.category:<9}{span.text!r}")


def command_ask(library: Library, text: str, refuse_below: float = 0.9) -> None:
    answer = run(Probe(text, "IN_SCOPE"), library, refuse_below=refuse_below,
                 require_grounded=True)
    print(f"Q  {text}")
    if answer.refused:
        print(f"A  (refused: {answer.refusal_reason})")
        return
    print(f"A  {answer.text}")
    print(f"   {answer.cited_path} [{answer.rec_id}]")
    if library.is_stale(answer.cited_path):
        print("   WARNING: that edition has been superseded.")


def command_writeup(library: Library) -> None:
    """What seventeen days established, including the unflattering parts."""
    print("what this repo measured")
    print("=" * 74)
    rows = (
        ("1", "A synthetic ECG, so a missed R peak is unambiguous"),
        ("2", "Mains hum beat motion artefact at equal SNR, which is the "
              "opposite of\n     what this file first claimed"),
        ("3", "A moving average delays a peak by (w-1)/2 and destroys it "
              "before it\n     stops delaying it"),
        ("4", "R-peak detection and HRV, scored against peaks the "
              "generator placed"),
        ("5", "Forty honest studies gave age odds ratios from 0.66 to 3.90"),
        ("6", "At a 0.9% event rate the model scored 99.1% and found "
              "nothing"),
        ("7", "Every threshold is an unstated cost ratio, and it moves the "
              "cut\n     further than any model change does"),
        ("8", "Four distortions, one AUC to four decimals, a hundredfold "
              "range of ECE"),
        ("9", "Imputation invents values in the one direction the data "
              "cannot reveal"),
        ("10", "94.9% of identifiers caught, 64.7% of notes clean"),
        ("11", "Fixing a bug that leaked surnames changed relaxed and token "
               "recall\n     by 0.0000 each, to four decimal places"),
        ("12", "The negation rules score 1.000 on the sentences they were "
               "written\n     for and 0.730 on sentences they were not"),
        ("13", "No chunking puts a recommendation in the same chunk as the "
               "section\n     that reverses it"),
        ("14", "BM25's score is mostly a measurement of how long the "
               "question was"),
        ("15", "A negated answer scores 0.959 on word overlap; traceable "
               "and correct\n     are different properties"),
        ("16", "Flipping the order two editions were indexed moved 26 of 28 "
               "answers\n     onto a withdrawn guideline, with no score "
               "changing"),
    )
    for day, finding in rows:
        print(f"  Day {day:<3} {finding}")

    print()
    print("what the whole thing does")
    print("=" * 74)
    cases = make_cases(library, n=300, seed=0)
    _outcomes, report = run_all(cases, library)
    for label, key, _note in STAGES:
        print(f"  {label:<26}{report.rate(key):>9.3f}")
    print(f"  {'end to end':<26}{report.rate('end_to_end'):>9.3f}")

    print()
    print("what it cannot do")
    print("=" * 74)
    print("  - every measurement in this repo was taken on data this repo")
    print("    generated. The de-identifier was scored against spans this")
    print("    file wrote down, the negation rules against sentences this")
    print("    file wrote, and the retriever against a guideline this file")
    print("    invented. Nothing here has met a real note.")
    print()
    print("  - the generator is extractive. It cannot paraphrase, so it")
    print("    cannot hallucinate, so Day 15's checker never fires. Put a")
    print("    language model in its place and the checker starts earning")
    print("    its keep - against failures this pipeline cannot currently")
    print("    commit, and on top of every failure it already does.")
    print()
    print("  - the two guards do not see a date, a scope declaration or a")
    print("    cross-reference, because the text does not carry them. Day")
    print("    13 listed what an index would have to hold and this")
    print("    pipeline holds none of it.")
    print()
    print("  - the numbers above are choosable. Half the notes use")
    print("    phrasings the negation rules were not built for, and that")
    print("    fraction is a parameter of make_cases. Set it to zero and")
    print("    the assertion stage reports 1.000.")
    print()
    print("  - and it is not a medical device. Using any of this on a")
    print("    patient would need regulatory approval, clinical validation")
    print("    and expertise this repo does not contain.")


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print(BANNER)
    print()
    command = argv[1] if len(argv) > 1 else "run"
    argument = " ".join(argv[2:]) if len(argv) > 2 else ""
    library = make_library(seed=0)

    if command == "note":
        command_note(library)
    elif command == "run":
        command_run(library)
        print()
        print("=" * 74)
        command_editions(library)
    elif command == "deid":
        command_deid(argument or
                     "Patient: Astrid Bergström   MRN 4482910\n"
                     "Reviewed on 14 March 2024 by Dr Okonkwo.")
    elif command == "ask":
        command_ask(library, argument or "newly diagnosed stage 1 disease")
    elif command == "writeup":
        command_writeup(library)
    else:
        print(f"unknown command {command!r}")
        print("  note | run | deid <text> | ask <text> | writeup")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
