"""Day 11 - Evaluating de-identification when a single leak is a failure.

Day 10 built a detector and scored it. This day asks whether the score
meant anything, and finds four reasons to doubt it.

**The metric you pick moves the number more than the detector does.**
Strict span matching, token matching and relaxed (any-overlap) matching
score the same output three different ways. Relaxed matching is the one
that flatters, and for this task it is not lenient - it is wrong. Every
relaxed hit that is not a strict hit is a surname still on the page.

**"We found no leaks" is not a rate.** Zero leaks in n notes bounds the
leak rate at roughly 3/n, and no lower. A flawless audit of 300 notes is
consistent with one note in a hundred leaking.

**The gold standard is built by correcting the system's own output.**
That is standard practice and it saves real annotator time, and what
comes back is the detector plus some of its errors. Scoring against it
inflates the note-level rate far more than the span-level one - the
metric that matters is the metric distorted most.

**And the identifiers were never the whole problem.** Age, year of
admission, department and diagnosis all legitimately survive a perfect
Safe Harbor de-identification, and 98% of these patients are unique on
that tuple alone. Getting to k >= 5 means throwing away the year and the
department, and the patients with the rarest diagnosis stay identifiable
after everyone else is safe.

NOT A MEDICAL DEVICE. Every note in this repo is invented by it. No
patient data, real or derived, appears anywhere.
"""

# --- reused from day10 ------------------------------------------------------


from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

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


class _Builder:
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

    def add(self, text: str, category: str | None = None) -> "_Builder":
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

DEPARTMENTS = ("Cardiology", "Respiratory Medicine", "Nephrology",
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
        builder = _Builder()
        given = str(rng.choice(GIVEN_NAMES))
        surname = str(rng.choice(SURNAMES))
        clinician = str(rng.choice(SURNAMES))
        age = int(rng.integers(24, 96))
        condition = str(rng.choice(CONDITIONS) if condition_weights is None
                        else rng.choice(CONDITIONS, p=condition_weights))
        medication = str(rng.choice(MEDICATIONS))
        department = str(rng.choice(DEPARTMENTS))

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


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. Every note here is invented by this file.")

    notes = make_notes(n=300, seed=0)
    strict = evaluate(notes)
    print()
    print(f"corpus {len(notes)} notes, {strict.total_truth} identifiers")
    print(f"Day 10's detector, unchanged: strict recall {strict.recall:.3f}, "
          f"{strict.clean_notes}/{strict.n_notes} notes clean")

    print()
    print("=" * 74)
    print("1. one output, three metrics, three numbers")
    print("=" * 74)
    token = token_scores(notes)
    relaxed = relaxed_scores(notes)
    print(f"  {'metric':<34}{'denominator':>13}{'recall':>9}")
    print(f"  {'token, any overlap':<34}{token['tokens']:>13}"
          f"{token['recall']:>9.3f}")
    print(f"  {'span, any overlap (relaxed)':<34}{relaxed['spans']:>13}"
          f"{relaxed['recall']:>9.3f}")
    print(f"  {'span, fully covered (strict)':<34}{strict.total_truth:>13}"
          f"{strict.recall:>9.3f}")
    print()
    print("  the detector did not change between those rows. Nothing about")
    print("  the system moved; only the question moved.")

    print()
    print("=" * 74)
    print("2. partial credit, for a task where there is no such thing")
    print("=" * 74)
    partials = partial_matches(notes)
    print("  spans counted as hits by relaxed matching and missed by")
    print(f"  strict matching: {len(partials)}")
    print()
    print(f"  {'truth':<28}{'what survives redaction':<28}")
    for _note_id, truth, leftover in partials[:6]:
        print(f"  {truth.text:<28}{leftover.strip():<28}")
    print()
    print("  every one of those is a relaxed-match success with a leak in it.")
    print("  Day 10 found the cause by reading its own miss list: the name")
    print("  patterns are written [a-z], which is ASCII, so they stop at the")
    print("  first letter carrying a diacritic. The rule fires, the redaction")
    print("  marker goes in, and the distinctive part of the name stays.")
    print()
    print("  relaxed matching is reported as the lenient metric. For this")
    print("  task it is not lenient, it is wrong: it scores a leak as a hit.")
    print()
    print("  so fix it - [^\\W\\d_] is a letter in any script - and watch what")
    print("  each metric says the fix was worth:")
    print()
    print(f"  {'name patterns':<20}{'strict':>9}{'relaxed':>9}{'token':>9}"
          f"{'clean':>9}{'partials':>10}")
    for label, detector in (("ASCII, as shipped", detect),
                            ("letter-aware", detect_unicode)):
        scored = evaluate(notes, detector)
        print(f"  {label:<20}{scored.recall:>9.4f}"
              f"{relaxed_scores(notes, detector)['recall']:>9.4f}"
              f"{token_scores(notes, detector)['recall']:>9.4f}"
              f"{scored.clean_rate:>9.3f}"
              f"{len(partial_matches(notes, detector)):>10}")
    print()
    print("  the relaxed and token columns are identical to four decimal")
    print("  places. Not close - identical. Both were already counting those")
    print("  spans as hits, so neither can register a repair to them.")
    print()
    print("  and strict recall after the fix is exactly the relaxed figure")
    print("  from before it. That is what relaxed matching was reporting all")
    print("  along: the score the system would get once its leaks were")
    print("  fixed, presented as the score it had.")

    print()
    print("=" * 74)
    print("3. 'we found no leaks' is not a rate")
    print("=" * 74)
    print(f"  {'notes audited':>15}{'leaks found':>13}{'95% upper bound':>18}"
          f"{'3/n':>9}")
    for n in (30, 100, 300, 1000, 10000):
        bound = leak_rate_upper_bound(0, n)
        print(f"  {n:>15}{0:>13}{bound:>18.4f}{rule_of_three(n):>9.4f}")
    print()
    print("  a flawless audit of 300 notes is consistent with 1 note in 100")
    print("  leaking. To bound the rate below 1 in 1000 you must audit three")
    print("  thousand notes and find nothing in any of them.")
    print()
    leaking = len({note_id for note_id, _ in strict.misses})
    print(f"  and this detector did not find nothing. {leaking} of "
          f"{len(notes)} notes leaked:")
    print(f"    point estimate     {leaking / len(notes):.3f}")
    print(f"    95% upper bound    {leak_rate_upper_bound(leaking, len(notes)):.3f}")

    print()
    print("=" * 74)
    print("4. the gold standard is built by correcting the system's output")
    print("=" * 74)
    print("  run the rules, hand a human the pre-annotated text, ask them to")
    print("  fix it. It saves real annotator time. The human catches each")
    print("  genuine miss with probability p - anchoring, fatigue, and the")
    print("  fact that an unmarked span in a marked document does not draw")
    print("  the eye.")
    print()
    print(f"  {'gold standard':<30}{'recall':>9}{'clean rate':>13}")
    print(f"  {'true (the generator record)':<30}{strict.recall:>9.3f}"
          f"{strict.clean_rate:>13.3f}")
    for probability in (0.75, 0.5, 0.25):
        gold = anchored_gold(notes, catch_probability=probability, seed=1)
        scored = evaluate(gold)
        label = f"annotator catches {probability:.0%}"
        print(f"  {label:<30}{scored.recall:>9.3f}{scored.clean_rate:>13.3f}")
    print()
    print("  the span recall barely moves and the clean rate moves a great")
    print("  deal, and that is the shape of the problem: the misses the")
    print("  annotator failed to catch simply stop existing, and a note with")
    print("  one uncaught miss is recorded as a clean note.")
    print()
    print("  the metric that matters is the metric this distorts most.")

    print()
    print("=" * 74)
    print("5. and the identifiers were never the whole problem")
    print("=" * 74)
    print("  suppose Day 10's detector were perfect. Every name, MRN, date,")
    print("  phone, address and email is gone. What is left in the note is")
    print("  the age (redacted only above 89), the year, the department and")
    print("  the diagnosis - all of it legitimately retained, all of it the")
    print("  reason the note was released at all.")
    print()
    print(f"  {'retained fields':<36}{'classes':>9}{'k':>4}{'unique':>9}")
    for label, fields in GENERALISATIONS:
        row = k_anonymity(notes, fields)
        print(f"  {label:<36}{row['classes']:>9}{row['k']:>4}"
              f"{row['unique_rate']:>9.1%}")
    print()
    print("  98% of these patients are alone in their equivalence class on")
    print("  the first row. An attacker holding an age, a year, a department")
    print("  and a diagnosis - an obituary, a social media post, a colleague")
    print("  with a memory - joins straight onto the record.")
    print()
    print("  getting the unique rate near zero costs the year and the")
    print("  department. That is not a tuning knob, it is the deletion of")
    print("  most of what a researcher wanted the corpus for.")

    print()
    print("=" * 74)
    print("6. and the average hides who is at risk")
    print("=" * 74)
    skewed = make_notes(n=300, seed=0, condition_weights=SKEWED_CONDITIONS)
    fields = ("band", "year", "condition")
    overall = k_anonymity(skewed, fields)
    print(f"  fields: age band, year, diagnosis. Overall unique "
          f"{overall['unique_rate']:.1%}")
    print()
    print(f"  {'diagnosis':<40}{'patients':>9}{'unique':>9}")
    for row in uniqueness_by_condition(skewed, fields):
        print(f"  {row['condition']:<40}{row['patients']:>9}"
              f"{row['rate']:>9.1%}")
    print()
    print("  the overall figure is an average over a population most of whom")
    print("  are safe. The patients with the rarest diagnosis are the ones")
    print("  the join lands on, and a rare diagnosis is what makes a record")
    print("  worth looking for in the first place.")
    print()
    print("  Day 6 said accuracy measures prevalence. This is the same")
    print("  sentence: a privacy metric averaged over a cohort measures the")
    print("  common cases, and the rare ones are the exposure.")

    print()
    print("=" * 74)
    print("what to report, and what it costs to say it")
    print("=" * 74)
    print("  not 'recall 0.98'. Three numbers, none of them flattering:")
    print()
    print(f"    leak rate per note   {leaking / len(notes):.3f} "
          f"(95% upper bound {leak_rate_upper_bound(leaking, len(notes)):.3f})")
    print(f"    strict span recall   {strict.recall:.3f}, against ground truth")
    print("                         that is not a corrected system output")
    unique_rate = k_anonymity(notes, GENERALISATIONS[0][1])["unique_rate"]
    print(f"    unique on retained   {unique_rate:.1%}")
    print("                         quasi-identifiers")
    print()
    print("  Day 12 stays in the text and changes the question. Nothing")
    print("  below is about identity: the problem is that a note saying")
    print("  'no evidence of pneumonia' contains the word pneumonia, and a")
    print("  system reading it as a diagnosis is wrong in the direction that")
    print("  puts a patient on a treatment.")
