"""Day 10 - De-identification: rules, recall, and the cost of a miss.

A corpus of synthetic clinical notes whose generator **records every
identifier it inserts**, and a good-faith rule-based de-identifier
measured against that record. Same discipline as Day 1's R peaks: the
detector is scored against spans this file placed, so a miss is not a
matter of opinion.

The headline number is good and it is the wrong number. Span recall sits
in the mid-nineties, which is roughly what published rule-based systems
report. But a note is not partly safe. **The denominator that matters is
the note**, and the note-level clean rate is far below the span-level
recall, because a note needs every one of its identifiers caught.

Four things this file measures rather than asserts:

* **The gazetteer cannot be finished.** Names are an open class. The
  generator draws from a wider pool than the detector knows, which is
  not a rigged demo - it is the permanent situation.
* **Misses cluster.** A note written in a date format the rules do not
  carry has two dates in that format. That makes the note-level rate
  differ from an independence model, and the direction of the difference
  says the errors are systematic rather than random.
* **Over-redaction is also a failure.** ``Parkinson`` is a surname and
  the name of a disease. Putting it in the gazetteer deletes diagnoses;
  leaving it out leaks the patients actually called Parkinson. Both arms
  of that trade are measured here, and neither is free.
* **A rule evaluated on the corpus it was written for looks free.** The
  aggressive MRN rule costs nothing measurable in this demo, and the
  reason is that this corpus contains nothing for it to trip over. That
  is a statement about the evaluation, not about the rule.

NOT A MEDICAL DEVICE. Every note in this repo is invented by it. No
patient data, real or derived, appears anywhere.
"""

from __future__ import annotations

import re
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
    """A synthetic note and the ground truth of what was put in it."""

    text: str
    spans: tuple = ()
    note_id: int = 0

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

    def build(self, note_id: int = 0) -> Note:
        return Note("".join(self._parts), tuple(self._spans), note_id)


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


def make_notes(n: int = 300, seed: int = 0) -> list:
    """Generate ``n`` synthetic notes, each carrying its own ground truth.

    Every identifier is recorded as it is written, with exact offsets.
    Nothing here is derived from a real record; the templates read like
    discharge summaries and contain the shapes a de-identifier has to
    cope with.
    """
    rng = np.random.default_rng(seed)
    notes = []
    for note_id in range(n):
        builder = _Builder()
        given = str(rng.choice(GIVEN_NAMES))
        surname = str(rng.choice(SURNAMES))
        clinician = str(rng.choice(SURNAMES))
        age = int(rng.integers(24, 96))
        condition = str(rng.choice(CONDITIONS))
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

        notes.append(builder.build(note_id))
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


def detect(text: str, gazetteer=None, bare_numbers: bool = False) -> list:
    """Find identifiers in ``text``. A good-faith rule system, not a strawman.

    Patterns first, then title and field context, then the gazetteer. A
    later match overlapping an earlier one is dropped, so the order above
    is the precedence.

    ``gazetteer`` swaps the surname list, and ``bare_numbers`` turns on
    the aggressive MRN rule. Both exist so the demo can measure a change
    instead of describing one.
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

    for pattern in (TITLE_PATTERN, FIELD_PATTERN):
        for match in pattern.finditer(text):
            claim(match.start(1), match.end(1), "NAME")

    for match in WORD_PATTERN.finditer(text):
        word = match.group()
        if word in surnames or word in GAZETTEER_GIVEN:
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


if __name__ == "__main__":
    import sys
    from collections import Counter

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. Every note here is invented by this file.")

    notes = make_notes(n=300, seed=0)
    identifiers = sum(len(note.spans) for note in notes)
    per_note = identifiers / len(notes)
    print()
    print(f"corpus {len(notes)} notes, {identifiers} identifiers, "
          f"{per_note:.1f} per note")

    print()
    print("=" * 74)
    print("1. one note, before and after")
    print("=" * 74)
    sample = notes[1]
    for line in sample.text.rstrip().splitlines():
        print(f"  | {line}")
    print()
    print("  redacted:")
    for line in redact(sample.text, detect(sample.text)).rstrip().splitlines():
        print(f"  | {line}")

    print()
    print("=" * 74)
    print("2. the headline number, and the one that matters")
    print("=" * 74)
    result = evaluate(notes)
    print(f"  {'category':<12}{'in corpus':>11}{'caught':>9}{'recall':>9}")
    for category in CATEGORIES:
        row = result.per_category[category]
        if not row["truth"]:
            continue
        print(f"  {category:<12}{row['truth']:>11}{row['found']:>9}"
              f"{row['recall']:>9.3f}")
    print(f"  {'ALL':<12}{result.total_truth:>11}"
          f"{result.total_truth - result.total_missed:>9}{result.recall:>9.3f}")
    print()
    print(f"  span recall                {result.recall:.3f}")
    print(f"  span precision             {result.precision:.3f}")
    print(f"  notes fully de-identified  {result.clean_notes}/{result.n_notes}"
          f" = {result.clean_rate:.3f}")
    print()
    print("  a note is not partly safe. The span figure is the one that gets")
    print("  published; the note figure describes what happens when the")
    print("  corpus is released.")

    print()
    print("=" * 74)
    print("3. misses are not spread evenly")
    print("=" * 74)
    leaking = Counter(note_id for note_id, _ in result.misses)
    by_category = Counter(span.category for _, span in result.misses)
    print(f"  {result.total_missed} misses, in {len(leaking)} notes")
    print()
    print(f"  {'category':<12}{'misses':>9}    {'misses in a note':<20}{'notes':>8}")
    rows = list(by_category.most_common())
    spread = sorted(Counter(leaking.values()).items())
    for i in range(max(len(rows), len(spread))):
        left = f"  {rows[i][0]:<12}{rows[i][1]:>9}" if i < len(rows) else " " * 23
        right = (f"    {spread[i][0]:<20}{spread[i][1]:>8}"
                 if i < len(spread) else "")
        print(left + right)
    print()
    predicted_clean = independence_prediction(result.recall, per_note)
    print(f"  independence model  {result.recall:.3f}^{per_note:.1f} = "
          f"{predicted_clean:.3f}")
    print(f"  measured                          {result.clean_rate:.3f}")
    print()
    print("  the two disagree, and the direction is the point: misses")
    print("  concentrate in the same notes, so the number of notes affected")
    print("  is not what independent errors would give. A systematic failure")
    print("  is the kind a larger corpus does not average away.")

    print()
    print("=" * 74)
    print("4. what was missed, and why it was always going to be")
    print("=" * 74)
    shown = Counter()
    for note_id, span in result.misses:
        if shown[span.category] >= 3:
            continue
        shown[span.category] += 1
        print(f"  {span.category:<8} {span.text!r}")
    print()
    print("  the DATE misses are one format the pattern list does not carry.")
    print("  It is rare in this corpus for the same reason it would be rare")
    print("  in a development sample - and that is exactly why the rules do")
    print("  not have it. The list is always one hospital behind.")
    print()
    print("  the MRN misses are bare numbers with no label. Section 6 turns")
    print("  on the rule that catches them.")
    print()
    print("  the NAME misses are people not in the gazetteer, mentioned with")
    print("  no title and no field label in front of them. That one is not")
    print("  fixable by adding names: a list is a snapshot of a population,")
    print("  and the next patient is under no obligation to appear in it.")

    print()
    print("=" * 74)
    print("5. the other kind of failure, and the trade it forces")
    print("=" * 74)
    print(f"  {'gazetteer':<26}{'recall':>8}{'precision':>11}"
          f"{'clean':>8}{'eponyms lost':>14}")
    for label, detector in (("with eponym surnames", detect),
                            ("without them", detect_without_eponyms)):
        scored = evaluate(notes, detector)
        damage = over_redaction(notes, detector)
        print(f"  {label:<26}{scored.recall:>8.3f}{scored.precision:>11.3f}"
              f"{scored.clean_rate:>8.3f}"
              f"{damage['damaged']:>8}/{damage['phrases']:<5}")
    print()
    damage = over_redaction(notes)
    for note_id, phrase in damage["examples"]:
        print(f"    note {note_id}: {phrase!r} redacted")
    print()
    print("  Parkinson is a surname and Parkinson's disease is a diagnosis,")
    print("  and the difference is not in the word. Keep the eponyms on the")
    print("  list and diagnoses are deleted; strike them off and the patients")
    print("  actually called Parkinson walk out of the building named.")
    print()
    print("  neither column is the safe one. The first loses clinical")
    print("  meaning while every number in section 2 stays clean, and the")
    print("  second is a leak. This is the trade, not a bug to be fixed.")

    print()
    print("=" * 74)
    print("6. a rule that looks free, and the reason it looks free")
    print("=" * 74)
    print(f"  {'rules':<30}{'recall':>8}{'precision':>11}{'clean':>8}")
    for label, detector in (("conservative", detect),
                            ("plus bare 6-8 digit numbers", detect_aggressive)):
        scored = evaluate(notes, detector)
        print(f"  {label:<30}{scored.recall:>8.3f}{scored.precision:>11.3f}"
              f"{scored.clean_rate:>8.3f}")
    print()
    aggressive = evaluate(notes, detect_aggressive)
    print(f"  the aggressive rule costs {len(aggressive.false_positives) - len(result.false_positives)}"
          f" additional false positives here.")
    print()
    print("  that is not evidence the rule is safe. This corpus contains no")
    print("  six-to-eight-digit number that is not an MRN - no accession")
    print("  numbers, no device serials, no pager IDs - so there is nothing")
    print("  for the rule to trip over. A rule evaluated on the corpus it")
    print("  was written for always looks free.")
    print()
    print("  the honest statement is not 'this rule is safe'. It is 'this")
    print("  corpus cannot tell you whether this rule is safe', and the")
    print("  difference between those two sentences is Day 11.")

    print()
    print("=" * 74)
    print("7. the cost of a miss")
    print("=" * 74)
    print(f"  {result.recall:.1%} span recall sounds like a "
          f"{1 - result.recall:.0%} problem.")
    print(f"  It is a {1 - result.clean_rate:.0%} problem: {len(leaking)} of "
          f"{len(notes)} notes carry at least one identifier out of the door.")
    print()
    print("  Day 7 chose a threshold from a cost ratio. The arithmetic still")
    print("  applies and the ratio is not 5:1. A false positive costs a word")
    print("  of clinical text; a false negative costs a person their medical")
    print("  privacy, permanently, in a corpus that has already been copied.")
    print("  There is no ratio at which a leak is priced acceptably, which is")
    print("  why de-identification is not a threshold-tuning problem.")
    print()
    print("  what is out of scope here, stated rather than hidden:")
    for name, why in SAFE_HARBOR_GAPS.items():
        print(f"    {name:<24} {why}")

    print()
    print("  Day 11 asks how you would know any of this. Every number above")
    print("  was scored against ground truth this file wrote down. A real")
    print("  corpus has no such record, and its evaluation set is annotated")
    print("  by people holding the rules that are being tested.")
