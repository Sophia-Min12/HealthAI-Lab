"""Day 12 - Clinical abbreviations, negation, and "no evidence of".

Nothing in this day is about identity. The problem is simpler and older:
a note saying ``no evidence of pneumonia`` contains the word
``pneumonia``, and a keyword search cannot tell the difference.

Four things measured here rather than asserted:

* **Roughly half of all concept mentions in clinical text are not
  assertions of the concept.** A keyword search for a diagnosis is right
  about the word and wrong about the patient, and its precision is the
  prevalence of positive mentions - which is Day 6's sentence again, in
  a different register.

* **Negation detection is easy.** A NegEx-style rule set recovers most
  of it in about forty lines. That is the reassuring half.

* **The residual errors are not randomly distributed.** They concentrate
  in hedges - ``cannot rule out``, ``suspicion of`` - which are neither
  assertions nor denials, and which a two-class system must round to one
  of them. Rounding them to ABSENT takes a patient off a pathway. The
  headline accuracy barely moves either way, which is the problem.

* **Abbreviations are ambiguous by department, and conditioning on the
  department is a trick that works until it doesn't.** ``MS`` is mitral
  stenosis in cardiology and morphine sulfate at the end of life, on the
  same ward, in the same week. The department rule fails on exactly
  those notes and the headline accuracy cannot see it.

NOT A MEDICAL DEVICE. Every note in this repo is invented by it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

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

DEPARTMENTS = ("Cardiology", "Respiratory Medicine", "Neurology",
               "Rheumatology", "Palliative Care", "Haematology")


class _Builder:
    """Assembles a sentence while recording where each concept landed."""

    def __init__(self) -> None:
        self._parts: list = []
        self._length = 0
        self._mentions: list = []

    def add(self, text: str, concept: str | None = None,
            assertion: str | None = None) -> "_Builder":
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
        builder = _Builder()
        department = str(rng.choice(DEPARTMENTS))
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
        builder = _Builder()
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


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("NOT A MEDICAL DEVICE. Every note here is invented by this file.")

    sentences = make_sentences(n=600, seed=0)
    counts = class_counts(sentences)
    mentions = sum(counts.values())
    print()
    print(f"corpus {len(sentences)} sentences, {mentions} concept mentions")

    print()
    print("=" * 74)
    print("1. what a mention of a diagnosis actually says")
    print("=" * 74)
    print(f"  {'assertion':<14}{'mentions':>10}{'share':>9}   meaning")
    meanings = {
        "PRESENT": "the patient has it",
        "ABSENT": "the patient does not",
        "UNCERTAIN": "nobody knows yet",
        "FAMILY": "a relative has it",
        "HISTORICAL": "the patient had it",
    }
    for name in ASSERTIONS:
        print(f"  {name:<14}{counts[name]:>10}{counts[name] / mentions:>9.1%}"
              f"   {meanings[name]}")
    print()
    naive = score(sentences, keyword_present)
    print(f"  a keyword search calls every one of those PRESENT, so its")
    print(f"  accuracy is the PRESENT share exactly: {naive.accuracy:.4f}")
    print(f"  and it puts {naive.false_actionable} patients on a pathway they")
    print("  do not belong on.")
    print()
    print("  Day 6 said accuracy measures prevalence. Here it measures the")
    print("  prevalence of positive mentions, which is a fact about how")
    print("  clinicians write rather than about who is ill.")

    print()
    print("=" * 74)
    print("2. negation detection is the easy part")
    print("=" * 74)
    full = score(sentences, classify)
    no_scope = score(sentences, lambda text, mention:
                     classify(text, mention, use_terminators=False))
    print(f"  {'system':<38}{'accuracy':>10}")
    for label, report in (("keyword presence", naive),
                          ("NegEx-style rules, no scope terminators", no_scope),
                          ("NegEx-style rules", full)):
        print(f"  {label:<38}{report.accuracy:>10.4f}")
    print()
    print("  about forty lines of trigger lists and one scope rule, and the")
    print("  scope rule is worth "
          f"{full.accuracy - no_scope.accuracy:.3f} of that on its own:")
    print()
    for sentence in sentences:
        if len(sentence.mentions) < 2 or "but" not in sentence.text:
            continue
        print(f"    {sentence.text}")
        for mention in sentence.mentions:
            with_scope = classify(sentence.text, mention)
            without = classify(sentence.text, mention, use_terminators=False)
            print(f"      {mention.concept:<22}truth {mention.assertion:<9}"
                  f"with scope {with_scope:<9}without {without}")
        break
    print()
    print("      'but' ends the denial. Without the terminator list the")
    print("      second concept inherits the first one's negation, and the")
    print("      note reports a symptom the patient has as one they deny.")
    print()
    print("  and it needs to look forwards as well as backwards. The first")
    print("  run of this demo put all 18 of its errors in one template:")
    print("    'cough resolved, however atrial fibrillation persists.'")
    print("  the cue for the first concept is after it, so a backwards rule")
    print("  saw an empty prefix and called a resolved problem current.")

    print()
    print("=" * 74)
    print("3. the hedges, and which way you round them")
    print("=" * 74)
    two_class = score(sentences, lambda text, mention:
                      classify(text, mention, use_hedges=False))
    print("  a two-class detector has nowhere to put 'cannot rule out'. The")
    print("  phrase contains 'rule out', which is on every negation list, so")
    print("  it lands on ABSENT:")
    print()
    print(f"  {'system':<24}{'accuracy':>10}{'missed actionable':>20}"
          f"{'false actionable':>19}")
    for label, report in (("with a hedge class", full),
                          ("two-class", two_class)):
        print(f"  {label:<24}{report.accuracy:>10.4f}"
              f"{report.missed_actionable:>20}{report.false_actionable:>19}")
    print()
    print(f"  the accuracy falls by {full.accuracy - two_class.accuracy:.3f}.")
    print(f"  Underneath it, {two_class.missed_actionable} mentions moved from")
    print("  'nobody knows yet' to 'the patient does not have it'.")
    print()
    print("  'cannot rule out pulmonary embolism' is the sentence that")
    print("  orders the scan. Read as a denial it cancels the scan, and an")
    print("  accuracy figure adds that error to the harmless kind and")
    print("  reports the total.")

    print()
    print("=" * 74)
    print("4. and now the uncomfortable part")
    print("=" * 74)
    print(f"  the rules score {full.accuracy:.4f} on the corpus above.")
    print()
    print("  I wrote the trigger lists while looking at the templates that")
    print("  generate that corpus. The score measures the overlap between")
    print("  two things written by the same person on the same afternoon.")
    print()
    print("  so: a second set of sentence forms, ordinary clinical")
    print("  shorthand, none of it consulted while writing the rules.")
    print()
    heldout = make_heldout_sentences(n=600, seed=1)
    held = score(heldout, classify)
    print(f"  {'sentence forms':<28}{'accuracy':>10}{'missed actionable':>20}")
    print(f"  {'the rules were written for':<28}{full.accuracy:>10.4f}"
          f"{full.missed_actionable:>20}")
    print(f"  {'held out':<28}{held.accuracy:>10.4f}"
          f"{held.missed_actionable:>20}")
    print()
    print(f"  {'true class':<14}{'n':>6}{'accuracy':>10}")
    for name, row in sorted(per_class_accuracy(heldout, classify).items()):
        print(f"  {name:<14}{row['n']:>6}{row['accuracy']:>10.3f}")
    print()
    print("  PRESENT scores 1.000 because PRESENT is what the classifier")
    print("  returns when no rule fires, and on unfamiliar text no rule")
    print("  fires. It is not recognising anything. It is defaulting, and")
    print("  being right by the arrangement of the classes.")
    print()
    print("  neither figure is 'the accuracy of negation detection'. The")
    print(f"  first is an upper bound and the second depends on how")
    print("  adversarial I felt when choosing the held-out forms - I could")
    print("  have chosen a set that scores 0.41, and the first draft of this")
    print("  section did. That the number is choosable is the finding, not a")
    print("  caveat attached to it.")

    print()
    print("=" * 74)
    print("5. abbreviations, and a trick that works until it does not")
    print("=" * 74)
    print(f"  {'abbreviation':<6}  senses")
    for abbreviation, senses in ABBREVIATIONS.items():
        rendered = ", ".join(f"{sense} ({dept})"
                             for dept, sense in senses.items())
        print(f"  {abbreviation:<6}  {rendered}")
    print()
    uses = make_abbreviation_uses(n=600, seed=0)
    print(f"  {'expansion rule':<24}{'overall':>9}{'in department':>16}"
          f"{'cross-department':>19}")
    for label, expander in (("most frequent sense", expand_most_frequent),
                            ("use the department", expand_by_department)):
        row = score_expansion(uses, expander)
        print(f"  {label:<24}{row['accuracy']:>9.3f}"
              f"{row['in_department']:>16.3f}"
              f"{row['cross_department']:>19.3f}")
    row = score_expansion(uses, expand_by_department)
    print()
    print(f"  the department rule scores {row['accuracy']:.3f}, and it is not")
    print(f"  a 94% system. It is a 100% system on {row['n'] - row['cross_n']}")
    print(f"  notes and a 0% system on {row['cross_n']}.")
    print()
    print("  those are the notes where a cardiology ward wrote MS for")
    print("  morphine sulfate at the end of life. The rule is not slightly")
    print("  wrong there; it is wrong every single time, because the thing")
    print("  it conditions on is exactly the thing that has changed.")
    print()
    print("  Day 11 said a privacy metric averaged over a cohort measures")
    print("  the common cases. This is the same shape: the cases a rule")
    print("  handles perfectly dominate the average, and the cases it")
    print("  cannot handle at all are invisible inside it.")

    print()
    print("=" * 74)
    print("6. and the two halves multiply")
    print("=" * 74)
    pipeline = score_pipeline(make_combined(n=600, seed=2))
    print("  a concept written as an abbreviation, in a negated sentence.")
    print("  Both stages have to be right (classes drawn uniformly here, so")
    print("  the assertion figure is not comparable with section 4's).")
    print()
    print(f"  {'expansion stage':<22}{pipeline['expansion']:>9.3f}")
    print(f"  {'assertion stage':<22}{pipeline['assertion']:>9.3f}")
    print(f"  {'product of the two':<22}{pipeline['product']:>9.3f}")
    print(f"  {'measured end to end':<22}{pipeline['joint']:>9.3f}")
    print()
    print("  two respectable stages, and a pipeline that is right about")
    print(f"  {pipeline['joint']:.0%} of the mentions it is asked about.")
    print("  Neither stage's report contains that number, and the two teams")
    print("  that built them would both be reporting honestly.")
    print()
    print("  Day 17 runs five stages.")

    print()
    print("=" * 74)
    print("what Level 3 established")
    print("=" * 74)
    print("  Day 10  94.9% of identifiers caught, 64.7% of notes clean. A")
    print("          note is not partly safe, and the denominator that gets")
    print("          published is not the one that describes the release.")
    print("  Day 11  the metric moved the number further than the detector")
    print("          did. Fixing a bug that leaked surnames changed relaxed")
    print("          and token recall by 0.0000 each, to four decimals.")
    print("  Day 12  half of all mentions of a diagnosis are not assertions")
    print("          of it, and the rules that sort them score 1.000 on the")
    print("          sentences they were written for and 0.730 on sentences")
    print("          they were not.")
    print()
    print("  the through-line: every measurement in this level was taken on")
    print("  a corpus that the thing being measured had already seen. Level")
    print("  4 keeps that problem and adds a generator on top of it, where")
    print("  the failure mode is not a wrong label but a fluent paragraph")
    print("  with a citation attached to it.")
