# Day 12 · Clinical Abbreviations, Negation, and "No Evidence Of"

> **NOT A MEDICAL DEVICE.** Every note in this repo is invented by it.

Nothing here is about identity. The problem is simpler and older: a note saying **"no evidence of pneumonia" contains the word pneumonia**, and a keyword search cannot tell the difference.

## What a mention of a diagnosis actually says

```
assertion       mentions    share   meaning
PRESENT              264    35.2%   the patient has it
ABSENT               298    39.8%   the patient does not
UNCERTAIN             77    10.3%   nobody knows yet
FAMILY                54     7.2%   a relative has it
HISTORICAL            56     7.5%   the patient had it
```

**Under two in five mentions of a diagnosis assert that the patient has it.** A keyword search calls every one of them PRESENT, so its accuracy is the PRESENT share *exactly* — 0.3525, pinned by a test as a closed form rather than a measurement — and it puts 408 patients on a pathway they do not belong on.

Day 6 said accuracy measures prevalence. Here it measures the prevalence of positive mentions, which is a fact about how clinicians write rather than about who is ill.

## Negation detection is the easy part

```
system                                  accuracy
keyword presence                          0.3525
NegEx-style rules, no scope terminators   0.8865
NegEx-style rules                         1.0000
```

About forty lines of trigger lists and one scope rule, and **the scope rule is worth 0.113 of that on its own**:

```
No shortness of breath but myocardial infarction is present.
  shortness of breath   truth ABSENT   with scope ABSENT   without ABSENT
  myocardial infarction truth PRESENT  with scope PRESENT  without ABSENT
```

`but` ends the denial. Without the terminator list the second concept inherits the first one's negation, and a symptom the patient *has* is recorded as one they deny.

It also has to look forwards. The first run of this demo put **all 18 of its errors in one template** — `"cough resolved, however atrial fibrillation persists."` The cue for the first concept comes after it, so a backwards-only rule saw an empty prefix and called a resolved problem current. English marks the past after the noun as readily as before it; a one-directional rule set will always have a class of failures shaped like that.

## The hedges, and which way you round them

A two-class detector has nowhere to put `cannot rule out`. The phrase contains `rule out`, which is on every negation list, so it lands on ABSENT.

```
system                    accuracy   missed actionable   false actionable
with a hedge class          1.0000                   0                  0
two-class                   0.8972                  50                  0
```

The accuracy falls by 0.103. Underneath it, **50 mentions moved from "nobody knows yet" to "the patient does not have it."**

*"Cannot rule out pulmonary embolism"* is the sentence that orders the scan. Read as a denial, it cancels the scan — and an accuracy figure adds that error to the harmless kind and reports the total.

## And now the uncomfortable part

The rules score **1.0000** on the corpus above.

I wrote the trigger lists while looking at the templates that generate that corpus. The score measures the overlap between two things written by the same person on the same afternoon. So: a second set of sentence forms, ordinary clinical shorthand — `Nil {c}.` · `{c}: none.` · `Doubt {c}.` · `?{c}` · `FHx {c}.` · `Longstanding {c}.` — none of it consulted while writing the rules.

```
sentence forms                accuracy   missed actionable
the rules were written for      1.0000                   0
held out                        0.7300                   0

true class         n  accuracy
ABSENT           165     0.582
FAMILY            45     0.467
HISTORICAL        61     0.525
PRESENT          246     1.000
UNCERTAIN         83     0.518
```

**PRESENT scores 1.000 because PRESENT is what the classifier returns when no rule fires**, and on unfamiliar text no rule fires. It is not recognising anything. It is defaulting, and being right by the arrangement of the classes.

Neither figure is "the accuracy of negation detection". The first is an upper bound; the second depends on how adversarial I felt when choosing the held-out forms — the first draft of this section chose a set that scores **0.41**, because every form in it was one no rule could reach. That the number is choosable is the finding, not a caveat attached to it.

## Abbreviations, and a trick that works until it does not

```
MS     mitral stenosis (Cardiology), multiple sclerosis (Neurology), morphine sulfate (Palliative Care)
PT     prothrombin time (Haematology), physical therapy (Rheumatology)
RA     right atrium (Cardiology), rheumatoid arthritis (Rheumatology), room air (Respiratory Medicine)
CVA    cerebrovascular accident (Neurology), costovertebral angle (Haematology)
DM     diabetes mellitus (Cardiology), dermatomyositis (Rheumatology)
```

```
expansion rule            overall   in department   cross-department
most frequent sense         0.530           0.535              0.444
use the department          0.940           1.000              0.000
```

The department rule scores 0.940, and **it is not a 94% system. It is a 100% system on 564 notes and a 0% system on 36.**

Those 36 are the notes where a cardiology ward wrote MS for morphine sulfate at the end of life. The rule is not slightly wrong there; it is wrong every single time, because the thing it conditions on is exactly the thing that has changed. Set the cross-department rate to zero and the rule reports a flawless 1.000 — a test asserts that, because it is what evaluating on a corpus without the hard cases looks like.

Day 11 said a privacy metric averaged over a cohort measures the common cases. Same shape: the cases a rule handles perfectly dominate the average, and the cases it cannot handle at all are invisible inside it.

## And the two halves multiply

A concept written as an abbreviation, in a negated sentence. Both stages have to be right.

```
expansion stage           0.950
assertion stage           0.613
product of the two        0.583
measured end to end       0.577
```

Two respectable stages, and a pipeline right about **58%** of the mentions it is asked about. Neither stage's report contains that number, and the two teams that built them would both be reporting honestly.

Day 17 runs five stages.

## What Level 3 established

| Day | Finding |
|-----|---------|
| 10 | 94.9% of identifiers caught, 64.7% of notes clean. A note is not partly safe, and the denominator that gets published is not the one that describes the release |
| 11 | The metric moved the number further than the detector did. Fixing a bug that leaked surnames changed relaxed and token recall by 0.0000 each, to four decimals |
| 12 | Under two in five mentions of a diagnosis assert it, and the rules that sort them score 1.000 on the sentences they were written for and 0.730 on sentences they were not |

**Every measurement in this level was taken on a corpus the thing being measured had already seen.**

## Run it

```bash
python clinical_text.py   # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Level 4 keeps that problem and adds a generator on top of it, where the failure mode is not a wrong label but a fluent paragraph with a citation attached to it.
