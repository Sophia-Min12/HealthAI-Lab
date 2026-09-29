# Day 11 · Evaluating De-identification When a Single Leak Is a Failure

> **NOT A MEDICAL DEVICE.** Every note in this repo is invented by it. No patient data, real or derived, appears anywhere.

Day 10 built a detector and scored it: **94.9% span recall, 194 of 300 notes clean.** This day asks whether that score meant anything, and finds four reasons to doubt it.

## 1. One output, three metrics, three numbers

```
metric                              denominator   recall
token, any overlap                         4853    0.977
span, any overlap (relaxed)                2698    0.958
span, fully covered (strict)               2698    0.949
```

The detector did not change between those rows. Nothing about the system moved; only the question moved.

## 2. Partial credit, for a task where there is no such thing

24 spans are hits under relaxed matching and misses under strict matching. Here is what survives redaction in each:

```
truth                       what survives redaction
Bergström                   öm
Astrid Bergström            öm
Lucia Bergström             öm
Robert Bergström            öm
```

Day 10 found the cause by reading its own miss list. Every character class in the module is written `[a-z]`, which is ASCII, so the name patterns stop at the first letter carrying a diacritic. The rule fires, the redaction marker goes in, and the distinctive part of the name stays on the page.

So fix it — `[^\W\d_]` is a letter in any script — and watch what each metric says the fix was worth:

```
name patterns          strict  relaxed    token    clean  partials
ASCII, as shipped      0.9489   0.9577   0.9765    0.647        24
letter-aware           0.9577   0.9577   0.9765    0.673         0
```

**The relaxed and token columns are identical to four decimal places.** Not close — identical, and a test pins them with `assertEqual`. Both metrics were already counting those spans as hits, so neither can register their repair. A team reporting either one would have seen a bug fix that leaked surnames register as no change whatsoever.

And strict recall *after* the fix is exactly the relaxed figure from *before* it. That is what relaxed matching was reporting all along: the score the system would get once its leaks were fixed, presented as the score it had.

## 3. "We found no leaks" is not a rate

```
  notes audited  leaks found   95% upper bound      3/n
             30            0            0.0950   0.1000
            100            0            0.0295   0.0300
            300            0            0.0099   0.0100
           1000            0            0.0030   0.0030
          10000            0            0.0003   0.0003
```

Clopper–Pearson, computed by bisecting a directly summed binomial CDF. At zero events it has the closed form `1 − α^(1/n)`, and the **rule of three** — `3/n` — is within 2% of it for any n worth running. Both are pinned by tests.

A flawless audit of 300 notes is consistent with **1 note in 100 leaking**. To bound the rate below 1 in 1000 you must audit three thousand notes and find nothing in any of them.

And this detector did not find nothing:

```
106 of 300 notes leaked
  point estimate     0.353
  95% upper bound    0.401
```

## 4. The gold standard is built by correcting the system's own output

Run the rules, hand a human the pre-annotated text, ask them to fix it. It is standard practice and it saves real annotator time. The human catches each genuine miss with probability *p* — anchoring, fatigue, and the fact that an unmarked span in a marked document does not draw the eye.

```
gold standard                    recall   clean rate
true (the generator record)       0.949        0.647
annotator catches 75%             0.963        0.733
annotator catches 50%             0.975        0.820
annotator catches 25%             0.989        0.907
```

The span recall barely moves and the clean rate moves a great deal. The misses the annotator failed to catch simply stop existing, and a note with one uncaught miss is recorded as a clean note.

**The metric that matters is the metric this distorts most.**

At the limit the arithmetic is stark: an annotator who catches *nothing* the system missed produces a gold standard that is the system's own output, and the system scores **exactly 1.0** against it. A test asserts that equality, because it is not an approximation.

## 5. And the identifiers were never the whole problem

Suppose Day 10's detector were perfect. Every name, MRN, date, phone, address and email is gone. What is left is the age (redacted only above 89), the year, the department and the diagnosis — all legitimately retained, all of it the reason the note was released at all.

```
retained fields                       classes   k   unique
age, year, dept, diagnosis                297   1    98.0%
age band, year, dept, diagnosis           282   1    88.3%
age band, year, diagnosis                 206   1    43.7%
age band, diagnosis                        63   1     0.7%
diagnosis only                              8  29     0.0%
```

**98% of these patients are alone in their equivalence class on the first row.** An attacker holding an age, a year, a department and a diagnosis — an obituary, a social media post, a colleague with a memory — joins straight onto the record.

Getting the unique rate near zero costs the year *and* the department. That is not a tuning knob; it is the deletion of most of what a researcher wanted the corpus for.

## 6. And the average hides who is at risk

On a cohort with a realistic tail of diagnoses (`age band, year, diagnosis`, overall unique **36.3%**):

```
diagnosis                                patients   unique
iron deficiency anaemia                         4   100.0%
chronic obstructive pulmonary disease           8    75.0%
essential hypertension                         21    61.9%
congestive heart failure                       29    65.5%
community-acquired pneumonia                   31    54.8%
atrial fibrillation                            51    33.3%
chronic kidney disease stage 3                 62    30.6%
type 2 diabetes mellitus                       94    14.9%
```

The overall figure averages over a population most of whom are safe. The patients with the rarest diagnosis are the ones the join lands on — and a rare diagnosis is what makes a record worth looking for in the first place.

Day 6 said accuracy measures prevalence. This is the same sentence: **a privacy metric averaged over a cohort measures the common cases, and the rare ones are the exposure.**

## What to report

Not "recall 0.98". Three numbers, none of them flattering:

| | |
|---|---|
| **Leak rate per note** | 0.353 (95% upper bound 0.401) |
| **Strict span recall** | 0.949, against ground truth that is not a corrected system output |
| **Unique on retained quasi-identifiers** | 98.0% |

## Run it

```bash
python evaluation.py      # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Day 12 stays in the text and changes the question. Nothing below is about identity: the problem is that a note saying *"no evidence of pneumonia"* contains the word *pneumonia*, and a system reading that as a diagnosis is wrong in the direction that puts a patient on a treatment.
