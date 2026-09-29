# Day 17 · Capstone: The Full Pipeline, a CLI, and the Honest Writeup

> **NOT A MEDICAL DEVICE.** Every note, every guideline and every evidence grade in this repo is invented by it. No patient data, real or derived, appears anywhere.

Seventeen days wired into one path: a synthetic clinical note goes in, and either a cited recommendation or a refusal comes out.

```bash
python capstone.py note           # one case, every stage shown
python capstone.py run            # the stage table and the compounding
python capstone.py deid "<text>"  # de-identify a note
python capstone.py ask "<text>"   # one question against the guideline
python capstone.py writeup        # what seventeen days established
```

## One note, four stages

```
Nephrology - Clinic Letter                  Nephrology - Clinic Letter

Patient: Linnea Vasquez   MRN 3820174       Patient: [NAME]   [MRN]
Date of review: 09.02.2021                  Date of review: [DATE]

This 71-year-old patient was reviewed       This 71-year-old patient was reviewed
by Dr Vasquez.                              by Dr [NAME].
With concurrent stage 3 disease,            With concurrent stage 3 disease,
confirmed on review.                        confirmed on review.

Contact: (261) 977-9494                     Contact: [PHONE]
```

```
2. the assertion, read from the redacted text
     finding   With concurrent stage 3 disease
     recorded  PRESENT       read as  PRESENT

3. ask the guideline?
     the note says the criterion applies: True
     the pipeline asked:                  True

4. the answer
     In adults with concurrent stage 3 disease, do not routinely offer
     screening in asymptomatic people. [Strength: conditional - Evidence: C]
     SG-1 (2024) 2.2 Investigations [R3]
     every claim traced to that chunk: True
```

**Every stage after the first works on the de-identified text**, which is the only order it is allowed to run in — and it means a de-identification bug is also a retrieval bug.

## The number this file exists to produce

300 notes. 143 written in phrasings Day 12's rules were not built against; 56 with a finding this guideline does not cover.

```
stage                       correct   what it is
de-identify the note          0.767   Day 11's letter-aware detector, note-level
read the assertion            0.653   Day 12's negation rules
ask, or decline to ask        0.783   actionable findings only
retrieve and refuse           0.973   Day 14's score per term
check the answer              0.973   Day 15's claim checks

product of the five           0.372
measured end to end           0.493
```

The product is what you get by assuming the stages fail independently, and they do not: a note whose identifiers leak can still have its assertion read correctly, and a misread assertion usually takes the query decision down with it. The measurement is better than the model, and **both are far below every stage in the table above them**.

No stage's report contains the bottom line. Five teams could each publish their column and every one of them would be reporting honestly. Day 12 previewed this with two stages; this is five.

### The numbers are choosable, and here is the knob

`make_cases` takes `held_out_rate` and `out_of_scope_rate`. Set both to zero and three of the five stages report **1.000** — a test asserts exactly that, because it is what evaluating on a corpus built to suit the system looks like.

## Two ways to end up quoting a withdrawn guideline

```
insertion order             end to end   stale citations
current added first              0.493                 4
superseded added first           0.390                88
```

**The top row is the correct configuration and it is not zero.** Those four notes record `an estimated GFR below 15 mL/min` — the threshold the *2019* edition used. The 2019 edition is the better lexical match for them, so it wins, and the retriever is working exactly as specified while it happens.

> A clinician who records the old criterion gets the old guideline back.

The bottom row is Day 16's finding arriving end to end, and it is sharper here: **80 of the 88 answers that move are notes recording one of the 4 criteria the revision did not change** — word for word the same in both editions, so there is nothing to choose between them and insertion order decides. Where the note names a threshold the revision *did* move, the current edition wins on merit. The retriever gets the right answer for a reason that has nothing to do with the date, on exactly the questions where the date happens to be encoded in the words.

And every one of those stale answers is verified: grounded, quoted correctly, and withdrawn.

## What seventeen days established

| Day | Finding |
|-----|---------|
| 1 | A synthetic ECG, so a missed R peak is unambiguous |
| 2 | Mains hum beat motion artefact at equal SNR — the opposite of what the file first claimed |
| 3 | A moving average delays a peak by `(w-1)/2` and destroys it before it stops delaying it |
| 4 | R-peak detection and HRV, scored against peaks the generator placed |
| 5 | Forty honest studies gave age odds ratios from 0.66 to 3.90 |
| 6 | At a 0.9% event rate the model scored 99.1% and found nothing |
| 7 | Every threshold is an unstated cost ratio, and it moves the cut further than any model change |
| 8 | Four distortions, one AUC to four decimals, a hundredfold range of ECE |
| 9 | Imputation invents values in the one direction the data cannot reveal |
| 10 | 94.9% of identifiers caught, 64.7% of notes clean |
| 11 | Fixing a bug that leaked surnames changed relaxed and token recall by 0.0000 each |
| 12 | The negation rules score 1.000 on the sentences they were written for and 0.730 on sentences they were not |
| 13 | No chunking puts a recommendation in the same chunk as the section that reverses it |
| 14 | BM25's score is mostly a measurement of how long the question was |
| 15 | A negated answer scores 0.959 on word overlap; traceable and correct are different properties |
| 16 | Flipping the index order moved 26 of 28 answers onto a withdrawn guideline, with no score changing |
| 17 | Five respectable stages, one pipeline correct on half its notes |

## What it cannot do

- **Every measurement in this repo was taken on data this repo generated.** The de-identifier was scored against spans this file wrote down, the negation rules against sentences this file wrote, and the retriever against a guideline this file invented. Nothing here has met a real note.
- **The generator is extractive.** It cannot paraphrase, so it cannot hallucinate, so Day 15's checker never fires — measured on Day 16 as a column of zeros. Put a language model in its place and the checker starts earning its keep, against failures this pipeline cannot currently commit and *on top of* every failure it already does.
- **The guards do not see a date, a scope declaration or a cross-reference**, because the text does not carry them. Day 13 listed what an index would have to hold, and this pipeline holds none of it.
- **It is not a medical device.** Using any of this on a patient would need regulatory approval, clinical validation, and expertise this repo does not contain.

## Run it

```bash
python capstone.py run    # the stage table
pytest                    # from the repo root
python -m unittest        # from inside this folder
```
