# Day 15 · Measuring Groundedness: Every Claim Traced to a Line

> **NOT A MEDICAL DEVICE.** The condition, the recommendations and the evidence grades in this folder are all invented by it.

Day 14 made the system cite. This day asks whether the citation means anything.

## Six ways to be wrong while quoting the source

```
source, 3.2.1 First-line therapy:
  R7. In adults with an estimated GFR above 60 mL/min, offer a thiazide-like
      diuretic as first-line therapy. [Strength: strong - Evidence: B]

FAITHFUL    In adults with an estimated GFR above 60 mL/min, offer a thiazide-...
NEGATED     In adults with an estimated GFR above 60 mL/min, do not offer a th...
NUMBER      In adults with an estimated GFR above 120 mL/min, offer a thiazide...
RECOMBINED  In adults with an estimated GFR below 30 mL/min, offer a thiazide-...
INVENTED    In adults with an estimated GFR above 60 mL/min, offer intravenous magnesium.
MISCITED    (the faithful text, citing a different section)
MIXED       (the faithful text, plus one fabricated sentence)
```

Look at RECOMBINED. `above 60` has become `below 30` — both clauses are in the document, and the pairing is an invention that reverses who the drug is for.

## What word overlap says about them

```
kind           n  supported  overlap  passes overlap >= 0.75
FAITHFUL      20         20    1.000                      20
NEGATED       20          0    0.959                      20
NUMBER        20          0    0.905                      20
RECOMBINED    20          4    0.793                      11
INVENTED      20          0    0.758                      11
MISCITED      20          0    0.495                       4
MIXED         20          0    0.517                       0
```

**A negated answer scores 0.959.** It reverses the instruction by adding two words, both of which most pipelines drop as stopwords. A changed threshold scores 0.905, and the one token that moved is the only thing the recommendation was about. At a conventional 0.75 cutoff, **every single negated and number-substituted answer passes.**

Overlap is a statement about vocabulary. Every failure mode above keeps the vocabulary.

## The checks a similarity skips

Not a cleverer similarity — a list of the things a similarity does not look at, applied per claim:

1. **Polarity**, against the recommendation the claim actually restates
2. **Numbers** in the claim must appear in the cited chunk
3. **Unfamiliar terms** must not
4. **Recombination** — the eligibility clause and the action must come from the *same* recommendation

```
kind           n  supported  strict pass
FAITHFUL      20         20           20
NEGATED       20          0            0
NUMBER        20          0            0
RECOMBINED    20          4            4
INVENTED      20          0            0
MISCITED      20          0            0
MIXED         20          0            0

which check caught it                               count
term not in source                                     52
number not in source                                   42
polarity reversed                                      20
population and action from different recommendations    2
```

### The `supported` column is not always zero, and that is a correction

The eight eligibility clauses repeat, so **four of the twenty recombined answers reproduce a pairing the document genuinely contains.** Labelling those unsupported would have been wrong, and the first version did.

The ground truth is now computed by comparing the generator's own recorded fields for equality — never by running the checker being tested. Day 12 spent a section on that circularity; this is the same trap one level up.

### Two bugs the attribution table found

- **Polarity was checked against the wrong thing.** The first version asked whether *any* recommendation in the chunk had matching polarity, which any chunk containing one "do not" satisfies. It passed 45% of the flipped claims. The question is not whether the chunk contains a negative sentence; it is whether *this instruction* is negative in the source.
- **Recombination was checked against a narrowed set.** Filtering candidates by polarity before the recombination check discarded the recommendation the population had been lifted from, so the check had nothing left to compare and passed. Both now have regression tests.

## The two measures, side by side

```
measure                             separation
word overlap, mean over claims           0.935
word overlap, worst claim                0.935
claim checks, all must pass              1.000

 threshold  supported refused  unsupported passed
      0.70                  0                  71
      0.80                  0                  59
      0.90                  0                  36
      0.95                  0                  15
      1.00                  0                  15
```

At **1.00** — demanding that every single word of the answer appear in the source — 15 unsupported answers still pass, because every word of a negated answer *is* in the source. The measure has no headroom left and the errors are still there.

## Why the worst claim and not the average

```
In adults aged 18 and over with newly diagnosed stage 1 disease, measure serum
electrolytes before starting treatment. The guideline also advises routine
genetic testing.

  claim 0: overlap 1.000  ok
  claim 1: overlap 0.000  term not in source: guideline
  answer:  mean 0.500, worst 0.000
```

The two separation columns above are equal only because almost every answer here carries one claim. MIXED is the only kind where mean and minimum differ. That is a fact about this answer set and not a reason to average: a real answer is several sentences, and **averaging is how one fabrication among four quotations gets a pass.**

## And now the part that should be uncomfortable

```
kind                   n  overlap  strict pass
SCOPE_DROPPED         20    1.000           20
EXCEPTION_IGNORED      3    1.000            3

SCOPE_DROPPED      Measure serum electrolytes before starting treatment.
EXCEPTION_IGNORED  In adults who have not tolerated first-line therapy, offer a
                   supervised exercise programme.
```

**Both scorers pass every one of those, and both are right to.** Every word is in the cited chunk, nothing is negated, no number moved, and the population and the action come from the same recommendation.

The first drops the eligibility clause, so a recommendation for *some* patients is restated as advice for all of them — the thing Day 13 spent a day keeping inside one chunk. The second is correct in the chunk it cites and reversed for this patient in section 4, which Day 13 measured as unreachable by any chunking.

So the strict checker scores **1.000 on the six failure modes it was written for and passes both of the two it was not.** That ratio is not a property of the checker. It is a property of the list of failure modes, and I wrote both lists.

> **Groundedness is not the property anybody wants. It is the property that is easy to define.** An answer can be traceable to the line it cites and still be the wrong answer for the patient in front of the clinician, and no amount of tracing closes that gap.

## Run it

```bash
python groundedness.py    # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Day 16 stops adding failure modes by hand and asks what a generator does with the questions Day 14 could not refuse.
