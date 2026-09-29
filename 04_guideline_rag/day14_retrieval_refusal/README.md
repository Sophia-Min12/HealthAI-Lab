# Day 14 · Retrieval That Must Cite, and Refuse When It Cannot

> **NOT A MEDICAL DEVICE.** The condition, the recommendations and the evidence grades in this folder are all invented by it.

BM25 over Day 13's section chunks, and a rule about when to say nothing.

## A retriever has no "nothing" option

**The easy case** — *"How should neonates be managed?"* No term in it appears anywhere in the document, so every chunk scores 0.00 and the ranking is just the chunks in document order. A score of zero is a usable signal.

**The hard case** — *"In adults with an estimated GFR below 30 mL/min, what is the recommended dialysis schedule?"* The document has no dialysis chapter. It answers anyway, and it answers confidently:

```
  7.71  4.1 Pregnancy
  7.42  3.1 Non-pharmacological management
  5.35  4.3 Frailty and older people
```

That is not a bug. Ranking is what a retriever does, and a ranked list of the least-bad chunks is what it returns when none of them is any good. Somebody downstream has to decide the best of them is not good enough, and **nothing in the ranking says so**.

## When the answer is in there, retrieval finds it

```
k        hit@k
1        0.925
2        1.000
3        1.000
5        1.000
```

So the retrieval half is not the problem. Everything below is about the other half: deciding whether to use what came back.

## The score is mostly a measurement of the question's length

Every answerable question was asked twice — a short form and a long one, both wanting the same chunk:

```
[short] electrolytes serum measure?                            top score  6.39
[long ] In adults aged 18 and over with newly diagnosed...     top score 20.04

criterion             short     long    ratio
top_score              6.63    17.62     2.66
margin                 4.57     8.23     1.80
score_per_term         2.21     1.05     0.48
```

BM25 normalises for the length of the **document** — that is what `b` is for, and a test pins it. Nothing anywhere normalises for the length of the **query**, because a longer query genuinely does carry more evidence. The trouble is that it carries more of it whether or not the document can answer:

```
answerable, short     4.35  service referral consider?
unanswerable, long    7.71  In adults with an estimated GFR below 30 mL/min...
```

A single absolute threshold set between those two numbers **refuses the question the document answers and accepts the one it does not.**

## So refuse on something that is not a length

```
criterion           separation  long/short
top_score                0.965        2.66
margin                   0.943        1.80
score_per_term           0.985        0.48
```

*Separation* is the probability that a random answerable question scores above a random unanswerable one — the rank-based AUC, computed directly. 0.5 is no information at all.

The **margin** between the best chunk and the second best is the criterion that ought to be immune to query length. It is not: it still runs 1.8× higher on long questions, and it separates 0.943 against the raw score's 0.965 — *worse*, not better. Both halves of that are the opposite of what this file predicted before the table was printed, and the prediction is still in the git history.

Dividing by the number of query terms is the crude fix and the one that works, separating 0.985. It also **over-corrects** — short questions now score 2.1× what long ones do. It is the best available criterion here and it is not a principled one.

## And now it is Day 7 again

```
 threshold  refused, answerable  answered, unanswerable  correct
      0.00                 0.0%                  100.0%       37
      0.35                 0.0%                   50.0%       37
      0.71                 0.0%                   20.0%       37
      1.06                25.0%                    0.0%       27
      1.41                50.0%                    0.0%       17
      2.12                67.5%                    0.0%       12
      2.83                95.0%                    0.0%        2
```

No row has both middle columns at zero, and that is not an artefact of the grid. Searching **every threshold any question actually produces**, the best either error can be made together is 0.0% refused and 10.0% answered, at 0.84. A test asserts it over the same exhaustive set, so it is a statement about the data rather than about the spacing of a table.

Every threshold is a statement about which error is worse, and the statement is being made whether or not anybody writes it down.

A refusal costs a clinician a lookup. A confident citation of a chunk that does not answer the question costs them the belief that the system checked. Those are not the same price, and the ratio between them is not 1.

## What an answer has to carry

At threshold 0.90 on score-per-term: 35 answered, 5 refused, 1 of 10 out-of-scope questions answered anyway.

```
Q measure serum electrolytes?
  -> 2.1 Initial assessment [R1, R2]
Q In adults aged 18 and over with newly diagnosed stage 1 disease...
  -> 2.1 Initial assessment [R1, R2]
Q routinely screening asymptomatic?
  -> 2.2 Investigations [R3, R4]
```

A section path and the recommendation ids the chunk contains. Without both, an answer cannot be checked at all.

## The ones that get through anything

```
0.99  Which antibiotic is first-line for meningitis?
      retrieved: 3.2.1 First-line therapy
0.77  In adults with an estimated GFR below 30 mL/min, what is the
      recommended dialysis schedule?
      retrieved: 4.1 Pregnancy
```

These are out of scope and they are not nonsense. They ask about adults with this condition, in this document's own vocabulary, about something it does not cover. **No threshold on a lexical score separates them, because lexically they belong.**

### A determinism bug worth keeping

The first version built each short question from the three highest-IDF terms of the recommendation, deduplicating with `set()`. Python randomises string hashing per process, so the term order — and therefore every number in this file — changed from run to run. `dict.fromkeys` plus an explicit tie-breaker fixed it, and a test now runs the generator in three subprocesses and asserts the outputs are identical.

The repo's rule is that randomness is always seeded. A set is not random, and it was not seeded either.

## Run it

```bash
python retrieval.py       # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Day 15 has to establish what it would mean for an answer to be *supported* at all. Day 16 puts a generator on the end and asks what happens to the questions that got through.
