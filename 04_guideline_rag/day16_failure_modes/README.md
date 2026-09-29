# Day 16 · Failure Modes: Out-of-Scope Questions and Confident Nonsense

> **NOT A MEDICAL DEVICE.** The condition, the recommendations and the evidence grades in this folder are all invented by it.

Day 14 built a refusal. Day 15 built a groundedness check. This day puts both in front of a generator and asks what still gets through.

**The generator here has no model in it.** It is extractive and template-driven, and every word it emits is lifted from the chunk it cites — a test asserts that, term by term. It *cannot* hallucinate a word. Everything below survives that restriction, and a fluent model would add failures to these rather than replace them.

## Out of scope, answered fluently

```
Q  Which antibiotic is first-line for meningitis?
A  In adults aged 18 and over with newly diagnosed stage 1 disease, offer a
   thiazide-like diuretic as first-line therapy.
   SG-1 (2024) 3.2.1 First-line therapy [R9]

Q  In adults with an estimated GFR below 30 mL/min, what is the recommended
   dialysis schedule?
A  In adults with an estimated GFR below 30 mL/min, do not offer routine dose
   escalation.
   SG-1 (2024) 4.1 Pregnancy [R14]
```

The citation is real, the section exists, the recommendation is quoted correctly, and the answer is to a question nobody asked.

## What the guards catch

```
guard                delivered  refused  harmful  in-scope refused
no guard                    28        5        8                 0
refusal only                24        9        4                 0
groundedness only           28        5        8                 0
both guards                 24        9        4                 0

kind             n  delivered  harmful
IN_SCOPE        20         20        0
OUT_OF_SCOPE    10          1        1
EXCEPTION        3          3        3
```

The five refusals in the first row are not a guard — they are the cases where the top chunk held no recommendation, so there was nothing to quote.

**The groundedness guard changes nothing: 8 harmful without it, 8 with it.** That is not a defect in Day 15's checker. This generator only quotes, so it cannot produce an ungrounded answer, and a guard against a failure the system cannot commit fires zero times. It would earn its keep against a generator that paraphrases — and every failure below would survive it either way.

### A sentence the generator wrote, caught by a test

The first version emitted `"The guideline does not state."` when the top chunk held no recommendation. A test asserting that every emitted word appears in the cited chunk caught it: that sentence was written by the generator, not quoted. Semantically it was always a refusal, so it is one now — which is what turned the groundedness column into a column of zeros and made the finding above visible.

## And now change nothing at all

Both editions of the guideline are in the index, because a document store contains what has been put in it, and withdrawing the old one is an administrative act that happens later than the revision does. The two editions differ in their numeric thresholds and in nothing else.

Same question set, same guards, same scores — with the chunks added to the index in the other order:

```
insertion order           guard            delivered  cite the withdrawn
current added first       no guard                28                   0
                          refusal only            24                   0
                          groundedness only       28                   0
                          both guards             24                   0

superseded added first    no guard                28                  26
                          refusal only            24                  23
                          groundedness only       28                  26
                          both guards             24                  23
```

**26 of 28 answers now cite a withdrawn guideline.** The delivery and refusal counts are identical in both orders — a test asserts that — so nothing about the system's confidence changed. Only which document it quoted.

BM25 has no notion of recency. The two editions say almost the same words, so they score almost the same, and the tie is broken by which chunk the index happened to see first.

Neither guard moves. Every one of those answers is **perfectly, verifiably grounded** in a chunk that is real, quoted correctly, and out of date — and a test runs Day 15's checker over them to confirm it passes every one.

> A guard that reads the text cannot see a date.

## The exceptions, end to end

```
Q  In adults with an estimated GFR above 60 mL/min, what does the guideline
   recommend about offering a thiazide-like diuretic as first-line therapy?
A  In adults with an estimated GFR above 60 mL/min, offer a thiazide-like
   diuretic as first-line therapy.
   cited       SG-1 (2024) 3.2.1 First-line therapy
   reversed in 4.2 Renal impairment, which was not retrieved
```

Three for three, both guards passing. Day 13 measured that no chunking puts a recommendation in the same chunk as the section that reverses it. Day 15 measured that a groundedness check passes the answer anyway. **This is what that costs at the point of use.**

## What would actually help, and what it is not

None of the fixes is a better score or a stricter checker:

| | |
|---|---|
| **Edition metadata** | A withdrawn document is not ranked lower — it is not in the index |
| **Effective dates** | So "current" is a property of a chunk and not of the order it was loaded |
| **Cross-reference links** | So retrieving 3.2.1 retrieves 4.2 with it, because the document says they belong together |
| **Scope declarations** | So "this guideline does not cover paediatrics" is a fact *in* the index rather than an absence *from* it |

Every one of those is a line from [Day 13's list of what the index has to carry](../day13_indexing_structure/README.md#what-the-index-has-to-carry). The guards in this file operate on text that has already lost them.

## Run it

```bash
python failure_modes.py   # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Day 17 assembles the whole thing and counts what it costs.
