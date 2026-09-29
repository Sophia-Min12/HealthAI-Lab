# Day 13 · Indexing a Clinical Guideline With Its Section Structure

> **NOT A MEDICAL DEVICE.** *Synthetic Cardiometabolic Syndrome* is not a real condition and none of the recommendations in this folder is real clinical guidance. The drug classes are real, the indication is fiction, and the document exists to be indexed.

A guideline is not prose. It is a numbered tree in which a recommendation carries three things written in different places: **what to do**, **who to do it to**, and **how sure anyone is**.

```
R1. In adults aged 18 and over with newly diagnosed stage 1 disease,
    measure serum electrolytes before starting treatment.
    [Strength: strong - Evidence: B]

section       2.1 Initial assessment
who           In adults aged 18 and over with newly diagnosed stage 1 disease
what          measure serum electrolytes before starting treatment
how sure      strength strong, evidence B
```

Three separable statements in one sentence. *"Offer an ACE inhibitor"* and *"in adults with an estimated GFR below 30"* are different instructions, and a retriever that returns one without the other has not lost detail — **it has changed the guidance.**

## Fixed-size chunking, at the sizes people use

```
strategy           chunks  mean  intact  severed  orphan  one section  recs/chunk
fixed 200              32   198     30%       14       4          50%         1.2
fixed 400              16   395     85%        3       1          19%         1.8
fixed 800               8   785    100%        0       0          12%         3.2
fixed 1600              4  1544    100%        0       0           0%         6.0
by section             14   380    100%        0       0         100%         1.4
```

At 200 characters, **14 of 20 recommendations arrive severed from their eligibility clause**, and four chunks contain a bare `[Strength: ... Evidence: ...]` tag attached to nothing:

```
R4 arrives as: ...[Strength: conditional - Evidence: C]

               2.2 Investigations
               This section should be read alongside the accompanyi...
```

The fix is not subtle and not clever: **chunk on the headings the document already has.** A guideline is a tree and the tree is written down in the text.

### Read the last two columns before reaching for a bigger chunk

At 1600 characters nothing is severed — because a chunk that size spans four sections. None of them can be attributed to one section, and retrieving any chunk hands back **six recommendations to answer a question about one**.

Chunking by section is the only row good in both directions at once, and it is good in both *by not choosing a size at all*. The document chose it.

An earlier version of this file measured section attribution as "does the chunk contain a heading", which scored the 1600-character chunks at 100%. A chunk straddling four headings has no section path; containing one is not the same as belonging to one.

## And the part chunking cannot fix

```
strategy                    exceptions  together   apart
fixed 400                            3         0       3
fixed 1600                           3         0       3
by section                           3         0       3

R5  in 3.1 Non-pharmacological management
    reversed in 4.1 Pregnancy, 3225 characters later
R7  in 3.2.1 First-line therapy
    reversed in 4.2 Renal impairment, 2826 characters later
R10 in 3.2.2 Second-line therapy
    reversed in 4.3 Frailty and older people, 2433 characters later
```

**No chunking gets those into one chunk, and the chunker is not the reason.** Section 3 says what to do and section 4 says when not to, because that is how guidelines are written — the general case first, the exceptions in their own chapter.

A retriever asked *"what is first-line therapy"* will rank section 3 above section 4 on every lexical measure there is, and return a recommendation the document contradicts between 2433 and 3225 characters further down.

The only chunking that keeps them together is the one that gives up on chunking — a single chunk holding the whole document. A test asserts exactly that, because it is the proof that the obstacle is the genre and not the code.

## What the index has to carry

Storing the text is not enough. Each chunk needs:

| | |
|---|---|
| **The section path** | so `3.2.1 First-line` is distinguishable from `3.2.2 Second-line` when both chunks say "offer an ACE inhibitor" |
| **The recommendation ids it contains** | so a retrieved chunk can be checked for whether it is whole |
| **The grades** | so a strong recommendation from evidence A is not presented identically to a conditional one from evidence D |
| **Cross-references** | so section 4 can be pulled in when section 3 is retrieved |

```
strength      evidence   count
conditional   B              3
conditional   C              4
conditional   D              2
strong        A              2
strong        B              3
strong        C              6
```

A chunk that arrives without its grade reads as advice, and the 2 conditional-from-D recommendations then read exactly like the 2 strong-from-A ones. The tag is the only thing separating them, and it is one character range away from being in a different chunk.

### A note carried forward from Day 12

Two of the eligibility clauses are:

```
In adults with stage 2 disease and established end-organ damage
In adults with stage 2 disease and no evidence of end-organ damage
```

They share every content word and mean opposite things. Day 12 spent a day on that; Day 15 has to score a retriever on it.

## Run it

```bash
python guideline.py       # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Day 14 retrieves against this index and has to decide what to do when the best chunk is not good enough.
