# Day 10 · De-identification: Rules, Recall, and the Cost of a Miss

> **NOT A MEDICAL DEVICE.** Every note in this repo is invented by it. No patient data, real or derived, appears anywhere.

300 synthetic notes whose generator **records every identifier it inserts**, and a good-faith rule-based de-identifier scored against that record. Same discipline as Day 1's R peaks: a miss is not a matter of opinion.

```
Nephrology - Discharge Summary            Nephrology - Discharge Summary

Patient: Emeka Graves   MRN 4804184       Patient: [NAME]   [MRN]
Date of admission: 04/14/2019             Date of admission: [DATE]

This 30-year-old patient was admitted     This 30-year-old patient was admitted
under the care of Dr Abadi with COPD.     under the care of Dr [NAME] with COPD.
Background includes Paget's disease       Background includes [NAME]'s disease
of bone, stable.                          of bone, stable.
Case discussed with Rasmussen at MDT.     Case discussed with [NAME] at MDT.

Address: 153 Fairhaven Road, Larkfield    Address: [ADDRESS]
Contact: 843.983.4415                     Contact: [PHONE]
```

Look at the fifth line of the output. `Paget's disease of bone` has become `[NAME]'s disease of bone`. Nothing leaked; the diagnosis is gone.

## The headline number is good and it is the wrong number

```
category      in corpus   caught   recall
NAME               1026      969    0.944
MRN                 300      259    0.863
DATE                600      560    0.933
PHONE               300      300    1.000
AGE                  27       27    1.000
ADDRESS             300      300    1.000
EMAIL               145      145    1.000
ALL                2698     2560    0.949

span recall                0.949
span precision             0.976
notes fully de-identified  194/300 = 0.647
```

**94.9% span recall. 64.7% of notes.**

A note is not partly safe. The span figure is the one that gets published; the note figure describes what happens when the corpus is released. With nine identifiers in the average note, a note needs every one of them caught, and 5% per identifier compounds into a third of the corpus.

## Misses are not spread evenly

```
138 misses, in 106 notes

category       misses    misses in a note       notes
NAME               57    1                         79
MRN                41    2                         22
DATE               40    3                          5

independence model  0.949^9.0 = 0.624
measured                          0.647
```

The independence model gets the order of magnitude right and the number wrong, and **the direction is the point**. Misses concentrate: a note written in a date format the rules do not carry has *two* dates in that format. Fewer notes are hit than independent errors would give, because the same notes absorb several.

That is not reassuring. A systematic failure is the kind a larger corpus does not average away.

## What was missed, and why it was always going to be

```
DATE     '3/23/19'
NAME     'Bergström'
NAME     'Astrid Bergström'
MRN      '1741227'
NAME     'Hodgkin'
```

- **The date misses** are one format the pattern list does not carry. It is rare in this corpus for exactly the reason it would be rare in a development sample — and that is *why* the rules do not have it. The list is always one hospital behind.
- **The MRN misses** are bare numbers with no label. A seven-digit run in free text is indistinguishable from an accession number without reading the sentence.
- **The name misses** are people not in the gazetteer, mentioned with no title and no field label in front of them. This one is not fixable by adding names. Names are an open class; a list is a snapshot of a population, and the next patient is under no obligation to appear in it.

## The other kind of failure, and the trade it forces

`Parkinson` is a surname. `Parkinson's disease` is a diagnosis. The difference is not in the word.

```
gazetteer                   recall  precision   clean  eponyms lost
with eponym surnames         0.949      0.976   0.647      63/92
without them                 0.943      1.000   0.603       0/92
```

Keep the eponyms on the list and **63 of 92 diagnoses are deleted** — while every number in the recall table stays clean, because over-redaction is invisible to recall by construction. Strike them off and precision goes to a perfect 1.000, and the patients actually *called* Parkinson walk out of the building named: recall falls, and the clean-note rate falls with it.

**Neither column is the safe one.** The first destroys clinical meaning silently; the second is a leak. This is the trade, not a bug awaiting a fix.

### The same bug, found in the address rule

The first version of the address pattern used `\s`, which matches a newline. `, Larkfield` ran straight into the `Contact:` label on the line below:

```
Address on file: [ADDRESS]: [PHONE]     <- a field name eaten
```

It swallowed a label, **scored a clean hit on the address**, and appeared in no recall figure anywhere. Over-capture is invisible to recall by construction — the section above arriving early and uninvited. The pattern now uses `[ \t]`, and a test pins it.

## A rule that looks free, and the reason it looks free

The conservative rules skip bare numbers. Turn on `\b\d{6,8}\b`:

```
rules                           recall  precision   clean
conservative                     0.949      0.976   0.647
plus bare 6-8 digit numbers      0.964      0.977   0.733

the aggressive rule costs 0 additional false positives here.
```

Recall up, clean notes up, **zero cost**. Ship it?

No. This corpus contains no six-to-eight-digit number that is not an MRN — no accession numbers, no device serials, no pager IDs — so there is nothing for the rule to trip over. A rule evaluated on the corpus it was written for always looks free. Add two such numbers and it bills for them immediately, which a test does.

The honest statement is not *"this rule is safe."* It is *"this corpus cannot tell you whether this rule is safe"* — and the distance between those sentences is Day 11.

## The cost of a miss

94.9% span recall sounds like a 5% problem. It is a **35% problem**: 106 of 300 notes carry at least one identifier out of the door.

Day 7 chose a threshold from a cost ratio, and that arithmetic still applies — but the ratio is not 5:1. A false positive costs a word of clinical text. A false negative costs a person their medical privacy, permanently, in a corpus that has already been copied. There is no ratio at which a leak is priced acceptably, which is why de-identification is not a threshold-tuning problem.

### What is out of scope, stated rather than hidden

HIPAA Safe Harbor lists eighteen identifiers. This day implements seven.

| Not implemented | Why it matters |
|---|---|
| Device identifiers | Serial numbers of implanted devices |
| Vehicle identifiers | Licence plates, in accident narratives |
| Biometric identifiers | Fingerprints, voiceprints — not text at all |
| URLs and IP addresses | Common in pasted referral text |
| **Any other unique number** | The catch-all clause, and the reason a rule list can never be declared complete |

## Run it

```bash
python deid.py            # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

Every number above was scored against ground truth this file wrote down. A real corpus has no such record, and its evaluation set is annotated by people holding the rules that are being tested. **Day 11 asks how you would know any of this.**
