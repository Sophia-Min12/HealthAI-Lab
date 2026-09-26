# HealthAI-Lab

![tests](https://github.com/Sophia-Min12/HealthAI-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — signal processing, risk scoring, de-identification, and clinical retrieval, on data that is entirely synthetic.**

> ECG filtering and R-peak detection, logistic regression as a risk score you can read, de-identification of clinical text, and a guideline-grounded RAG demo. The domain where "the model is 94% accurate" is the beginning of the question, not the end.

**Environment**: Python 3.10+ · NumPy · `pytest` as the test runner.

---

## ⚠️ What this repo is not

**Not a medical device. Not clinical advice. Not validated on patients.**

Every signal in this repo is **synthesised by code in this repo**, and every clinical note is invented. No patient data, real or derived, appears anywhere — which is both an ethical requirement and a practical one, since synthetic data is the only kind whose ground truth is known exactly. When the R-peak detector is tested, it is tested against peaks this repo *placed*, so a miss is unambiguous.

This is a lab for learning the methods. Using any of it on real patients would require regulatory approval, clinical validation, and expertise this repo does not contain.

---

## 🧭 The rules this repo is built on

**Accuracy is the wrong headline.** A screening test for a condition with 1% prevalence is 99% accurate by saying "no" to everyone. Level 2 reports sensitivity, specificity, PPV and the confusion matrix, and treats a single accuracy figure as a warning sign.

**Say which errors you are choosing.** Every threshold trades false negatives against false positives, and in medicine those costs are wildly unequal and *directional*. A model that will not say which one it is minimising has not been specified.

**Calibration is not accuracy.** A risk score that says "30%" should be right about 30% of the time. Level 2 measures that separately, because a model can rank perfectly and still be numerically useless as a probability.

---

## 🗺️ Curriculum Roadmap

**Level 0 · Repo Setup**
- [x] **Day 0** — Repo scaffold, CI, and curriculum roadmap

**Level 1 · Biosignals**
- [x] **Day 1** — A synthetic ECG, and why synthetic is the honest choice
- [x] **Day 2** — Noise: baseline wander, mains hum, motion artefact
- [ ] **Day 3** — Filtering: moving average, and what it does to the peaks
- [ ] **Day 4** — R-peak detection and heart-rate variability

**Level 2 · Risk Scoring**
- [ ] **Day 5** — Logistic regression from scratch, and why the coefficients are readable
- [ ] **Day 6** — Beyond accuracy: sensitivity, specificity, PPV, and prevalence
- [ ] **Day 7** — Thresholds: choosing which error you would rather make
- [ ] **Day 8** — Calibration: a 30% risk that is right 30% of the time
- [ ] **Day 9** — Missing data, and the ways imputation lies

**Level 3 · Clinical Text**
- [ ] **Day 10** — De-identification: rules, recall, and the cost of a miss
- [ ] **Day 11** — Evaluating de-identification when a single leak is a failure
- [ ] **Day 12** — Clinical abbreviations, negation, and "no evidence of"

**Level 4 · Guideline RAG**
- [ ] **Day 13** — Indexing a clinical guideline with its section structure
- [ ] **Day 14** — Retrieval that must cite, and refuse when it cannot
- [ ] **Day 15** — Measuring groundedness: every claim traced to a line
- [ ] **Day 16** — Failure modes: out-of-scope questions and confident nonsense
- [ ] **Day 17** — Capstone: the full pipeline, a CLI, and the honest writeup

---

## 📐 Conventions

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: helpers from earlier days are copied forward with a `# reused from dayNN` comment.
- All data is synthetic and generated from a **seeded** random number generator, so every number in every README is reproducible.
- Every detector is tested against ground truth the generator recorded, never against its own output.

## 🔗 Sibling labs

- [NLP-Lab](https://github.com/Sophia-Min12/NLP-Lab) — tokenization and retrieval, complete. Level 3's de-identification is its regex tokenizer pointed at a harder target.
- [RAG-Lab](https://github.com/Sophia-Min12/RAG-Lab) — Level 4 here is that repo's pipeline with a refusal requirement bolted on.
- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — fitting a logistic regression is gradient descent on a convex loss.

## License

[MIT](LICENSE) © Sophia Min
