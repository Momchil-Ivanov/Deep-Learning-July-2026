# TrialCriteriaNER: detailed project design

## 1. Idea and goal

**TrialCriteriaNER** is a deep-learning project for automatic recognition of
medical entities in clinical-trial inclusion and exclusion criteria.

Eligibility criteria are usually free text. For example:

> Patients with diabetes and HbA1c above 7% receiving metformin.

The desired structured output is:

- `diabetes` → `Condition`
- `HbA1c` → `Measurement`
- `7%` → `Value`
- `metformin` → `Drug`

The main research question is:

> Can a compact transformer identify medical entities in eligibility criteria
> more accurately than a classical CRF model?

The project extracts information from text. It does not decide whether a
specific patient should join a trial and is not a medical device.

## 2. Real-world problem

Participant selection is an important and labor-intensive part of clinical
trials. Criteria are diverse and contain medical terms, numeric thresholds,
temporal conditions, and negations. Manually converting them into structured
queries is slow and hard to scale.

Named Entity Recognition (NER) is the first step toward structuring this text.
Extracted entities can later support:

- search and filtering of clinical trials;
- preparation of formal eligibility rules;
- electronic prescreening under clinician supervision;
- analysis of the complexity and frequency of different criterion types.

## 3. Why a neural network is used

A neural network is justified by the nature of the data:

1. **Meaning depends on context.** The same word can play different roles
   depending on surrounding words.
2. **Medical language has many paraphrases.** A lexicon or regular expressions
   cannot reliably enumerate every formulation.
3. **Both boundary and type must be recovered.** For example, the model must
   treat `body mass index` as one span rather than unrelated words.
4. **Transformers use bidirectional context.** Each token representation depends
   on text before and after it.
5. **Transfer learning reduces the need for a huge labeled set.** Pretrained
   DistilBERT already encodes language representations and is fine-tuned on the
   CHIA labels.

The usefulness of the neural network is not assumed. DistilBERT is compared with
a classical Conditional Random Field (CRF) on the same locked test set.

## 4. Why this topic fits the exam project

- It solves a clearly defined real problem tied directly to clinical trials.
- The deep-learning model is essential to the solution, not a cosmetic add-on.
- There is a public, expert-annotated dataset without personal patient records.
- Gold-standard labels enable measurable precision, recall, and F1.
- Published results exist for comparison.
- The scope is realistic for the deadline and a GTX 1050 Ti with 4 GB VRAM.
- The project supports meaningful visualizations, tests, and error analysis.

Other considered topics were riskier:

- Trial-termination prediction is ambiguous because `Completed` does not mean
  clinical success, and registry fields may leak post-outcome information.
- Patient-to-trial matching lacks easily available real patient profiles and
  reliable relevance labels.
- Reinforcement learning has no natural agent, environment, and reward here and
  would add complexity without scientific necessity.

## 5. Data: CHIA

We use the public
[CHIA](https://doi.org/10.1038/s41597-020-00620-0) corpus built from
ClinicalTrials.gov criteria:

- 1,000 Phase IV interventional clinical trials;
- 12,409 annotated eligibility criteria;
- 44,616 raw text-bound annotations in the archive used here;
- 30 raw annotation types, including model entities and quality/error tags.

This project focuses on entity recognition. Relationship extraction is out of
scope.

The official `bigbio/chia` Hugging Face dataset uses a Python loading script.
`datasets==5.0.0` no longer supports those scripts, so direct `load_dataset`
fails with an incompatibility error. Instead of downgrading the library, the
loader downloads a revision-pinned `data/chia_without_scope.zip` archive and
parses the original BRAT `.txt` / `.ann` files. Acquisition and parsing are
therefore explicit and reproducible.

For comparison with the published transformer benchmark we use the same 11 major
entity types:

- `Condition`, `Device`, `Drug`, `Measurement`;
- `Mood`, `Observation`, `Person`;
- `Pregnancy_considerations`, `Procedure`, `Temporal`, `Value`.

These classes contain 34,396 entities before overlap resolution. The remaining
10,220 raw annotations are construct or quality/error categories and are not used
as model labels.

Each record keeps:

- `NCT ID`;
- inclusion or exclusion type;
- original text;
- character offsets;
- entity type.

Before modeling we check empty texts, invalid offsets, duplicate annotations,
unknown labels, class imbalance, and nested/overlapping spans.

The completed audit confirmed:

- 2,000 inclusion/exclusion documents from exactly 1,000 NCT IDs;
- 12,409 non-empty criteria lines;
- 1,000 inclusion and 1,000 exclusion documents;
- zero duplicate entity IDs;
- zero invalid spans and zero text mismatches for the 11 model labels;
- 2 repaired offsets where the annotation text had exactly one match;
- 2,523 overlapping model-entity pairs, of which 2,515 were nested;
- 3 empty documents and 66 documents without any of the 11 model entities.

## 6. BIO representation

Character spans are converted to token labels:

- `B-Type` — beginning of an entity;
- `I-Type` — continuation of an entity;
- `O` — token outside any entity.

Example:

| Token | BIO label |
|---|---|
| HbA1c | B-Measurement |
| above | O |
| 7 | B-Value |
| % | I-Value |

Standard BIO cannot keep two overlapping entities on one token. We therefore use
a predetermined, documented policy: keep the longest span and report how many
annotations are dropped by the transformation.

The implemented policy ranks entities by envelope length and covered length,
keeps the longest span under true character overlap, and does not merge merely
adjacent spans. This reduces 34,396 model entities to 32,434 with zero remaining
overlapping pairs. The published research notebook also merges adjacent spans and
treats discontinuous entities as one envelope; applied to the current archive,
that code yields 31,996 entities versus 31,944 in the paper. Our policy is closer
to a standard NER definition, and the difference is documented explicitly.

A WordPiece token can cross the boundary between two adjacent character spans.
For example, one `[UNK]` token may cover the end of `Measurement` and the start
of `Value`. In that case the token is assigned to the entity with larger
character overlap under a deterministic tie-break. All 12,409 criteria passed
token alignment without error.

Train, validation, and test splits are created by whole `NCT ID` groups, not by
individual criteria. This prevents text from the same trial appearing in multiple
subsets.

We use seed `42` and an 800/100/100 NCT split. Resulting sizes:

- train: 9,806 criteria and 25,625 entities;
- validation: 1,293 criteria and 3,285 entities;
- test: 1,310 criteria and 3,524 entities;
- NCT leakage: 0.

The DistilBERT token audit showed median 15, p95 56, p99 96, and maximum 507
tokens. Only 48 criteria exceed 128 tokens and 5 exceed 256. Training
`max_length=128` remains suitable for 4 GB VRAM, while long criteria are handled
with overflow windows rather than silent truncation.

## 7. Models

### 7.1. CRF baseline

The CRF uses regex tokenization and local features:

- lowercase form;
- prefixes and suffixes of length 2 and 3;
- token shape, length, capitalization, and digits;
- beginning/end-of-sequence markers;
- lowercase, shape, title-case, and uppercase features of neighboring tokens.

This is a classical sequence-labeling baseline. It shows what hand-crafted local
dependencies can achieve without a transformer.

A smoke test with 300 training criteria and 10 L-BFGS iterations confirmed that
feature extraction, BIO prediction, entity-level evaluation, and strict/relaxed
quality checks work end to end.

The full model was trained only on the 800 training NCT IDs: 9,806 criteria and
141,141 regex tokens. Fixed settings were `c1=0.1`, `c2=0.1`, 100 L-BFGS
iterations, and `all_possible_transitions=True`. No tuning was done on the test
set.

Results:

- validation strict precision `0.6710`, recall `0.6008`, F1 `0.6339`;
- validation relaxed precision `0.7980`, recall `0.7145`, F1 `0.7540`;
- test strict precision `0.6343`, recall `0.5486`, F1 `0.5884`;
- test relaxed precision `0.7844`, recall `0.6783`, F1 `0.7275`;
- test strict F1 by criteria type: `0.6167` for exclusion and `0.5410` for
  inclusion.

CPU training took about 82.1 seconds. The pickle model is about 3.27 MB and is
stored in Git-ignored `checkpoints/crf_baseline.pkl`. The detailed per-type JSON
report is in `artifacts/crf_baseline_metrics.json`. The strongest test strict
results are for `Person` and `Value`; rarer or more ambiguous types such as
`Device`, `Mood`, `Observation`, and `Pregnancy_considerations` remain harder.

### 7.2. DistilBERT

Pipeline:

1. Text is split into WordPiece tokens.
2. Character offsets are aligned to token positions.
3. DistilBERT builds a contextual embedding for each token.
4. A linear classification head outputs BIO-class probabilities.
5. Cross-entropy trains the model; padding and special tokens get label `-100`
   and do not enter the loss.
6. Predicted BIO labels are reconstructed into text spans.
7. Criteria longer than `max_length=128` are kept via overflowing windows with
   `stride=0`, instead of silent truncation.

Initial settings for 4 GB VRAM:

- `max_length=128`;
- physical batch size 1–2;
- gradient accumulation to an effective batch around 16;
- AdamW learning rate about `2e-5`;
- weight decay `0.01`;
- warmup about 10%;
- 3–5 epochs with early stopping;
- gradient checkpointing if needed;
- checkpoint and evaluation after every epoch.

FP16 is used only if a short benchmark shows stability and benefit. The GTX
1050 Ti has no Tensor Cores, so FP16 is not automatically faster.

The Stage 6 CUDA smoke test confirmed the pipeline before full training:

- 64 train and 32 validation criteria; locked test set unused;
- 3 AdamW steps with batch size 1 and gradient accumulation 4;
- loss: `3.106 → 2.984 → 2.817`;
- peak allocated VRAM about `1.29 GB`;
- checkpoint reload logit difference `0.0`;
- after resume, parameters change and training state is restored;
- the full train split yields `9,856` windows and `42` overflow criteria;
- smoke validation F1 is low by design (`strict 0.0208`) because the goal is
  pipeline correctness, not a final score.

The report is in `artifacts/distilbert_smoke_metrics.json`. Checkpoints are in
Git-ignored `checkpoints/distilbert-smoke/`.

Full training (Stage 7) used the same locked train/validation splits and float32
settings suitable for the GTX 1050 Ti:

- batch size 1, gradient accumulation 16 (effective batch 16);
- AdamW `2e-5`, weight decay `0.01`, linear warmup 10%;
- maximum 5 epochs with patience 2 on validation strict F1;
- checkpoint after every epoch in `checkpoints/distilbert/latest/` and best copy
  in `checkpoints/distilbert/best/`;
- `--resume-from` allows continuation on another machine.

Validation results:

- epoch 1: loss `1.2754`, strict F1 `0.5924`;
- epoch 2: loss `0.5759`, strict F1 `0.6453`;
- epoch 3: loss `0.4617`, strict F1 `0.6542`;
- epoch 4: loss `0.3927`, strict F1 `0.6541`;
- epoch 5: loss `0.3506`, strict F1 `0.6594`, relaxed F1 `0.7828`.

The best checkpoint is epoch 5. Validation strict F1 `0.6594` beats the CRF
baseline `0.6339` by `+0.0255`. Training took about 52.2 minutes with peak
allocated VRAM about `1.29 GB`. The locked test set was unused at this stage.

The detailed report is in `artifacts/distilbert_training_metrics.json`.

### 7.3. Locked test comparison

After selecting the best DistilBERT checkpoint by validation, a one-shot
evaluation was run on the locked test set (100 NCT IDs, 1,310 criteria). No
additional training or test-set tuning was performed.

Results:

- CRF: strict F1 `0.5884`, relaxed F1 `0.7275`;
- DistilBERT: strict F1 `0.6192`, relaxed F1 `0.7583`;
- delta: `+0.0308` strict and `+0.0307` relaxed in favor of DistilBERT;
- inclusion strict F1: CRF `0.5410`, DistilBERT `0.5503`;
- exclusion strict F1: CRF `0.6167`, DistilBERT `0.6628`.

The short error analysis shows that DistilBERT reduces missed entities
(`610` vs `1,068` for CRF) but produces more spurious predictions
(`1,019` vs `570`). Boundary errors remain common for both models. The full JSON
report with per-type metrics and example errors is in
`artifacts/test_evaluation.json`.

These results answer the research question: the compact DistilBERT recognizes
medical entities in eligibility criteria more accurately than the classical CRF
on the same held-out test set.

### 7.4. Final notebook and reproducibility

The exam research notebook is `notebooks/TrialCriteriaNER.ipynb`. It summarizes
the problem statement, previous research, data, training curves, locked-test
comparison, error analysis, and conclusion in English. Heavy training runs are
not repeated in the notebook; it loads locked JSON artifacts and demonstrates the
BIO pipeline on a short example.

Reproducibility commands:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
python -m pytest
python scripts\evaluate_test.py
```

Stage 9 closes the project: notebook, documentation, and regression tests are
aligned with the final results.

## 8. Staged work and quality gates

We do not move to the next stage until the current one is verified.

1. **Environment and GPU** — imports, dependency check, and a real CUDA tensor op.
2. **CHIA loader** — schema, sizes, missing data, and manual record checks.
3. **BIO alignment** — unit tests and round-trip examples.
4. **Data split** — automatic check for shared NCT IDs.
5. **CRF** — mini-run before full training.
6. **DistilBERT smoke test** — loss, gradients, VRAM, evaluation, and checkpoint
   save/resume on a small sample.
7. **Full training** — monitor loss, F1, time, and VRAM.
8. **Test evaluation** — once, after model selection on the validation set.
9. **Reproducibility** — clean notebook run, tests, and README.

On failure we stop at that stage and document:

- the exact error or incorrect behavior;
- the proven or most likely cause;
- what was checked and ruled out;
- the applied correction;
- the result of the rerun and regression check;
- a safe fallback if the issue remains.

## 9. Evaluation

The primary metric is entity-level strict micro-F1. We also report:

- strict precision, recall, and F1;
- relaxed overlap precision, recall, and F1;
- F1 by entity type;
- results for inclusion and exclusion criteria;
- CRF versus DistilBERT on the same test set.

Token accuracy is not the main metric because the dominant `O` class can produce
misleadingly high values.

Error analysis groups:

- wrong boundary;
- wrong entity type;
- missed entity;
- spurious entity;
- truncation;
- nested-annotation artifacts.

## 10. Previous research

The minimum core sources are:

1. Kury et al.,
   [Chia, a large annotated corpus of clinical trial eligibility criteria](https://doi.org/10.1038/s41597-020-00620-0).
2. Zhang et al.,
   [Transformer-Based Named Entity Recognition for Parsing Clinical Trial Eligibility Criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373041/).

The second study selects 11 major CHIA types, keeps the longest span under
overlap, and splits 1,000 trials into 800/100/100. It reports strict F1 `0.6339`
for BERT and `0.6578` for the best RoBERTa-MIMIC-Trial model. We follow the same
high-level protocol, but our split seed and scientifically stricter overlap
policy do not fully reproduce the original preprocessing. Comparison with
published F1 is therefore indicative, while CRF versus DistilBERT on our shared
test set is the direct experimental comparison. DistilBERT is a smaller model
chosen for 4 GB VRAM, so it is not assumed in advance to match the best published
result.

## 11. Checkpoints and continuing elsewhere

Physical location does not change training. A checkpoint should contain model
weights, optimizer state, scheduler state, trainer state, and random state.
Resume continues from the last saved step.

The code uses relative paths. The environment is restored from
`requirements.txt`. Checkpoints are not committed to GitHub because they usually
exceed the 100 MB limit. They can be synced separately via MEGA, Google Drive, or
Hugging Face Hub.

A short save/resume test was completed before full training. On another NVIDIA
machine the checkpoint remains portable if the same training settings are kept.

## 12. Verified local environment

On 27 July 2026 the following checks passed:

- Python 3.11.9, 64-bit;
- NVIDIA driver 582.53;
- NVIDIA GeForce GTX 1050 Ti, 4 GB, compute capability 6.1;
- PyTorch 2.13.0 with CUDA 12.6 runtime;
- `torch.cuda.is_available() == True`;
- successful CUDA matrix operation;
- successful import of all direct dependencies;
- `pip check` with no dependency conflicts.

A CUDA 12.6 PyTorch wheel is used because Pascal `sm_61` is not included in
current CUDA 13 wheels.

## 13. Limitations and ethics

- CHIA covers Phase IV trials; results may not generalize to all phases and
  therapeutic areas.
- BIO conversion loses some nested structure.
- Rare entity classes are likely to have lower F1.
- Registry text is not equivalent to an electronic health record.
- The model is not validated for clinical decision-making.
- Predictions must be reviewed by a qualified specialist in any real use.

## 14. Alignment with exam criteria

- **Problem statement:** a real, clearly bounded clinical-trial NLP problem.
- **Layout:** structured Jupyter notebook and separate source modules.
- **Code quality:** functions, type hints, docstrings, and tests.
- **Previous research:** at least the two core sources and result comparison.
- **Data:** reproducible loading, cleaning, BIO formatting, and data audit.
- **Testing:** unit tests, leakage checks, validation/test split, and smoke tests.
- **Visualization:** class distributions, training curves, per-class F1, and
  example predictions.
- **Communication:** English notebook and README, clear limitations, and a
  research story from problem to conclusion.
