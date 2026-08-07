# TrialCriteriaNER

Structuring clinical-trial eligibility criteria with transformer-based named
entity recognition.

## Project goal

Clinical-trial eligibility criteria are commonly written as free text.
TrialCriteriaNER investigates whether a compact DistilBERT token classifier can
extract structured medical entities more accurately than a classical
Conditional Random Field (CRF) baseline.

Example:

> Patients with diabetes and HbA1c above 7% receiving metformin.

Expected entities include:

- `diabetes` — `Condition`
- `HbA1c` — `Measurement`
- `7%` — `Value`
- `metformin` — `Drug`

The project is an information-extraction research prototype. It does not decide
whether a real patient is eligible for a trial and is not a medical device.

## Research question

Can a compact transformer identify medical entities in clinical-trial
eligibility criteria more accurately than a CRF model on the same held-out
CHIA test set?

## Data

The project uses the public
[CHIA corpus](https://doi.org/10.1038/s41597-020-00620-0), which contains
expert-annotated eligibility criteria from 1,000 Phase IV interventional trials
registered on ClinicalTrials.gov.

The loader downloads a revision-pinned `chia_without_scope.zip` archive and
parses the original BRAT `.txt`/`.ann` files directly. This avoids the obsolete
Hugging Face loading script, which is incompatible with `datasets>=4`.

For direct comparison with the published transformer benchmark, the modeling
task uses its 11 major entity types: `Condition`, `Device`, `Drug`,
`Measurement`, `Mood`, `Observation`, `Person`,
`Pregnancy_considerations`, `Procedure`, `Temporal`, and `Value`.

The raw annotations are converted to BIO token labels. Train, validation, and
test data are split by NCT ID to prevent criteria from the same trial from
appearing in multiple subsets.

## Models

1. CRF with lexical and contextual token features.
2. DistilBERT with a token-classification head.

Both models are evaluated on the same locked test set using strict entity-level
precision, recall, and F1. Relaxed overlap F1 and per-class results are also
reported.

## Step-by-step validation

Each stage has a quality gate:

1. Python, dependencies, PyTorch, and CUDA.
2. CHIA loading and schema validation.
3. Character-span to BIO-token alignment tests.
4. NCT-level split leakage checks.
5. CRF mini-run and full baseline.
6. DistilBERT smoke test, memory check, and checkpoint resume.
7. Full training and validation.
8. Final held-out evaluation and reproducibility check.

If a gate fails, the project records the error, explains the cause, applies a
targeted correction, and reruns related regression checks before continuing.

## Environment

The verified development environment is:

- Windows 10
- Python 3.11.9
- NVIDIA GeForce GTX 1050 Ti, 4 GB
- PyTorch 2.13.0 with CUDA 12.6

CUDA 12.6 wheels are intentional: the GTX 1050 Ti is a Pascal `sm_61` GPU, which
is not included in current CUDA 13 PyTorch wheels.

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected GPU output includes `True` and `NVIDIA GeForce GTX 1050 Ti`.

Run the data audit:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
python scripts\audit_chia.py --output artifacts\chia_audit.json
python scripts\prepare_chia.py
python scripts\train_crf.py --smoke-test
python scripts\train_crf.py
python scripts\smoke_distilbert.py
python scripts\train_distilbert.py
python scripts\evaluate_test.py
```

`PYTHONUTF8` is required in the current workspace because its Windows path
contains Unicode characters that cannot be represented by the default CP1251
encoding.

The CRF smoke command uses 300 training criteria and 10 L-BFGS iterations. The
full CRF command uses the locked 800-trial training split and writes
reproducible metrics to `artifacts/crf_baseline_metrics.json`. The serialized
model is written to `checkpoints/crf_baseline.pkl`, which is intentionally
ignored by Git.

`smoke_distilbert.py` is a CUDA quality gate: 64 train / 32 validation criteria,
3 AdamW optimizer steps with gradient accumulation 4, checkpoint save/load, and
resume for one additional step. It does not use the locked test set.

`train_distilbert.py` trains on the full locked train split, validates every
epoch, keeps the best checkpoint by validation strict F1, and supports
`--resume-from` for continuing on another machine. The locked test set stays
unused until Stage 8.

`evaluate_test.py` is the one-shot Stage 8 comparison on the locked 100-trial
test split. It performs no additional training.

## Repository structure

```text
.
├── artifacts/                 # Small reproducible outputs; model files ignored
├── docs/
│   └── PROJECT_DESIGN.md      # Detailed English design explanation
├── notebooks/
│   └── TrialCriteriaNER.ipynb # Exam-facing research notebook
├── scripts/
│   ├── audit_chia.py           # Reproducible data-quality audit
│   ├── prepare_chia.py         # Split, overlap, and BIO quality gate
│   ├── train_crf.py            # CRF smoke/full training and evaluation
│   ├── smoke_distilbert.py     # DistilBERT CUDA / resume quality gate
│   ├── train_distilbert.py     # Full DistilBERT training + early stopping
│   └── evaluate_test.py        # One-shot locked test comparison
├── src/
│   └── trial_criteria_ner/
│       ├── baseline.py         # CRF features and entity-level metrics
│       ├── data.py             # CHIA download, parsing, and audit
│       ├── error_analysis.py   # Boundary/type/miss/spurious buckets
│       ├── preprocessing.py    # Splits, overlap policy, and BIO alignment
│       ├── transformer_data.py # Windowed DistilBERT BIO inputs
│       └── transformer_train.py# Train/eval/checkpoint helpers
├── tests/                     # Unit and data-integrity tests
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

Open [`notebooks/TrialCriteriaNER.ipynb`](notebooks/TrialCriteriaNER.ipynb) for the
exam-facing research narrative, plots, and conclusion.

## Current status

Stage 1 is complete:

- repository and branch verified;
- isolated virtual environment created;
- dependencies installed without conflicts;
- CUDA detected by PyTorch;
- a real matrix multiplication completed successfully on the GTX 1050 Ti.

Stages 2 and 3 are also complete:

- project documentation and package structure created;
- revision-pinned CHIA BRAT loader implemented;
- 2,000 documents from 1,000 NCT IDs validated;
- 12,409 non-empty criteria confirmed;
- 34,396 entities across the 11 benchmark labels retained;
- CRLF offsets normalized and two uniquely recoverable offsets repaired;
- zero invalid spans and zero text mismatches remain for model entities;
- six data tests pass.

The complete generated audit is stored in
[`artifacts/chia_audit.json`](artifacts/chia_audit.json).

Stage 4 is complete:

- 12,409 criterion-level examples created with unique IDs;
- deterministic split of 800/100/100 NCT IDs with zero leakage;
- longest-span overlap policy leaves 32,434 non-overlapping entities;
- all criteria align to DistilBERT WordPiece BIO labels;
- median token length is 15, p95 is 56, and only 48 criteria exceed 128 tokens;
- 13 unit and integrity tests pass.

The preprocessing report is stored in
[`artifacts/preprocessing_audit.json`](artifacts/preprocessing_audit.json).

Stage 5 is complete:

- linear-chain CRF trained on all 9,806 training criteria using 141,141 tokens;
- L-BFGS with `c1=0.1`, `c2=0.1`, 100 iterations, and all possible transitions;
- test strict precision `0.6343`, recall `0.5486`, and F1 `0.5884`;
- test relaxed precision `0.7844`, recall `0.6783`, and F1 `0.7275`;
- test strict F1 is `0.6167` for exclusion and `0.5410` for inclusion criteria;
- training took approximately 82.1 seconds on CPU;
- the serialized CRF is approximately 3.27 MB.

The complete per-class baseline report is stored in
[`artifacts/crf_baseline_metrics.json`](artifacts/crf_baseline_metrics.json).

Stage 6 is complete:

- windowed DistilBERT BIO dataset retains all overflow windows (`stride=0`);
- special tokens and padding use label `-100` and are excluded from the loss;
- CUDA smoke on GTX 1050 Ti: losses `3.106 → 2.984 → 2.817` over 3 steps;
- peak allocated VRAM about `1.29 GB` (float32, batch size 1, accum 4);
- checkpoint reload logit difference `0.0`; resume continues and changes weights;
- smoke validation F1 remains low by design (`strict 0.0208`, `relaxed 0.2284`);
- full train split yields `9,856` windows with `42` overflow criteria;
- the locked test set was not used.

The smoke report is stored in
[`artifacts/distilbert_smoke_metrics.json`](artifacts/distilbert_smoke_metrics.json).

Stage 7 is complete:

- DistilBERT trained on all 9,856 train windows for 5 epochs (no early stop);
- AdamW `2e-5`, weight decay `0.01`, batch size 1, gradient accumulation 16;
- best checkpoint at epoch 5 with validation strict F1 `0.6594` and relaxed
  F1 `0.7828`;
- beats the CRF validation strict F1 `0.6339` by `+0.0255`;
- training took about 52.2 minutes; peak allocated VRAM about `1.29 GB`;
- the locked test set was not used.

The training report is stored in
[`artifacts/distilbert_training_metrics.json`](artifacts/distilbert_training_metrics.json).
Best weights live in Git-ignored `checkpoints/distilbert/best/`.

Stage 8 is complete:

- one-shot evaluation on the locked 100 NCT / 1,310-criterion test split;
- no additional training was performed;
- CRF test strict F1 `0.5884`, relaxed F1 `0.7275`;
- DistilBERT test strict F1 `0.6192`, relaxed F1 `0.7583`;
- DistilBERT wins by `+0.0308` strict and `+0.0307` relaxed F1;
- DistilBERT reduces missed entities (`610` vs CRF `1068`) but has more
  spurious predictions (`1019` vs `570`);
- exclusion criteria remain easier than inclusion for both models.

The comparison report is stored in
[`artifacts/test_evaluation.json`](artifacts/test_evaluation.json).

Stage 9 is complete:

- exam-facing notebook `notebooks/TrialCriteriaNER.ipynb` summarizes the
  problem, data, models, plots, previous-research comparison, and conclusion;
- DistilBERT answers the research question positively on the locked test set
  (`0.6192` vs CRF `0.5884` strict F1);
- unit/regression tests and notebook integrity checks pass.

## Documentation

See [docs/PROJECT_DESIGN.md](docs/PROJECT_DESIGN.md) for the detailed
motivation, architecture, training plan, evaluation strategy, limitations, and
checkpoint portability notes.

Start with the notebook for the graded research narrative:
[notebooks/TrialCriteriaNER.ipynb](notebooks/TrialCriteriaNER.ipynb).

## Primary references

- Kury et al.,
  [Chia, a large annotated corpus of clinical trial eligibility criteria](https://doi.org/10.1038/s41597-020-00620-0).
- Zhang et al.,
  [Transformer-Based Named Entity Recognition for Parsing Clinical Trial Eligibility Criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373041/).
