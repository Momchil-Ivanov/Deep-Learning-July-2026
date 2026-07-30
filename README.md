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
```

`PYTHONUTF8` is required in the current workspace because its Windows path
contains Unicode characters that cannot be represented by the default CP1251
encoding.

## Repository structure

```text
.
├── artifacts/                 # Small reproducible outputs; model files ignored
├── docs/
│   └── PROJECT_DESIGN_BG.md   # Detailed Bulgarian design explanation
├── scripts/
│   ├── audit_chia.py           # Reproducible data-quality audit
│   └── prepare_chia.py         # Split, overlap, and BIO quality gate
├── src/
│   └── trial_criteria_ner/    # Reusable Python package
├── tests/                     # Unit and data-integrity tests
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

The Jupyter notebook, BIO pipeline, models, and evaluation code will be added
incrementally after their quality gates pass.

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

## Documentation

See [docs/PROJECT_DESIGN_BG.md](docs/PROJECT_DESIGN_BG.md) for the detailed
motivation, architecture, training plan, evaluation strategy, limitations, and
checkpoint portability notes in Bulgarian.

## Primary references

- Kury et al.,
  [Chia, a large annotated corpus of clinical trial eligibility criteria](https://doi.org/10.1038/s41597-020-00620-0).
- Zhang et al.,
  [Transformer-Based Named Entity Recognition for Parsing Clinical Trial Eligibility Criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373041/).
