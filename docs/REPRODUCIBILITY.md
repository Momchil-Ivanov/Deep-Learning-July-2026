# Reproducibility and examiner verification

This guide separates fast verification of the committed evidence from full model
retraining. The fast path requires neither the CHIA download nor model
checkpoints.

## 1. Fresh Windows clone

Use Python 3.11 in PowerShell:

```powershell
git clone `
  --branch clinical-trial-ner `
  --single-branch `
  https://github.com/Momchil-Ivanov/Deep-Learning-July-2026.git `
  trial-criteria-ner
Set-Location trial-criteria-ner

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

`PYTHONUTF8` prevents encoding failures when the repository path contains
characters outside the active Windows code page.

## 2. Fast offline verification

Run:

```powershell
python -m pip check
python -m pytest
python scripts\verify_submission.py
```

The submission validator checks the seven committed JSON reports, metric ranges,
strict/relaxed relationships, CRF/DistilBERT comparison arithmetic, locked-test
isolation, split consistency, and saved notebook execution outputs. A successful
run ends with a JSON object whose `status` is `"ok"`.

The tests and validator do not download CHIA and do not require a GPU or model
weights.

## 3. Execute the research notebook

The repository includes the notebook with saved tables and plots. Verify that it
also executes from a clean clone:

```powershell
python -m nbconvert `
  --to notebook `
  --execute notebooks\TrialCriteriaNER.ipynb `
  --output TrialCriteriaNER.verified.ipynb `
  --output-dir artifacts `
  --ExecutePreprocessor.timeout=300
```

The generated `artifacts\TrialCriteriaNER.verified.ipynb` is ignored by Git. The
live DistilBERT demo prints a clear skip message when the Git-ignored best
checkpoint is absent; the rest of the notebook still executes.

## 4. CUDA verification

Full DistilBERT training requires a CUDA GPU. The recorded environment used an
NVIDIA GeForce GTX 1050 Ti with the CUDA 12.6 PyTorch build:

```powershell
python -c "import torch; available=torch.cuda.is_available(); print(torch.__version__); print(available); print(torch.cuda.get_device_name(0) if available else 'CUDA unavailable')"
```

The fast checks remain valid on a CPU-only examiner machine.

## 5. Full experiment reproduction

Run the quality gates in order:

```powershell
python scripts\audit_chia.py --output artifacts\chia_audit.json
python scripts\prepare_chia.py

python scripts\train_crf.py --smoke-test
python scripts\train_crf.py

python scripts\smoke_distilbert.py
python scripts\train_distilbert.py

python scripts\evaluate_test.py
python scripts\verify_submission.py
```

The data scripts download a revision-pinned CHIA BRAT archive into the local
cache. The split is deterministic by NCT identifier (`seed=42`): 800 train, 100
validation, and 100 test trials, with zero leakage.

The smoke runs prove that each training path works before the expensive runs.
The full DistilBERT run takes about 52 minutes on the verified GTX 1050 Ti.
`evaluate_test.py` must remain last because it opens the locked test split after
model selection is complete.

Expected final strict F1 scores are approximately:

- CRF: `0.5884`;
- DistilBERT: `0.6192`;
- DistilBERT improvement: `+0.0308`.

Small timing differences across machines are expected. The split composition,
entity counts, and deterministic preprocessing audits should remain unchanged.

## 6. Checkpoints and portability

Training writes:

- `checkpoints\crf_baseline.pkl`;
- `checkpoints\distilbert\latest`;
- `checkpoints\distilbert\best`.

These files are excluded from Git because the DistilBERT weights exceed GitHub's
file-size limit. This does not prevent verification of the committed experiment
reports or notebook. Exact prediction regeneration requires running the full
training commands first.

To continue training on another machine, copy the entire desired checkpoint
directory, recreate the pinned environment, and pass it to:

```powershell
python scripts\train_distilbert.py `
  --resume-from checkpoints\distilbert\latest
```

The checkpoint includes model, tokenizer, optimizer, scheduler, epoch history,
and random-number-generator state.
