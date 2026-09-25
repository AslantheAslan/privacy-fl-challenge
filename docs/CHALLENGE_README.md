# Technical Take-Home Challenge
## Privacy-Preserving Clinical AI and Cross-Silo Federated Learning

### Purpose

This challenge evaluates applied machine learning, clinical NLP, privacy reasoning, federated-learning fundamentals, software engineering, and independent problem solving. The benchmark uses only generated synthetic data. It contains no patient data and must not be interpreted clinically.

You have **seven calendar days** from receipt of the challenge. The expected effort is approximately **8–12 hours**. We assess the quality of your decisions and implementation, not only the final benchmark score.

## Scenario

Three independently governed hospital nodes want to collaborate without pooling patient-level data:

- `BERLIN_NODE`
- `CHENNAI_NODE`
- `HYDERABAD_NODE`

Documentation style, terminology, units, demographics, disease prevalence, and outcome prevalence differ across the nodes. This deliberate non-IID structure reflects a cross-silo federated-learning setting.

The public data comprise:

- 120 labelled training cases
- 30 labelled validation cases, supplied as separate input and ground-truth files
- a private hidden test set used only by the reviewers

Each case contains a synthetic clinical note, a small set of already structured features, and—where labels are released—ground truth for de-identification, structured extraction, and a synthetic 30-day readmission endpoint.

## Your tasks

### 1. Multimodal-ready clinical de-identification

Identify all protected entities in `note_text` and produce both character-offset spans and a de-identified note. The current release is text-based, but your design should explain how it could be extended to other modalities such as scanned documents or images.

The required PII labels are:

- `PATIENT_NAME`
- `DATE_OF_BIRTH`
- `ENCOUNTER_DATE`
- `ADDRESS`
- `PHONE_NUMBER`
- `PATIENT_ID`
- `CLINICIAN_NAME`
- `EMAIL`

Replace each detected span with its exact label placeholder, for example `[PATIENT_NAME]`. Preserve clinically meaningful content and avoid unnecessary redaction.

### 2. Structured extraction and standardization

Extract the following canonical fields from each note:

```json
{
  "diagnoses": ["atrial_fibrillation"],
  "medications": ["apixaban"],
  "heart_rate_bpm": 112,
  "systolic_bp_mmhg": 128,
  "creatinine_mg_dl": 1.14,
  "hemoglobin_g_dl": 12.6,
  "lvef_percent": 45,
  "smoking_status": "former",
  "allergy": "penicillin"
}
```

The notes contain abbreviations, brands, alternative wording, negations, missing values, decimal commas, and different units. All outputs must use the canonical vocabulary and units described in `DATA_DICTIONARY.md`.

### 3. Cross-silo federated learning

Develop a binary model for `readmission_30d`. At minimum, compare:

1. one independently trained local model per hospital;
2. a federated model across the three hospital nodes;
3. a centralized model as a reference.

A correct implementation of FedAvg is sufficient. You may implement it directly or use a framework. Patient-level rows from one node must not be made available to another node in the federated experiment.

Your analysis should address:

- client weighting and aggregation;
- non-IID data and site-specific performance;
- convergence and random-seed stability;
- whether the centralized comparison is fair;
- what information is exchanged between clients and server.

The standard inference output contains one `readmission_probability` per evaluation case. The hidden benchmark evaluates discrimination and calibration. The reviewers separately inspect whether the federated experiment is genuine and technically sound.

### 4. Privacy extension

Implement or rigorously prototype **one** additional privacy mechanism, for example:

- differential privacy;
- secure aggregation or secure multi-party computation;
- homomorphic encryption;
- another well-justified privacy-preserving approach.

A superficial library call is not sufficient. Define the protected asset, adversary, trust assumptions, privacy claim, utility or computational cost, and remaining failure modes. Explicitly distinguish federated learning from a formal privacy guarantee.

## Required repository interface

Your repository must run with this command:

```bash
python run_submission.py \
  --train data/train.jsonl \
  --input data/validation_inputs.jsonl \
  --output outputs/validation_predictions.jsonl \
  --artifacts-dir outputs/artifacts
```

The same command will be used with the hidden input file. Do not assume that hidden labels are available. The evaluator runtime has no network access.

Your script must write:

- one JSON object per input case to the requested `--output` path;
- `experiment_summary.json` to `--artifacts-dir`;
- `privacy_summary.json` to `--artifacts-dir`.

The exact prediction schema is documented in `SUBMISSION_SCHEMA.md` and `schemas/prediction.schema.json`.

A deliberately limited starter implementation is included. It is intended only to demonstrate the interface and should be replaced or substantially improved.

## Local validation

Install and run the starter:

```bash
python -m pip install -r requirements.txt
make evaluate
```

This creates predictions and a detailed report under `outputs/`. The public evaluator provides the same automated metrics used on the hidden set.

## Runtime and reproducibility constraints

The final solution must be reproducible in a clean environment. Hidden evaluation is planned with approximately:

- 4 CPU cores;
- 16 GB RAM;
- no GPU;
- no runtime network access;
- a maximum inference/training runtime of 30 minutes after installation or image build.

Provide a working `requirements.txt` and a functional `Dockerfile` whose entry point accepts the standard arguments above. The image should remain below 8 GB. Any model weights needed at runtime must be legally redistributable and available inside the built image; runtime downloads are not possible. Fix important random seeds and document unavoidable nondeterminism.

## Required repository contents

Include:

- `README.md` with setup and execution instructions;
- a functional `Dockerfile` using the standard entry point;
- clear, modular source code;
- automated tests for important behavior and edge cases;
- `REPORT.md`, based on the supplied template;
- `AI_USAGE.md`;
- retained Git commit history showing your normal development process.

Do not commit credentials, API keys, external private data, real clinical data, or model artefacts that you do not have the right to redistribute.

## Use of AI tools

Use of ChatGPT, Claude, GitHub Copilot, or comparable tools is permitted. Document the tools, their role, and how you verified or changed their output in `AI_USAGE.md`. Undisclosed use is viewed less favorably than transparent, critical use.

## Evaluation

The hidden automated benchmark contributes **40 of 100 points**:

- de-identification: 15 points;
- structured extraction: 15 points;
- readmission prediction: 10 points.

The remaining **60 points** are assigned through code and report review:

- federated-learning implementation and experimental design: 20;
- privacy mechanism and threat-model reasoning: 15;
- software engineering, tests, and reproducibility: 15;
- scientific analysis and communication: 5;
- Git history and independent development process: 5.

Raw model performance is not the sole or dominant criterion. A simpler, correct, well-tested and well-reasoned solution may score higher than a complex but opaque system.

## Submission

Submit:

1. a Git repository URL with accessible commit history;
2. the commit hash to be evaluated;
3. any execution notes not already covered in the README.

Do not submit or generate any real patient data.
