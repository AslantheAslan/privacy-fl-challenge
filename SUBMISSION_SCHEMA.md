# Submission Schema

Write one JSON object per input case to the path specified by `--output`.

```json
{
  "case_id": "VAL-BER-0001",
  "pii_entities": [
    {"start": 42, "end": 54, "label": "PATIENT_NAME"}
  ],
  "deidentified_text": "... [PATIENT_NAME] ...",
  "extracted_clinical_data": {
    "diagnoses": ["atrial_fibrillation"],
    "medications": ["apixaban"],
    "heart_rate_bpm": 112,
    "systolic_bp_mmhg": 128,
    "creatinine_mg_dl": 1.14,
    "hemoglobin_g_dl": 12.6,
    "lvef_percent": 45,
    "smoking_status": "former",
    "allergy": "penicillin"
  },
  "readmission_probability": 0.37
}
```

Requirements:

- Include exactly one record for every input `case_id`.
- Use zero-based, end-exclusive character offsets.
- PII spans must not overlap.
- Render `deidentified_text` by replacing each submitted span with `[LABEL]`.
- Use canonical extraction terms and units from `DATA_DICTIONARY.md`.
- Use `null` for absent numeric or categorical values, not an empty string.
- `readmission_probability` must be a finite number from 0 through 1.
- Output ordering is not important.

## `experiment_summary.json`

This file is manually reviewed and should contain, at minimum:

- algorithms and feature sets for each local model;
- federated algorithm, number of rounds, local epochs, optimizer, client weighting and random seeds;
- local, federated and centralized validation metrics, overall and by site;
- communication payload and what leaves each client;
- convergence and non-IID observations;
- limitations.

## `privacy_summary.json`

This file is manually reviewed and should contain, at minimum:

- mechanism and implementation status;
- protected asset;
- adversary and trust assumptions;
- privacy parameters or cryptographic configuration;
- empirical utility or runtime comparison;
- exact privacy claim and what it does not guarantee;
- remaining attack surface and limitations.
