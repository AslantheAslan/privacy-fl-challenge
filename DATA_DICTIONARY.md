# Data Dictionary

All records are synthetic and generated. No field originates from a patient or hospital information system.

## Input record

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string | Unique synthetic case identifier. |
| `hospital_id` | enum | `BERLIN_NODE`, `CHENNAI_NODE`, or `HYDERABAD_NODE`. |
| `note_text` | string | Synthetic unstructured clinical note containing clinical facts and generated PII. |
| `structured_features.age_years` | integer | Age at encounter. |
| `structured_features.sex` | enum | `female` or `male`. |
| `structured_features.prior_admissions_12m` | integer | Synthetic admissions in the preceding 12 months. |
| `structured_features.length_of_stay_days` | integer | Synthetic index length of stay. |
| `structured_features.emergency_admission` | boolean | Whether the synthetic index encounter was emergent. |

## De-identification labels

Offsets follow Python slicing: `note_text[start:end]`. The start offset is inclusive and the end offset is exclusive.

| Label | Scope |
|---|---|
| `PATIENT_NAME` | Generated patient name. |
| `DATE_OF_BIRTH` | Generated date of birth. |
| `ENCOUNTER_DATE` | Generated encounter/admission date. |
| `ADDRESS` | Generated residential address. |
| `PHONE_NUMBER` | Generated contact number. |
| `PATIENT_ID` | Generated MRN/UHID/case identifier. |
| `CLINICIAN_NAME` | Generated clinician name. Repeated occurrences are separately labelled. |
| `EMAIL` | Generated clinician contact email. |

## Canonical diagnoses

- `atrial_fibrillation`
- `heart_failure`
- `hypertension`
- `type_2_diabetes`
- `chronic_kidney_disease`
- `coronary_artery_disease`
- `acute_coronary_syndrome`
- `pneumonia`
- `copd`

Only active patient diagnoses belong in the output. Negated, ruled-out, considered-only, and family-history concepts must not be extracted as active diagnoses.

## Canonical medications

- `apixaban`
- `rivaroxaban`
- `warfarin`
- `metoprolol`
- `bisoprolol`
- `furosemide`
- `ramipril`
- `amlodipine`
- `metformin`
- `insulin`
- `atorvastatin`
- `aspirin`
- `clopidogrel`
- `amiodarone`
- `digoxin`
- `azithromycin`

Return currently prescribed medications only. Brand names and common spelling variants should be mapped to the canonical generic name.

## Numeric fields and standard units

| Field | Standard unit | Public scoring tolerance |
|---|---:|---:|
| `heart_rate_bpm` | beats/min | ±3 |
| `systolic_bp_mmhg` | mmHg | ±5 |
| `creatinine_mg_dl` | mg/dL | ±0.10 |
| `hemoglobin_g_dl` | g/dL | ±0.30 |
| `lvef_percent` | % | ±3 |

Use JSON `null` when the note explicitly lacks the value or the value is not documented. Relevant conversions include approximately `1 mg/dL creatinine = 88.4 µmol/L` and `1 g/dL hemoglobin = 10 g/L`.

## Categorical extraction fields

`smoking_status` must be one of `never`, `former`, `current`, or `null`.

`allergy` must be one of `none`, `penicillin`, `nsaid`, `iodinated_contrast`, or `null`.

## Outcome

`readmission_30d` is a generated binary endpoint representing an unplanned readmission within 30 days. It is designed only for the technical exercise and has no clinical validity.
