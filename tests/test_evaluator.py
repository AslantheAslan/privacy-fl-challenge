import json
from pathlib import Path

from evaluator.evaluate import evaluate


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_perfect_submission_scores_one(tmp_path: Path) -> None:
    note = "Patient: Anna Keller"
    span = {"start": 9, "end": 20, "label": "PATIENT_NAME", "text": "Anna Keller"}
    base_input = {
        "hospital_id": "BERLIN_NODE",
        "note_text": note,
        "structured_features": {
            "age_years": 50,
            "sex": "female",
            "prior_admissions_12m": 0,
            "length_of_stay_days": 2,
            "emergency_admission": False
        }
    }
    extraction = {
        "diagnoses": ["hypertension"],
        "medications": ["ramipril"],
        "heart_rate_bpm": None,
        "systolic_bp_mmhg": None,
        "creatinine_mg_dl": None,
        "hemoglobin_g_dl": None,
        "lvef_percent": None,
        "smoking_status": "never",
        "allergy": "none"
    }
    inputs = [dict(base_input, case_id="X1"), dict(base_input, case_id="X2")]
    truth = [
        {
            "case_id": case_id,
            "hospital_id": "BERLIN_NODE",
            "pii_entities": [span],
            "deidentified_text": "Patient: [PATIENT_NAME]",
            "extracted_clinical_data": extraction,
            "readmission_30d": label
        }
        for case_id, label in (("X1", 1), ("X2", 0))
    ]
    predictions = [
        {
            "case_id": case_id,
            "pii_entities": [{k: span[k] for k in ("start", "end", "label")}],
            "deidentified_text": "Patient: [PATIENT_NAME]",
            "extracted_clinical_data": extraction,
            "readmission_probability": probability
        }
        for case_id, probability in (("X1", 1.0), ("X2", 0.0))
    ]
    input_path = tmp_path / "inputs.jsonl"
    truth_path = tmp_path / "truth.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    _write_jsonl(input_path, inputs)
    _write_jsonl(truth_path, truth)
    _write_jsonl(prediction_path, predictions)
    report = evaluate(input_path, truth_path, prediction_path)
    assert report["metrics"]["deidentification"]["score"] == 1.0
    assert report["metrics"]["structured_extraction"]["score"] == 1.0
    assert report["metrics"]["readmission_prediction"]["score"] == 1.0
