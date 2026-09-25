"""End-to-end: the standard command produces schema-valid, evaluable output."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evaluator.evaluate import evaluate
from src.pipeline import process_note, run

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/prediction.schema.json").read_text())
X_PROPS = SCHEMA["properties"]["extracted_clinical_data"]["properties"]


def _validate_prediction(record: dict, note: str) -> None:
    """Minimal JSON-schema check (no third-party validator in the runtime image)."""
    assert set(record) == set(SCHEMA["required"])
    assert isinstance(record["case_id"], str) and record["case_id"]
    previous_end = 0
    rendered = note
    for span in sorted(record["pii_entities"], key=lambda s: s["start"]):
        assert isinstance(span["start"], int) and isinstance(span["end"], int)
        assert previous_end <= span["start"] < span["end"] <= len(note)
        assert span["label"] in SCHEMA["properties"]["pii_entities"]["items"]["properties"]["label"]["enum"]
        previous_end = span["end"]
    for span in sorted(record["pii_entities"], key=lambda s: -s["start"]):
        rendered = rendered[: span["start"]] + f"[{span['label']}]" + rendered[span["end"] :]
    assert record["deidentified_text"] == rendered
    extracted = record["extracted_clinical_data"]
    assert set(extracted) == set(SCHEMA["properties"]["extracted_clinical_data"]["required"])
    for field in ("diagnoses", "medications"):
        assert len(set(extracted[field])) == len(extracted[field])
        assert set(extracted[field]) <= set(X_PROPS[field]["items"]["enum"])
    for field in ("smoking_status", "allergy"):
        assert extracted[field] in X_PROPS[field]["enum"]
    for field in ("heart_rate_bpm", "systolic_bp_mmhg", "creatinine_mg_dl", "hemoglobin_g_dl", "lvef_percent"):
        value = extracted[field]
        assert value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
    p = record["readmission_probability"]
    assert isinstance(p, float) and 0.0 <= p <= 1.0


@pytest.fixture(scope="module")
def submission(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("run")
    inputs = [json.loads(line) for line in (ROOT / "data/validation_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    # add hostile edge cases: missing features, unknown site, empty note
    inputs.append({"case_id": "EDGE-1", "hospital_id": "UNKNOWN_NODE", "note_text": "", "structured_features": {}})
    inputs.append({"case_id": "EDGE-2", "hospital_id": "BERLIN_NODE", "note_text": None})
    input_path = out / "inputs.jsonl"
    input_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in inputs) + "\n", encoding="utf-8")
    run(ROOT / "data/train.jsonl", input_path, out / "pred.jsonl", out / "artifacts", quick=True)
    predictions = [json.loads(line) for line in (out / "pred.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"dir": out, "inputs": inputs, "predictions": predictions}


def test_one_schema_valid_record_per_input(submission: dict) -> None:
    predictions = {p["case_id"]: p for p in submission["predictions"]}
    assert len(predictions) == len(submission["inputs"]) == len(submission["predictions"])
    for record in submission["inputs"]:
        _validate_prediction(predictions[record["case_id"]], record.get("note_text") or "")


def test_artifacts_contain_required_sections(submission: dict) -> None:
    experiment = json.loads((submission["dir"] / "artifacts/experiment_summary.json").read_text())
    for key in ("local_models", "federated_model", "centralized_model", "communication", "convergence",
                "non_iid_observations", "limitations", "random_seeds", "seed_stability"):
        assert key in experiment
    assert experiment["federated_model"]["rounds"] > 0
    privacy = json.loads((submission["dir"] / "artifacts/privacy_summary.json").read_text())
    for key in ("mechanism", "implementation_status", "protected_asset", "adversary", "trust_assumptions",
                "parameters", "privacy_claim", "what_is_not_guaranteed", "empirical_utility_and_cost",
                "remaining_attack_surface"):
        assert key in privacy


def test_validation_scores_with_official_evaluator(submission: dict, tmp_path: Path) -> None:
    real = [p for p in submission["predictions"] if not p["case_id"].startswith("EDGE")]
    path = tmp_path / "pred.jsonl"
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in real), encoding="utf-8")
    report = evaluate(ROOT / "data/validation_inputs.jsonl", ROOT / "data/validation_ground_truth.jsonl", path)
    assert report["validation_messages"] == []
    assert report["metrics"]["deidentification"]["score"] > 0.99
    assert report["metrics"]["structured_extraction"]["score"] > 0.99
    assert report["metrics"]["readmission_prediction"]["roc_auc"] > 0.65


def test_process_note_never_raises_on_odd_text() -> None:
    record = {"case_id": "X", "note_text": "\x00�" * 50 + "DOB: 99.99.9999 | HR 999 bpm | +++",
              "structured_features": {"age_years": "n/a"}}
    prediction, _ = process_note(record)
    assert prediction["extracted_clinical_data"]["heart_rate_bpm"] is None
