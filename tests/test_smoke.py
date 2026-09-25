from src.baseline import detect_pii, extract_clinical_data, render_deidentified


def test_deidentification_offsets_are_renderable() -> None:
    note = "Patient: Anna Keller | DOB: 01.02.1970 | Case ID: B-123456"
    spans = detect_pii(note)
    rendered = render_deidentified(note, spans)
    assert "[DATE_OF_BIRTH]" in rendered
    assert "[PATIENT_ID]" in rendered


def test_extraction_returns_complete_schema() -> None:
    result = extract_clinical_data("Atrial fibrillation. HR 110 bpm. Creatinine 1.20 mg/dL.")
    assert set(result) == {
        "diagnoses",
        "medications",
        "heart_rate_bpm",
        "systolic_bp_mmhg",
        "creatinine_mg_dl",
        "hemoglobin_g_dl",
        "lvef_percent",
        "smoking_status",
        "allergy",
    }
