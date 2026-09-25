"""Unit tests for structured extraction: vocabulary, negation, units, nulls."""
from __future__ import annotations

import pytest

from src.extraction import (
    extract_allergy,
    extract_clinical_data,
    extract_creatinine,
    extract_diagnoses,
    extract_heart_rate,
    extract_hemoglobin,
    extract_lvef,
    extract_medications,
    extract_smoking_status,
    extract_systolic_bp,
)

FIELDS = {
    "diagnoses", "medications", "heart_rate_bpm", "systolic_bp_mmhg", "creatinine_mg_dl",
    "hemoglobin_g_dl", "lvef_percent", "smoking_status", "allergy",
}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Diagnoses: Vorhofflimmern (AF); arterial hypertension.", ["atrial_fibrillation", "hypertension"]),
        ("Dx: HFrEF; T2DM; CKD stage III.", ["chronic_kidney_disease", "heart_failure", "type_2_diabetes"]),
        ("Problem list—NSTEMI; COAD.", ["acute_coronary_syndrome", "copd"]),
        ("Known diagnoses: stable ischemic heart disease; infective consolidation.", ["coronary_artery_disease", "pneumonia"]),
        ("Diagnoses [unstable angina; LV failure; DM2].", ["acute_coronary_syndrome", "heart_failure", "type_2_diabetes"]),
    ],
)
def test_diagnosis_synonyms_and_abbreviations(text: str, expected: list[str]) -> None:
    assert extract_diagnoses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Atrial fibrillation was considered but not confirmed.",
        "The patient denies a history of COPD.",
        "No history of heart failure.",
        "Family history of atrial fibrillation in a first-degree relative.",
        "Mother had type 2 diabetes.",
        "Pneumonia was ruled out.",
        "?pneumonia",
    ],
)
def test_negated_hypothetical_and_family_history_are_excluded(text: str) -> None:
    assert extract_diagnoses(text) == []


def test_negation_scope_does_not_leak_past_contrast() -> None:
    assert extract_diagnoses("No COPD but known hypertension.") == ["hypertension"]


def test_type1_diabetes_is_not_type2() -> None:
    assert extract_diagnoses("Diagnoses: type 1 diabetes mellitus.") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Medication list: Eliquis 5 mg b.i.d.; Lasix 40 mg; ecosprin 75 mg.", ["apixaban", "aspirin", "furosemide"]),
        ("Rx—Tab Xarelto 20 mg OD; frusemide 40 OD; atorva 40.", ["atorvastatin", "furosemide", "rivaroxaban"]),
        ("Current therapy: acetylsalicylic acid 100 mg; ASA 75 mg; basal insulin.", ["aspirin", "insulin"]),
        ("Medication [APX 5 mg BID; azithro 500; phenprocoumon/warfarin therapy].", ["apixaban", "azithromycin", "warfarin"]),
    ],
)
def test_medication_brands_and_variants(text: str, expected: list[str]) -> None:
    assert extract_medications(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Apixaban was discussed but was not started.",
        "Metformin was discontinued last month.",
        "Allergy: allergic to aspirin.",
        "Medication list: none documented.",
    ],
)
def test_inactive_or_allergy_drugs_are_excluded(text: str) -> None:
    assert extract_medications(text) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("HR 102 bpm", 102), ("pulse 82/min", 82), ("ventricular rate 73 per minute", 73),
        ("Heart rate: 88", 88), ("vital signs not captured in the export", None), ("RR 18/min", None),
    ],
)
def test_heart_rate(text: str, expected: int | None) -> None:
    assert extract_heart_rate(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("BP 123/72 mmHg", 123), ("blood pressure 142 over 74", 142), ("RR 135/85 mmHg", 135), ("RR 18/min", None)],
)
def test_systolic_bp(text: str, expected: int | None) -> None:
    assert extract_systolic_bp(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("creatinine 1,14 mg/dL", 1.14), ("serum creatinine 136 µmol/L", 1.54), ("Kreatinin 97 umol/l", 1.10),
        ("creatinine and hemoglobin unavailable", None), ("creatinine 0.95 mg/dL", 0.95),
    ],
)
def test_creatinine_units(text: str, expected: float | None) -> None:
    assert extract_creatinine(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("Hb 151 g/L", 15.1), ("hemoglobin 12.6 g/dL", 12.6), ("Hb 8,4 mmol/l", 13.5), ("HbA1c 7.2%", None)],
)
def test_hemoglobin_units(text: str, expected: float | None) -> None:
    assert extract_hemoglobin(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("EF=58%", 58), ("LVEF 35%", 35), ("left ventricular ejection fraction approximately 53 per cent", 53),
        ("echocardiographic ejection fraction not documented", None), ("EF 35-40%", 38),
        ("heart failure with reduced EF; SpO2 95%", None),
    ],
)
def test_lvef(text: str, expected: int | None) -> None:
    assert extract_lvef(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Smoking status: stopped smoking several years ago.", "former"), ("smoking=ex-smoker", "former"),
        ("Tobacco history: non-smoker.", "never"), ("NKDA; no tobacco use.", "never"),
        ("Smoking: ongoing tobacco use.", "current"), ("tobacco: actively smokes cigarettes.", "current"),
        ("Smoking: quit 2019.", "former"), ("Smoking: no.", "never"), ("No smoking data.", None),
        ("Smoking history not documented.", None), ("Diagnoses: HTN.", None),
    ],
)
def test_smoking_status(text: str, expected: str | None) -> None:
    assert extract_smoking_status(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Allergies: NKDA.", "none"), ("Drug allergy: no medication allergy documented.", "none"),
        ("Allergy: penicillin causes rash.", "penicillin"), ("Allergy: ibuprofen-associated urticaria.", "nsaid"),
        ("Allergy=allergic to non-steroidal anti-inflammatory drugs", "nsaid"),
        ("Drug allergy: contrast medium reaction.", "iodinated_contrast"), ("Diagnoses: HTN.", None),
    ],
)
def test_allergy(text: str, expected: str | None) -> None:
    assert extract_allergy(text) == expected


def test_full_record_schema_and_null_policy() -> None:
    result = extract_clinical_data("Nothing clinically relevant documented.")
    assert set(result) == FIELDS
    assert result["diagnoses"] == [] and result["medications"] == []
    assert all(result[f] is None for f in FIELDS - {"diagnoses", "medications"})
