"""Unit tests for the PII detector: formats, labelling logic, invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.deid import detect_pii
from src.format_shift import TRANSFORMS, perturb
from src.io_utils import read_jsonl
from src.spans import Span, render_deidentified, resolve_overlaps

ROOT = Path(__file__).resolve().parents[1]


def labelled(note: str, age: int | None = None) -> dict[str, str]:
    """Map covered text -> label for compact assertions."""
    return {note[s.start : s.end]: s.label for s in detect_pii(note, age_years=age)}


def test_berlin_summary_template() -> None:
    note = (
        "SYNTHETIC BERLIN CLINICAL SUMMARY\nPatient: Anna Keller | DOB: 01.02.1970 | Case ID: B-123456\n"
        "Address: Lindenallee 17, 10117 Berlin | Telephone: +49 30 0000 1111\n"
        "Encounter date: 03.04.2025 | Attending physician: Dr. Emil Brandt\n"
        "Electronically signed by Dr. Emil Brandt; contact emil.brandt@synthetic-clinic.example."
    )
    found = labelled(note)
    assert found["Anna Keller"] == "PATIENT_NAME"
    assert found["01.02.1970"] == "DATE_OF_BIRTH"
    assert found["03.04.2025"] == "ENCOUNTER_DATE"
    assert found["B-123456"] == "PATIENT_ID"
    assert found["Lindenallee 17, 10117 Berlin"] == "ADDRESS"
    assert found["+49 30 0000 1111"] == "PHONE_NUMBER"
    assert found["emil.brandt@synthetic-clinic.example"] == "EMAIL"
    # Repeated clinician mentions are labelled separately and exclude the title.
    clinician = [s for s in detect_pii(note) if s.label == "CLINICIAN_NAME"]
    assert [note[s.start : s.end] for s in clinician] == ["Emil Brandt", "Emil Brandt"]


def test_clinician_recovered_from_email_with_umlaut_transliteration() -> None:
    note = "Allergy record: NKDA. Author: Laura König <laura.koenig@synthetic-clinic.example>."
    assert labelled(note)["Laura König"] == "CLINICIAN_NAME"


@pytest.mark.parametrize(
    "date_text",
    ["2025-03-14", "14 March 2025", "March 14, 2025", "14-Mar-2025", "14.03.25", "14/03/2025"],
)
def test_date_formats_and_cue_typing(date_text: str) -> None:
    note = f"Patient: Anna Keller | Date of birth: 02.02.1960 | Admitted on: {date_text}."
    found = labelled(note)
    assert found["02.02.1960"] == "DATE_OF_BIRTH"
    assert found[date_text] == "ENCOUNTER_DATE"


def test_cueless_header_date_uses_age_consistency() -> None:
    # Header date without a cue, then an explicit DOB: header must be the encounter.
    note = "EXPORT // B-191271 // 21.02.2026\nAnton Adler, born 23.05.1963, 62 y, male."
    found = labelled(note, age=62)
    assert found["21.02.2026"] == "ENCOUNTER_DATE"
    assert found["23.05.1963"] == "DATE_OF_BIRTH"
    # Two cue-less dates: the one consistent with the age is the DOB.
    found = labelled("RECORD 23.05.1963 ... 21.02.2026", age=62)
    assert found["23.05.1963"] == "DATE_OF_BIRTH"
    assert found["21.02.2026"] == "ENCOUNTER_DATE"


@pytest.mark.parametrize("identifier", ["BER/2025/12", "UHID/12345/25", "MRN-12-34567", "HYD864557", "CHN-5425165"])
def test_identifier_schemes(identifier: str) -> None:
    assert labelled(f"MRN: {identifier} | Patient: Anna Keller")[identifier] == "PATIENT_ID"


@pytest.mark.parametrize("phone", ["(030) 1234 5678", "+91-98765-43210", "98765 43210", "+49 (0)30 12345678"])
def test_phone_formats_with_cue(phone: str) -> None:
    assert labelled(f"Tel.: {phone} | Address: Rosenweg 5, 10115 Berlin")[phone] == "PHONE_NUMBER"


@pytest.mark.parametrize(
    "name", ["Anna-Lena Schmidt-Weber", "Ana Sofia O'Connor", "NEUMANN, Anton", "Neumann, Anton", "Sri Lakshmi Rao"]
)
def test_complex_patient_names(name: str) -> None:
    assert labelled(f"Patient name: {name} | DOB: 01.02.1970")[name] == "PATIENT_NAME"


def test_german_academic_title_is_not_part_of_clinician_name() -> None:
    assert labelled("Attending: Dr. med. Hans Müller")["Hans Müller"] == "CLINICIAN_NAME"


def test_indian_address_variants() -> None:
    note = "Home: Flat 15-B, 31 Marina Cross Street, Chennai - 600 018. Mobile +91 00000 34235."
    assert labelled(note)["Flat 15-B, 31 Marina Cross Street, Chennai - 600 018"] == "ADDRESS"


def test_no_over_redaction_of_clinical_content() -> None:
    note = (
        "Diagnoses: Vorhofflimmern (AF); hypertension. On assessment: BP 123/72 mmHg; HR 82 bpm. "
        "Medication list: Eliquis 5 mg b.i.d. No evidence of Parkinson disease. "
        "Family history includes Hodgkin lymphoma. Laboratory: creatinine 136 µmol/L; Hb 151 g/L. LVEF 45%."
    )
    assert detect_pii(note) == []


def test_spans_are_sorted_non_overlapping_and_renderable() -> None:
    record = json.loads((ROOT / "data/validation_inputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    spans = detect_pii(record["note_text"])
    assert spans == sorted(spans, key=lambda s: s.start)
    for first, second in zip(spans, spans[1:]):
        assert first.end <= second.start
    rendered = render_deidentified(record["note_text"], spans)
    assert "Anton Neumann" not in rendered and "[PATIENT_NAME]" in rendered


def test_empty_and_pii_free_notes() -> None:
    assert detect_pii("") == []
    assert detect_pii("Diagnoses: hypertension.") == []


def test_resolve_overlaps_prefers_priority_then_length() -> None:
    a = Span(0, 10, "PATIENT_ID")
    b = Span(5, 20, "PHONE_NUMBER")
    c = Span(12, 14, "EMAIL")
    assert resolve_overlaps([(1, a), (0, b), (2, c)]) == [b]
    assert resolve_overlaps([(0, a), (0, c)]) == [a, c]


def test_render_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        render_deidentified("abcdef", [Span(0, 3, "EMAIL"), Span(2, 4, "EMAIL")])


@pytest.mark.parametrize("transform", [*TRANSFORMS, "combined"])
def test_format_shift_robustness_on_validation(transform: str) -> None:
    """Every synthetic formatting variant must be de-identified exactly."""
    inputs = read_jsonl(ROOT / "data/validation_inputs.jsonl")
    truth = {r["case_id"]: r for r in read_jsonl(ROOT / "data/validation_ground_truth.jsonl")}
    for record in inputs:
        note, spans = perturb(record["note_text"], truth[record["case_id"]]["pii_entities"], transform, seed=3)
        predicted = {(s.start, s.end, s.label) for s in detect_pii(note, record["structured_features"]["age_years"])}
        assert predicted == {(s["start"], s["end"], s["label"]) for s in spans}, record["case_id"]
