"""Deliberately simple starter baseline.

Applicants are expected to replace or substantially improve this code. It does
not implement federated learning or a privacy extension.
"""
from __future__ import annotations

import re
from typing import Any

DATE_PATTERN = re.compile(
    r"\b(?:\d{2}[./-]\d{2}[./-](?:\d{2}|\d{4})|\d{2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2})\b"
)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,5}\)?[ /-]?){2,4}\d{3,8}")
ID_PATTERN = re.compile(r"\b(?:B-\d{6}|CHN-\d{7}|HYD\d{6}|BER/\d{4}/\d{2}|UHID/\d{5}/\d{2}|MRN-\d{2}-\d{5})\b")

DIAGNOSIS_KEYWORDS = {
    "atrial_fibrillation": ("atrial fibrillation",),
    "heart_failure": ("heart failure",),
    "hypertension": ("hypertension",),
    "type_2_diabetes": ("type 2 diabetes", "type 2 dm"),
    "chronic_kidney_disease": ("chronic kidney disease",),
    "coronary_artery_disease": ("coronary artery disease",),
    "acute_coronary_syndrome": ("acute coronary syndrome",),
    "pneumonia": ("pneumonia",),
    "copd": ("chronic obstructive pulmonary disease",),
}
MEDICATIONS = (
    "apixaban",
    "rivaroxaban",
    "warfarin",
    "metoprolol",
    "bisoprolol",
    "furosemide",
    "ramipril",
    "amlodipine",
    "metformin",
    "insulin",
    "atorvastatin",
    "aspirin",
    "clopidogrel",
    "amiodarone",
    "digoxin",
    "azithromycin",
)


def _add_span(spans: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    if start < 0 or end <= start:
        return
    for span in spans:
        if start < span["end"] and end > span["start"]:
            return
    spans.append({"start": start, "end": end, "label": label})


def detect_pii(note: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in EMAIL_PATTERN.finditer(note):
        _add_span(spans, *match.span(), "EMAIL")
    for match in ID_PATTERN.finditer(note):
        _add_span(spans, *match.span(), "PATIENT_ID")

    # Contextual date labeling is simplistic by design.
    for match in DATE_PATTERN.finditer(note):
        context = note[max(0, match.start() - 20) : match.start()].lower()
        label = "DATE_OF_BIRTH" if "dob" in context or "born" in context else "ENCOUNTER_DATE"
        _add_span(spans, *match.span(), label)

    for match in PHONE_PATTERN.finditer(note):
        candidate = match.group(0)
        context = note[max(0, match.start() - 15) : match.start()].lower()
        if any(token in context for token in ("phone", "telephone", "mobile", "contact", "ph ")) and sum(ch.isdigit() for ch in candidate) >= 8:
            _add_span(spans, *match.span(), "PHONE_NUMBER")


    return sorted(spans, key=lambda item: (item["start"], item["end"]))


def render_deidentified(note: str, spans: list[dict[str, Any]]) -> str:
    output = note
    for span in reversed(spans):
        output = output[: span["start"]] + f"[{span['label']}]" + output[span["end"] :]
    return output


def _first_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def extract_clinical_data(note: str) -> dict[str, Any]:
    lower = note.lower()
    diagnoses = [
        canonical
        for canonical, keywords in DIAGNOSIS_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    medications = [medication for medication in MEDICATIONS if medication in lower]

    heart_rate = _first_number(r"(?:\bhr\b|pulse|ventricular rate)\s*[=:]?\s*(\d{2,3})", note)
    systolic_bp = _first_number(r"(?:\bbp\b|blood pressure)\s*[=:]?\s*(\d{2,3})(?:\s*/|\s+over\s+)", note)
    creatinine = _first_number(r"(?:serum\s+)?creatinine\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*mg/dl", note)
    hemoglobin = _first_number(r"(?:hemoglobin|\bhb\b)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*g/dl", note)
    lvef = _first_number(r"(?:lvef|\bef\b|ejection fraction(?: approximately)?)\s*[=:]?\s*(\d{2})", note)

    if any(token in lower for token in ("never smoked", "non-smoker", "no tobacco use")):
        smoking = "never"
    elif any(token in lower for token in ("former smoker", "ex-smoker", "stopped smoking")):
        smoking = "former"
    elif any(token in lower for token in ("current smoker", "actively smokes", "ongoing tobacco")):
        smoking = "current"
    else:
        smoking = None

    if any(token in lower for token in ("nkda", "no known drug allergies", "no medication allergy")):
        allergy = "none"
    elif "penicillin" in lower:
        allergy = "penicillin"
    elif any(token in lower for token in ("nsaid", "ibuprofen", "non-steroidal")):
        allergy = "nsaid"
    elif "contrast" in lower:
        allergy = "iodinated_contrast"
    else:
        allergy = None

    return {
        "diagnoses": sorted(diagnoses),
        "medications": sorted(medications),
        "heart_rate_bpm": int(heart_rate) if heart_rate is not None else None,
        "systolic_bp_mmhg": int(systolic_bp) if systolic_bp is not None else None,
        "creatinine_mg_dl": creatinine,
        "hemoglobin_g_dl": hemoglobin,
        "lvef_percent": int(lvef) if lvef is not None else None,
        "smoking_status": smoking,
        "allergy": allergy,
    }
