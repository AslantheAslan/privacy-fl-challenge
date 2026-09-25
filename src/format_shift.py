"""Span-aware synthetic format perturbations for robustness testing.

The hidden test split "includes additional formatting variants". We cannot see
them, so we stress-test the de-identifier by rewriting labelled notes in ways
that a different EMR export could plausibly produce, while recomputing the gold
character offsets exactly:

* PII values are re-rendered in another surface format (ISO / long / two-digit
  year dates, alternative ID schemes, national phone formats, three-token,
  hyphenated, inverted or upper-case names);
* field cues are relabelled (``DOB:`` -> ``Date of birth:``, ``MRN:`` ->
  ``Hospital No.:``) – these are non-PII edits;
* layout is flattened (new lines -> `` | ``).

Every transform is deterministic given its seed so results are reproducible.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable

MONTHS_SHORT = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS_FULL = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str
    label: str | None  # PII label of the replacement, or None for context edits


def parse_date(text: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", text)
    if match and match.group(2).title() in MONTHS_SHORT:
        return int(match.group(1)), MONTHS_SHORT.index(match.group(2).title()) + 1, int(match.group(3))
    return None


def apply_edits(note: str, spans: list[dict[str, Any]], edits: list[Edit]) -> tuple[str, list[dict[str, Any]]]:
    """Apply non-overlapping edits and return the new note and gold spans.

    PII spans that are not edited are shifted; an edit with a label replaces the
    gold span it covers exactly.
    """
    edits = sorted(edits, key=lambda e: e.start)
    for first, second in zip(edits, edits[1:]):
        if second.start < first.end:
            raise ValueError("edits overlap")
    edited_ranges = {(e.start, e.end) for e in edits if e.label}
    pieces: list[str] = []
    new_spans: list[dict[str, Any]] = []
    cursor = 0
    shift = 0
    events = sorted(
        [("edit", e.start, e) for e in edits]
        + [("span", s["start"], s) for s in spans if (s["start"], s["end"]) not in edited_ranges],
        key=lambda item: (item[1], 0 if item[0] == "edit" else 1),
    )
    for kind, _, item in events:
        if kind == "span":
            new_spans.append({"start": item["start"] + shift, "end": item["end"] + shift, "label": item["label"]})
            continue
        edit: Edit = item
        pieces.append(note[cursor : edit.start])
        new_start = edit.start + shift
        pieces.append(edit.text)
        if edit.label:
            new_spans.append({"start": new_start, "end": new_start + len(edit.text), "label": edit.label})
        shift += len(edit.text) - (edit.end - edit.start)
        cursor = edit.end
    pieces.append(note[cursor:])
    new_note = "".join(pieces)
    new_spans.sort(key=lambda s: s["start"])
    return new_note, new_spans


# ---------------------------------------------------------------------------
# Individual transforms: (note, spans, rng) -> list[Edit]
# ---------------------------------------------------------------------------
def _date_edits(style: Callable[[int, int, int], str]) -> Callable[[str, list, random.Random], list[Edit]]:
    def transform(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
        edits = []
        for span in spans:
            if span["label"] in ("DATE_OF_BIRTH", "ENCOUNTER_DATE"):
                parsed = parse_date(note[span["start"] : span["end"]])
                if parsed:
                    edits.append(Edit(span["start"], span["end"], style(*parsed), span["label"]))
        return edits

    return transform


def _ids(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    edits = []
    for span in spans:
        if span["label"] == "PATIENT_ID":
            style = rng.choice(("ber", "uhid", "mrn"))
            if style == "ber":
                text = f"BER/{rng.randint(2019, 2026)}/{rng.randint(10, 99)}"
            elif style == "uhid":
                text = f"UHID/{rng.randint(10000, 99999)}/{rng.randint(10, 99)}"
            else:
                text = f"MRN-{rng.randint(10, 99)}-{rng.randint(10000, 99999)}"
            edits.append(Edit(span["start"], span["end"], text, "PATIENT_ID"))
    return edits


def _phones(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    edits = []
    for span in spans:
        if span["label"] == "PHONE_NUMBER":
            original = note[span["start"] : span["end"]]
            if original.startswith("+49"):
                text = rng.choice(("(030) 1234 5678", "030/12345678", "+49-30-1234-5678", "+49 (0)30 12345678"))
            else:
                text = rng.choice(("98765 43210", "+91-98765-43210", "044 2345 6789", "+91 (44) 2345 6789"))
            edits.append(Edit(span["start"], span["end"], text, "PHONE_NUMBER"))
    return edits


_GIVEN = ("Anna-Lena", "Jean-Luc", "Maria José", "Sri Lakshmi", "Karl-Heinz", "Ana Sofia")
_FAMILY = ("Schmidt-Weber", "O'Connor", "Venkataraman", "Müller-Lüdenscheidt", "Da Silva", "Reddy-Rao")


def _names(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    replacement: dict[str, str] = {}
    edits = []
    for span in spans:
        if span["label"] in ("PATIENT_NAME", "CLINICIAN_NAME"):
            original = note[span["start"] : span["end"]]
            if original not in replacement:
                replacement[original] = f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}"
            edits.append(Edit(span["start"], span["end"], replacement[original], span["label"]))
    return edits


def _inverted_patient_name(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    edits = []
    for span in spans:
        if span["label"] == "PATIENT_NAME":
            given, _, family = note[span["start"] : span["end"]].partition(" ")
            text = f"{family.upper()}, {given}" if rng.random() < 0.5 else f"{family}, {given}"
            edits.append(Edit(span["start"], span["end"], text, "PATIENT_NAME"))
    return edits


_CUE_REWRITES: tuple[tuple[str, str], ...] = (
    (r"DOB: ", "Date of birth: "),
    (r"\(DOB ", "(D.O.B. "),
    (r"born ", "date of birth "),
    (r"Encounter date: ", "Admitted on: "),
    (r"Admission: ", "Date of admission: "),
    (r"Visit: ", "Date of visit: "),
    (r"DOA ", "Admitted "),
    (r"seen ", "visit "),
    (r"Telephone: ", "Tel.: "),
    (r"Mobile: ", "Mobile no.: "),
    (r"; ph ", "; phone "),
    (r"Case ID: ", "Patient ID: "),
    (r"MRN: ", "Hospital No.: "),
    (r"Patient: ", "Patient name: "),
    (r"Name: ", "Patient name: "),
    (r"Attending physician: Dr\. ", "Attending: Dr. med. "),
    (r"Treating clinician: ", "Seen by: "),
    (r"Consultant: Dr ", "Consultant in charge: Dr. "),
    (r"Address: ", "Home address: "),
    (r"Residence: ", "Address: "),
)


def _cues(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    occupied = [(s["start"], s["end"]) for s in spans]
    edits: list[Edit] = []
    for pattern, replacement in _CUE_REWRITES:
        for match in re.finditer(pattern, note):
            if any(match.start() < e and s < match.end() for s, e in occupied):
                continue
            if any(match.start() < e.end and e.start < match.end() for e in edits):
                continue
            edits.append(Edit(match.start(), match.end(), replacement, None))
    return edits


def _single_line(note: str, spans: list[dict[str, Any]], rng: random.Random) -> list[Edit]:
    return [Edit(m.start(), m.end(), " | ", None) for m in re.finditer(r"\n", note)]


TRANSFORMS: dict[str, Callable[[str, list, random.Random], list[Edit]]] = {
    "dates_iso": _date_edits(lambda d, m, y: f"{y:04d}-{m:02d}-{d:02d}"),
    "dates_long": _date_edits(lambda d, m, y: f"{d} {MONTHS_FULL[m - 1]} {y}"),
    "dates_us_long": _date_edits(lambda d, m, y: f"{MONTHS_FULL[m - 1]} {d}, {y}"),
    "dates_two_digit_year": _date_edits(lambda d, m, y: f"{d:02d}.{m:02d}.{y % 100:02d}"),
    "ids_alternative_schemes": _ids,
    "phones_national_formats": _phones,
    "names_complex": _names,
    "names_inverted": _inverted_patient_name,
    "cues_relabelled": _cues,
    "layout_single_line": _single_line,
}


def perturb(note: str, spans: list[dict[str, Any]], transform: str, seed: int = 0) -> tuple[str, list[dict[str, Any]]]:
    """Apply one named transform, or ``"combined"`` for all compatible ones."""
    rng = random.Random(f"{seed}:{transform}:{len(note)}")
    if transform != "combined":
        return apply_edits(note, spans, TRANSFORMS[transform](note, spans, rng))
    # Sequential composition: each step sees the output of the previous one.
    for name in ("ids_alternative_schemes", "phones_national_formats", "names_complex", "cues_relabelled",
                 "layout_single_line", "dates_iso"):
        note, spans = apply_edits(note, spans, TRANSFORMS[name](note, spans, rng))
    return note, spans
