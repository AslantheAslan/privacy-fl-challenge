"""JSONL input/output helpers and light-weight input validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

HOSPITALS: tuple[str, ...] = ("BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE")
STRUCTURED_FIELDS: tuple[str, ...] = (
    "age_years",
    "sex",
    "prior_admissions_12m",
    "length_of_stay_days",
    "emergency_admission",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines, with line-numbered errors."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def normalise_input_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a defensively-normalised copy of an input record.

    Hidden inputs may contain formatting variants; we never want a single
    malformed record to abort the whole run, so missing optional pieces are
    replaced by neutral defaults and reported via ``_input_warnings``.
    """
    warnings: list[str] = []
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("input record without a valid case_id")
    note = record.get("note_text")
    if not isinstance(note, str):
        warnings.append("note_text missing or not a string; treated as empty")
        note = ""
    hospital = record.get("hospital_id")
    if hospital not in HOSPITALS:
        warnings.append(f"unknown hospital_id {hospital!r}")
    features = record.get("structured_features")
    if not isinstance(features, dict):
        warnings.append("structured_features missing; defaults used")
        features = {}
    clean = dict(record)
    clean["note_text"] = note
    clean["hospital_id"] = hospital if isinstance(hospital, str) else "UNKNOWN"
    clean["structured_features"] = dict(features)
    clean["_input_warnings"] = warnings
    return clean
