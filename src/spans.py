"""Shared span utilities for de-identification.

All offsets are zero-based and end-exclusive (Python slicing semantics), which
is what the challenge evaluator expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

PII_LABELS: tuple[str, ...] = (
    "PATIENT_NAME",
    "DATE_OF_BIRTH",
    "ENCOUNTER_DATE",
    "ADDRESS",
    "PHONE_NUMBER",
    "PATIENT_ID",
    "CLINICIAN_NAME",
    "EMAIL",
)


@dataclass(frozen=True, order=True)
class Span:
    """A labelled character span. ``source`` records which rule produced it."""

    start: int
    end: int
    label: str
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span offsets {self.start}:{self.end}")
        if self.label not in PII_LABELS:
            raise ValueError(f"unsupported PII label {self.label!r}")

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def to_json(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label}


def resolve_overlaps(candidates: Iterable[tuple[int, Span]]) -> list[Span]:
    """Greedy non-overlapping selection.

    ``candidates`` are ``(priority, span)`` pairs; lower priority numbers win.
    Within the same priority, longer spans win. The result is sorted by start.
    """
    ordered = sorted(candidates, key=lambda item: (item[0], -(item[1].end - item[1].start), item[1].start))
    chosen: list[Span] = []
    for _, span in ordered:
        if any(span.overlaps(existing) for existing in chosen):
            continue
        chosen.append(span)
    return sorted(chosen, key=lambda span: (span.start, span.end))


def render_deidentified(note: str, spans: Iterable[Span | dict[str, Any]]) -> str:
    """Replace every span with ``[LABEL]``. Spans must not overlap."""
    normalized = sorted(
        (
            (item.start, item.end, item.label)
            if isinstance(item, Span)
            else (int(item["start"]), int(item["end"]), str(item["label"]))
            for item in spans
        ),
        key=lambda triple: triple[0],
    )
    pieces: list[str] = []
    cursor = 0
    for start, end, label in normalized:
        if start < cursor:
            raise ValueError("overlapping spans cannot be rendered")
        pieces.append(note[cursor:start])
        pieces.append(f"[{label}]")
        cursor = end
    pieces.append(note[cursor:])
    return "".join(pieces)
