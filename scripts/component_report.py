#!/usr/bin/env python3
"""Development harness: score de-identification and extraction on labelled data.

Uses the official evaluator's scoring functions so numbers match ``make evaluate``.
Prints every mismatching entity / field so that rules can be debugged quickly.

    python scripts/component_report.py --split train
    python scripts/component_report.py --split validation --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.evaluate import score_extraction, score_pii  # noqa: E402
from src.deid import detect_pii  # noqa: E402
from src.extraction import extract_clinical_data  # noqa: E402
from src.io_utils import read_jsonl  # noqa: E402
from src.spans import render_deidentified  # noqa: E402


def load_split(split: str) -> tuple[dict, dict]:
    if split == "train":
        records = read_jsonl(ROOT / "data/train.jsonl")
        inputs = {r["case_id"]: r for r in records}
        truth = {r["case_id"]: {"case_id": r["case_id"], "hospital_id": r["hospital_id"], **r["labels"]} for r in records}
    else:
        inputs = {r["case_id"]: r for r in read_jsonl(ROOT / "data/validation_inputs.jsonl")}
        truth = {r["case_id"]: r for r in read_jsonl(ROOT / "data/validation_ground_truth.jsonl")}
    return inputs, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    inputs, truth = load_split(args.split)

    predictions = {}
    for case_id, record in inputs.items():
        note = record["note_text"]
        spans = detect_pii(note, age_years=record["structured_features"].get("age_years"))
        predictions[case_id] = {
            "case_id": case_id,
            "pii_entities": [s.to_json() for s in spans],
            "deidentified_text": render_deidentified(note, spans),
            "extracted_clinical_data": extract_clinical_data(note),
        }

    errors: list[str] = []
    pii = score_pii(inputs, truth, predictions, errors)
    extraction = score_extraction(truth, predictions, errors)
    print(f"[{args.split}] de-id score {pii['score']:.4f}  char-F1 {pii['character_detection']['f1']:.4f}  "
          f"entity-F1 {pii['exact_entity']['f1']:.4f}  label-char-F1 {pii['label_aware_character']['f1']:.4f}")
    for label, m in pii["exact_entity"]["per_label"].items():
        print(f"    {label:15s} P={m['precision']:.3f} R={m['recall']:.3f} tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    print(f"[{args.split}] extraction score {extraction['score']:.4f}  dx-F1 {extraction['diagnoses']['f1']:.3f}  "
          f"rx-F1 {extraction['medications']['f1']:.3f}  numeric {extraction['numeric_average']:.3f}  "
          f"categorical {extraction['categorical_average']:.3f}")
    for field, m in extraction["numeric"].items():
        print(f"    {field:18s} score={m['score']:.3f} within_tol={m['within_tolerance_rate']:.3f}")
    for field, m in extraction["categorical"].items():
        print(f"    {field:18s} acc={m['accuracy']:.3f}")

    if not args.verbose:
        return
    for case_id, gt in truth.items():
        note = inputs[case_id]["note_text"]
        true_set = {(s["start"], s["end"], s["label"]) for s in gt["pii_entities"]}
        pred_set = {(s["start"], s["end"], s["label"]) for s in predictions[case_id]["pii_entities"]}
        for s, e, lab in sorted(true_set - pred_set):
            print(f"  FN {case_id} {lab:14s} {note[s:e]!r}")
        for s, e, lab in sorted(pred_set - true_set):
            print(f"  FP {case_id} {lab:14s} {note[s:e]!r}")
        pred_x = predictions[case_id]["extracted_clinical_data"]
        for field, true_value in gt["extracted_clinical_data"].items():
            pred_value = pred_x.get(field)
            if isinstance(true_value, list):
                if sorted(true_value) != sorted(pred_value or []):
                    print(f"  XF {case_id} {field}: true={sorted(true_value)} pred={sorted(pred_value or [])}")
            elif isinstance(true_value, (int, float)) and pred_value is not None:
                tol = {"heart_rate_bpm": 3, "systolic_bp_mmhg": 5, "creatinine_mg_dl": 0.1,
                       "hemoglobin_g_dl": 0.3, "lvef_percent": 3}[field]
                if abs(float(pred_value) - float(true_value)) > tol:
                    print(f"  XF {case_id} {field}: true={true_value} pred={pred_value}")
            elif true_value != pred_value:
                print(f"  XF {case_id} {field}: true={true_value} pred={pred_value}")


if __name__ == "__main__":
    main()
