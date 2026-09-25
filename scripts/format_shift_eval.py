#!/usr/bin/env python3
"""Robustness of de-identification under synthetic formatting shift.

Rewrites every labelled note (train + validation) with each transform from
``src.format_shift`` and scores the detector with the official PII metric.

    python scripts/format_shift_eval.py [--output reports/format_shift.json] [--verbose]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.evaluate import score_pii  # noqa: E402
from src.deid import detect_pii  # noqa: E402
from src.format_shift import TRANSFORMS, perturb  # noqa: E402
from src.io_utils import read_jsonl, write_json  # noqa: E402
from src.spans import render_deidentified  # noqa: E402


def labelled_notes() -> list[tuple[str, str, list[dict], int | None]]:
    rows = []
    for record in read_jsonl(ROOT / "data/train.jsonl"):
        rows.append((record["case_id"], record["note_text"], record["labels"]["pii_entities"],
                     record["structured_features"].get("age_years")))
    truth = {r["case_id"]: r for r in read_jsonl(ROOT / "data/validation_ground_truth.jsonl")}
    for record in read_jsonl(ROOT / "data/validation_inputs.jsonl"):
        rows.append((record["case_id"], record["note_text"], truth[record["case_id"]]["pii_entities"],
                     record["structured_features"].get("age_years")))
    return rows


def evaluate_transform(transform: str, verbose: bool = False, use_age: bool = True) -> dict:
    inputs, truth, predictions = {}, {}, {}
    for case_id, note, spans, age in labelled_notes():
        if transform == "identity":
            new_note, new_spans = note, [{k: s[k] for k in ("start", "end", "label")} for s in spans]
        else:
            new_note, new_spans = perturb(note, spans, transform, seed=13)
        inputs[case_id] = {"note_text": new_note}
        truth[case_id] = {"pii_entities": new_spans}
        predicted = detect_pii(new_note, age_years=age if use_age else None)
        predictions[case_id] = {"pii_entities": [s.to_json() for s in predicted],
                                "deidentified_text": render_deidentified(new_note, predicted)}
        if verbose:
            gold = {(s["start"], s["end"], s["label"]) for s in new_spans}
            pred = {(s.start, s.end, s.label) for s in predicted}
            for s, e, lab in sorted(gold - pred):
                print(f"  [{transform}] FN {case_id} {lab:14s} {new_note[s:e]!r}")
            for s, e, lab in sorted(pred - gold):
                print(f"  [{transform}] FP {case_id} {lab:14s} {new_note[s:e]!r}")
    metrics = score_pii(inputs, truth, predictions, [])
    return {
        "score": metrics["score"],
        "char_f1": metrics["character_detection"]["f1"],
        "char_leakage_rate": metrics["character_detection"]["leakage_rate"],
        "entity_f1": metrics["exact_entity"]["f1"],
        "label_char_f1": metrics["label_aware_character"]["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    results = {}
    for transform in ("identity", *TRANSFORMS, "combined"):
        results[transform] = evaluate_transform(transform, verbose=args.verbose)
        r = results[transform]
        print(f"{transform:26s} score={r['score']:.4f} charF1={r['char_f1']:.4f} "
              f"leak={r['char_leakage_rate']:.4f} entityF1={r['entity_f1']:.4f}")
    results["no_age_hint_identity"] = evaluate_transform("identity", use_age=False)
    if args.output:
        write_json(args.output, {"n_notes": len(labelled_notes()), "seed": 13, "results": results})


if __name__ == "__main__":
    main()
