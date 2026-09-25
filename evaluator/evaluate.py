#!/usr/bin/env python3
"""Evaluate submissions for the synthetic Privacy-Preserving Clinical AI challenge."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PII_LABELS = (
    "PATIENT_NAME",
    "DATE_OF_BIRTH",
    "ENCOUNTER_DATE",
    "ADDRESS",
    "PHONE_NUMBER",
    "PATIENT_ID",
    "CLINICIAN_NAME",
    "EMAIL",
)
LIST_FIELDS = ("diagnoses", "medications")
NUMERIC_TOLERANCES = {
    "heart_rate_bpm": 3.0,
    "systolic_bp_mmhg": 5.0,
    "creatinine_mg_dl": 0.10,
    "hemoglobin_g_dl": 0.30,
    "lvef_percent": 3.0,
}
CATEGORICAL_FIELDS = ("smoking_status", "allergy")
ALL_EXTRACTION_FIELDS = LIST_FIELDS + tuple(NUMERIC_TOLERANCES) + CATEGORICAL_FIELDS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            records.append(value)
    return records


def index_by_case(records: Iterable[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Record without a valid case_id in {source_name}")
        if case_id in indexed:
            raise ValueError(f"Duplicate case_id {case_id!r} in {source_name}")
        indexed[case_id] = record
    return indexed


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def normalize_token(value: Any) -> str:
    if value is None:
        return "__NULL__"
    return " ".join(str(value).strip().lower().replace("-", "_").split())


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [normalize_token(value)]
    return sorted({normalize_token(item) for item in value if str(item).strip()})


def numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_spans(spans: Any, note: str, case_id: str, errors: list[str]) -> list[dict[str, Any]]:
    if spans is None:
        return []
    if not isinstance(spans, list):
        errors.append(f"{case_id}: pii_entities must be a list")
        return []
    cleaned: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"{case_id}: pii_entities[{index}] is not an object")
            continue
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{case_id}: pii_entities[{index}] has non-integer offsets")
            continue
        if not 0 <= start < end <= len(note):
            errors.append(f"{case_id}: pii_entities[{index}] has out-of-bounds offsets")
            continue
        if label not in PII_LABELS:
            errors.append(f"{case_id}: pii_entities[{index}] has unsupported label {label!r}")
            continue
        cleaned.append({"start": start, "end": end, "label": label})
    cleaned.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    previous_end = -1
    non_overlapping: list[dict[str, Any]] = []
    for span in cleaned:
        if span["start"] < previous_end:
            errors.append(f"{case_id}: overlapping predicted PII spans; later overlap ignored")
            continue
        non_overlapping.append(span)
        previous_end = span["end"]
    return non_overlapping


def render_deidentified(note: str, spans: list[dict[str, Any]]) -> str:
    output = note
    for span in reversed(spans):
        output = output[: span["start"]] + f"[{span['label']}]" + output[span["end"] :]
    return output


def create_char_labels(note_length: int, spans: list[dict[str, Any]]) -> list[str | None]:
    labels: list[str | None] = [None] * note_length
    for span in spans:
        for position in range(span["start"], span["end"]):
            labels[position] = span["label"]
    return labels


def score_pii(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    char_tp = char_fp = char_fn = 0
    label_char_tp = label_char_fp = label_char_fn = 0
    entity_tp = entity_fp = entity_fn = 0
    render_matches = 0
    label_entity_counts: dict[str, dict[str, int]] = {label: {"tp": 0, "fp": 0, "fn": 0} for label in PII_LABELS}

    for case_id, gt in ground_truth.items():
        note = str(inputs[case_id]["note_text"])
        true_spans = validate_spans(gt.get("pii_entities", []), note, f"GT/{case_id}", errors)
        pred_record = predictions.get(case_id, {})
        pred_spans = validate_spans(pred_record.get("pii_entities", []), note, case_id, errors)

        true_chars = create_char_labels(len(note), true_spans)
        pred_chars = create_char_labels(len(note), pred_spans)
        for true_label, pred_label in zip(true_chars, pred_chars):
            true_positive = true_label is not None
            predicted_positive = pred_label is not None
            if true_positive and predicted_positive:
                char_tp += 1
            elif predicted_positive:
                char_fp += 1
            elif true_positive:
                char_fn += 1

            if true_label is not None and pred_label == true_label:
                label_char_tp += 1
            else:
                if pred_label is not None:
                    label_char_fp += 1
                if true_label is not None:
                    label_char_fn += 1

        true_set = {(item["start"], item["end"], item["label"]) for item in true_spans}
        pred_set = {(item["start"], item["end"], item["label"]) for item in pred_spans}
        entity_tp += len(true_set & pred_set)
        entity_fp += len(pred_set - true_set)
        entity_fn += len(true_set - pred_set)
        for label in PII_LABELS:
            true_label_set = {item for item in true_set if item[2] == label}
            pred_label_set = {item for item in pred_set if item[2] == label}
            label_entity_counts[label]["tp"] += len(true_label_set & pred_label_set)
            label_entity_counts[label]["fp"] += len(pred_label_set - true_label_set)
            label_entity_counts[label]["fn"] += len(true_label_set - pred_label_set)

        predicted_text = pred_record.get("deidentified_text")
        if isinstance(predicted_text, str) and predicted_text == render_deidentified(note, pred_spans):
            render_matches += 1
        elif predicted_text is not None and not isinstance(predicted_text, str):
            errors.append(f"{case_id}: deidentified_text must be a string")

    char_precision, char_recall, char_f1 = f1_from_counts(char_tp, char_fp, char_fn)
    label_char_precision, label_char_recall, label_char_f1 = f1_from_counts(label_char_tp, label_char_fp, label_char_fn)
    entity_precision, entity_recall, entity_f1 = f1_from_counts(entity_tp, entity_fp, entity_fn)
    per_label: dict[str, Any] = {}
    macro_values: list[float] = []
    for label, counts in label_entity_counts.items():
        precision, recall, f1 = f1_from_counts(counts["tp"], counts["fp"], counts["fn"])
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, **counts}
        macro_values.append(f1)
    render_exact_rate = safe_div(render_matches, len(ground_truth))
    composite = 0.55 * char_f1 + 0.25 * entity_f1 + 0.15 * label_char_f1 + 0.05 * render_exact_rate
    return {
        "score": composite,
        "character_detection": {
            "precision": char_precision,
            "recall": char_recall,
            "f1": char_f1,
            "tp_chars": char_tp,
            "fp_chars": char_fp,
            "fn_chars": char_fn,
            "leakage_rate": safe_div(char_fn, char_tp + char_fn),
            "false_discovery_rate": safe_div(char_fp, char_tp + char_fp),
        },
        "label_aware_character": {
            "precision": label_char_precision,
            "recall": label_char_recall,
            "f1": label_char_f1,
        },
        "exact_entity": {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1,
            "tp": entity_tp,
            "fp": entity_fp,
            "fn": entity_fn,
            "macro_f1": float(np.mean(macro_values)),
            "per_label": per_label,
        },
        "deidentified_text_render_exact_rate": render_exact_rate,
    }


def score_set_field(
    field: str,
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tp = fp = fn = 0
    for case_id, gt in ground_truth.items():
        true_values = set(normalize_list(gt.get("extracted_clinical_data", {}).get(field)))
        pred_values = set(normalize_list(predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field)))
        tp += len(true_values & pred_values)
        fp += len(pred_values - true_values)
        fn += len(true_values - pred_values)
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_numeric_field(
    field: str,
    tolerance: float,
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_scores: list[float] = []
    absolute_errors: list[float] = []
    exact_within_tolerance = 0
    for case_id, gt in ground_truth.items():
        true_value = numeric_value(gt.get("extracted_clinical_data", {}).get(field))
        pred_value = numeric_value(predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field))
        if true_value is None and pred_value is None:
            case_scores.append(1.0)
            exact_within_tolerance += 1
        elif true_value is None or pred_value is None:
            case_scores.append(0.0)
        else:
            error = abs(pred_value - true_value)
            absolute_errors.append(error)
            if error <= tolerance:
                score = 1.0
                exact_within_tolerance += 1
            else:
                score = max(0.0, 1.0 - (error - tolerance) / (2.0 * tolerance))
            case_scores.append(score)
    return {
        "score": float(np.mean(case_scores)) if case_scores else 0.0,
        "within_tolerance_rate": safe_div(exact_within_tolerance, len(case_scores)),
        "mean_absolute_error_nonmissing_pairs": float(np.mean(absolute_errors)) if absolute_errors else None,
        "tolerance": tolerance,
    }


def score_categorical_field(
    field: str,
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    correct = 0
    total = len(ground_truth)
    for case_id, gt in ground_truth.items():
        true_value = normalize_token(gt.get("extracted_clinical_data", {}).get(field))
        pred_value = normalize_token(predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field))
        correct += int(true_value == pred_value)
    return {"accuracy": safe_div(correct, total), "correct": correct, "n": total}


def score_extraction(
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    for case_id, record in predictions.items():
        extracted = record.get("extracted_clinical_data")
        if extracted is not None and not isinstance(extracted, dict):
            errors.append(f"{case_id}: extracted_clinical_data must be an object")

    diagnoses = score_set_field("diagnoses", ground_truth, predictions)
    medications = score_set_field("medications", ground_truth, predictions)
    numeric = {
        field: score_numeric_field(field, tolerance, ground_truth, predictions)
        for field, tolerance in NUMERIC_TOLERANCES.items()
    }
    categorical = {field: score_categorical_field(field, ground_truth, predictions) for field in CATEGORICAL_FIELDS}
    numeric_average = float(np.mean([value["score"] for value in numeric.values()]))
    categorical_average = float(np.mean([value["accuracy"] for value in categorical.values()]))
    composite = 0.22 * diagnoses["f1"] + 0.22 * medications["f1"] + 0.40 * numeric_average + 0.16 * categorical_average
    return {
        "score": composite,
        "diagnoses": diagnoses,
        "medications": medications,
        "numeric": numeric,
        "categorical": categorical,
        "numeric_average": numeric_average,
        "categorical_average": categorical_average,
    }


def prediction_probability(record: dict[str, Any], case_id: str, errors: list[str]) -> float:
    value = numeric_value(record.get("readmission_probability"))
    if value is None or not 0.0 <= value <= 1.0:
        errors.append(f"{case_id}: invalid readmission_probability; 0.5 used for scoring")
        return 0.5
    return value


def safe_auc(y_true: list[int], y_prob: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def score_prediction(
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    case_ids = list(ground_truth)
    y_true = [int(ground_truth[case_id]["readmission_30d"]) for case_id in case_ids]
    y_prob = [prediction_probability(predictions.get(case_id, {}), case_id, errors) for case_id in case_ids]
    auc = safe_auc(y_true, y_prob)
    average_precision = float(average_precision_score(y_true, y_prob)) if any(y_true) else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    ll = float(log_loss(y_true, clipped, labels=[0, 1]))
    auc_for_score = auc if auc is not None else 0.5
    composite = 0.55 * auc_for_score + 0.25 * average_precision + 0.20 * (1.0 - brier)

    by_site: dict[str, Any] = {}
    site_to_pairs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for case_id, truth, probability in zip(case_ids, y_true, y_prob):
        site_to_pairs[str(ground_truth[case_id].get("hospital_id", "UNKNOWN"))].append((truth, probability))
    for site, pairs in sorted(site_to_pairs.items()):
        site_truth = [item[0] for item in pairs]
        site_prob = [item[1] for item in pairs]
        by_site[site] = {
            "n": len(pairs),
            "prevalence": float(np.mean(site_truth)),
            "roc_auc": safe_auc(site_truth, site_prob),
            "average_precision": float(average_precision_score(site_truth, site_prob)) if any(site_truth) else None,
            "brier_score": float(brier_score_loss(site_truth, site_prob)),
        }
    valid_site_aucs = [metrics["roc_auc"] for metrics in by_site.values() if metrics["roc_auc"] is not None]
    return {
        "score": composite,
        "roc_auc": auc,
        "average_precision": average_precision,
        "brier_score": brier,
        "log_loss": ll,
        "prevalence": float(np.mean(y_true)),
        "by_site": by_site,
        "worst_site_roc_auc": min(valid_site_aucs) if valid_site_aucs else None,
    }


def evaluate(inputs_path: Path, ground_truth_path: Path, predictions_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    inputs = index_by_case(read_jsonl(inputs_path), "inputs")
    ground_truth = index_by_case(read_jsonl(ground_truth_path), "ground truth")
    predictions = index_by_case(read_jsonl(predictions_path), "predictions")

    missing_inputs = sorted(set(ground_truth) - set(inputs))
    if missing_inputs:
        raise ValueError(f"Ground-truth cases missing from inputs: {missing_inputs[:5]}")
    missing_predictions = sorted(set(ground_truth) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(ground_truth))
    if missing_predictions:
        errors.append(f"Missing predictions for {len(missing_predictions)} cases; empty/default predictions used")
    if extra_predictions:
        errors.append(f"Predictions contain {len(extra_predictions)} unknown case IDs; ignored")

    pii = score_pii(inputs, ground_truth, predictions, errors)
    extraction = score_extraction(ground_truth, predictions, errors)
    prediction = score_prediction(ground_truth, predictions, errors)
    automated_score = 0.375 * pii["score"] + 0.375 * extraction["score"] + 0.25 * prediction["score"]
    automated_points = {
        "deidentification": 15.0 * pii["score"],
        "structured_extraction": 15.0 * extraction["score"],
        "readmission_prediction": 10.0 * prediction["score"],
    }
    automated_points["total_out_of_40"] = sum(automated_points.values())
    return {
        "n_cases": len(ground_truth),
        "missing_prediction_case_ids": missing_predictions,
        "extra_prediction_case_ids": extra_predictions,
        "validation_messages": errors,
        "metrics": {
            "deidentification": pii,
            "structured_extraction": extraction,
            "readmission_prediction": prediction,
            "automated_composite_0_to_1": automated_score,
            "automated_points": automated_points,
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(f"Cases scored: {report['n_cases']}")
    print(f"De-identification score:       {metrics['deidentification']['score']:.4f}")
    print(f"Structured extraction score:   {metrics['structured_extraction']['score']:.4f}")
    print(f"Readmission prediction score:  {metrics['readmission_prediction']['score']:.4f}")
    print(f"Automated benchmark:           {metrics['automated_composite_0_to_1']:.4f}")
    print(f"Automated points:              {metrics['automated_points']['total_out_of_40']:.2f} / 40")
    if report["validation_messages"]:
        print(f"Validation messages:           {len(report['validation_messages'])}")
        for message in report["validation_messages"][:10]:
            print(f"  - {message}")
        if len(report["validation_messages"]) > 10:
            print("  - ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.inputs, args.ground_truth, args.predictions)
    except (OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print_summary(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
