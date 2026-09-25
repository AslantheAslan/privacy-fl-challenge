#!/usr/bin/env python3
"""Standard challenge entry point using the deliberately simple starter baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.baseline import detect_pii, extract_clinical_data, render_deidentified


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_dict(record: dict[str, Any]) -> dict[str, Any]:
    features = dict(record.get("structured_features", {}))
    features["hospital_id"] = record.get("hospital_id", "UNKNOWN")
    return features


def train_centralized_baseline(train_records: list[dict[str, Any]]) -> Pipeline:
    x_train = [feature_dict(record) for record in train_records]
    y_train = [int(record["labels"]["readmission_30d"]) for record in train_records]
    model = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()

    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.input)
    model = train_centralized_baseline(train_records)
    probabilities = model.predict_proba([feature_dict(record) for record in evaluation_records])[:, 1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record, probability in zip(evaluation_records, probabilities):
            spans = detect_pii(record["note_text"])
            prediction = {
                "case_id": record["case_id"],
                "pii_entities": spans,
                "deidentified_text": render_deidentified(record["note_text"], spans),
                "extracted_clinical_data": extract_clinical_data(record["note_text"]),
                "readmission_probability": float(np.clip(probability, 0.0, 1.0)),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiment_summary = {
        "implementation": "starter_baseline",
        "local_models": None,
        "federated_model": None,
        "centralized_model": {"implemented": True, "algorithm": "logistic_regression"},
        "non_iid_analysis": "TODO",
        "notes": "Replace this starter with local, federated, and centralized experiments.",
    }
    (args.artifacts_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2), encoding="utf-8"
    )
    privacy_summary = {
        "mechanism": None,
        "threat_model": "TODO",
        "privacy_guarantee": "TODO",
        "utility_analysis": "TODO",
        "limitations": "TODO",
    }
    (args.artifacts_dir / "privacy_summary.json").write_text(
        json.dumps(privacy_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
