# REPORT.md — Suggested Structure

## 1. Executive summary

State the final approach and the most important findings in no more than 200 words.

## 2. System architecture

Describe the de-identification, extraction, prediction, federated-learning and privacy components. Include a compact diagram if useful.

## 3. De-identification

Explain the method, error analysis, over-redaction controls, and how the design could extend to image or document modalities.

## 4. Structured extraction and standardization

Explain terminology normalization, negation handling, unit conversion, missing values, and validation results.

## 5. Federated-learning experiment

Document local, federated and centralized models, data partitioning, rounds, local updates, aggregation, convergence, site-specific metrics, and the effect of non-IID data.

## 6. Privacy extension and threat model

Define the adversary, protected information, trust assumptions, mechanism, parameter choices, privacy claim, utility or computational cost, and limitations.

## 7. Reproducibility and testing

List deterministic settings, tests, runtime, environment, known nondeterminism and failure handling.

## 8. Limitations and next steps

Be specific. Distinguish benchmark limitations from limitations of your own implementation.
