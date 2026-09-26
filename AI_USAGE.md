# AI Tool Usage Disclosure

## Tool

* **Claude (Anthropic), model Claude Opus 5.5**, used through claude.ai with a code-execution workspace,
  25–26 September 2026. Commits written in that session carry a `Co-Authored-By: Claude` trailer.

## What the tool did

Claude was used as an implementation partner for most of the repository. Concretely it:

* analysed the training notes (templates, PII cue contexts, diagnosis/medication surface forms) and proposed the
  rule-based design for de-identification and extraction, then implemented `src/deid.py`,
  `src/extraction.py` and `src/spans.py`;
* designed the span-aware formatting-shift generator (`src/format_shift.py`) and used it to find and fix four
  de-identification weaknesses (honorific stripping, sentence-final two-digit-year dates, inverted names in
  structural position, flattened layout);
* ran the feature-set exploration with repeated CV and chose the compact feature set;
* implemented the FL stack (`src/fl/`), the secure-aggregation protocol and primitives (`src/privacy/`), the
  DP mechanism and accountants, the experiment battery and the pipeline;
* wrote the 156 tests, the Dockerfile, Makefile, README and REPORT, and generated `reports/`.

## How the output was verified

* **Official metrics**: every component was scored with the challenge's own evaluator functions
  (`scripts/component_report.py`, `make evaluate`); per-entity and per-field errors were inspected, not only
  aggregates.
* **Independent checks of cryptographic constants**: the RFC 3526 prime was checked to be a 2048-bit safe prime
  with the generator in the prime-order subgroup (Miller–Rabin, also a test). A wrong test assumption about a
  quadratic non-residue was caught by the test suite and corrected (the code was right, the test was not).
* **Equivalence tests** instead of trusting implementations: FedAvg with one client equals local SGD; federated
  standardisation equals pooled statistics; FedAvg reaches the Newton optimum's objective; secure aggregation
  equals plaintext aggregation; process and in-process backends agree; exact DP accounting is never looser
  than the RDP bound and inverts the noise calibration.
* **Scepticism about validation numbers**: the starter scored higher on the 30-case validation split for
  readmission; this was checked under identical cross-validation (starter AUC 0.696 vs 0.842) before
  concluding it was noise, and both results are reported.
* **Reproducibility**: a fresh clone in a clean virtualenv from `requirements.txt` reproduced the predictions
  byte-for-byte. The Docker image could not be built in the sandbox (no Docker daemon) and is untested there.
* A wrong semantic expectation in one of Claude's own extraction tests (`"No smoking data"` → `never`) was
  found on review and corrected to `null`, with the extractor changed accordingly.
