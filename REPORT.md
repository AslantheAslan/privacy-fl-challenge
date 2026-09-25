# REPORT — Privacy-Preserving Clinical AI and Cross-Silo Federated Learning

All numbers below are reproduced by `make report` (outputs in `reports/validation_run/` and
`reports/format_shift_deid.json`). The standard command writes the same summaries to `--artifacts-dir`.

## 1. Executive summary

A transparent, dependency-light pipeline: rule-based, cue-driven **de-identification** and **clinical
extraction** (auditable, deterministic, no model weights), and an L2 logistic-regression **readmission model
trained with FedAvg** across three hospital processes that each load only their own rows. Every aggregate a
hospital releases passes through **secure aggregation** implemented from first principles (pairwise + self
masking, Shamir dropout recovery, 2048-bit DH, stdlib only); record-level **differential privacy** is
implemented with exact accounting and evaluated as an option.

On the public validation split the automated score is **37.44 / 40** (starter: 31.84): de-identification and
extraction are 1.000, readmission 0.744. De-identification remains 1.000 under ten synthetic formatting shifts.
In site-stratified CV, **federated (AUC 0.842, Brier 0.147) matches centralized (0.837, 0.147) and beats
local-only models (0.809, 0.170)**. Secure aggregation is lossless (max parameter difference 7·10⁻¹⁰) at ~0.4 s
extra runtime; DP costs ~0.015 AUC at ε = 4 and ~0.10 at ε = 1 (δ = 10⁻³).

## 2. System architecture

```
                    ┌──────────────────── hospital node k (own OS process) ────────────────────┐
 train.jsonl ─rows of│ notes ─► PII detector ─► de-identified text                              │
   site k only       │   └────► clinical extractor ─► features ─► private X_k, y_k              │
                     │ releases ONLY: moments / n_k·θ_k / loss, masked in Z_2^64 (SecAgg)       │
                     └───────────────────────────────┬─────────────────────────────────────────┘
                                                     │ JSON bytes (no pickle), audit-logged
                                   ┌─────────────────▼─────────────────┐
                                   │ FedAvg server: routes encrypted   │
                                   │ shares, sums masked vectors,      │
                                   │ θ ← Σ n_k θ_k / Σ n_k             │
                                   └───────────────────────────────────┘
```

| Component | Module | Key design choice |
|---|---|---|
| De-identification | `src/deid.py`, `src/spans.py` | prioritised rule passes + greedy non-overlap resolver; every span carries its rule (`source`) |
| Extraction | `src/extraction.py` | lexicons (abbrev. case-sensitive), NegEx-style clause scope, unit conversion, explicit null policy |
| Features | `src/features.py` | computed inside each node; standardisation from aggregated moments |
| FL runtime | `src/fl/{client,server,transport,logreg}.py` | clients hold private arrays, answer an allow-listed set of aggregate requests |
| Experiments | `src/fl/experiments.py` | local vs federated vs centralized, CV, seeds, convergence, DP, SecAgg benchmarks |
| Privacy | `src/privacy/{crypto,secagg,dp,summary}.py` | SecAgg (used) + DP-GD (evaluated) |

## 3. De-identification

**Method.** Six prioritised passes: e-mail → phone (international `+CC…`, or national after a phone cue) →
identifiers (site schemes, generic `PREFIX[-/]digits` with ≥5 digits, or any token after an ID cue) → dates
(ISO, numeric, month-name and cue-gated two-digit-year formats) → addresses (anchored on a German 5-digit PLZ
or Indian 6-digit PIN plus city, expanded left to the field boundary, cue words stripped) → names. Dates are
typed by the *nearest preceding* cue within the same field (`DOB/born` vs `Admission/seen/Encounter`); cue-less
dates fall back to consistency with the structured `age_years`. Names require two Title-case tokens after a
patient/clinician cue or title, a structural pattern (`Name, born …`, `Name / MRN`), or a match with a
clinician e-mail local part (`laura.koenig@` → `Laura König`, with umlaut transliteration). Every confirmed
name is then propagated to its repeated mentions, which the dictionary labels separately.

**Over-redaction control.** No pass fires on a bare capitalised word: clinical eponyms (`Parkinson disease`,
`Hodgkin lymphoma`), brands (`Eliquis`), BP ratios (`123/72`) and lab values are never redacted (unit test
`test_no_over_redaction_of_clinical_content`). Titles (`Dr.`, `Dr. med.`) are excluded from spans as in the
gold standard.

**Results.** Train and validation: character-F1, entity-F1 and label-aware F1 all 1.000 (0 leaked characters).
Because the hidden split contains *additional formatting variants*, `src/format_shift.py` rewrites all 150
labelled notes with exact gold offsets under ten transforms (ISO / long / US / two-digit-year dates, `BER/…`,
`UHID/…`, `MRN-…` IDs, national phone formats, hyphenated/three-token/`O'Connor` names, inverted and
upper-case surnames, relabelled cues, single-line layout, and all combined). The first stress run exposed four
real weaknesses (honorific `Sri` stripped from names, sentence-final two-digit-year dates, inverted names in
structural position, flattened layout); after fixing them every transform scores 1.000
(`reports/format_shift_deid.json`, enforced by tests). This is evidence of robustness to *anticipated* shifts
only; truly novel layouts could still defeat the rules.

**Multimodal extension.** The detector is a function from text to character spans, so other modalities plug
in through a text layer that keeps provenance: (1) scanned documents/images → OCR (e.g. Tesseract/docTR) that
emits words with bounding boxes; (2) run the same detector on the reconstructed text; (3) map each span back to
the union of its words' boxes and burn in black rectangles (plus strip EXIF/DICOM headers, whose tags are
structured PII and can be handled by allow-lists). Faces or burned-in text in medical images need a vision
detector feeding the same span/box abstraction. Low OCR confidence should trigger over-redaction of the whole
line, because recall matters more than precision for PII.

## 4. Structured extraction and standardisation

* **Terminology.** Canonical diagnoses and drugs have lexicons of synonyms, abbreviations (case-sensitive:
  `AF`, `HTN`, `T2DM`, `HFrEF`, `NSTEMI`, `COAD`…), German terms (`Vorhofflimmern`) and brands/variants
  (`Eliquis`, `Xarelto`, `Lasix`/`frusemide`, `ecosprin`, `ASA`, `azithro`, `APX`). Type-1 diabetes is
  explicitly prevented from mapping to `type_2_diabetes`.
* **Negation / context.** Clauses are split on sentence punctuation, `;`, `|` and new lines. A concept is
  dropped if a pre-trigger (`denies`, `no evidence of`, `?`) occurs within six tokens before it and after the
  last contrastive conjunction, if a post-trigger follows (`considered but not confirmed`, `not started`,
  `discontinued`), if the clause is family history, or — for drugs — if the clause is an allergy statement.
* **Units.** µmol/L creatinine ÷ 88.4, mmol/L × 1000/88.4; Hb g/L ÷ 10, mmol/L × 1.611; unit-less values are
  disambiguated by magnitude; decimal commas accepted; plausibility ranges guard against mis-reads
  (`RR 18/min` is never a blood pressure, `HbA1c` is never Hb, `SpO2 95%` is never LVEF; LVEF ranges → midpoint).
* **Missing values.** Absent or explicitly undocumented values (`not documented`, `unavailable`,
  `not captured`) → `null`; `No smoking data` → `null`, not `never`.
* **Results.** 1.000 on train and validation for every field (diagnosis/medication F1, all numeric fields
  within tolerance, both categorical fields). 60+ unit tests cover the variants above.

## 5. Federated-learning experiment

**Data partitioning.** Each `HospitalClient` is constructed from its own site's rows only (it raises if given
another site's row). For the submitted model each hospital runs in a separate OS process that reads the
training file itself and keeps only its site; the server holds nothing but pipes. Messages are JSON with
tagged arrays (no pickle), and a transport log records the fields, shapes and bytes of every message.

**Model and features.** L2 logistic regression on 13 features chosen by repeated CV on the training split:
age, prior admissions, length of stay, emergency admission, site indicator (3), and six comorbidities from the
NLP extraction (heart failure, CKD, AF, COPD, T2DM, CAD). Labs/vitals were tested (`extended` set) and did not
improve CV performance. Standardisation uses pooled mean/std computed *federatedly* from securely aggregated
per-feature `(count, sum, sum²)` — provably identical to pooled statistics (test).

**Training.** FedAvg: 60 rounds × 2 local epochs of mini-batch SGD (batch 16, lr 0.2), client weight
`n_k / Σn`, init `N(0, 0.01²)`. L2 = 0.05 selected at run time by *federated* site-stratified CV log-loss over
{0.01, 0.02, 0.05, 0.1, 0.2}. Each node sends `[n_k·θ_k, n_k]` so one secure sum yields the weighted mean.

**Fair comparison.** Local, federated and centralized models share model class, features, L2, optimiser,
learning rate, batch size, seed and number of passes over each row; federated and centralized share
preprocessing exactly. Local models can only use their own statistics. The centralized model is a
*non-deployable upper reference* (it requires pooling rows). The exact pooled optimum (Newton) is used for
convergence analysis.

**Evaluation design.** 5-fold CV *within each site*, stratified by label, repeated 3×, so every node has its own
train/test split as it would in deployment. The 30-case validation split has 6 positives (1 in Hyderabad), so
its per-site AUCs are uninterpretable; it is reported with bootstrap CIs as a sanity check only.

| Site-stratified CV (mean ± sd over repeats) | ROC AUC | AP | Brier | log-loss | Berlin AUC | Chennai AUC | Hyderabad AUC |
|---|---|---|---|---|---|---|---|
| Local (each site scores its own patients) | 0.809 ± 0.004 | 0.682 | 0.170 | 0.505 | 0.642 | 0.863 | 0.872 |
| **Federated (FedAvg)** | **0.842 ± 0.018** | **0.770** | **0.147** | **0.457** | 0.649 | 0.884 | 0.926 |
| Centralized (pooled reference) | 0.837 ± 0.016 | 0.767 | 0.147 | 0.457 | 0.645 | 0.883 | 0.927 |
| Starter baseline (same CV) | 0.696 | 0.568 | 0.218 | 0.621 | – | – | – |

| Validation holdout (30 cases, 6 positives; mean over 5 seeds) | ROC AUC [95% CI, seed 0] | Brier | log-loss | mean predicted |
|---|---|---|---|---|
| Local | 0.767 [0.52, 0.99] | 0.128 | 0.421 | 0.269 |
| Federated | 0.761 [0.50, 0.97] | 0.124 | 0.404 | 0.274 |
| Centralized | 0.765 [0.50, 0.97] | 0.124 | 0.404 | 0.275 |

The starter's higher validation AUC (0.819, CI 0.56–1.00) does not survive cross-validation, and its
`class_weight="balanced"` makes it badly calibrated (mean prediction 0.50 vs prevalence 0.20; Brier 0.213).

**Non-IID effects.** Label skew (prevalence Berlin 0.36, Chennai 0.44, Hyderabad 0.23) and strong coverage
skew: **Berlin has no CKD case and one patient with a prior admission** in training (Chennai: 6 CKD cases,
4 patients with prior admissions); mean
age is 68 in Berlin vs 58–59 elsewhere. A Berlin-only model therefore cannot learn two of the strongest risk
factors; federation improves Hyderabad (0.872 → 0.926) and Chennai (0.863 → 0.884) and, above all, calibration
at every site. Berlin stays hardest for every regime (≈0.65), which points to outcome noise at that site rather
than a federation deficit. Personalisation (FedAvg + 2/5 local fine-tuning epochs) did not help
(0.843/0.826 AUC): the site indicator already carries site baselines and fine-tuning on ~32 rows adds variance.

**Convergence and client drift.** Pooled optimum objective 0.4388. With E = 2 the global objective is within
10⁻³ after 17 rounds and ends 3·10⁻⁴ above the optimum (mini-batch noise floor). Larger E converges in fewer
rounds but to a worse fixed point — the non-IID client-drift signature: final gap 0.0003 (E = 1, 2),
0.0011 (E = 5), 0.0013 (E = 10). Uniform weighting behaves similarly (sites are near-equal in size) but optimises
a site-reweighted objective, so its gap to the pooled optimum is slightly larger. E = 2 is the chosen trade-off.

**Seed stability.** Across 10 seeds: final objective 0.4391 ± 0.0001, parameter sd ≤ 0.021, largest per-patient
prediction sd 0.016, holdout AUC 0.759 ± 0.013. Secure-aggregation keys use the OS CSPRNG, yet outputs are
bit-reproducible because masks cancel exactly (a clean virtualenv reproduced the predictions byte-for-byte).

**What is exchanged.** Per training round each node sends 15 numbers (`n_k·θ_k`, `n_k`) — masked — and
receives the 14-parameter global model; once, 39 moment values for standardisation. Under secure aggregation
the node also sends encrypted Shamir shares and unmasking shares. Submitted run: ≈217 kB up / 188 kB down per
node for 60 rounds (vs ≈14 kB plaintext). No notes, features, labels, row-level gradients or per-row
predictions ever leave a node (enforced by the client's task allow-list and a test over the transport log).

## 6. Privacy extension and threat model

**Federated learning is not a privacy guarantee.** FedAvg keeps rows on site, but updates are deterministic
functions of patient data and can be inverted or used for membership inference. We therefore add:

**Secure aggregation (implemented, used for the submitted model).** Semi-honest Bonawitz et al. (2017):
each node publishes two DH public keys (RFC 3526 2048-bit group, subgroup-checked), Shamir-shares (t = 2 of 3,
over GF(2⁵²¹−1)) its mask key and a fresh per-round self-mask seed to peers under encrypt-then-MAC
(SHAKE-256 keystream + HMAC-SHA256). Each input is fixed-point encoded (24 fractional bits) in Z_2⁶⁴ and hidden
by a self mask plus pairwise masks from SHAKE-256 with per-round domain separation. The server unmasks with
reconstructed self masks, or — if a node drops — with its reconstructed mask key; a node never reveals both
shares for the same peer in a round, and a node whose key was reconstructed is excluded afterwards.

* *Protected asset:* each hospital's per-round model, sample size, loss and feature moments.
* *Adversary:* honest-but-curious server, possibly colluding with one hospital; passive eavesdropper.
* *Claim:* the server's view (even with one colluding node) is simulatable from the sum of the other nodes'
  vectors; it learns Σ n_kθ_k, Σ n_k, pooled moments and pooled loss — nothing per hospital.
* *Not guaranteed:* the sum and the global model are revealed (server + one hospital learn the other two's
  sum); no protection against a malicious server (key substitution, inconsistent dropout claims — needs PKI and
  the consistency round), poisoned updates (which SecAgg makes harder to detect), or inference from the final
  model.
* *Cost:* lossless (max |Δθ| = 6.7·10⁻¹⁰ vs plaintext); 0.05 s → 0.47 s for 60 rounds; ≈7× bytes; one node
  dropping at round 30 is recovered and training completes on two nodes (test + benchmark).

**Differential privacy (implemented, evaluated, off by default).** Local full-batch DP-GD with per-example
clipping (C = 1) and Gaussian noise; add/remove-one-record adjacency with n_k public. T full-batch Gaussian steps
compose exactly to μ-GDP with μ = √T/σ, converted exactly to (ε, δ); the RDP bound is reported as a check
(always ≥ exact, e.g. 4.49 vs 4.00). The guarantee holds against any observer of the node's messages and,
by post-processing, the final model and predictions.

| ε (δ = 10⁻³), T = 30 steps | σ | CV AUC | CV Brier | holdout AUC |
|---|---|---|---|---|
| 0.5 | 25.3 | 0.671 ± 0.077 | 0.241 | 0.682 |
| 1 | 14.1 | 0.747 ± 0.038 | 0.190 | 0.721 |
| 2 | 7.9 | 0.800 ± 0.014 | 0.163 | 0.753 |
| 4 | 4.5 | 0.827 ± 0.006 | 0.148 | 0.739 |
| 8 | 2.6 | 0.831 ± 0.005 | 0.144 | 0.740 |
| ∞ (same optimiser) | 0 | 0.845 ± 0.002 | 0.144 | 0.764 |

With ~40 records per hospital, noise on the summed gradient is large relative to signal, so utility drops
sharply below ε ≈ 2. A deployment needing a formal patient-level bound should use ε ∈ [4, 8] with SecAgg; the
submission keeps DP off to maximise utility and states this explicitly in `privacy_summary.json`.

**Remaining attack surface.** Model inversion/membership inference on the released model (without DP);
malicious-server attacks; poisoning; traffic analysis of dropouts; non-constant-time pure-Python big-integer
cryptography (prototype, not audited); residual quasi-identifiers (age, rare diagnosis combinations) in
de-identified text.

## 7. Reproducibility and testing

* `requirements.txt` pins numpy 2.4.4, scikit-learn 1.8.0, pytest 9.1.1 (Python 3.11). Cryptography uses only
  the standard library; no model weights; no network access at runtime.
* `Dockerfile`: `python:3.11-slim`, non-root user, BLAS threads pinned to 1, offline test gate during build,
  standard entry point. (The container could not be built in the development sandbox — no Docker daemon — but a
  clean virtualenv built from `requirements.txt` on a fresh clone reproduced the predictions byte-for-byte.)
* Seeds: submitted model seed 7; model seeds 0–4 for comparisons; CV fold seeds 1000–1002; stability seeds
  0–9. SecAgg randomness does not affect outputs (exact mask cancellation; fixed-point rounding ≤ 2⁻²⁴).
* Runtime: ≈11 s for the standard command (≈19 s with `--eval-labels`) on 2 CPUs, far below the 30-minute limit.
* Failure handling: malformed input records, unknown sites, empty/non-string notes and extractor exceptions
  degrade to conservative defaults with per-case warnings in the summary; non-finite probabilities are replaced
  by 0.5; if subprocesses are forbidden, training falls back to in-process nodes (reported in the summary).
* Tests (`make test`, 156 tests, ≈13 s): de-id formats/labelling/invariants and all format-shift transforms;
  extraction vocabulary, negation, units, nulls; DH group safety, subgroup checks, AEAD tamper/replay, Shamir,
  fixed point; SecAgg correctness, dropout, no double reveal, fresh masks, mask uniformity; FedAvg aggregation
  rule, single-client equivalence, pooled-statistics standardisation, convergence to the pooled optimum,
  lossless SecAgg, no row-level payloads, process/in-process equivalence; exact-vs-RDP accounting, noise
  calibration, clipping bound; schema-valid end-to-end output for hostile inputs. `make lint` (ruff) is clean.

## 8. Limitations and next steps

*Benchmark limitations.* 120 templated training notes (~40 per site) with an explicit synthetic site effect:
CV intervals are wide, validation per-site metrics are meaningless, and neither the rules nor the risk model say
anything about real clinical text or risk.

*Implementation limitations and next steps.*
1. Rule-based NLP generalises only to anticipated variation; next: a small token-classification model
   (e.g. a distilled clinical NER) *behind* the rules as a recall safety net, trained federatedly on
   de-identified synthetic augmentations from `src/format_shift.py`.
2. Secure aggregation is semi-honest and simulated on one host; next: signed key advertisement (PKI), the
   consistency round for malicious security, real network transport with timeouts, and a vetted library
   (X25519 + ChaCha20-Poly1305).
3. DP accounting assumes public n_k and full-batch steps; next: Poisson-subsampled DP-SGD with a PRV accountant,
   and distributed DP (noise added under SecAgg) to reduce per-node noise.
4. FedAvg shows client drift for E ≥ 5; with more sites or stronger skew, FedProx/SCAFFOLD should be evaluated.
5. Readmission model is linear; with more data, interactions or gradient-boosted trees via federated
   histogram aggregation would be natural extensions.
