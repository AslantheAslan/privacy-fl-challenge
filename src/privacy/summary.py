"""Builds ``privacy_summary.json`` from the threat model and measured results."""
from __future__ import annotations

from typing import Any

from .crypto import FRACTIONAL_BITS, PRIVATE_KEY_BITS


def build_privacy_summary(cfg: Any, final: dict[str, Any], secagg: dict[str, Any], dp: dict[str, Any]) -> dict[str, Any]:
    dp_rows = dp["results"]
    return {
        "mechanism": "Secure aggregation (pairwise + self masking with Shamir-based dropout recovery, "
                     "Bonawitz et al. 2017, semi-honest variant); record-level DP-GD evaluated as an extension",
        "implementation_status": {
            "secure_aggregation": "implemented from first principles (src/privacy/crypto.py, secagg.py); "
                                  "ON for the submitted model and for federated standardisation",
            "differential_privacy": "implemented (src/privacy/dp.py) with exact GDP and RDP accounting; evaluated "
                                    "over a range of epsilons; OFF for the submitted model (utility trade-off below)",
            "maturity": "research prototype: correct and tested, but not audited, not constant-time",
        },
        "federated_learning_is_not_a_privacy_guarantee": (
            "FedAvg alone only keeps raw rows on site. The model updates it exchanges are deterministic functions of "
            "patient data and can leak them (gradient inversion, membership inference, property inference). Secure "
            "aggregation removes access to *individual hospitals'* updates but still reveals their sum and the final "
            "model; only differential privacy gives a formal, quantifiable bound on what any output reveals about "
            "one patient."),
        "protected_asset": {
            "secure_aggregation": "each hospital's per-round model (n_k * theta_k), its sample size n_k, its local "
                                  "loss and its feature moments (count/sum/sum-of-squares) used for standardisation",
            "differential_privacy": "whether any single patient record was part of a hospital's training data "
                                    "(add/remove-one-record adjacency, per hospital)",
        },
        "adversary": {
            "in_scope": [
                "honest-but-curious aggregation server that follows the protocol but inspects everything it sees",
                "server colluding with at most t-1 = 1 hospital",
                "passive network eavesdropper (shares are AEAD-encrypted end-to-end between hospitals)",
                "for DP: any party observing messages, the final model or its predictions (post-processing)",
            ],
            "out_of_scope": [
                "actively malicious server (key substitution, inconsistent dropout claims to different clients) — "
                "needs PKI signatures and the consistency round of the malicious-secure protocol",
                "malicious hospitals sending poisoned updates (secure aggregation makes poisoning harder to detect)",
                "compromised hospital infrastructure, side channels (timing, memory)",
            ],
        },
        "trust_assumptions": [
            f"at least t = 2 of n = 3 hospitals complete each round (threshold t = {2})",
            "public keys are authentic (in deployment: certificates/PKI; here the relaying server is trusted to "
            "forward keys unmodified — the semi-honest assumption)",
            "discrete-log / CDH hardness in the RFC 3526 2048-bit MODP group (~112-bit security)",
            "SHAKE-256 behaves as a PRG / random oracle; OS CSPRNG (secrets) for keys and seeds",
            "for DP: n_k treated as public; clipping bound enforced by the hospital's own code",
        ],
        "parameters": {
            "secure_aggregation": {
                "clients": 3, "threshold": 2,
                "key_agreement": f"finite-field DH, RFC 3526 group 14 (2048-bit safe prime), {PRIVATE_KEY_BITS}-bit "
                                 "private exponents, subgroup check on every public key, SHA-256 KDF with domain "
                                 "separation",
                "share_encryption": "encrypt-then-MAC: SHAKE-256 keystream, HMAC-SHA256 tag, 128-bit random nonce",
                "masking_ring": "Z_2^64 (uint64 wrap-around)",
                "fixed_point": f"{FRACTIONAL_BITS} fractional bits (resolution {2.0 ** -FRACTIONAL_BITS:.1e}), "
                               "overflow-checked with 8 bits of head-room for the sum",
                "prg": "SHAKE-256 with per-round and per-purpose domain separation (masks never reused)",
                "secret_sharing": "Shamir over GF(2^521 - 1)",
                "rekeying": "a hospital whose mask key was reconstructed after dropping out is excluded from later "
                            "rounds",
            },
            "differential_privacy": {
                "mechanism": "local full-batch DP-GD: per-example clipping + Gaussian noise on the summed gradient",
                "clip_norm": dp["clip_norm"],
                "delta": dp["delta"],
                "rounds": dp["rounds"],
                "local_epochs": dp["local_epochs"],
                "steps_per_record": dp["steps_per_record"],
                "accounting": "exact: T Gaussian steps compose to mu-GDP with mu = sqrt(T)/sigma (no subsampling), "
                              "converted exactly to (epsilon, delta); RDP bound reported as a cross-check",
            },
        },
        "privacy_claim": (
            "Secure aggregation: in every aggregation round, the view of a semi-honest server — even jointly with "
            "any one hospital — can be simulated from the sum of the remaining hospitals' vectors alone. The server "
            "therefore learns sum_k n_k*theta_k and sum_k n_k (hence the global model), the pooled feature moments "
            "and the pooled training loss, but no individual hospital's model, statistics or sample size. "
            "Differential privacy (when enabled): for every patient record, the joint distribution of everything "
            "its hospital ever sends — and hence of the final model and all predictions — is "
            "(epsilon, delta)-indistinguishable from the case where that record is absent, with epsilon as "
            "reported below for delta = 1e-3."),
        "what_is_not_guaranteed": [
            "The aggregate itself is revealed: with three hospitals, the server plus one hospital learns the sum of "
            "the other two; two colluding hospitals plus the server learn the third's update exactly.",
            "Without DP, the global model and predictions can leak information about individual patients "
            "(membership/attribute inference); secure aggregation offers no protection here.",
            "Every hospital receives the global model each round and can difference it against its own update.",
            "No protection against a malicious server or poisoned updates (see adversary.out_of_scope).",
            "De-identification is rule-based; residual quasi-identifiers (age, rare diagnosis combinations) remain "
            "in de-identified text, and recall on unseen formats is empirical, not guaranteed.",
        ],
        "empirical_utility_and_cost": {
            "secure_aggregation": {
                "max_abs_parameter_difference_vs_plaintext_fedavg": secagg["max_abs_parameter_difference_secure_vs_plain"],
                "runtime_seconds": secagg["runtime_seconds"],
                "client_to_server_bytes": secagg["client_to_server_bytes"],
                "dropout_test": secagg["dropout_test"],
                "submitted_model_training": {"backend": final["backend"], "seconds": round(final["seconds"], 3)},
                "interpretation": "Lossless: identical model up to fixed-point rounding. Cost is dominated by one-time "
                                  "2048-bit DH operations; per-round overhead is a few milliseconds and ~7x bytes.",
            },
            "differential_privacy": {
                "table": dp_rows,
                "interpretation": (
                    "Utility degrades gracefully down to epsilon ~ 4 and sharply below epsilon ~ 2: with only ~40 "
                    "records per hospital the Gaussian noise on the summed gradient is large relative to the signal. "
                    "The submitted model therefore uses secure aggregation without DP; a deployment that needs a "
                    "formal patient-level bound should run with epsilon in [4, 8], ideally with more data per site."),
            },
        },
        "remaining_attack_surface": [
            "membership / attribute inference against the released global model and predictions (mitigated only "
            "with DP enabled)",
            "server lying about dropouts to different clients (malicious setting) — would require signed, "
            "consistency-checked dropout lists",
            "model poisoning / backdoors by a hospital (secure aggregation hides individual updates from robust "
            "aggregation rules)",
            "traffic analysis: participation and timing reveal which hospital dropped; message sizes are constant",
            "prototype cryptography in pure Python (non-constant-time big-integer arithmetic)",
            "residual PII in de-identified notes if a hidden format defeats the detector",
        ],
        "limitations": [
            "Semi-honest security only; authentication of public keys is assumed, not implemented.",
            "Federation simulated on one host (separate processes and serialised messages, no real network).",
            "DP accounting treats n_k as public and assumes per-example clipping is enforced correctly on-site.",
        ],
        "configuration_reference": {"rounds": cfg.rounds, "local_epochs": cfg.local_epochs, "l2": cfg.l2},
    }
