"""Dependency-free cryptographic building blocks for secure aggregation.

Only the Python standard library is used (``hashlib``, ``hmac``, ``secrets``) so
the evaluator image needs no extra packages. These primitives are standard
constructions, but this module is a research prototype: it has not been
audited, is not constant-time, and should be replaced by a vetted library
(e.g. X25519 + ChaCha20-Poly1305 from ``cryptography``) in production.

* Key agreement: finite-field Diffie–Hellman in the RFC 3526 2048-bit MODP
  group (safe prime, generator 2 of the prime-order subgroup), 256-bit private
  exponents, SHA-256 key derivation with domain separation, and a subgroup
  membership check on every received public key.
* Authenticated encryption: encrypt-then-MAC with a SHAKE-256 keystream and
  HMAC-SHA256 over (associated data, nonce, ciphertext).
* PRG: SHAKE-256 expanded to uint64 words (masks live in Z_{2^64}).
* Secret sharing: Shamir t-of-n over the Mersenne prime 2^521 - 1.
* Fixed-point encoding of real vectors into Z_{2^64} (two's complement).
"""
from __future__ import annotations

import hashlib
import hmac
import random
import secrets
from dataclasses import dataclass

import numpy as np

# RFC 3526, group 14 (2048-bit MODP). p is a safe prime: p = 2q + 1.
MODP_2048_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
MODP_2048_ORDER = (MODP_2048_PRIME - 1) // 2
GENERATOR = 2
SHAMIR_PRIME = 2**521 - 1
PRIVATE_KEY_BITS = 256


class CryptoError(Exception):
    """Raised on authentication failure or invalid key material."""


def _random_bits(bits: int, rng: random.Random | None) -> int:
    # ``rng`` is only for deterministic unit tests; production uses ``secrets``.
    return rng.getrandbits(bits) if rng is not None else secrets.randbits(bits)


def _random_bytes(count: int, rng: random.Random | None) -> bytes:
    return rng.randbytes(count) if rng is not None else secrets.token_bytes(count)


# ---------------------------------------------------------------------------
# Diffie–Hellman key agreement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DHKeyPair:
    secret: int
    public: int


def dh_keygen(rng: random.Random | None = None) -> DHKeyPair:
    secret = _random_bits(PRIVATE_KEY_BITS, rng) | (1 << (PRIVATE_KEY_BITS - 1))
    return DHKeyPair(secret=secret, public=pow(GENERATOR, secret, MODP_2048_PRIME))


def validate_public_key(public: int) -> None:
    if not 1 < public < MODP_2048_PRIME - 1 or pow(public, MODP_2048_ORDER, MODP_2048_PRIME) != 1:
        raise CryptoError("public key is not in the prime-order subgroup")


def dh_agree(secret: int, peer_public: int, context: bytes) -> bytes:
    """Derive a 32-byte symmetric key bound to ``context`` (domain separation)."""
    validate_public_key(peer_public)
    shared = pow(peer_public, secret, MODP_2048_PRIME)
    return hashlib.sha256(b"secagg-kdf-v1|" + context + b"|" + shared.to_bytes(256, "big")).digest()


# ---------------------------------------------------------------------------
# Authenticated encryption (encrypt-then-MAC)
# ---------------------------------------------------------------------------
_NONCE_BYTES = 16
_TAG_BYTES = 32


def seal(key: bytes, plaintext: bytes, associated_data: bytes, rng: random.Random | None = None) -> bytes:
    nonce = _random_bytes(_NONCE_BYTES, rng)
    enc_key = hashlib.sha256(b"enc|" + key).digest()
    mac_key = hashlib.sha256(b"mac|" + key).digest()
    stream = hashlib.shake_256(enc_key + nonce).digest(len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(mac_key, associated_data + nonce + ciphertext, hashlib.sha256).digest()
    return nonce + ciphertext + tag


def open_sealed(key: bytes, blob: bytes, associated_data: bytes) -> bytes:
    if len(blob) < _NONCE_BYTES + _TAG_BYTES:
        raise CryptoError("ciphertext too short")
    nonce, ciphertext, tag = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:-_TAG_BYTES], blob[-_TAG_BYTES:]
    enc_key = hashlib.sha256(b"enc|" + key).digest()
    mac_key = hashlib.sha256(b"mac|" + key).digest()
    expected = hmac.new(mac_key, associated_data + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise CryptoError("authentication tag mismatch")
    stream = hashlib.shake_256(enc_key + nonce).digest(len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


# ---------------------------------------------------------------------------
# Pseudo-random generator for masks
# ---------------------------------------------------------------------------
def prg_uint64(seed: bytes, domain: bytes, length: int) -> np.ndarray:
    raw = hashlib.shake_256(b"secagg-prg-v1|" + domain + b"|" + seed).digest(8 * length)
    return np.frombuffer(raw, dtype="<u8").astype(np.uint64)


# ---------------------------------------------------------------------------
# Shamir secret sharing over GF(2^521 - 1)
# ---------------------------------------------------------------------------
Share = tuple[int, int]


def shamir_split(secret: int, n_shares: int, threshold: int, rng: random.Random | None = None) -> list[Share]:
    if not 0 <= secret < SHAMIR_PRIME:
        raise ValueError("secret out of field range")
    if not 1 <= threshold <= n_shares:
        raise ValueError("need 1 <= threshold <= n_shares")
    coefficients = [secret] + [_random_bits(520, rng) % SHAMIR_PRIME for _ in range(threshold - 1)]
    shares = []
    for x in range(1, n_shares + 1):
        y = 0
        for coefficient in reversed(coefficients):  # Horner
            y = (y * x + coefficient) % SHAMIR_PRIME
        shares.append((x, y))
    return shares


def shamir_reconstruct(shares: list[Share]) -> int:
    if len({x for x, _ in shares}) != len(shares):
        raise ValueError("duplicate share indices")
    secret = 0
    for i, (x_i, y_i) in enumerate(shares):
        numerator, denominator = 1, 1
        for j, (x_j, _) in enumerate(shares):
            if i != j:
                numerator = numerator * (-x_j) % SHAMIR_PRIME
                denominator = denominator * (x_i - x_j) % SHAMIR_PRIME
        secret = (secret + y_i * numerator * pow(denominator, -1, SHAMIR_PRIME)) % SHAMIR_PRIME
    return secret


# ---------------------------------------------------------------------------
# Fixed-point encoding into Z_{2^64}
# ---------------------------------------------------------------------------
FRACTIONAL_BITS = 24
_MAX_ABS = 2.0 ** (63 - FRACTIONAL_BITS - 8)  # head-room for summing up to 256 clients


def encode_fixed_point(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(vector)) or np.any(np.abs(vector) >= _MAX_ABS):
        raise ValueError("value out of fixed-point range")
    return np.round(vector * 2.0**FRACTIONAL_BITS).astype(np.int64).astype(np.uint64)


def decode_fixed_point(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.uint64).astype(np.int64).astype(float) / 2.0**FRACTIONAL_BITS
