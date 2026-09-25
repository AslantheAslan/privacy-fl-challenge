"""Cryptographic primitives and the secure-aggregation protocol."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.privacy.crypto import (
    MODP_2048_ORDER,
    MODP_2048_PRIME,
    CryptoError,
    decode_fixed_point,
    dh_agree,
    dh_keygen,
    encode_fixed_point,
    open_sealed,
    seal,
    shamir_reconstruct,
    shamir_split,
    validate_public_key,
)
from src.privacy.secagg import SecAggClient, SecAggError, SecAggServer, _pairwise_mask


def _miller_rabin(n: int, rounds: int = 16) -> bool:
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    rng = random.Random(0)
    for _ in range(rounds):
        x = pow(rng.randrange(2, n - 2), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def test_dh_group_is_a_safe_prime_with_generator_in_subgroup() -> None:
    assert MODP_2048_PRIME.bit_length() == 2048
    assert _miller_rabin(MODP_2048_PRIME) and _miller_rabin(MODP_2048_ORDER)
    assert pow(2, MODP_2048_ORDER, MODP_2048_PRIME) == 1


def test_dh_agreement_is_symmetric_and_context_bound() -> None:
    rng = random.Random(1)
    a, b = dh_keygen(rng), dh_keygen(rng)
    assert dh_agree(a.secret, b.public, b"ctx") == dh_agree(b.secret, a.public, b"ctx")
    assert dh_agree(a.secret, b.public, b"ctx") != dh_agree(a.secret, b.public, b"other")


def _non_residue() -> int:
    """Smallest element outside the prime-order subgroup (small-subgroup attack input)."""
    return next(a for a in range(3, 1000) if pow(a, MODP_2048_ORDER, MODP_2048_PRIME) != 1)


@pytest.mark.parametrize("bad", [0, 1, MODP_2048_PRIME - 1, MODP_2048_PRIME, _non_residue()])
def test_invalid_public_keys_are_rejected(bad: int) -> None:
    with pytest.raises(CryptoError):
        validate_public_key(bad)


def test_authenticated_encryption_detects_tampering_and_wrong_context() -> None:
    key = bytes(range(32))
    blob = seal(key, b"share payload", b"aad", random.Random(2))
    assert open_sealed(key, blob, b"aad") == b"share payload"
    tampered = blob[:20] + bytes([blob[20] ^ 1]) + blob[21:]
    with pytest.raises(CryptoError):
        open_sealed(key, tampered, b"aad")
    with pytest.raises(CryptoError):
        open_sealed(key, blob, b"different aad")


def test_shamir_threshold_reconstruction() -> None:
    secret = 2**255 + 12345
    shares = shamir_split(secret, 5, 3, random.Random(3))
    assert shamir_reconstruct(shares[:3]) == secret
    assert shamir_reconstruct([shares[4], shares[1], shares[2]]) == secret
    assert shamir_reconstruct(shares[:2]) != secret  # below threshold
    with pytest.raises(ValueError):
        shamir_reconstruct([shares[0], shares[0]])


def test_fixed_point_roundtrip_and_overflow_guard() -> None:
    values = np.array([-1234.5, -1e-3, 0.0, 3.14159, 1e6])
    assert np.allclose(decode_fixed_point(encode_fixed_point(values)), values, atol=2**-23)
    with pytest.raises(ValueError):
        encode_fixed_point(np.array([1e12]))
    with pytest.raises(ValueError):
        encode_fixed_point(np.array([np.nan]))


# ---------------------------------------------------------------------------
def _setup(n: int = 3, threshold: int = 2, seed: int = 0):
    rng = random.Random(seed)
    clients = {i: SecAggClient(i, threshold, rng=rng) for i in range(n)}
    keys = {i: c.advertise_keys() for i, c in clients.items()}
    for c in clients.values():
        c.receive_public_keys(keys)
    outboxes = {i: c.share_mask_key() for i, c in clients.items()}
    for v in clients:
        clients[v].receive_shares("mask_key", 0, {u: outboxes[u][v] for u in clients if u != v})
    return clients, SecAggServer(threshold, keys)


def _round(clients, server, round_id, inputs, drop=()):
    outboxes = {i: c.share_self_mask(round_id) for i, c in clients.items()}
    for v in clients:
        clients[v].receive_shares("self_mask", round_id, {u: outboxes[u][v] for u in clients if u != v})
    participants = sorted(clients)
    masked = {i: clients[i].masked_input(round_id, inputs[i], participants) for i in participants if i not in drop}
    survivors, dropped = sorted(masked), sorted(drop)
    responses = {i: clients[i].unmask_response(round_id, survivors, dropped) for i in survivors}
    return server.unmask(round_id, masked, participants, responses), masked


def test_secure_sum_matches_plain_sum_over_rounds() -> None:
    clients, server = _setup()
    rng = np.random.default_rng(0)
    for round_id in (1, 2, 3):
        inputs = {i: rng.normal(size=16) * 50 for i in clients}
        total, masked = _round(clients, server, round_id, inputs)
        assert np.allclose(total, sum(inputs.values()), atol=1e-6)
        for i in clients:  # what the server receives is not the encoded input
            assert not np.array_equal(masked[i], encode_fixed_point(inputs[i]))


def test_dropout_recovery_returns_sum_of_survivors() -> None:
    clients, server = _setup()
    inputs = {i: np.full(4, float(i + 1)) for i in clients}
    total, _ = _round(clients, server, 1, inputs, drop={2})
    assert np.allclose(total, inputs[0] + inputs[1], atol=1e-6)


def test_too_many_dropouts_abort() -> None:
    clients, server = _setup()
    with pytest.raises(SecAggError):
        _round(clients, server, 1, {i: np.ones(3) for i in clients}, drop={1, 2})


def test_client_never_reveals_both_shares_for_one_peer_in_a_round() -> None:
    clients, _ = _setup()
    outboxes = {i: c.share_self_mask(1) for i, c in clients.items()}
    for v in clients:
        clients[v].receive_shares("self_mask", 1, {u: outboxes[u][v] for u in clients if u != v})
    clients[0].unmask_response(1, survivors=[0, 1, 2], dropped=[])
    with pytest.raises(SecAggError):
        clients[0].unmask_response(1, survivors=[0, 1], dropped=[2])  # asks for mask key of a "survivor"


def test_tampered_share_is_rejected() -> None:
    rng = random.Random(5)
    clients = {i: SecAggClient(i, 2, rng=rng) for i in range(3)}
    keys = {i: c.advertise_keys() for i, c in clients.items()}
    for c in clients.values():
        c.receive_public_keys(keys)
    blob = clients[0].share_mask_key()[1]
    with pytest.raises(SecAggError):
        clients[1].receive_shares("mask_key", 0, {0: blob[:-1] + bytes([blob[-1] ^ 1])})
    with pytest.raises(SecAggError):  # replay into the wrong round / purpose
        clients[1].receive_shares("self_mask", 7, {0: blob})


def test_pairwise_masks_are_fresh_each_round() -> None:
    seed = bytes(32)
    assert not np.array_equal(_pairwise_mask(seed, 1, 8), _pairwise_mask(seed, 2, 8))


def test_masked_vector_is_statistically_independent_of_input() -> None:
    """With fresh masks, masked outputs for very different inputs look alike (uniform high bits)."""
    clients, _ = _setup(seed=9)
    outboxes = {i: c.share_self_mask(1) for i, c in clients.items()}
    for v in clients:
        clients[v].receive_shares("self_mask", 1, {u: outboxes[u][v] for u in clients if u != v})
    masked = clients[0].masked_input(1, np.zeros(4096), [0, 1, 2])
    high_bits = (masked >> np.uint64(63)).astype(float)
    assert abs(high_bits.mean() - 0.5) < 0.05  # zero input would otherwise have all-zero high bits
