"""Secure aggregation with pairwise + self masks and dropout recovery.

Protocol (semi-honest variant of Bonawitz et al., CCS 2017, "Practical Secure
Aggregation for Privacy-Preserving Machine Learning"):

Setup (once)
  S0 AdvertiseKeys   each client u publishes two DH public keys:
                     c_u^PK (to encrypt share traffic) and s_u^PK (pairwise masks).
  S1 ShareMaskKey    u Shamir-shares its mask secret s_u^SK (t-of-n) and sends each
                     share to peer v encrypted under KA(c_u^SK, c_v^PK). The server
                     only routes ciphertexts.
Per aggregation round r
  R1 ShareSelfMask   u draws a fresh self-mask seed b_u^r, Shamir-shares it the same way.
  R2 MaskedInput     u sends y_u = x_u + PRG(b_u^r) + sum_{v != u} sgn(u,v) PRG(s_uv, r)
                     over Z_{2^64}, with s_uv = KA(s_u^SK, s_v^PK) and sgn = +1 if u < v
                     else -1. The round index is mixed into the PRG so masks are never
                     reused across rounds.
  R3 Unmask          for each survivor u the peers reveal their share of b_u^r; for each
                     client that dropped after S1 they reveal their share of s_u^SK. An
                     honest client never reveals both shares for the same u in a round
                     (this is what keeps a slow-but-alive client's input hidden).
  Server             sum_u y_u, minus reconstructed self masks, minus the pairwise masks
                     that survivors shared with dropped clients.

Guarantee (semi-honest, n clients, threshold t): the server's view — even
jointly with any set of fewer than t clients — reveals nothing about the inputs
of the other clients beyond their sum, provided at least t clients survive.
With the three hospital nodes we use t = 2: one node may drop out, and the
server colluding with one node learns the sum of the other two, never a single
node's update. Malicious (actively deviating) parties are out of scope; the
full paper adds signatures and consistency checks for that setting.
"""
from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass, field

import numpy as np

from .crypto import (
    CryptoError,
    DHKeyPair,
    Share,
    decode_fixed_point,
    dh_agree,
    dh_keygen,
    encode_fixed_point,
    open_sealed,
    prg_uint64,
    seal,
    shamir_reconstruct,
    shamir_split,
)


class SecAggError(Exception):
    """Protocol violation, too many dropouts or a refused unmasking request."""


def _pair_seed(secret: int, peer_public: int, low: int, high: int) -> bytes:
    return dh_agree(secret, peer_public, f"mask|{low}|{high}".encode())


def _pairwise_mask(seed: bytes, round_id: int, length: int) -> np.ndarray:
    return prg_uint64(seed, f"pairwise|round={round_id}".encode(), length)


def _self_mask(seed: int, round_id: int, length: int) -> np.ndarray:
    return prg_uint64(seed.to_bytes(32, "big"), f"self|round={round_id}".encode(), length)


@dataclass
class SecAggClient:
    """Client-side protocol state. Holds its own secrets and peers' shares."""

    client_id: int
    threshold: int
    rng: random.Random | None = None  # None -> OS CSPRNG. Seeded RNG only for tests.
    _enc_keys: DHKeyPair = field(init=False)
    _mask_keys: DHKeyPair = field(init=False)
    _peer_keys: dict[int, tuple[int, int]] = field(default_factory=dict, init=False)
    _mask_key_shares: dict[int, Share] = field(default_factory=dict, init=False)
    _self_mask_shares: dict[tuple[int, int], Share] = field(default_factory=dict, init=False)
    _self_mask_seed: dict[int, int] = field(default_factory=dict, init=False)
    _revealed: dict[tuple[int, int], str] = field(default_factory=dict, init=False)
    # DH-derived keys are fixed after setup; per-round freshness comes from the
    # round index mixed into the PRG, so caching them is safe and avoids
    # repeated 2048-bit exponentiations.
    _channel_cache: dict[int, bytes] = field(default_factory=dict, init=False)
    _pair_cache: dict[int, bytes] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._enc_keys = dh_keygen(self.rng)
        self._mask_keys = dh_keygen(self.rng)

    # -- S0 ---------------------------------------------------------------
    def advertise_keys(self) -> dict[str, int]:
        return {"c_pk": self._enc_keys.public, "s_pk": self._mask_keys.public}

    def receive_public_keys(self, keys: dict[int, dict[str, int]]) -> None:
        if len(keys) < self.threshold:
            raise SecAggError("fewer clients than the threshold")
        self._peer_keys = {cid: (k["c_pk"], k["s_pk"]) for cid, k in keys.items()}

    def _channel_key(self, peer: int) -> bytes:
        if peer not in self._channel_cache:
            low, high = sorted((self.client_id, peer))
            self._channel_cache[peer] = dh_agree(
                self._enc_keys.secret, self._peer_keys[peer][0], f"channel|{low}|{high}".encode()
            )
        return self._channel_cache[peer]

    def _pair_seed(self, peer: int) -> bytes:
        if peer not in self._pair_cache:
            low, high = sorted((self.client_id, peer))
            self._pair_cache[peer] = _pair_seed(self._mask_keys.secret, self._peer_keys[peer][1], low, high)
        return self._pair_cache[peer]

    def _encrypt_shares(self, kind: str, round_id: int, shares: list[Share]) -> dict[int, bytes]:
        participants = sorted(self._peer_keys)
        out: dict[int, bytes] = {}
        for peer, share in zip(participants, shares):
            payload = json.dumps({"from": self.client_id, "to": peer, "kind": kind, "round": round_id,
                                  "share": [share[0], str(share[1])]}).encode()
            if peer == self.client_id:
                self._store_share(kind, round_id, self.client_id, share)
                continue
            aad = f"{kind}|{round_id}|{self.client_id}->{peer}".encode()
            out[peer] = seal(self._channel_key(peer), payload, aad, self.rng)
        return out

    def _store_share(self, kind: str, round_id: int, owner: int, share: Share) -> None:
        if kind == "mask_key":
            self._mask_key_shares[owner] = share
        else:
            self._self_mask_shares[(round_id, owner)] = share

    def receive_shares(self, kind: str, round_id: int, ciphertexts: dict[int, bytes]) -> None:
        for sender, blob in ciphertexts.items():
            aad = f"{kind}|{round_id}|{sender}->{self.client_id}".encode()
            try:
                payload = json.loads(open_sealed(self._channel_key(sender), blob, aad))
            except CryptoError as exc:
                raise SecAggError(f"share from client {sender} failed authentication") from exc
            if payload["to"] != self.client_id or payload["from"] != sender or payload["round"] != round_id:
                raise SecAggError("share routed to the wrong recipient/round")
            self._store_share(kind, round_id, sender, (int(payload["share"][0]), int(payload["share"][1])))

    # -- S1 ---------------------------------------------------------------
    def share_mask_key(self) -> dict[int, bytes]:
        shares = shamir_split(self._mask_keys.secret, len(self._peer_keys), self.threshold, self.rng)
        return self._encrypt_shares("mask_key", 0, shares)

    # -- R1 ---------------------------------------------------------------
    def share_self_mask(self, round_id: int) -> dict[int, bytes]:
        seed = int.from_bytes(self.rng.randbytes(32) if self.rng else secrets.token_bytes(32), "big")
        self._self_mask_seed[round_id] = seed
        shares = shamir_split(seed, len(self._peer_keys), self.threshold, self.rng)
        return self._encrypt_shares("self_mask", round_id, shares)

    # -- R2 ---------------------------------------------------------------
    def masked_input(self, round_id: int, vector: np.ndarray, participants: list[int]) -> np.ndarray:
        if self.client_id not in participants:
            raise SecAggError("client is not a registered participant")
        encoded = encode_fixed_point(vector)
        masked = encoded + _self_mask(self._self_mask_seed[round_id], round_id, len(encoded))
        for peer in participants:
            if peer == self.client_id:
                continue
            mask = _pairwise_mask(self._pair_seed(peer), round_id, len(encoded))
            masked = masked + mask if self.client_id < peer else masked - mask
        return masked  # uint64 arithmetic wraps modulo 2^64

    # -- R3 ---------------------------------------------------------------
    def unmask_response(self, round_id: int, survivors: list[int], dropped: list[int]) -> dict[str, dict[int, Share]]:
        if set(survivors) & set(dropped):
            raise SecAggError("a client cannot be both a survivor and dropped")
        response: dict[str, dict[int, Share]] = {"self_mask": {}, "mask_key": {}}
        for owner in survivors:
            if self._revealed.get((round_id, owner)) == "mask_key":
                raise SecAggError(f"refusing: mask key of {owner} already revealed this round")
            self._revealed[(round_id, owner)] = "self_mask"
            response["self_mask"][owner] = self._self_mask_shares[(round_id, owner)]
        for owner in dropped:
            if self._revealed.get((round_id, owner)) == "self_mask":
                raise SecAggError(f"refusing: self-mask of {owner} already revealed this round")
            self._revealed[(round_id, owner)] = "mask_key"
            response["mask_key"][owner] = self._mask_key_shares[owner]
        return response


@dataclass
class SecAggServer:
    """Server-side unmasking. Never sees an individual unmasked input."""

    threshold: int
    public_keys: dict[int, dict[str, int]] = field(default_factory=dict)

    def unmask(
        self,
        round_id: int,
        masked_inputs: dict[int, np.ndarray],
        participants: list[int],
        responses: dict[int, dict[str, dict[int, Share]]],
    ) -> np.ndarray:
        survivors = sorted(masked_inputs)
        dropped = sorted(set(participants) - set(survivors))
        if len(survivors) < self.threshold or len(responses) < self.threshold:
            raise SecAggError("not enough surviving clients to unmask")
        length = len(next(iter(masked_inputs.values())))
        total = np.zeros(length, dtype=np.uint64)
        for vector in masked_inputs.values():
            total = total + vector
        for owner in survivors:
            shares = [r["self_mask"][owner] for r in responses.values() if owner in r["self_mask"]]
            if len(shares) < self.threshold:
                raise SecAggError(f"not enough self-mask shares for client {owner}")
            total = total - _self_mask(shamir_reconstruct(shares[: self.threshold]), round_id, length)
        for owner in dropped:
            shares = [r["mask_key"][owner] for r in responses.values() if owner in r["mask_key"]]
            if len(shares) < self.threshold:
                raise SecAggError(f"not enough mask-key shares for dropped client {owner}")
            secret = shamir_reconstruct(shares[: self.threshold])
            for survivor in survivors:
                low, high = sorted((owner, survivor))
                seed = _pair_seed(secret, self.public_keys[survivor]["s_pk"], low, high)
                mask = _pairwise_mask(seed, round_id, length)
                # survivor added +mask if survivor < owner, else -mask: undo it
                total = total - mask if survivor < owner else total + mask
        return decode_fixed_point(total)
