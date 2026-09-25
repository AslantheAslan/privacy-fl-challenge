"""Message transport between the FL server and hospital clients.

Every request/response crosses the client boundary as *bytes* produced by
``encode_message``: a JSON document in which NumPy arrays, bytes and big
integers are explicitly tagged. There is no pickle anywhere on this path, so a
client cannot be tricked into executing code by a malicious server (and vice
versa), and the ``TransportLog`` records exactly which fields — with shapes,
dtypes and byte counts — left each node. That log is what the experiment
summary reports as "what leaves each client".

Two endpoint implementations share one interface:

* ``InProcessEndpoint`` – the client object lives in this process but is only
  reachable through encode/decode (fast; used for experiments and tests);
* ``ProcessEndpoint`` – the client runs in a separate OS process that loads
  *only its own site's rows* from the training file (strongest isolation; used
  by ``run_submission.py`` to train the submitted model).
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Safe serialisation
# ---------------------------------------------------------------------------
def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__nd__": base64.b64encode(np.ascontiguousarray(value).tobytes()).decode(),
                "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode()}
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        value = int(value)
        return value if abs(value) < 2**53 else {"__int__": str(value)}
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    raise TypeError(f"type {type(value).__name__} is not allowed on the wire")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if "__nd__" in value:
            return np.frombuffer(base64.b64decode(value["__nd__"]), dtype=np.dtype(value["dtype"])).reshape(
                value["shape"]).copy()
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        if "__int__" in value:
            return int(value["__int__"])
        if "__dict__" in value:
            return {_hashable(_decode(k)): _decode(v) for k, v in value["__dict__"]}
        raise ValueError("unknown tagged object")
    return value


def _hashable(key: Any) -> Any:
    return tuple(key) if isinstance(key, list) else key


def encode_message(payload: dict[str, Any]) -> bytes:
    return json.dumps(_encode(payload), separators=(",", ":")).encode()


def decode_message(blob: bytes) -> dict[str, Any]:
    return _decode(json.loads(blob.decode()))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def _describe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array": list(value.shape), "dtype": value.dtype.str}
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value[:3]] + (["..."] if len(value) > 3 else [])
    if isinstance(value, bytes):
        return f"<{len(value)} bytes ciphertext>"
    return type(value).__name__


@dataclass
class TransportLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, direction: str, client: str, msg_type: str, payload: dict[str, Any], size: int) -> None:
        self.entries.append({"direction": direction, "client": client, "type": msg_type, "bytes": size,
                             "fields": _describe(payload)})

    def bytes_by(self, direction: str) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            if entry["direction"] == direction:
                totals[entry["client"]] = totals.get(entry["client"], 0) + entry["bytes"]
        return totals

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, dict[str, Any]] = {}
        for entry in self.entries:
            key = f"{entry['direction']}:{entry['type']}"
            item = by_type.setdefault(key, {"messages": 0, "bytes": 0, "example_fields": entry["fields"]})
            item["messages"] += 1
            item["bytes"] += entry["bytes"]
        return {
            "client_to_server_bytes": self.bytes_by("client->server"),
            "server_to_client_bytes": self.bytes_by("server->client"),
            "by_message_type": by_type,
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
class Endpoint(Protocol):
    name: str

    def request(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class InProcessEndpoint:
    """Client reachable only through serialised messages."""

    def __init__(self, name: str, handler: Callable[[str, dict[str, Any]], dict[str, Any]], log: TransportLog):
        self.name = name
        self._handler = handler
        self._log = log

    def request(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        outbound = encode_message({"type": msg_type, "payload": payload})
        self._log.record("server->client", self.name, msg_type, payload, len(outbound))
        message = decode_message(outbound)
        reply = encode_message(self._handler(message["type"], message["payload"]))
        decoded = decode_message(reply)
        self._log.record("client->server", self.name, msg_type, decoded, len(reply))
        return decoded

    def close(self) -> None:
        return None


def _client_process_main(conn: Any, factory: Callable[[], Any]) -> None:
    client = factory()
    while True:
        message = decode_message(conn.recv_bytes())
        if message["type"] == "__shutdown__":
            conn.close()
            return
        try:
            reply = {"ok": True, "payload": client.handle(message["type"], message["payload"])}
        except Exception as exc:  # report, never hang the server
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        conn.send_bytes(encode_message(reply))


class ProcessEndpoint:
    """Client running in its own OS process; the server holds only a pipe."""

    def __init__(self, name: str, factory: Callable[[], Any], log: TransportLog):
        self.name = name
        self._log = log
        context = mp.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(target=_client_process_main, args=(child, factory), daemon=True)
        self._process.start()
        child.close()

    def request(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        outbound = encode_message({"type": msg_type, "payload": payload})
        self._log.record("server->client", self.name, msg_type, payload, len(outbound))
        self._conn.send_bytes(outbound)
        raw = self._conn.recv_bytes()
        reply = decode_message(raw)
        if not reply["ok"]:
            raise RuntimeError(f"client {self.name} failed: {reply['error']}")
        self._log.record("client->server", self.name, msg_type, reply["payload"], len(raw))
        return reply["payload"]

    def close(self) -> None:
        try:
            self._conn.send_bytes(encode_message({"type": "__shutdown__", "payload": {}}))
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
