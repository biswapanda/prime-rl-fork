from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import uuid
from pathlib import Path

import zmq

from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.protocol import DeltaManifest, canonical_json

PROTOCOL = b"prime.delta.v1"
DEFAULT_MAX_MESSAGE_BYTES = 256 * 1024 * 1024 + 16 * 1024 * 1024


def _mac(secret: bytes, frames: tuple[bytes, ...]) -> bytes:
    digest = hmac.new(secret, digestmod=hashlib.sha256)
    for frame in frames:
        digest.update(len(frame).to_bytes(8, "big"))
        digest.update(frame)
    return digest.digest()


def _write_atomic(path: Path, body: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
    with temporary.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ZmqDeltaReceiver:
    def __init__(
        self,
        bind_endpoint: str,
        *,
        store: FileSystemDeltaStore,
        spool_dir: Path,
        secret: bytes,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ):
        if not secret:
            raise ValueError("ZeroMQ delta receiver requires an HMAC secret")
        self.bind_endpoint = bind_endpoint
        self.store = store
        self.spool_dir = spool_dir
        self.secret = secret
        self.max_message_bytes = max_message_bytes
        self.endpoint = ""
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    def __enter__(self) -> ZmqDeltaReceiver:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ZeroMQ delta receiver is already running")
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, name="prime-delta-zmq", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise TimeoutError("ZeroMQ delta receiver did not bind")
        if self._startup_error is not None:
            raise RuntimeError("ZeroMQ delta receiver failed to start") from self._startup_error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("ZeroMQ delta receiver did not stop")
            self._thread = None

    def _reply(self, socket: zmq.Socket, identity: bytes, status: bytes, metadata: bytes) -> None:
        frames = (PROTOCOL, status, metadata)
        socket.send_multipart([identity, *frames, _mac(self.secret, frames)])

    def _transaction_dir(self, run_id: str, transfer_id: str) -> Path:
        uuid.UUID(run_id)
        uuid.UUID(transfer_id)
        return self.spool_dir / run_id / transfer_id

    def _handle(self, command: bytes, metadata: bytes, body: bytes) -> bytes:
        import json

        try:
            details = json.loads(metadata)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("metadata is not valid JSON") from error
        if not isinstance(details, dict):
            raise ValueError("metadata must be an object")

        if command == b"HELLO":
            if set(details) != {"message_id", "run_id", "transfer_id"}:
                raise ValueError("HELLO metadata is invalid")
            self._transaction_dir(details["run_id"], details["transfer_id"])
            return canonical_json({"message_id": details["message_id"], "state": "ready"})

        if command == b"MANIFEST":
            manifest = DeltaManifest.from_bytes(body)
            if details != {
                "message_id": details.get("message_id"),
                "run_id": manifest.run_id,
                "transfer_id": manifest.transfer_id,
                "manifest_hash": manifest.manifest_hash,
            }:
                raise ValueError("MANIFEST metadata is invalid")
            transaction = self._transaction_dir(manifest.run_id, manifest.transfer_id)
            transaction.mkdir(parents=True, exist_ok=True)
            path = transaction / "manifest.json"
            if path.exists() and path.read_bytes() != body:
                raise ValueError("manifest conflicts with staged transfer")
            if not path.exists():
                _write_atomic(path, body)
            return canonical_json({"message_id": details["message_id"], "state": "manifest"})

        run_id = details.get("run_id")
        transfer_id = details.get("transfer_id")
        transaction = self._transaction_dir(run_id, transfer_id)
        manifest = DeltaManifest.from_bytes((transaction / "manifest.json").read_bytes())

        if command == b"PART":
            if set(details) != {"message_id", "run_id", "transfer_id", "seq", "sha256"}:
                raise ValueError("PART metadata is invalid")
            seq = details["seq"]
            if not isinstance(seq, int) or seq < 0 or seq >= len(manifest.parts):
                raise ValueError("PART sequence is invalid")
            descriptor = manifest.parts[seq]
            if details["sha256"] != descriptor.sha256:
                raise ValueError("PART metadata checksum is invalid")
            manifest.verify_part(descriptor, body)
            path = transaction / f"part-{seq:06d}.bin"
            if path.exists() and path.read_bytes() != body:
                raise ValueError("part conflicts with staged transfer")
            if not path.exists():
                _write_atomic(path, body)
            return canonical_json({"message_id": details["message_id"], "state": "part", "seq": seq})

        if command == b"SEAL":
            if set(details) != {"message_id", "run_id", "transfer_id", "manifest_hash"}:
                raise ValueError("SEAL metadata is invalid")
            if details["manifest_hash"] != manifest.manifest_hash:
                raise ValueError("SEAL manifest hash is invalid")
            parts = tuple((transaction / f"part-{part.seq:06d}.bin").read_bytes() for part in manifest.parts)
            self.store.publish(manifest, parts)
            return canonical_json({"message_id": details["message_id"], "state": "staged"})

        raise ValueError("unknown ZeroMQ delta command")

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 4)
        socket.setsockopt(zmq.SNDHWM, 4)
        socket.setsockopt(zmq.MAXMSGSIZE, self.max_message_bytes)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        try:
            socket.bind(self.bind_endpoint)
            self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            socket.close()
            context.term()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    received = socket.recv_multipart()
                except zmq.Again:
                    continue
                if len(received) != 6:
                    continue
                identity, protocol, command, metadata, body, received_mac = received
                frames = (protocol, command, metadata, body)
                if protocol != PROTOCOL or not hmac.compare_digest(received_mac, _mac(self.secret, frames)):
                    reply = canonical_json({"error": "authentication failed", "message_id": ""})
                    self._reply(socket, identity, b"NACK", reply)
                    continue
                try:
                    reply = self._handle(command, metadata, body)
                    self._reply(socket, identity, b"ACK", reply)
                except BaseException as error:
                    message_id = ""
                    try:
                        import json

                        parsed = json.loads(metadata)
                        message_id = parsed.get("message_id", "") if isinstance(parsed, dict) else ""
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    reply = canonical_json({"error": str(error), "message_id": message_id})
                    self._reply(socket, identity, b"NACK", reply)
        finally:
            socket.close()
            context.term()


class ZmqDeltaSender:
    def __init__(
        self,
        endpoint: str,
        *,
        secret: bytes,
        timeout_ms: int = 5_000,
        attempts: int = 3,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ):
        if not secret:
            raise ValueError("ZeroMQ delta sender requires an HMAC secret")
        if timeout_ms <= 0 or attempts <= 0:
            raise ValueError("ZeroMQ timeout and attempts must be positive")
        self.endpoint = endpoint
        self.secret = secret
        self.timeout_ms = timeout_ms
        self.attempts = attempts
        self.max_message_bytes = max_message_bytes

    def _request(self, socket: zmq.Socket, command: bytes, details: dict[str, object], body: bytes = b"") -> dict:
        import json

        message_id = str(uuid.uuid4())
        metadata = canonical_json({**details, "message_id": message_id})
        frames = (PROTOCOL, command, metadata, body)
        for attempt in range(self.attempts):
            try:
                socket.send_multipart([*frames, _mac(self.secret, frames)])
            except zmq.Again:
                if attempt + 1 < self.attempts:
                    time.sleep(min(0.1 * 2**attempt, 1.0))
                    continue
                raise TimeoutError(f"ZeroMQ delta receiver timed out while sending {command.decode()}") from None
            if socket.poll(self.timeout_ms, zmq.POLLIN):
                reply = socket.recv_multipart()
                if len(reply) != 4:
                    raise RuntimeError("invalid ZeroMQ delta reply")
                protocol, status, reply_metadata, received_mac = reply
                reply_frames = (protocol, status, reply_metadata)
                if protocol != PROTOCOL or not hmac.compare_digest(received_mac, _mac(self.secret, reply_frames)):
                    raise RuntimeError("ZeroMQ delta reply authentication failed")
                parsed = json.loads(reply_metadata)
                if not isinstance(parsed, dict) or parsed.get("message_id") not in ("", message_id):
                    continue
                if status == b"NACK":
                    raise RuntimeError(
                        f"ZeroMQ delta receiver rejected request: {parsed.get('error', 'unknown error')}"
                    )
                if status != b"ACK":
                    raise RuntimeError("invalid ZeroMQ delta reply status")
                return parsed
            if attempt + 1 < self.attempts:
                time.sleep(min(0.1 * 2**attempt, 1.0))
        raise TimeoutError(f"ZeroMQ delta receiver timed out for {command.decode()}")

    def send(self, manifest: DeltaManifest, parts: tuple[bytes, ...]) -> None:
        manifest.validate()
        if len(parts) != len(manifest.parts):
            raise ValueError("part count does not match manifest")
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.MAXMSGSIZE, self.max_message_bytes)
        socket.setsockopt(zmq.IDENTITY, f"prime-delta-{uuid.uuid4()}".encode())
        socket.connect(self.endpoint)
        identity = {"run_id": manifest.run_id, "transfer_id": manifest.transfer_id}
        try:
            self._request(socket, b"HELLO", identity)
            self._request(
                socket, b"MANIFEST", {**identity, "manifest_hash": manifest.manifest_hash}, manifest.to_bytes()
            )
            for descriptor, body in zip(manifest.parts, parts, strict=True):
                manifest.verify_part(descriptor, body)
                self._request(
                    socket,
                    b"PART",
                    {**identity, "seq": descriptor.seq, "sha256": descriptor.sha256},
                    body,
                )
            self._request(socket, b"SEAL", {**identity, "manifest_hash": manifest.manifest_hash})
        finally:
            socket.close()
            context.term()
