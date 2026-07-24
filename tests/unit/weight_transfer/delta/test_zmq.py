import time
from pathlib import Path

import pytest

from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart
from prime_rl.weight_transfer.delta.zmq import ZmqDeltaReceiver, ZmqDeltaSender


def make_manifest(parts: tuple[bytes, ...]) -> DeltaManifest:
    return DeltaManifest.create(
        run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
        transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        baseline_fingerprint="sha256:" + "1" * 64,
        tensor_table_hash="sha256:" + "2" * 64,
        base_version=0,
        target_version=1,
        parts=tuple(DeltaPart.from_bytes(seq=i, body=body, tensor_entries=1) for i, body in enumerate(parts)),
        target_evidence={},
    )


def test_zmq_delivers_identical_artifact_and_replay_is_idempotent(tmp_path: Path) -> None:
    parts = (b"one", b"two")
    manifest = make_manifest(parts)
    store = FileSystemDeltaStore(tmp_path / "committed")
    with ZmqDeltaReceiver(
        "tcp://127.0.0.1:0",
        store=store,
        spool_dir=tmp_path / "spool",
        secret=b"test-secret",
    ) as receiver:
        sender = ZmqDeltaSender(receiver.endpoint, secret=b"test-secret", timeout_ms=2_000)
        sender.send(manifest, parts)
        sender.send(manifest, parts)

    assert store.load(manifest.run_id, manifest.transfer_id) == (manifest, parts)


def test_zmq_rejects_wrong_hmac_secret(tmp_path: Path) -> None:
    parts = (b"one",)
    manifest = make_manifest(parts)
    with ZmqDeltaReceiver(
        "tcp://127.0.0.1:0",
        store=FileSystemDeltaStore(tmp_path / "committed"),
        spool_dir=tmp_path / "spool",
        secret=b"correct-secret",
    ) as receiver:
        sender = ZmqDeltaSender(receiver.endpoint, secret=b"wrong-secret", timeout_ms=500, attempts=1)
        with pytest.raises(RuntimeError, match="authentication"):
            sender.send(manifest, parts)


def test_zmq_unreachable_receiver_has_bounded_send_timeout() -> None:
    manifest = make_manifest((b"one",))
    sender = ZmqDeltaSender(
        "tcp://127.0.0.1:1",
        secret=b"test-secret",
        timeout_ms=50,
        attempts=1,
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="while sending"):
        sender.send(manifest, (b"one",))

    assert time.monotonic() - started < 2
