from pathlib import Path

import pytest

from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart


def manifest_for(parts: tuple[bytes, ...]) -> DeltaManifest:
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


def test_filesystem_publish_load_and_duplicate_are_idempotent(tmp_path: Path) -> None:
    store = FileSystemDeltaStore(tmp_path)
    parts = (b"first", b"second")
    manifest = manifest_for(parts)

    published = store.publish(manifest, parts)
    duplicate = store.publish(manifest, parts)

    assert duplicate == published
    assert store.load(manifest.run_id, manifest.transfer_id) == (manifest, parts)
    assert (published / "COMMITTED").is_file()


def test_filesystem_rejects_same_transfer_with_different_bytes(tmp_path: Path) -> None:
    store = FileSystemDeltaStore(tmp_path)
    manifest = manifest_for((b"first",))
    store.publish(manifest, (b"first",))

    with pytest.raises(ValueError, match="different|sha256"):
        store.publish(manifest, (b"other",))


def test_filesystem_load_rejects_corrupt_part(tmp_path: Path) -> None:
    store = FileSystemDeltaStore(tmp_path)
    manifest = manifest_for((b"first",))
    published = store.publish(manifest, (b"first",))
    (published / "part-000000.bin").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="size|sha256"):
        store.load(manifest.run_id, manifest.transfer_id)
