import json
from dataclasses import replace

import pytest

from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart


def part(seq: int, body: bytes) -> DeltaPart:
    return DeltaPart.from_bytes(seq=seq, body=body, tensor_entries=1)


def test_manifest_is_canonical_and_round_trips() -> None:
    manifest = DeltaManifest.create(
        run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
        transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        baseline_fingerprint="sha256:" + "1" * 64,
        tensor_table_hash="sha256:" + "2" * 64,
        base_version=0,
        target_version=1,
        parts=(part(0, b"part-zero"), part(1, b"part-one")),
        target_evidence={"model.layers.0.weight": "sha256:" + "3" * 64},
    )

    encoded = manifest.to_bytes()

    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")).encode()
    assert DeltaManifest.from_bytes(encoded) == manifest
    assert manifest.manifest_hash.startswith("sha256:")


@pytest.mark.parametrize(
    ("base_version", "target_version", "message"),
    [
        (1, 1, "target_version"),
        (2, 4, "target_version"),
        (-1, 0, "base_version"),
    ],
)
def test_manifest_rejects_invalid_version_chain(base_version: int, target_version: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeltaManifest.create(
            run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
            transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
            model_id="Qwen/Qwen3-0.6B",
            checkpoint_revision="0123456789abcdef",
            baseline_fingerprint="sha256:" + "1" * 64,
            tensor_table_hash="sha256:" + "2" * 64,
            base_version=base_version,
            target_version=target_version,
            parts=(part(0, b"part-zero"),),
            target_evidence={},
        )


def test_manifest_rejects_non_contiguous_part_sequence() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        DeltaManifest.create(
            run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
            transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
            model_id="Qwen/Qwen3-0.6B",
            checkpoint_revision="0123456789abcdef",
            baseline_fingerprint="sha256:" + "1" * 64,
            tensor_table_hash="sha256:" + "2" * 64,
            base_version=0,
            target_version=1,
            parts=(part(0, b"part-zero"), part(2, b"part-two")),
            target_evidence={},
        )


def test_manifest_detects_part_descriptor_tampering() -> None:
    manifest = DeltaManifest.create(
        run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
        transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        baseline_fingerprint="sha256:" + "1" * 64,
        tensor_table_hash="sha256:" + "2" * 64,
        base_version=0,
        target_version=1,
        parts=(part(0, b"part-zero"),),
        target_evidence={},
    )
    tampered = replace(manifest.parts[0], size=manifest.parts[0].size + 1)

    with pytest.raises(ValueError, match="descriptor|size"):
        manifest.verify_part(tampered, b"part-zero")
