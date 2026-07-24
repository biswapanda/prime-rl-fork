import os
import uuid
from pathlib import Path

import boto3
import torch
from botocore.config import Config
from safetensors.torch import save_file

from prime_rl.weight_transfer.delta.checkpoint import (
    apply_parts_to_shadow_checkpoint,
    build_checkpoint_delta,
    checkpoint_fingerprint,
)
from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.s3 import S3DeltaStore
from prime_rl.weight_transfer.delta.zmq import ZmqDeltaReceiver, ZmqDeltaSender


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text('{"model_type":"qwen3"}\n')


def test_identical_artifact_reaches_identical_checkpoint_through_all_transports(tmp_path: Path) -> None:
    endpoint = os.environ["DELTA_S3_ENDPOINT"]
    bucket = os.environ["DELTA_S3_BUCKET"]
    base = tmp_path / "w0"
    target = tmp_path / "w1"
    base_tensors = {
        "model.embed_tokens.weight": torch.arange(4096, dtype=torch.bfloat16).reshape(64, 64),
        "model.layers.0.weight": torch.arange(1024, dtype=torch.float32).reshape(32, 32),
    }
    target_tensors = {name: value.clone() for name, value in base_tensors.items()}
    target_tensors["model.embed_tokens.weight"][2, 3] = -4
    target_tensors["model.embed_tokens.weight"][7, 9] = 11
    target_tensors["model.layers.0.weight"][20:24, 4] = -8
    write_checkpoint(base, base_tensors)
    write_checkpoint(target, target_tensors)

    manifest, parts = build_checkpoint_delta(
        base,
        target,
        run_id=str(uuid.uuid4()),
        transfer_id=str(uuid.uuid4()),
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        base_version=0,
        target_version=1,
    )

    filesystem = FileSystemDeltaStore(tmp_path / "filesystem-artifacts")
    filesystem.publish(manifest, parts)
    fs_manifest, fs_parts = filesystem.load(manifest.run_id, manifest.transfer_id)

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    s3 = S3DeltaStore(s3_client, bucket=bucket, prefix=f"cross-transport/{uuid.uuid4()}")
    s3.publish(manifest, parts)
    s3_manifest, s3_parts = s3.load(manifest.run_id, manifest.transfer_id)

    zmq_store = FileSystemDeltaStore(tmp_path / "zmq-artifacts")
    with ZmqDeltaReceiver(
        "tcp://127.0.0.1:0",
        store=zmq_store,
        spool_dir=tmp_path / "zmq-spool",
        secret=b"localhost-integration-only",
    ) as receiver:
        ZmqDeltaSender(receiver.endpoint, secret=b"localhost-integration-only").send(manifest, parts)
    zmq_manifest, zmq_parts = zmq_store.load(manifest.run_id, manifest.transfer_id)

    assert fs_manifest.manifest_hash == s3_manifest.manifest_hash == zmq_manifest.manifest_hash
    assert fs_parts == s3_parts == zmq_parts == parts

    expected = checkpoint_fingerprint(target)
    for name, carrier_manifest, carrier_parts in (
        ("filesystem", fs_manifest, fs_parts),
        ("s3", s3_manifest, s3_parts),
        ("zmq", zmq_manifest, zmq_parts),
    ):
        shadow = tmp_path / f"w1-{name}"
        apply_parts_to_shadow_checkpoint(base, shadow, carrier_manifest, carrier_parts)
        assert checkpoint_fingerprint(shadow) == expected
