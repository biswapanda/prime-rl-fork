import os
import uuid

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart
from prime_rl.weight_transfer.delta.s3 import S3DeltaStore


def test_rustfs_round_trip_and_duplicate_publish() -> None:
    endpoint = os.environ.get("DELTA_S3_ENDPOINT")
    bucket = os.environ.get("DELTA_S3_BUCKET")
    if not endpoint or not bucket:
        pytest.skip("DELTA_S3_ENDPOINT and DELTA_S3_BUCKET are required")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client.create_bucket(Bucket=bucket)

    body = b"rustfs-delta-integration"
    manifest = DeltaManifest.create(
        run_id=str(uuid.uuid4()),
        transfer_id=str(uuid.uuid4()),
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        baseline_fingerprint="sha256:" + "1" * 64,
        tensor_table_hash="sha256:" + "2" * 64,
        base_version=0,
        target_version=1,
        parts=(DeltaPart.from_bytes(seq=0, body=body, tensor_entries=1),),
        target_evidence={},
    )
    store = S3DeltaStore(client, bucket=bucket, prefix=f"integration/{uuid.uuid4()}")

    store.publish(manifest, (body,))
    store.publish(manifest, (body,))

    assert store.load(manifest.run_id, manifest.transfer_id) == (manifest, (body,))
