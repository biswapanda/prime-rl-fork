from io import BytesIO

import pytest

from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart
from prime_rl.weight_transfer.delta.s3 import S3DeltaStore


class PreconditionFailed(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class NotFound(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.put_order: list[str] = []

    def put_object(self, *, Bucket, Key, Body, Metadata, IfNoneMatch):
        assert IfNoneMatch == "*"
        identity = (Bucket, Key)
        if identity in self.objects:
            raise PreconditionFailed
        self.objects[identity] = (bytes(Body), Metadata)
        self.put_order.append(Key)

    def head_object(self, *, Bucket, Key):
        try:
            body, metadata = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise NotFound from error
        return {"ContentLength": len(body), "Metadata": metadata}

    def get_object(self, *, Bucket, Key):
        try:
            body, metadata = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise NotFound from error
        return {"Body": BytesIO(body), "ContentLength": len(body), "Metadata": metadata}


def make_manifest(body: bytes = b"part") -> DeltaManifest:
    return DeltaManifest.create(
        run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
        transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        baseline_fingerprint="sha256:" + "1" * 64,
        tensor_table_hash="sha256:" + "2" * 64,
        base_version=0,
        target_version=1,
        parts=(DeltaPart.from_bytes(seq=0, body=body, tensor_entries=1),),
        target_evidence={},
    )


def test_s3_publish_is_manifest_and_commit_last_and_idempotent() -> None:
    client = FakeS3()
    store = S3DeltaStore(client, bucket="delta", prefix="prime")
    manifest = make_manifest()

    store.publish(manifest, (b"part",))
    store.publish(manifest, (b"part",))

    assert client.put_order[0].endswith("/parts/000000-" + manifest.parts[0].sha256 + ".bin")
    assert client.put_order[1].endswith("/manifest.json")
    assert client.put_order[2].endswith("/COMMITTED")
    assert store.load(manifest.run_id, manifest.transfer_id) == (manifest, (b"part",))


def test_s3_rejects_conflicting_existing_object() -> None:
    client = FakeS3()
    store = S3DeltaStore(client, bucket="delta", prefix="prime")
    manifest = make_manifest()
    store.publish(manifest, (b"part",))
    part_key = client.put_order[0]
    client.objects[("delta", part_key)] = (b"other", {"sha256": "0" * 64, "size": "5"})

    with pytest.raises(ValueError, match="conflict"):
        store.publish(manifest, (b"part",))
