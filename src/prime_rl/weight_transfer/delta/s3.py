from __future__ import annotations

import hashlib
from typing import Any

from prime_rl.weight_transfer.delta.protocol import DeltaManifest


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    details = response.get("Error")
    if not isinstance(details, dict):
        return None
    code = details.get("Code")
    return str(code) if code is not None else None


def _is_missing(error: BaseException) -> bool:
    return _error_code(error) in {"404", "NoSuchKey", "NotFound"}


class S3DeltaStore:
    def __init__(self, client: Any, *, bucket: str, prefix: str = ""):
        if not bucket:
            raise ValueError("S3 bucket must be non-empty")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    @classmethod
    def from_endpoint(
        cls,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        prefix: str = "",
    ) -> S3DeltaStore:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("S3 delta transfer requires the prime-rl[delta-s3] extra") from error
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"mode": "standard", "max_attempts": 5},
        )
        client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name, config=config)
        return cls(client, bucket=bucket, prefix=prefix)

    def _transfer_prefix(self, run_id: str, transfer_id: str) -> str:
        identity = f"{run_id}/{transfer_id}"
        return f"{self.prefix}/{identity}" if self.prefix else identity

    def _matching_existing(self, key: str, body: bytes, digest: str) -> bool:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except BaseException as error:
            if _is_missing(error):
                return False
            raise
        metadata = response.get("Metadata") or {}
        return (
            response.get("ContentLength") == len(body)
            and metadata.get("sha256") == digest
            and metadata.get("size") == str(len(body))
        )

    def _put_immutable(self, key: str, body: bytes) -> None:
        digest = hashlib.sha256(body).hexdigest()
        metadata = {"sha256": digest, "size": str(len(body))}
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                Metadata=metadata,
                IfNoneMatch="*",
            )
        except BaseException as error:
            if self._matching_existing(key, body, digest):
                return
            code = _error_code(error)
            if code in {"412", "PreconditionFailed"}:
                raise ValueError(f"S3 immutable object conflict: {key}") from error
            raise

    def publish(self, manifest: DeltaManifest, parts: tuple[bytes, ...]) -> str:
        manifest.validate()
        if len(parts) != len(manifest.parts):
            raise ValueError("part count does not match manifest")
        transfer_prefix = self._transfer_prefix(manifest.run_id, manifest.transfer_id)
        for descriptor, body in zip(manifest.parts, parts, strict=True):
            manifest.verify_part(descriptor, body)
            key = f"{transfer_prefix}/parts/{descriptor.seq:06d}-{descriptor.sha256}.bin"
            self._put_immutable(key, body)
        self._put_immutable(f"{transfer_prefix}/manifest.json", manifest.to_bytes())
        self._put_immutable(f"{transfer_prefix}/COMMITTED", manifest.manifest_hash.encode() + b"\n")
        return f"s3://{self.bucket}/{transfer_prefix}"

    def _get(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read()
        metadata = response.get("Metadata") or {}
        digest = hashlib.sha256(body).hexdigest()
        if response.get("ContentLength") != len(body):
            raise ValueError(f"S3 object length mismatch: {key}")
        if metadata.get("sha256") != digest or metadata.get("size") != str(len(body)):
            raise ValueError(f"S3 object checksum metadata mismatch: {key}")
        return body

    def load(self, run_id: str, transfer_id: str) -> tuple[DeltaManifest, tuple[bytes, ...]]:
        transfer_prefix = self._transfer_prefix(run_id, transfer_id)
        committed = self._get(f"{transfer_prefix}/COMMITTED").decode().strip()
        manifest_body = self._get(f"{transfer_prefix}/manifest.json")
        manifest = DeltaManifest.from_bytes(manifest_body)
        if manifest.run_id != run_id or manifest.transfer_id != transfer_id:
            raise ValueError("manifest identity does not match S3 key")
        if committed != manifest.manifest_hash:
            raise ValueError("S3 commit marker does not match manifest hash")
        parts: list[bytes] = []
        for descriptor in manifest.parts:
            key = f"{transfer_prefix}/parts/{descriptor.seq:06d}-{descriptor.sha256}.bin"
            body = self._get(key)
            manifest.verify_part(descriptor, body)
            parts.append(body)
        return manifest, tuple(parts)
