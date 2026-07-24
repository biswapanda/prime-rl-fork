from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

SCHEMA = "prime.delta.v1"
ENCODING = "absolute-values"
COMPRESSION = "none"


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256_ref(body: bytes) -> str:
    return f"sha256:{_sha256(body)}"


def _validate_uuid(value: str, field: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    if str(parsed) != value.lower():
        raise ValueError(f"{field} must use canonical UUID form")


def _validate_hash_ref(value: str, field: str) -> None:
    prefix, separator, digest = value.partition(":")
    if separator != ":" or prefix != "sha256" or len(digest) != 64:
        raise ValueError(f"{field} must be a sha256 reference")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a sha256 reference") from error


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@dataclass(frozen=True, slots=True)
class DeltaPart:
    seq: int
    id: str
    size: int
    sha256: str
    tensor_entries: int

    @classmethod
    def from_bytes(cls, *, seq: int, body: bytes, tensor_entries: int) -> DeltaPart:
        if seq < 0:
            raise ValueError("part seq must be non-negative")
        if tensor_entries < 0:
            raise ValueError("tensor_entries must be non-negative")
        digest = _sha256(body)
        return cls(
            seq=seq,
            id=f"sha256:{digest}",
            size=len(body),
            sha256=digest,
            tensor_entries=tensor_entries,
        )

    @classmethod
    def from_dict(cls, value: object) -> DeltaPart:
        if not isinstance(value, dict):
            raise ValueError("part descriptor must be an object")
        expected = {"seq", "id", "size", "sha256", "tensor_entries"}
        if set(value) != expected:
            raise ValueError("part descriptor has unknown or missing fields")
        part = cls(
            seq=value["seq"],
            id=value["id"],
            size=value["size"],
            sha256=value["sha256"],
            tensor_entries=value["tensor_entries"],
        )
        part.validate()
        return part

    def validate(self) -> None:
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 0:
            raise ValueError("part seq must be a non-negative integer")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("part size must be a non-negative integer")
        if not isinstance(self.tensor_entries, int) or isinstance(self.tensor_entries, bool) or self.tensor_entries < 0:
            raise ValueError("tensor_entries must be a non-negative integer")
        _validate_hash_ref(self.id, "part id")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("part sha256 must be a hex digest")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("part sha256 must be a hex digest") from error
        if self.id != f"sha256:{self.sha256}":
            raise ValueError("part id and sha256 must match")

    def to_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "id": self.id,
            "size": self.size,
            "sha256": self.sha256,
            "tensor_entries": self.tensor_entries,
        }


@dataclass(frozen=True, slots=True)
class DeltaManifest:
    schema: str
    run_id: str
    transfer_id: str
    model_id: str
    checkpoint_revision: str
    baseline_fingerprint: str
    tensor_table_hash: str
    base_version: int
    target_version: int
    kind: str
    encoding: str
    compression: str
    parts: tuple[DeltaPart, ...]
    target_evidence: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        transfer_id: str,
        model_id: str,
        checkpoint_revision: str,
        baseline_fingerprint: str,
        tensor_table_hash: str,
        base_version: int,
        target_version: int,
        parts: tuple[DeltaPart, ...],
        target_evidence: dict[str, str],
    ) -> DeltaManifest:
        manifest = cls(
            schema=SCHEMA,
            run_id=run_id,
            transfer_id=transfer_id,
            model_id=model_id,
            checkpoint_revision=checkpoint_revision,
            baseline_fingerprint=baseline_fingerprint,
            tensor_table_hash=tensor_table_hash,
            base_version=base_version,
            target_version=target_version,
            kind="patch",
            encoding=ENCODING,
            compression=COMPRESSION,
            parts=parts,
            target_evidence=dict(sorted(target_evidence.items())),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_bytes(cls, body: bytes) -> DeltaManifest:
        try:
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("manifest is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        expected = {
            "schema",
            "run_id",
            "transfer_id",
            "model",
            "base_version",
            "target_version",
            "kind",
            "encoding",
            "compression",
            "parts",
            "target_evidence",
        }
        if set(value) != expected:
            raise ValueError("manifest has unknown or missing fields")
        model = value["model"]
        if not isinstance(model, dict) or set(model) != {
            "id",
            "checkpoint_revision",
            "baseline_fingerprint",
            "tensor_table_hash",
        }:
            raise ValueError("manifest model descriptor is invalid")
        parts = value["parts"]
        if not isinstance(parts, list):
            raise ValueError("manifest parts must be a list")
        target_evidence = value["target_evidence"]
        if not isinstance(target_evidence, dict) or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in target_evidence.items()
        ):
            raise ValueError("target_evidence must map tensor names to hashes")
        manifest = cls(
            schema=value["schema"],
            run_id=value["run_id"],
            transfer_id=value["transfer_id"],
            model_id=model["id"],
            checkpoint_revision=model["checkpoint_revision"],
            baseline_fingerprint=model["baseline_fingerprint"],
            tensor_table_hash=model["tensor_table_hash"],
            base_version=value["base_version"],
            target_version=value["target_version"],
            kind=value["kind"],
            encoding=value["encoding"],
            compression=value["compression"],
            parts=tuple(DeltaPart.from_dict(part) for part in parts),
            target_evidence=dict(sorted(target_evidence.items())),
        )
        manifest.validate()
        if manifest.to_bytes() != body:
            raise ValueError("manifest JSON must use canonical serialization")
        return manifest

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        _validate_uuid(self.run_id, "run_id")
        _validate_uuid(self.transfer_id, "transfer_id")
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValueError("model_id must be non-empty")
        if not self.checkpoint_revision or not isinstance(self.checkpoint_revision, str):
            raise ValueError("checkpoint_revision must be non-empty")
        _validate_hash_ref(self.baseline_fingerprint, "baseline_fingerprint")
        _validate_hash_ref(self.tensor_table_hash, "tensor_table_hash")
        if not isinstance(self.base_version, int) or isinstance(self.base_version, bool) or self.base_version < 0:
            raise ValueError("base_version must be a non-negative integer")
        if self.target_version != self.base_version + 1:
            raise ValueError("target_version must equal base_version + 1")
        if self.kind != "patch":
            raise ValueError("kind must be patch")
        if self.encoding != ENCODING:
            raise ValueError(f"encoding must be {ENCODING}")
        if self.compression != COMPRESSION:
            raise ValueError(f"compression must be {COMPRESSION}")
        for expected_seq, part in enumerate(self.parts):
            part.validate()
            if part.seq != expected_seq:
                raise ValueError("part sequences must be contiguous from zero")
        if len({part.id for part in self.parts}) != len(self.parts):
            raise ValueError("part IDs must be unique")
        for name, digest in self.target_evidence.items():
            if not name:
                raise ValueError("target evidence tensor name must be non-empty")
            _validate_hash_ref(digest, f"target_evidence[{name}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "transfer_id": self.transfer_id,
            "model": {
                "id": self.model_id,
                "checkpoint_revision": self.checkpoint_revision,
                "baseline_fingerprint": self.baseline_fingerprint,
                "tensor_table_hash": self.tensor_table_hash,
            },
            "base_version": self.base_version,
            "target_version": self.target_version,
            "kind": self.kind,
            "encoding": self.encoding,
            "compression": self.compression,
            "parts": [part.to_dict() for part in self.parts],
            "target_evidence": dict(sorted(self.target_evidence.items())),
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def manifest_hash(self) -> str:
        return _sha256_ref(self.to_bytes())

    def verify_part(self, descriptor: DeltaPart, body: bytes) -> None:
        expected = self.parts[descriptor.seq] if 0 <= descriptor.seq < len(self.parts) else None
        if expected != descriptor:
            raise ValueError("part descriptor does not match manifest")
        if len(body) != descriptor.size:
            raise ValueError("part size does not match manifest")
        if _sha256(body) != descriptor.sha256:
            raise ValueError("part sha256 does not match manifest")
