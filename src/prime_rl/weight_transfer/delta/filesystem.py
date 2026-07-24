from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from prime_rl.weight_transfer.delta.protocol import DeltaManifest


def _validated_id(value: str, field: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a UUID") from error
    canonical = str(parsed)
    if canonical != value.lower():
        raise ValueError(f"{field} must use canonical UUID form")
    return canonical


def _write_fsynced(path: Path, body: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class FileSystemDeltaStore:
    def __init__(self, root: Path):
        self.root = root

    def transfer_path(self, run_id: str, transfer_id: str) -> Path:
        return self.root / _validated_id(run_id, "run_id") / _validated_id(transfer_id, "transfer_id")

    def publish(self, manifest: DeltaManifest, parts: tuple[bytes, ...]) -> Path:
        manifest.validate()
        if len(parts) != len(manifest.parts):
            raise ValueError("part count does not match manifest")
        for descriptor, body in zip(manifest.parts, parts, strict=True):
            manifest.verify_part(descriptor, body)

        destination = self.transfer_path(manifest.run_id, manifest.transfer_id)
        if destination.exists():
            existing_manifest, existing_parts = self.load(manifest.run_id, manifest.transfer_id)
            if existing_manifest == manifest and existing_parts == parts:
                return destination
            raise ValueError("transfer already exists with different manifest or part bytes")

        run_dir = destination.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary = run_dir / f".{manifest.transfer_id}.tmp-{uuid.uuid4()}"
        temporary.mkdir(mode=0o700)
        try:
            for descriptor, body in zip(manifest.parts, parts, strict=True):
                _write_fsynced(temporary / f"part-{descriptor.seq:06d}.bin", body)
            _fsync_directory(temporary)
            _write_fsynced(temporary / "manifest.json", manifest.to_bytes())
            _fsync_directory(temporary)
            _write_fsynced(temporary / "COMMITTED", manifest.manifest_hash.encode() + b"\n")
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(run_dir)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def load(self, run_id: str, transfer_id: str) -> tuple[DeltaManifest, tuple[bytes, ...]]:
        transfer = self.transfer_path(run_id, transfer_id)
        committed = transfer / "COMMITTED"
        if not committed.is_file():
            raise FileNotFoundError(f"transfer is not committed: {transfer}")
        manifest = DeltaManifest.from_bytes((transfer / "manifest.json").read_bytes())
        if manifest.run_id != _validated_id(run_id, "run_id") or manifest.transfer_id != _validated_id(
            transfer_id, "transfer_id"
        ):
            raise ValueError("manifest identity does not match transfer path")
        if committed.read_text().strip() != manifest.manifest_hash:
            raise ValueError("commit marker does not match manifest hash")
        bodies: list[bytes] = []
        for descriptor in manifest.parts:
            body = (transfer / f"part-{descriptor.seq:06d}.bin").read_bytes()
            manifest.verify_part(descriptor, body)
            bodies.append(body)
        return manifest, tuple(bodies)
