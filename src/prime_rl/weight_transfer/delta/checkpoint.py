from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prime_rl.weight_transfer.delta.codec import TensorDelta, decode_part, encode_part, tensor_sha256
from prime_rl.weight_transfer.delta.protocol import DeltaManifest, DeltaPart, canonical_json


def _weight_files(checkpoint: Path) -> tuple[Path, ...]:
    files = tuple(sorted(checkpoint.glob("*.safetensors")))
    if not files:
        raise ValueError(f"checkpoint contains no safetensors files: {checkpoint}")
    return files


def _tensor_locations(checkpoint: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for path in _weight_files(checkpoint):
        for name in load_file(path, device="cpu"):
            if name in locations:
                raise ValueError(f"checkpoint contains duplicate tensor name: {name}")
            locations[name] = path
    return locations


def _load_tensors(checkpoint: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for path in _weight_files(checkpoint):
        for name, tensor in load_file(path, device="cpu").items():
            if name in tensors:
                raise ValueError(f"checkpoint contains duplicate tensor name: {name}")
            tensors[name] = tensor
    return tensors


def tensor_table_hash(checkpoint: Path) -> str:
    tensors = _load_tensors(checkpoint)
    table = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in sorted(tensors.items())
    ]
    return f"sha256:{hashlib.sha256(canonical_json(table)).hexdigest()}"


def checkpoint_fingerprint(checkpoint: Path) -> str:
    tensors = _load_tensors(checkpoint)
    table = [(name, tensor_sha256(tensor)) for name, tensor in sorted(tensors.items())]
    return f"sha256:{hashlib.sha256(canonical_json(table)).hexdigest()}"


def build_checkpoint_delta(
    base_checkpoint: Path,
    target_checkpoint: Path,
    *,
    run_id: str,
    transfer_id: str,
    model_id: str,
    checkpoint_revision: str,
    base_version: int,
    target_version: int,
) -> tuple[DeltaManifest, tuple[bytes, ...]]:
    base_tensors = _load_tensors(base_checkpoint)
    target_tensors = _load_tensors(target_checkpoint)
    if set(base_tensors) != set(target_tensors):
        raise ValueError("base and target checkpoint tensor names differ")

    encoded_parts: list[bytes] = []
    target_evidence: dict[str, str] = {}
    for name in sorted(base_tensors):
        base = base_tensors[name]
        target = target_tensors[name]
        if base.shape != target.shape or base.dtype != target.dtype:
            raise ValueError(f"base and target tensor metadata differ: {name}")
        delta = TensorDelta.between(name, base, target)
        if delta.changed_elements == 0:
            continue
        encoded_parts.append(encode_part((delta,)))
        target_evidence[name] = delta.target_hash

    target_evidence["__checkpoint__"] = checkpoint_fingerprint(target_checkpoint)
    descriptors = tuple(
        DeltaPart.from_bytes(seq=seq, body=body, tensor_entries=len(decode_part(body)))
        for seq, body in enumerate(encoded_parts)
    )
    manifest = DeltaManifest.create(
        run_id=run_id,
        transfer_id=transfer_id,
        model_id=model_id,
        checkpoint_revision=checkpoint_revision,
        baseline_fingerprint=checkpoint_fingerprint(base_checkpoint),
        tensor_table_hash=tensor_table_hash(base_checkpoint),
        base_version=base_version,
        target_version=target_version,
        parts=descriptors,
        target_evidence=target_evidence,
    )
    return manifest, tuple(encoded_parts)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def apply_parts_to_shadow_checkpoint(
    base_checkpoint: Path,
    shadow_checkpoint: Path,
    manifest: DeltaManifest,
    parts: tuple[bytes, ...],
) -> Path:
    manifest.validate()
    if checkpoint_fingerprint(base_checkpoint) != manifest.baseline_fingerprint:
        raise ValueError("base checkpoint fingerprint does not match manifest")
    if tensor_table_hash(base_checkpoint) != manifest.tensor_table_hash:
        raise ValueError("base checkpoint tensor table does not match manifest")
    if len(parts) != len(manifest.parts):
        raise ValueError("part count does not match manifest")

    expected_target = manifest.target_evidence.get("__checkpoint__")
    if shadow_checkpoint.exists():
        if expected_target is not None and checkpoint_fingerprint(shadow_checkpoint) == expected_target:
            return shadow_checkpoint
        raise FileExistsError(f"shadow checkpoint already exists with different contents: {shadow_checkpoint}")

    temporary = shadow_checkpoint.parent / f".{shadow_checkpoint.name}.tmp-{uuid.uuid4()}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_checkpoint, temporary, copy_function=shutil.copy2)
    try:
        entries: dict[str, TensorDelta] = {}
        for descriptor, body in zip(manifest.parts, parts, strict=True):
            manifest.verify_part(descriptor, body)
            decoded = decode_part(body)
            if len(decoded) != descriptor.tensor_entries:
                raise ValueError("part tensor entry count does not match manifest")
            for entry in decoded:
                if entry.name in entries:
                    raise ValueError(f"duplicate delta tensor across parts: {entry.name}")
                entries[entry.name] = entry

        locations = _tensor_locations(temporary)
        unknown = set(entries) - set(locations)
        if unknown:
            raise ValueError(f"delta contains tensors not present in base checkpoint: {sorted(unknown)}")

        by_file: dict[Path, list[TensorDelta]] = {}
        for entry in entries.values():
            by_file.setdefault(locations[entry.name], []).append(entry)
        for path, file_entries in by_file.items():
            tensors = load_file(path, device="cpu")
            for entry in sorted(file_entries, key=lambda value: value.name):
                tensors[entry.name] = entry.apply(tensors[entry.name])
                expected = manifest.target_evidence.get(entry.name)
                if expected != tensor_sha256(tensors[entry.name]):
                    raise ValueError(f"target evidence mismatch for {entry.name}")
            replacement = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
            save_file(tensors, replacement, metadata={"format": "pt"})
            _fsync_file(replacement)
            os.replace(replacement, path)

        for path in temporary.iterdir():
            if path.is_file():
                _fsync_file(path)
        _fsync_directory(temporary)
        actual_target = checkpoint_fingerprint(temporary)
        if expected_target is not None and actual_target != expected_target:
            raise ValueError("shadow checkpoint fingerprint does not match target evidence")
        os.rename(temporary, shadow_checkpoint)
        _fsync_directory(shadow_checkpoint.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return shadow_checkpoint
