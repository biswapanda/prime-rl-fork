from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from prime_rl.weight_transfer.delta.codec import TensorDelta, decode_part, tensor_sha256
from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.protocol import DeltaManifest
from prime_rl.weight_transfer.delta.s3 import S3DeltaStore


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    model_id: str
    checkpoint_revision: str
    baseline_fingerprint: str
    tensor_table_hash: str
    version: int
    last_manifest_hash: str | None = None
    last_transfer_id: str | None = None


@dataclass(frozen=True, slots=True)
class UndoEntry:
    name: str
    positions: torch.Tensor | None
    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class StagedDelta:
    transport: str
    manifest: DeltaManifest
    entries: tuple[TensorDelta, ...]
    state: str = "staged"
    undo: tuple[UndoEntry, ...] = ()


class DeltaRuntime:
    """Transactional, exact-next delta activation for one vLLM worker."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._identity: RuntimeIdentity | None = None
        self._staged: StagedDelta | None = None
        self._lock = threading.RLock()

    def initialize(
        self,
        *,
        model_id: str,
        checkpoint_revision: str,
        baseline_fingerprint: str,
        tensor_table_hash: str,
        version: int,
    ) -> dict[str, object]:
        candidate = RuntimeIdentity(
            model_id=model_id,
            checkpoint_revision=checkpoint_revision,
            baseline_fingerprint=baseline_fingerprint,
            tensor_table_hash=tensor_table_hash,
            version=version,
        )
        with self._lock:
            if self._identity is not None and self._identity != candidate:
                raise ValueError("delta runtime is already initialized with a different identity")
            self._identity = candidate
            return self.status()

    def _parameters(self) -> dict[str, nn.Parameter]:
        return dict(self._model.named_parameters())

    def _validate_manifest(self, manifest: DeltaManifest) -> RuntimeIdentity:
        manifest.validate()
        identity = self._identity
        if identity is None:
            raise RuntimeError("delta runtime is not initialized")
        if (
            manifest.model_id != identity.model_id
            or manifest.checkpoint_revision != identity.checkpoint_revision
            or manifest.baseline_fingerprint != identity.baseline_fingerprint
            or manifest.tensor_table_hash != identity.tensor_table_hash
        ):
            raise ValueError("delta manifest does not match the active model identity")
        if manifest.base_version != identity.version or manifest.target_version != identity.version + 1:
            raise ValueError(
                "delta manifest is not the exact-next version "
                f"(active={identity.version}, base={manifest.base_version}, target={manifest.target_version})"
            )
        if "__checkpoint__" not in manifest.target_evidence:
            raise ValueError("delta manifest is missing target checkpoint evidence")
        return identity

    def _stage(
        self,
        manifest: DeltaManifest,
        parts: tuple[bytes, ...],
        *,
        transport: str,
    ) -> dict[str, object]:
        identity = self._identity
        if identity is None:
            raise RuntimeError("delta runtime is not initialized")
        if identity.last_transfer_id == manifest.transfer_id:
            if (
                identity.last_manifest_hash != manifest.manifest_hash
                or identity.version != manifest.target_version
                or identity.baseline_fingerprint != manifest.target_evidence.get("__checkpoint__")
            ):
                raise ValueError("committed transfer identity conflicts with delta manifest")
            return {
                "state": "committed",
                "version": identity.version,
                "target_version": identity.version,
                "transfer_id": manifest.transfer_id,
                "committed_transfer_id": manifest.transfer_id,
                "manifest_hash": identity.last_manifest_hash,
            }
        self._validate_manifest(manifest)
        if len(parts) != len(manifest.parts):
            raise ValueError("part count does not match manifest")
        entries: list[TensorDelta] = []
        for descriptor, body in zip(manifest.parts, parts, strict=True):
            manifest.verify_part(descriptor, body)
            decoded = decode_part(body)
            if len(decoded) != descriptor.tensor_entries:
                raise ValueError("part tensor entry count does not match manifest")
            entries.extend(decoded)
        if len({entry.name for entry in entries}) != len(entries):
            raise ValueError("delta contains duplicate tensor names")

        parameters = self._parameters()
        for entry in entries:
            parameter = parameters.get(entry.name)
            if parameter is None:
                raise ValueError(f"delta tensor is not present in the live model: {entry.name}")
            if tuple(parameter.shape) != entry.shape or str(parameter.dtype).removeprefix("torch.") != entry.dtype:
                raise ValueError(f"live tensor metadata does not match delta for {entry.name}")

        candidate = StagedDelta(
            transport=transport,
            manifest=manifest,
            entries=tuple(entries),
        )
        if self._staged is not None:
            if (
                self._staged.transport == candidate.transport
                and self._staged.manifest.manifest_hash == candidate.manifest.manifest_hash
            ):
                return self.status()
            raise RuntimeError("another delta transfer is already staged")
        self._staged = candidate
        return self.status()

    def stage_from_filesystem(
        self,
        *,
        root: Path,
        run_id: str,
        transfer_id: str,
        transport: str = "filesystem",
    ) -> dict[str, object]:
        if transport not in ("filesystem", "zmq"):
            raise ValueError("filesystem staging transport must be filesystem or zmq")
        with self._lock:
            manifest, parts = FileSystemDeltaStore(root).load(run_id, transfer_id)
            return self._stage(manifest, parts, transport=transport)

    def stage_from_s3(
        self,
        *,
        bucket: str,
        run_id: str,
        transfer_id: str,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        prefix: str = "",
    ) -> dict[str, object]:
        with self._lock:
            store = S3DeltaStore.from_endpoint(
                bucket=bucket,
                endpoint_url=endpoint_url,
                region_name=region_name,
                prefix=prefix,
            )
            manifest, parts = store.load(run_id, transfer_id)
            return self._stage(manifest, parts, transport="s3")

    @torch.no_grad()
    def activate(self, transfer_id: str) -> dict[str, object]:
        with self._lock:
            staged = self._staged
            if staged is None or staged.manifest.transfer_id != transfer_id:
                identity = self._identity
                if identity is not None and identity.last_transfer_id == transfer_id:
                    return {
                        "state": "committed",
                        "version": identity.version,
                        "target_version": identity.version,
                        "transfer_id": transfer_id,
                        "committed_transfer_id": transfer_id,
                        "manifest_hash": identity.last_manifest_hash,
                    }
                raise ValueError("requested delta transfer is not staged")
            if staged.state == "activated":
                return self.status()
            if staged.state != "staged":
                raise RuntimeError(f"cannot activate delta in state {staged.state}")
            self._validate_manifest(staged.manifest)

            parameters = self._parameters()
            undo: list[UndoEntry] = []
            self._staged = replace(staged, state="activating", undo=())
            try:
                for entry in staged.entries:
                    parameter = parameters[entry.name]
                    base = parameter.detach().to(device="cpu").contiguous()
                    target = entry.apply(base)
                    positions = entry.decoded_positions()
                    if positions is None:
                        prior_values = base
                    else:
                        prior_values = base.reshape(-1)[positions].clone()
                    undo.append(
                        UndoEntry(
                            name=entry.name,
                            positions=positions,
                            values=prior_values,
                        )
                    )
                    self._staged = replace(
                        staged,
                        state="activating",
                        undo=tuple(undo),
                    )
                    parameter.copy_(target.to(device=parameter.device))
                self._verify_target(staged)
            except Exception as activation_error:
                self._staged = replace(staged, state="poisoned", undo=tuple(undo))
                try:
                    self._restore(tuple(undo))
                except Exception as restore_error:
                    activation_error.add_note(f"local delta rollback failed: {restore_error!r}")
                    raise
                self._staged = replace(staged, state="staged", undo=())
                raise

            self._staged = replace(staged, state="activated", undo=tuple(undo))
            return self.status()

    def _verify_target(self, staged: StagedDelta) -> None:
        parameters = self._parameters()
        for entry in staged.entries:
            actual = tensor_sha256(parameters[entry.name])
            expected = staged.manifest.target_evidence.get(entry.name)
            if actual != entry.target_hash or actual != expected:
                raise ValueError(f"live target evidence mismatch for {entry.name}")

    @torch.no_grad()
    def _restore(self, undo: tuple[UndoEntry, ...]) -> None:
        parameters = self._parameters()
        for item in reversed(undo):
            parameter = parameters[item.name]
            if item.positions is None:
                parameter.copy_(item.values.to(device=parameter.device))
                if tensor_sha256(parameter) != tensor_sha256(item.values):
                    raise RuntimeError(f"failed to restore delta tensor for {item.name}")
            else:
                positions = item.positions.to(device=parameter.device)
                parameter.reshape(-1)[positions] = item.values.to(device=parameter.device)
                restored = parameter.detach().reshape(-1)[positions].to(device="cpu").contiguous()
                expected = item.values.to(device="cpu").contiguous()
                if not torch.equal(restored.view(torch.uint8), expected.view(torch.uint8)):
                    raise RuntimeError(f"failed to restore delta tensor positions for {item.name}")

    def commit(self, transfer_id: str) -> dict[str, object]:
        with self._lock:
            staged = self._staged
            identity = self._identity
            assert identity is not None
            if staged is None and identity.last_transfer_id == transfer_id:
                return {
                    "state": "committed",
                    "version": identity.version,
                    "target_version": identity.version,
                    "transfer_id": transfer_id,
                    "committed_transfer_id": transfer_id,
                    "manifest_hash": identity.last_manifest_hash,
                }
            if staged is None or staged.manifest.transfer_id != transfer_id:
                raise ValueError("requested delta transfer is not staged")
            if staged.state != "activated":
                raise RuntimeError("delta transfer must be activated before commit")
            self._identity = replace(
                identity,
                baseline_fingerprint=staged.manifest.target_evidence["__checkpoint__"],
                version=staged.manifest.target_version,
                last_manifest_hash=staged.manifest.manifest_hash,
                last_transfer_id=transfer_id,
            )
            committed = {
                "state": "committed",
                "version": staged.manifest.target_version,
                "target_version": staged.manifest.target_version,
                "transfer_id": transfer_id,
                "committed_transfer_id": transfer_id,
                "manifest_hash": staged.manifest.manifest_hash,
            }
            self._staged = None
            return committed

    @torch.no_grad()
    def rollback(self, transfer_id: str) -> dict[str, object]:
        with self._lock:
            staged = self._staged
            if staged is None:
                identity = self._identity
                if identity is not None and identity.last_transfer_id == transfer_id:
                    raise RuntimeError("cannot roll back a committed delta transfer")
                return self.status() | {
                    "state": "rolled_back",
                    "rolled_back_transfer_id": transfer_id,
                }
            if staged.manifest.transfer_id != transfer_id:
                raise ValueError("requested delta transfer is not staged")
            if staged.state in ("activated", "activating", "poisoned"):
                try:
                    self._restore(staged.undo)
                except Exception:
                    self._staged = replace(staged, state="poisoned")
                    raise
            elif staged.state != "staged":
                raise RuntimeError(f"cannot roll back delta in state {staged.state}")
            self._staged = None
            return self.status() | {
                "state": "rolled_back",
                "rolled_back_transfer_id": transfer_id,
            }

    def status(self) -> dict[str, object]:
        identity = self._identity
        staged = self._staged
        return {
            "initialized": identity is not None,
            "model_id": None if identity is None else identity.model_id,
            "checkpoint_revision": None if identity is None else identity.checkpoint_revision,
            "baseline_fingerprint": None if identity is None else identity.baseline_fingerprint,
            "tensor_table_hash": None if identity is None else identity.tensor_table_hash,
            "version": None if identity is None else identity.version,
            "last_manifest_hash": None if identity is None else identity.last_manifest_hash,
            "last_transfer_id": None if identity is None else identity.last_transfer_id,
            "state": "idle" if staged is None else staged.state,
            "transport": None if staged is None else staged.transport,
            "transfer_id": (
                None
                if staged is None and identity is None
                else identity.last_transfer_id
                if staged is None
                else staged.manifest.transfer_id
            ),
            "manifest_hash": (
                None
                if staged is None and identity is None
                else identity.last_manifest_hash
                if staged is None
                else staged.manifest.manifest_hash
            ),
            "target_version": (
                None
                if staged is None and identity is None
                else identity.version
                if staged is None
                else staged.manifest.target_version
            ),
        }
