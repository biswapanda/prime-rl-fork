from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn

from prime_rl.weight_transfer.delta.runtime import DeltaRuntime

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


def _json_object(body: str, *, operation: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{operation} payload must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{operation} payload must be an object")
    return value


def _exact_fields(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    operation: str,
) -> None:
    optional = optional or set()
    if not required <= value.keys() or not value.keys() <= required | optional:
        raise ValueError(f"{operation} payload has unknown or missing fields")


def _required_string(value: dict[str, object], field: str, *, operation: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{operation} {field} must be a non-empty string")
    return result


def _optional_string(value: dict[str, object], field: str) -> str | None:
    result = value.get(field)
    if result is not None and not isinstance(result, str):
        raise ValueError(f"delta source {field} must be a string")
    return result


def _configured_filesystem_root(requested: str) -> Path:
    configured = os.environ.get("PRIME_DELTA_FILESYSTEM_ROOT")
    if not configured:
        raise RuntimeError("PRIME_DELTA_FILESYSTEM_ROOT is required for filesystem delta staging")
    root = Path(requested).resolve()
    allowed = Path(configured).resolve()
    if not root.is_relative_to(allowed):
        raise ValueError("delta source root is outside PRIME_DELTA_FILESYSTEM_ROOT")
    return root


def _configured_s3_value(source: dict[str, object], field: str, environment: str) -> str:
    requested = _required_string(source, field, operation="delta source").rstrip("/")
    configured = os.environ.get(environment, "").rstrip("/")
    if not configured:
        raise RuntimeError(f"{environment} is required for S3 delta staging")
    if requested != configured:
        raise ValueError(f"delta source {field} does not match {environment}")
    return configured


class DeltaWeightUpdateWorker(Worker):
    """Prime delta lifecycle exposed through vLLM's native collective RPC."""

    def _prime_delta_model(self) -> nn.Module:
        model = self.model_runner.model
        if hasattr(model, "runnable"):
            model = model.runnable
        if not isinstance(model, nn.Module):
            raise TypeError("vLLM model runner did not expose a torch module")
        return model

    def _prime_delta_runtime(self) -> DeltaRuntime:
        runtime = getattr(self, "_prime_delta_runtime_instance", None)
        if runtime is None:
            runtime = DeltaRuntime(self._prime_delta_model())
            self._prime_delta_runtime_instance = runtime
        return runtime

    def liveness_probe(self) -> None:
        return None

    def initialize_delta_runtime(self, identity_json: str) -> dict[str, object]:
        identity = _json_object(identity_json, operation="delta identity")
        fields = {
            "model_id",
            "checkpoint_revision",
            "baseline_fingerprint",
            "tensor_table_hash",
            "version",
        }
        _exact_fields(identity, required=fields, operation="delta identity")
        if not all(isinstance(identity[name], str) and identity[name] for name in fields - {"version"}):
            raise ValueError("delta identity string fields must be non-empty")
        version = identity["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("delta identity version must be a non-negative integer")
        return self._prime_delta_runtime().initialize(
            model_id=identity["model_id"],
            checkpoint_revision=identity["checkpoint_revision"],
            baseline_fingerprint=identity["baseline_fingerprint"],
            tensor_table_hash=identity["tensor_table_hash"],
            version=version,
        )

    def stage_delta(self, source_json: str) -> dict[str, object]:
        source = _json_object(source_json, operation="delta source")
        transport = source.get("transport")
        common = {"transport", "run_id", "transfer_id"}
        if transport in ("filesystem", "zmq"):
            _exact_fields(
                source,
                required=common | {"root"},
                operation="delta source",
            )
            return self._prime_delta_runtime().stage_from_filesystem(
                root=_configured_filesystem_root(_required_string(source, "root", operation="delta source")),
                run_id=_required_string(source, "run_id", operation="delta source"),
                transfer_id=_required_string(source, "transfer_id", operation="delta source"),
                transport=transport,
            )
        if transport == "s3":
            _exact_fields(
                source,
                required=common | {"bucket"},
                optional={"endpoint_url", "region_name", "prefix"},
                operation="delta source",
            )
            prefix = _optional_string(source, "prefix") or ""
            if prefix.startswith("/") or ".." in Path(prefix).parts:
                raise ValueError("delta source prefix must be a relative S3 key prefix")
            return self._prime_delta_runtime().stage_from_s3(
                bucket=_configured_s3_value(
                    source,
                    "bucket",
                    "PRIME_DELTA_S3_BUCKET",
                ),
                run_id=_required_string(source, "run_id", operation="delta source"),
                transfer_id=_required_string(source, "transfer_id", operation="delta source"),
                endpoint_url=_configured_s3_value(
                    source,
                    "endpoint_url",
                    "PRIME_DELTA_S3_ENDPOINT",
                ),
                region_name=_optional_string(source, "region_name") or "us-east-1",
                prefix=prefix,
            )
        raise ValueError("delta source transport must be filesystem, s3 or zmq")

    def activate_delta(self, transfer_id: str) -> dict[str, object]:
        return self._prime_delta_runtime().activate(transfer_id)

    def commit_delta(self, transfer_id: str) -> dict[str, object]:
        return self._prime_delta_runtime().commit(transfer_id)

    def rollback_delta(self, transfer_id: str) -> dict[str, object]:
        return self._prime_delta_runtime().rollback(transfer_id)

    def get_delta_state(self) -> dict[str, object]:
        return self._prime_delta_runtime().status()
