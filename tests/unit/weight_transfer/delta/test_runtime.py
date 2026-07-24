import uuid
from pathlib import Path

import pytest
import torch
from torch import nn

from prime_rl.weight_transfer.delta.checkpoint import build_checkpoint_delta
from prime_rl.weight_transfer.delta.filesystem import FileSystemDeltaStore
from prime_rl.weight_transfer.delta.runtime import DeltaRuntime


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)


def artifact(tmp_path: Path):
    from safetensors.torch import save_file

    base = tmp_path / "base"
    target = tmp_path / "target"
    base.mkdir()
    target.mkdir()
    base_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    target_weight = base_weight.clone()
    target_weight[1, 2] = -99
    save_file({"proj.weight": base_weight}, base / "model.safetensors")
    save_file({"proj.weight": target_weight}, target / "model.safetensors")
    run_id = str(uuid.uuid4())
    transfer_id = str(uuid.uuid4())
    manifest, parts = build_checkpoint_delta(
        base,
        target,
        run_id=run_id,
        transfer_id=transfer_id,
        model_id="tiny",
        checkpoint_revision="revision",
        base_version=0,
        target_version=1,
    )
    store = FileSystemDeltaStore(tmp_path / "store")
    store.publish(manifest, parts)
    return base_weight, target_weight, manifest, store


def initialized_runtime(model: TinyModel, manifest) -> DeltaRuntime:
    runtime = DeltaRuntime(model)
    runtime.initialize(
        model_id=manifest.model_id,
        checkpoint_revision=manifest.checkpoint_revision,
        baseline_fingerprint=manifest.baseline_fingerprint,
        tensor_table_hash=manifest.tensor_table_hash,
        version=0,
    )
    return runtime


def test_stage_activate_commit_advances_exactly_one_version(tmp_path):
    base, target, manifest, store = artifact(tmp_path)
    model = TinyModel()
    model.proj.weight.data.copy_(base)
    runtime = initialized_runtime(model, manifest)

    staged = runtime.stage_from_filesystem(
        root=store.root,
        run_id=manifest.run_id,
        transfer_id=manifest.transfer_id,
        transport="filesystem",
    )
    assert staged["state"] == "staged"
    assert torch.equal(model.proj.weight, base)

    activated = runtime.activate(manifest.transfer_id)
    assert activated["state"] == "activated"
    assert torch.equal(model.proj.weight, target)
    assert runtime.status()["version"] == 0

    committed = runtime.commit(manifest.transfer_id)
    assert committed["state"] == "committed"
    assert runtime.status()["version"] == 1
    assert runtime.status()["transfer_id"] == manifest.transfer_id
    assert runtime.commit(manifest.transfer_id) == committed
    retried = runtime.stage_from_filesystem(
        root=store.root,
        run_id=manifest.run_id,
        transfer_id=manifest.transfer_id,
        transport="filesystem",
    )
    assert retried == committed


def test_identical_stage_retry_is_idempotent(tmp_path):
    base, _, manifest, store = artifact(tmp_path)
    model = TinyModel()
    model.proj.weight.data.copy_(base)
    runtime = initialized_runtime(model, manifest)
    kwargs = {
        "root": store.root,
        "run_id": manifest.run_id,
        "transfer_id": manifest.transfer_id,
        "transport": "filesystem",
    }

    first = runtime.stage_from_filesystem(**kwargs)
    second = runtime.stage_from_filesystem(**kwargs)

    assert first == second
    assert second["state"] == "staged"


def test_rollback_restores_base_and_does_not_advance_version(tmp_path):
    base, target, manifest, store = artifact(tmp_path)
    model = TinyModel()
    model.proj.weight.data.copy_(base)
    runtime = initialized_runtime(model, manifest)
    runtime.stage_from_filesystem(
        root=store.root,
        run_id=manifest.run_id,
        transfer_id=manifest.transfer_id,
        transport="zmq",
    )
    runtime.activate(manifest.transfer_id)
    assert torch.equal(model.proj.weight, target)

    rolled_back = runtime.rollback(manifest.transfer_id)
    assert rolled_back["state"] == "rolled_back"
    assert torch.equal(model.proj.weight, base)
    assert runtime.status()["version"] == 0


def test_stage_rejects_stale_or_skipped_version(tmp_path):
    _, _, manifest, store = artifact(tmp_path)
    model = TinyModel()
    runtime = DeltaRuntime(model)
    runtime.initialize(
        model_id=manifest.model_id,
        checkpoint_revision=manifest.checkpoint_revision,
        baseline_fingerprint=manifest.baseline_fingerprint,
        tensor_table_hash=manifest.tensor_table_hash,
        version=1,
    )

    with pytest.raises(ValueError, match="exact-next"):
        runtime.stage_from_filesystem(
            root=store.root,
            run_id=manifest.run_id,
            transfer_id=manifest.transfer_id,
            transport="filesystem",
        )


def test_stage_rejects_unknown_or_mismatched_tensor_before_mutation(tmp_path):
    base, _, manifest, store = artifact(tmp_path)
    model = TinyModel().to(torch.bfloat16)
    model.proj.weight.data.copy_(base.to(torch.bfloat16))
    runtime = initialized_runtime(model, manifest)

    with pytest.raises(ValueError, match="metadata"):
        runtime.stage_from_filesystem(
            root=store.root,
            run_id=manifest.run_id,
            transfer_id=manifest.transfer_id,
            transport="filesystem",
        )
    assert torch.equal(model.proj.weight, base.to(torch.bfloat16))


def test_initialize_is_idempotent_but_rejects_conflicting_identity(tmp_path):
    _, _, manifest, _ = artifact(tmp_path)
    runtime = DeltaRuntime(TinyModel())
    kwargs = {
        "model_id": manifest.model_id,
        "checkpoint_revision": manifest.checkpoint_revision,
        "baseline_fingerprint": manifest.baseline_fingerprint,
        "tensor_table_hash": manifest.tensor_table_hash,
        "version": 0,
    }
    assert runtime.initialize(**kwargs) == runtime.initialize(**kwargs)

    with pytest.raises(ValueError, match="already initialized"):
        runtime.initialize(**{**kwargs, "version": 1})
