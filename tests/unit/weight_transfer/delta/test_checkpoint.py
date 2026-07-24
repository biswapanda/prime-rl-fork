from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prime_rl.weight_transfer.delta.checkpoint import (
    apply_parts_to_shadow_checkpoint,
    build_checkpoint_delta,
    checkpoint_fingerprint,
)


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text('{"model_type":"qwen3"}\n')


def test_shadow_checkpoint_apply_preserves_base_and_matches_target(tmp_path: Path) -> None:
    base = tmp_path / "w0"
    target = tmp_path / "w1-full"
    shadow = tmp_path / "w1-delta"
    base_tensors = {
        "model.embed_tokens.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
        "model.layers.0.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }
    target_tensors = {name: tensor.clone() for name, tensor in base_tensors.items()}
    target_tensors["model.embed_tokens.weight"][1, 2] = -7
    target_tensors["model.layers.0.weight"][3, 0] = 99
    write_checkpoint(base, base_tensors)
    write_checkpoint(target, target_tensors)

    manifest, parts = build_checkpoint_delta(
        base,
        target,
        run_id="e281ae31-d195-4b27-9d6f-ce0eae5f5e55",
        transfer_id="8cce5da8-e869-4576-a542-a579305a3fd0",
        model_id="Qwen/Qwen3-0.6B",
        checkpoint_revision="0123456789abcdef",
        base_version=0,
        target_version=1,
    )
    apply_parts_to_shadow_checkpoint(base, shadow, manifest, parts)

    assert checkpoint_fingerprint(shadow) == checkpoint_fingerprint(target)
    assert torch.equal(
        load_file(base / "model.safetensors")["model.embed_tokens.weight"], base_tensors["model.embed_tokens.weight"]
    )
    assert torch.equal(
        load_file(shadow / "model.safetensors")["model.layers.0.weight"],
        target_tensors["model.layers.0.weight"],
    )
