from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims

if TYPE_CHECKING:
    from prime_rl.trainer.rl.broadcast.base import WeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
) -> WeightBroadcast:
    if config.type == "nccl":
        if config.dynamo is not None and not config.quantize_in_weight_transfer:
            from prime_rl.trainer.rl.broadcast.dynamo_nccl import DynamoNCCLWeightBroadcast

            return DynamoNCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
        from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast

        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast

        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "nixl":
        from prime_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast

        return NIXLWeightBroadcast(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
