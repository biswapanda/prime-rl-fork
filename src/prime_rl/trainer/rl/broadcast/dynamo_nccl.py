from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, cast

import httpx
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.utils import get_world
from prime_rl.utils.client import NCCL_READY_MARKER
from prime_rl.utils.dynamo import (
    DynamoDiscoveryPending,
    DynamoWorker,
    client_headers,
    discover_dynamo_workers,
    topology_fingerprint,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


class PrimeWeightSource:
    """Adapts Prime's checkpoint conversion to vLLM's generic WeightSource."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.model: nn.Module | None = None
        self.dtype = dtype
        self._metadata = None

    def set_model(self, model: nn.Module) -> None:
        if model is not self.model:
            self._metadata = None
        self.model = model

    def _items(self) -> Iterator[tuple[str, Tensor]]:
        if self.model is None:
            raise RuntimeError("Prime weight source has no model")
        from prime_rl.trainer.conversion_utils import get_max_layer_num
        from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers, preprocess_layer_checkpoint

        state_dict = self.model.state_dict()
        layer_prefix = get_layer_prefix(self.model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            resolved = {}
            for name, tensor in layer_state_dict.items():
                if isinstance(tensor, DTensor):
                    tensor = cast(DTensor, tensor.to(self.dtype)).full_tensor()
                resolved[name] = tensor
            yield from preprocess_layer_checkpoint(self.model, resolved, layer_idx).items()

    def metadata(self):
        from vllm.distributed.weight_transfer import ParamMeta

        if self._metadata is None:
            if self.model is None:
                raise RuntimeError("Prime weight source has no model")
            from prime_rl.trainer.conversion_utils import get_max_layer_num
            from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers, preprocess_layer_checkpoint

            state_dict = {
                name: torch.empty(
                    tuple(tensor.shape),
                    dtype=self.dtype if isinstance(tensor, DTensor) else tensor.dtype,
                    device="meta",
                )
                for name, tensor in self.model.state_dict().items()
            }
            layer_prefix = get_layer_prefix(self.model.config)
            num_layers = get_max_layer_num(state_dict, layer_prefix)
            metadata = []
            for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
                converted = preprocess_layer_checkpoint(self.model, layer_state_dict, layer_idx)
                metadata.extend(
                    ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in converted.items()
                )
            self._metadata = metadata
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._items()


class DynamoVLLMWeightSyncClient:
    def __init__(self, workers: tuple[DynamoWorker, ...], headers: dict[str, str], timeout: float) -> None:
        self.workers = workers
        self.clients = [
            httpx.Client(base_url=worker.system_url.rstrip("/"), headers=headers, timeout=timeout) for worker in workers
        ]

    @staticmethod
    def _validate(response: httpx.Response, path: str) -> None:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dynamo route {path} returned a non-object response")
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", f"Dynamo route {path} failed"))

    def _fanout(self, path: str, bodies: list[dict[str, Any]]) -> None:
        def request(client: httpx.Client, body: dict[str, Any]) -> None:
            self._validate(client.post(path, json=body), path)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [executor.submit(request, client, body) for client, body in zip(self.clients, bodies)]
            for future in futures:
                future.result()

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        rank_offset = 1
        bodies = []
        for worker in self.workers:
            worker_init = {**init_info, "rank_offset": rank_offset}
            bodies.append({"init_info": worker_init})
            rank_offset += worker.world_size
        self._fanout("/engine/update/init_weight_transfer_engine", bodies)

    def start_weight_update(self) -> None:
        self._fanout("/engine/update/start_weight_update", [{} for _ in self.clients])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        self._fanout(
            "/engine/update/update_weights",
            [{"update_info": update_info} for _ in self.clients],
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self._fanout(
            "/engine/update/finish_weight_update",
            [{"weight_version": weight_version} for _ in self.clients],
        )

    def close(self) -> None:
        for client in self.clients:
            client.close()


def _discover(config: NCCLWeightBroadcastConfig) -> tuple[DynamoWorker, ...]:
    assert config.dynamo is not None
    dynamo = config.dynamo
    headers = client_headers(
        dynamo.headers,
        dynamo.headers_from_env,
        dynamo.api_key_var,
    )
    deadline = time.monotonic() + config.timeout
    last_error: Exception | None = None
    previous_fingerprint = None
    while time.monotonic() < deadline:
        try:
            workers = discover_dynamo_workers(
                dynamo.discovery_url,
                dynamo.model_name,
                headers=headers,
                timeout=min(30.0, max(1.0, deadline - time.monotonic())),
            )
            discovered_world_size = sum(worker.world_size for worker in workers)
            if discovered_world_size != config.inference_world_size:
                raise DynamoDiscoveryPending(
                    f"Dynamo world size {discovered_world_size} does not match expected {config.inference_world_size}"
                )
            fingerprint = topology_fingerprint(workers)
            if fingerprint == previous_fingerprint:
                return workers
            previous_fingerprint = fingerprint
        except httpx.HTTPStatusError as error:
            if error.response.status_code < 500:
                raise
            previous_fingerprint = None
            last_error = error
        except (DynamoDiscoveryPending, httpx.TransportError) as error:
            previous_fingerprint = None
            last_error = error
        time.sleep(1)
    raise TimeoutError("Dynamo workers did not become ready before trainer initialization") from last_error


class DynamoNCCLWeightBroadcast(WeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(output_dir)
        if config.dynamo is None:
            raise ValueError("Dynamo native NCCL requires Dynamo discovery configuration")
        if config.quantize_in_weight_transfer:
            raise ValueError("Dynamo native NCCL does not support quantized transfer")

        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

        self.world = get_world()
        self.workers = _discover(config)
        dynamo = config.dynamo
        headers = client_headers(
            dynamo.headers,
            dynamo.headers_from_env,
            dynamo.api_key_var,
        )
        self.client = DynamoVLLMWeightSyncClient(self.workers, headers, config.timeout)
        self.source = PrimeWeightSource(dtype)
        disable_nccl_p2p_if_unavailable()
        self.engine = WeightTransferTrainerFactory.trainer_init(
            NCCLTrainerInitInfo(
                master_address=config.host,
                master_port=config.port,
                world_size=1 + sum(worker.world_size for worker in self.workers),
                rank=self.world.rank,
            ),
            client=self.client,
            source=self.source,
        )
        self.dtype = dtype

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        self.source.set_model(model)
        save_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        if self.world.is_master:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "STABLE").touch()
            sync_wait_for_path(save_dir / NCCL_READY_MARKER, interval=0.1, log_interval=10)
        if self.world.world_size > 1:
            dist.barrier()
        self.engine.send_weights(weight_version=str(step))
        get_logger().debug(f"Dynamo workers committed weight version {step}")
