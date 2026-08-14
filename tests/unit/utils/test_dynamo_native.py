import asyncio
import json

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.trainer.rl.broadcast.dynamo_nccl import DynamoVLLMWeightSyncClient
from prime_rl.utils.dynamo import (
    REQUIRED_ROUTES,
    DynamoDiscoveryPending,
    DynamoInferencePool,
    DynamoWorker,
    parse_dynamo_workers,
)


def worker(component: str, instance_id: int, world_size: int = 1, *, admin_base_url: str | None = None) -> dict:
    return {
        "namespace": "dynamo",
        "component": component,
        "instance_id": instance_id,
        "model": "Qwen/Qwen3-0.6B",
        "system_url": f"http://{component}-{instance_id}:8080",
        "world_size": world_size,
        "weight_transfer_backend": "nccl",
        "routes": sorted(REQUIRED_ROUTES),
        "admin_base_url": admin_base_url,
    }


def test_client_config_rejects_two_admin_discovery_modes():
    with pytest.raises(ValueError, match="cannot be combined"):
        ClientConfig(
            dynamo_discovery_url="http://frontend:8001",
            admin_base_url=["http://worker:8000"],
        )


def test_parse_dynamo_workers_preserves_heterogeneous_topology():
    workers = parse_dynamo_workers(
        {
            "protocol_version": 1,
            "workers": [worker("prefill", 3, 2), worker("decode", 9, 4)],
        },
        "Qwen/Qwen3-0.6B",
    )
    assert [(item.component, item.world_size) for item in workers] == [("decode", 4), ("prefill", 2)]


def test_parse_dynamo_workers_preserves_collective_rpc_endpoint():
    workers = parse_dynamo_workers(
        {
            "protocol_version": 1,
            "workers": [worker("backend", 1, admin_base_url="http://backend-1:8120")],
        },
        "Qwen/Qwen3-0.6B",
    )
    assert workers[0].admin_base_url == "http://backend-1:8120"


def test_parse_dynamo_workers_waits_for_complete_capabilities():
    incomplete = worker("backend", 1)
    incomplete["routes"] = []
    with pytest.raises(DynamoDiscoveryPending, match="missing native routes"):
        parse_dynamo_workers(
            {"protocol_version": 1, "workers": [incomplete]},
            "Qwen/Qwen3-0.6B",
        )


def test_native_weight_client_assigns_cumulative_rank_offsets(monkeypatch):
    requests: list[tuple[str, str, dict]] = []
    real_client = httpx.Client

    def client_factory(*, base_url, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.url.host or "", request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"status": "ok"})

        return real_client(base_url=base_url, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.dynamo_nccl.httpx.Client", client_factory)
    workers = (
        DynamoWorker.model_validate(worker("prefill", 3, 2)),
        DynamoWorker.model_validate(worker("decode", 9, 4)),
    )
    client = DynamoVLLMWeightSyncClient(workers, {}, timeout=10)
    try:
        client.init_weight_transfer_engine({"master_address": "trainer", "world_size": 7})
    finally:
        client.close()

    assert sorted((host, body["init_info"]["rank_offset"]) for host, _, body in requests) == [
        ("decode-9", 3),
        ("prefill-3", 1),
    ]
    assert {path for _, path, _ in requests} == {"/engine/update/init_weight_transfer_engine"}


@pytest.mark.asyncio
async def test_dynamo_update_pauses_commits_and_resumes(tmp_path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if request.url.path.endswith("get_weight_version"):
            return httpx.Response(200, json={"weight_version": "3"})
        return httpx.Response(200, json={"status": "ok"})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1)),)
    pool._weight_update_timeout = 10
    pool._admin_clients = [httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(handler))]
    try:
        await pool.update_weights(tmp_path, step=3)
    finally:
        await pool._admin_clients[0].aclose()

    assert (tmp_path / "NCCL_READY").exists()
    assert requests[0] == (
        "/engine/control/pause_generation",
        {"mode": "abort", "clear_cache": True},
    )
    assert [path for path, _ in requests] == [
        "/engine/control/pause_generation",
        "/engine/control/is_paused",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]


@pytest.mark.asyncio
async def test_dynamo_update_fails_closed_when_pause_is_not_confirmed(tmp_path):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": False})
        return httpx.Response(200, json={"status": "ok"})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1)),)
    pool._weight_update_timeout = 10
    pool._admin_clients = [httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(handler))]
    try:
        with pytest.raises(RuntimeError, match="every pinned worker was paused"):
            await pool.update_weights(tmp_path, step=3)
    finally:
        await pool._admin_clients[0].aclose()

    assert "/engine/control/resume_generation" not in paths


@pytest.mark.asyncio
async def test_collective_rpc_initialization_preserves_engine_rank_spans():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", json.loads(request.content)))
        return httpx.Response(200, json={"results": [None]})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (
        DynamoWorker.model_validate(worker("prefill", 3, 2, admin_base_url="http://prefill-3:8120")),
        DynamoWorker.model_validate(worker("decode", 9, 4, admin_base_url="http://decode-9:8120")),
    )
    pool._collective_clients = [
        httpx.AsyncClient(base_url=item.admin_base_url, transport=httpx.MockTransport(handler)) for item in pool.workers
    ]
    try:
        await pool.init_nccl_broadcast(
            host="trainer",
            port=29501,
            timeout=1200,
            inference_world_size=6,
            quantize_in_weight_transfer=True,
        )
    finally:
        await asyncio.gather(*(client.aclose() for client in pool._collective_clients))

    assert sorted((host, body["kwargs"]["rank_offset"]) for host, body in requests) == [
        ("decode-9", 2),
        ("prefill-3", 0),
    ]
    assert {body["method"] for _, body in requests} == {"init_broadcaster"}


@pytest.mark.asyncio
async def test_collective_rpc_updates_prime_worker_extension(tmp_path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200, json={"results": [None]})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1, admin_base_url="http://worker:8120")),)
    pool._weight_transfer_mode = "collective_rpc"
    pool._collective_clients = [
        httpx.AsyncClient(base_url="http://worker:8120", transport=httpx.MockTransport(handler))
    ]
    try:
        await pool.update_weights(tmp_path, step=2)
    finally:
        await pool._collective_clients[0].aclose()

    assert [path for path, _ in requests] == ["/pause", "/collective_rpc", "/resume"]
    assert requests[1][1] == {"method": "update_weights_from_path", "args": [tmp_path.as_posix()]}
