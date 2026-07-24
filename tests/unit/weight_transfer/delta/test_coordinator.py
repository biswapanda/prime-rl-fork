import asyncio
import json

import httpx
import pytest

from prime_rl.weight_transfer.delta.coordinator import DeltaUpdateCoordinator


def client(
    events: list[str],
    *,
    fail_method: str | None = None,
    transient_failures: dict[str, int] | None = None,
) -> httpx.AsyncClient:
    remaining_failures = dict(transient_failures or {})

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pause":
            events.append("pause")
            assert request.url.params["mode"] == "wait"
            assert request.url.params["clear_cache"] == "true"
            return httpx.Response(200, json={"status": "paused"})
        if request.url.path == "/resume":
            events.append("resume")
            return httpx.Response(200, json={"status": "resumed"})

        body = json.loads(request.content)
        method = body["method"]
        events.append(method)
        transient = remaining_failures.get(method, 0)
        if method == fail_method or transient > 0:
            if transient > 0:
                remaining_failures[method] = transient - 1
            return httpx.Response(500, json={"error": "injected"})
        transfer_id = "22222222-2222-2222-2222-222222222222"
        states = {
            "initialize_delta_runtime": {"state": "idle", "version": 0},
            "stage_delta": {
                "state": "staged",
                "version": 0,
                "target_version": 1,
                "transfer_id": transfer_id,
                "manifest_hash": "sha256:" + "3" * 64,
            },
            "activate_delta": {
                "state": "activated",
                "version": 0,
                "target_version": 1,
                "transfer_id": transfer_id,
                "manifest_hash": "sha256:" + "3" * 64,
            },
            "get_delta_state": {
                "state": "activated",
                "version": 0,
                "target_version": 1,
                "transfer_id": transfer_id,
                "manifest_hash": "sha256:" + "3" * 64,
            },
            "commit_delta": {
                "state": "committed",
                "version": 1,
                "committed_transfer_id": transfer_id,
            },
            "rollback_delta": {
                "state": "rolled_back",
                "version": 0,
                "rolled_back_transfer_id": transfer_id,
            },
        }
        return httpx.Response(200, json={"results": [states[method]]})

    return httpx.AsyncClient(
        base_url="http://worker",
        transport=httpx.MockTransport(handler),
    )


IDENTITY = {
    "model_id": "Qwen/Qwen3-0.6B",
    "checkpoint_revision": "revision",
    "baseline_fingerprint": "sha256:" + "1" * 64,
    "tensor_table_hash": "sha256:" + "2" * 64,
    "version": 0,
}
SOURCE = {
    "transport": "filesystem",
    "root": "/deltas",
    "run_id": "11111111-1111-1111-1111-111111111111",
    "transfer_id": "22222222-2222-2222-2222-222222222222",
}


def test_coordinator_stages_before_pause_and_resumes_after_commit():
    events: list[str] = []

    async def scenario():
        admin = client(events)
        coordinator = DeltaUpdateCoordinator([admin])
        try:
            await coordinator.initialize(IDENTITY)
            return await coordinator.apply(SOURCE)
        finally:
            await admin.aclose()

    receipts = asyncio.run(scenario())

    assert receipts[0]["state"] == "committed"
    assert events == [
        "initialize_delta_runtime",
        "stage_delta",
        "pause",
        "activate_delta",
        "get_delta_state",
        "commit_delta",
        "resume",
    ]


def test_coordinator_rolls_back_and_resumes_on_activation_failure():
    events: list[str] = []

    async def scenario():
        admin = client(events, fail_method="activate_delta")
        coordinator = DeltaUpdateCoordinator([admin])
        try:
            await coordinator.apply(SOURCE)
        finally:
            await admin.aclose()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(scenario())
    assert events == [
        "stage_delta",
        "pause",
        "activate_delta",
        "rollback_delta",
        "resume",
    ]


def test_coordinator_does_not_pause_when_staging_fails():
    events: list[str] = []

    async def scenario():
        admin = client(events, fail_method="stage_delta")
        coordinator = DeltaUpdateCoordinator([admin])
        try:
            await coordinator.apply(SOURCE)
        finally:
            await admin.aclose()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(scenario())
    assert events == ["stage_delta", "rollback_delta"]


def test_coordinator_retries_ambiguous_commit_before_resuming():
    events: list[str] = []

    async def scenario():
        admin = client(events, transient_failures={"commit_delta": 1})
        coordinator = DeltaUpdateCoordinator([admin])
        try:
            return await coordinator.apply(SOURCE)
        finally:
            await admin.aclose()

    receipts = asyncio.run(scenario())

    assert receipts[0]["state"] == "committed"
    assert events[-3:] == ["commit_delta", "commit_delta", "resume"]


def test_coordinator_keeps_engine_paused_when_commit_cannot_be_confirmed():
    events: list[str] = []

    async def scenario():
        admin = client(events, fail_method="commit_delta")
        coordinator = DeltaUpdateCoordinator([admin])
        try:
            await coordinator.apply(SOURCE)
        finally:
            await admin.aclose()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(scenario())
    assert events[-3:] == ["commit_delta", "commit_delta", "commit_delta"]
    assert "resume" not in events
