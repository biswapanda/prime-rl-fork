from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx


class DeltaUpdateCoordinator:
    """Coordinate one transactional delta across discovered vLLM admin endpoints."""

    def __init__(self, admin_clients: list[httpx.AsyncClient], *, timeout_seconds: float = 1800):
        if not admin_clients:
            raise ValueError("delta coordinator requires at least one admin client")
        if timeout_seconds <= 0:
            raise ValueError("delta coordinator timeout must be positive")
        self._clients = tuple(admin_clients)
        self._timeout = httpx.Timeout(timeout_seconds)

    async def _collective(self, method: str, *args: str) -> list[dict[str, object]]:
        async def call(client: httpx.AsyncClient) -> list[dict[str, object]]:
            response = await client.post(
                "/collective_rpc",
                json={
                    "method": method,
                    "args": list(args),
                    "timeout": self._timeout.read,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("results"), list):
                raise ValueError(f"{method} returned an invalid collective RPC response")
            results = body["results"]
            if not results or any(not isinstance(result, dict) for result in results):
                raise ValueError(f"{method} returned an invalid worker receipt")
            return results

        outcomes = await asyncio.gather(
            *(call(client) for client in self._clients),
            return_exceptions=True,
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            first = errors[0]
            for additional in errors[1:]:
                first.add_note(f"additional collective failure: {additional!r}")
            raise first
        return [receipt for results in outcomes if isinstance(results, list) for receipt in results]

    @staticmethod
    def _uniform(
        receipts: list[dict[str, object]],
        expected: Mapping[str, object],
        *,
        operation: str,
    ) -> None:
        for receipt in receipts:
            mismatched = {
                field: (receipt.get(field), value) for field, value in expected.items() if receipt.get(field) != value
            }
            if mismatched:
                raise ValueError(f"{operation} returned a divergent worker receipt: {mismatched}")

    async def initialize(self, identity: Mapping[str, object]) -> list[dict[str, object]]:
        return await self._collective(
            "initialize_delta_runtime",
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        )

    async def _pause(self) -> None:
        async def pause(client: httpx.AsyncClient) -> None:
            response = await client.post(
                "/pause",
                params={"mode": "wait", "clear_cache": "true"},
                timeout=self._timeout,
            )
            response.raise_for_status()

        outcomes = await asyncio.gather(
            *(pause(client) for client in self._clients),
            return_exceptions=True,
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            raise errors[0]

    async def _resume(self) -> None:
        async def resume(client: httpx.AsyncClient) -> None:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = await client.post("/resume", timeout=self._timeout)
                    response.raise_for_status()
                    return
                except Exception as error:
                    last_error = error
                    if attempt < 2:
                        await asyncio.sleep(0.1 * 2**attempt)
            assert last_error is not None
            raise last_error

        outcomes = await asyncio.gather(
            *(resume(client) for client in self._clients),
            return_exceptions=True,
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            raise errors[0]

    async def _retry_collective(
        self,
        method: str,
        *args: str,
        attempts: int = 3,
    ) -> list[dict[str, object]]:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._collective(method, *args)
            except Exception as error:
                last_error = error
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.1 * 2**attempt)
        assert last_error is not None
        raise last_error

    async def _rollback(self, transfer_id: str) -> None:
        await asyncio.gather(
            *(self._collective_for_client(client, "rollback_delta", transfer_id) for client in self._clients)
        )

    async def apply(self, source: Mapping[str, object]) -> list[dict[str, object]]:
        source_json = json.dumps(source, sort_keys=True, separators=(",", ":"))
        transfer_id = source.get("transfer_id")
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("delta source requires a non-empty transfer_id")

        try:
            staged = await self._collective("stage_delta", source_json)
        except Exception as stage_error:
            try:
                await self._rollback(transfer_id)
            except Exception as rollback_error:
                stage_error.add_note(f"staging cleanup failed: {rollback_error!r}")
            raise
        first = staged[0]
        manifest_hash = first.get("manifest_hash")
        target_version = first.get("target_version")
        if not isinstance(manifest_hash, str) or not isinstance(target_version, int):
            raise ValueError("stage_delta did not return manifest identity")
        expected_transfer = {
            "transfer_id": transfer_id,
            "manifest_hash": manifest_hash,
            "target_version": target_version,
        }
        self._uniform(staged, expected_transfer, operation="stage_delta")
        stage_states = {receipt.get("state") for receipt in staged}
        if not stage_states <= {"staged", "committed"}:
            raise ValueError(f"stage_delta returned invalid states: {stage_states}")
        had_precommitted = "committed" in stage_states
        if stage_states == {"committed"}:
            self._uniform(
                staged,
                {"state": "committed", "version": target_version},
                operation="stage_delta",
            )
            return staged

        try:
            await self._pause()
        except Exception:
            await self._resume()
            raise

        try:
            activated = await self._collective("activate_delta", transfer_id)
            self._uniform(activated, expected_transfer, operation="activate_delta")
            active_states = {receipt.get("state") for receipt in activated}
            if not active_states <= {"activated", "committed"}:
                raise ValueError(f"activate_delta returned invalid states: {active_states}")
            verified = await self._collective("get_delta_state")
            self._uniform(verified, expected_transfer, operation="get_delta_state")
            verified_states = {receipt.get("state") for receipt in verified}
            if not verified_states <= {"activated", "idle"}:
                raise ValueError(f"get_delta_state returned invalid states: {verified_states}")
        except Exception as activation_error:
            if had_precommitted:
                activation_error.add_note("a worker was already committed; engines remain fenced for forward recovery")
                raise
            await self._rollback(transfer_id)
            await self._resume()
            raise

        committed = await self._retry_collective("commit_delta", transfer_id)
        self._uniform(
            committed,
            {
                "state": "committed",
                "version": target_version,
                "committed_transfer_id": transfer_id,
            },
            operation="commit_delta",
        )
        await self._resume()
        return committed

    async def _collective_for_client(
        self,
        client: httpx.AsyncClient,
        method: str,
        *args: str,
    ) -> None:
        response = await client.post(
            "/collective_rpc",
            json={"method": method, "args": list(args), "timeout": self._timeout.read},
            timeout=self._timeout,
        )
        response.raise_for_status()
