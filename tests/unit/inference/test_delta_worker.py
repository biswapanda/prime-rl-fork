import json
from types import SimpleNamespace

import pytest
from torch import nn

from prime_rl.inference.vllm.worker.delta import (
    DeltaWeightUpdateWorker,
    _configured_filesystem_root,
    _configured_s3_value,
)


def worker() -> DeltaWeightUpdateWorker:
    instance = DeltaWeightUpdateWorker()
    instance.model_runner = SimpleNamespace(model=nn.Linear(2, 2, bias=False))
    return instance


def test_worker_initialization_accepts_only_the_versioned_identity():
    instance = worker()
    payload = {
        "model_id": "Qwen/Qwen3-0.6B",
        "checkpoint_revision": "revision",
        "baseline_fingerprint": "sha256:" + "1" * 64,
        "tensor_table_hash": "sha256:" + "2" * 64,
        "version": 0,
    }

    status = instance.initialize_delta_runtime(json.dumps(payload))
    assert status["initialized"] is True
    assert status["version"] == 0

    with pytest.raises(ValueError, match="unknown or missing"):
        instance.initialize_delta_runtime(json.dumps(payload | {"extra": True}))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"transport": "unknown"},
        {
            "transport": "filesystem",
            "root": "/tmp/deltas",
            "run_id": "run",
            "transfer_id": "transfer",
            "extra": "rejected",
        },
    ],
)
def test_worker_rejects_malformed_stage_source(payload):
    with pytest.raises(ValueError):
        worker().stage_delta(json.dumps(payload))


def test_filesystem_delta_source_must_stay_under_configured_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("PRIME_DELTA_FILESYSTEM_ROOT", str(allowed))

    assert _configured_filesystem_root(str(allowed / "run")) == allowed / "run"
    with pytest.raises(ValueError, match="outside"):
        _configured_filesystem_root(str(tmp_path / "other"))


def test_s3_delta_source_must_match_startup_configuration(monkeypatch):
    monkeypatch.setenv("PRIME_DELTA_S3_ENDPOINT", "http://rustfs:9000")

    assert (
        _configured_s3_value(
            {"endpoint_url": "http://rustfs:9000/"},
            "endpoint_url",
            "PRIME_DELTA_S3_ENDPOINT",
        )
        == "http://rustfs:9000"
    )
    with pytest.raises(ValueError, match="does not match"):
        _configured_s3_value(
            {"endpoint_url": "http://attacker.invalid"},
            "endpoint_url",
            "PRIME_DELTA_S3_ENDPOINT",
        )
