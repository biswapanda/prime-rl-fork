import sys
from types import ModuleType
from unittest.mock import Mock, patch

from prime_rl.configs.trainer import DynamoWeightBroadcastConfig, NCCLWeightBroadcastConfig
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast


def dynamo_nccl_config(*, quantized: bool) -> NCCLWeightBroadcastConfig:
    return NCCLWeightBroadcastConfig(
        quantize_in_weight_transfer=quantized,
        dynamo=DynamoWeightBroadcastConfig(
            discovery_url="http://frontend:8001",
            model_name="Qwen/Qwen3-0.6B",
        ),
    )


def test_standard_dynamo_nccl_uses_native_transfer(tmp_path):
    expected = Mock()
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "prime_rl.trainer.rl.broadcast.dynamo_nccl.DynamoNCCLWeightBroadcast",
            return_value=expected,
        ),
    ):
        actual = setup_weight_broadcast(tmp_path, dynamo_nccl_config(quantized=False), Mock())

    assert actual is expected


def test_quantized_dynamo_nccl_uses_prime_extension_transfer(tmp_path):
    expected = Mock()
    nccl_module = ModuleType("prime_rl.trainer.rl.broadcast.nccl")
    nccl_module.NCCLWeightBroadcast = Mock(return_value=expected)
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch.dict(sys.modules, {"prime_rl.trainer.rl.broadcast.nccl": nccl_module}),
    ):
        actual = setup_weight_broadcast(tmp_path, dynamo_nccl_config(quantized=True), Mock())

    assert actual is expected
