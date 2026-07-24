import hashlib
import struct

import pytest
import torch

from prime_rl.weight_transfer.delta.codec import (
    PART_MAGIC,
    TensorDelta,
    decode_gap_positions,
    decode_part,
    encode_gap_positions,
    encode_part,
    tensor_sha256,
)


def test_bitwise_diff_preserves_signed_zero_and_nan_payloads() -> None:
    base_bits = torch.tensor([0x00000000, 0x7FC00001, 0x3F800000], dtype=torch.int32)
    target_bits = torch.tensor([-0x80000000, 0x7FC00002, 0x3F800000], dtype=torch.int32)
    base = base_bits.view(torch.float32)
    target = target_bits.view(torch.float32)

    delta = TensorDelta.between("edge", base, target, encoding="indices")
    restored = delta.apply(base)

    assert delta.changed_elements == 2
    assert torch.equal(restored.view(torch.int32), target_bits)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1, 3, 65_538], dtype=torch.int64),
        torch.tensor([2**32 + 3, 2**32 + 8], dtype=torch.int64),
    ],
)
def test_gap_positions_round_trip(positions: torch.Tensor) -> None:
    encoded, width = encode_gap_positions(positions)

    assert width in (16, 32, 64)
    assert torch.equal(decode_gap_positions(encoded, width), positions)


def test_codec_selects_dense_fallback_for_dense_change() -> None:
    base = torch.zeros(64, dtype=torch.bfloat16)
    target = torch.ones(64, dtype=torch.bfloat16)

    delta = TensorDelta.between("dense", base, target, encoding="auto")

    assert delta.encoding == "dense"
    assert torch.equal(delta.apply(base), target)


def test_part_round_trip_with_mixed_dtypes() -> None:
    bf16_base = torch.zeros(32, dtype=torch.bfloat16)
    bf16_target = bf16_base.clone()
    bf16_target[7] = 2
    fp32_base = torch.arange(16, dtype=torch.float32)
    fp32_target = fp32_base.clone()
    fp32_target[3] = -4

    encoded = encode_part(
        (
            TensorDelta.between("bf16", bf16_base, bf16_target),
            TensorDelta.between("fp32", fp32_base, fp32_target),
        )
    )
    decoded = {entry.name: entry for entry in decode_part(encoded)}

    assert encoded.startswith(PART_MAGIC)
    assert torch.equal(decoded["bf16"].apply(bf16_base), bf16_target)
    assert torch.equal(decoded["fp32"].apply(fp32_base), fp32_target)
    assert decoded["bf16"].target_hash == tensor_sha256(bf16_target)
    assert decoded["fp32"].target_hash == tensor_sha256(fp32_target)


def test_part_rejects_corrupt_body_before_decode() -> None:
    delta = TensorDelta.between("weight", torch.zeros(8), torch.ones(8))
    encoded = bytearray(encode_part((delta,)))
    encoded[-1] ^= 0xFF

    with pytest.raises(ValueError, match="checksum"):
        decode_part(bytes(encoded))


def test_part_rejects_oversized_declared_header() -> None:
    prefix = PART_MAGIC + struct.pack(">I", 2**31)
    encoded = prefix + hashlib.sha256(prefix).digest()

    with pytest.raises(ValueError, match="header"):
        decode_part(encoded, max_part_bytes=1024)
