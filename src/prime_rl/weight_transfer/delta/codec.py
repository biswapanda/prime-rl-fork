from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Literal

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from prime_rl.weight_transfer.delta.protocol import canonical_json

PART_MAGIC = b"PDELTA01"
PART_VERSION = 1
PART_DIGEST_BYTES = 32
DEFAULT_MAX_PART_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_HEADER_BYTES = 16 * 1024 * 1024
EncodingRequest = Literal["auto", "indices", "gap", "dense"]

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "bool": torch.bool,
}


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    return name


def _as_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = _as_cpu_contiguous(tensor)
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = _as_cpu_contiguous(tensor)
    semantic_header = canonical_json(
        {"dtype": _dtype_name(value.dtype), "shape": list(value.shape), "numel": value.numel()}
    )
    digest = hashlib.sha256(semantic_header)
    digest.update(_raw_tensor_bytes(value))
    return f"sha256:{digest.hexdigest()}"


def _changed_positions(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if base.shape != target.shape:
        raise ValueError("base and target shapes differ")
    if base.dtype != target.dtype:
        raise ValueError("base and target dtypes differ")
    base = _as_cpu_contiguous(base).reshape(-1)
    target = _as_cpu_contiguous(target).reshape(-1)
    element_size = base.element_size()
    base_bytes = base.view(torch.uint8).reshape(base.numel(), element_size)
    target_bytes = target.view(torch.uint8).reshape(target.numel(), element_size)
    return torch.nonzero(torch.any(base_bytes != target_bytes, dim=1), as_tuple=False).reshape(-1).to(torch.int64)


def _unsigned_dtype(width: int) -> torch.dtype:
    return {16: torch.uint16, 32: torch.uint32, 64: torch.uint64}[width]


def _minimum_unsigned_width(maximum: int) -> int:
    if maximum <= 2**16 - 1:
        return 16
    if maximum <= 2**32 - 1:
        return 32
    return 64


def encode_gap_positions(positions: torch.Tensor) -> tuple[torch.Tensor, int]:
    positions = _as_cpu_contiguous(positions).to(torch.int64).reshape(-1)
    if positions.numel() and (positions[0] < 0 or torch.any(positions[1:] <= positions[:-1])):
        raise ValueError("positions must be sorted, unique and non-negative")
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.uint16), 16
    gaps = torch.empty_like(positions)
    gaps[0] = positions[0]
    gaps[1:] = positions[1:] - positions[:-1] - 1
    width = _minimum_unsigned_width(int(gaps.max()))
    return gaps.to(_unsigned_dtype(width)), width


def decode_gap_positions(encoded: torch.Tensor, width: int) -> torch.Tensor:
    if width not in (16, 32, 64) or encoded.dtype != _unsigned_dtype(width):
        raise ValueError("gap position width and dtype disagree")
    gaps = _as_cpu_contiguous(encoded).to(torch.int64).reshape(-1)
    if gaps.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    return torch.cumsum(gaps + 1, dim=0) - 1


def _encode_absolute_positions(positions: torch.Tensor) -> tuple[torch.Tensor, int]:
    maximum = int(positions[-1]) if positions.numel() else 0
    width = 32 if maximum <= 2**32 - 1 else 64
    return positions.to(_unsigned_dtype(width)), width


@dataclass(frozen=True, slots=True)
class TensorDelta:
    name: str
    shape: tuple[int, ...]
    dtype: str
    encoding: str
    positions: torch.Tensor | None
    values: torch.Tensor
    target_hash: str
    changed_elements: int
    numel: int

    @classmethod
    def between(
        cls,
        name: str,
        base: torch.Tensor,
        target: torch.Tensor,
        *,
        encoding: EncodingRequest = "auto",
    ) -> TensorDelta:
        if not name:
            raise ValueError("tensor name must be non-empty")
        base = _as_cpu_contiguous(base)
        target = _as_cpu_contiguous(target)
        positions = _changed_positions(base, target)
        flat_target = target.reshape(-1)
        changed_values = flat_target[positions].contiguous()

        absolute, absolute_width = _encode_absolute_positions(positions)
        gaps, gap_width = encode_gap_positions(positions)
        dense_bytes = target.numel() * target.element_size()
        absolute_bytes = absolute.numel() * absolute.element_size() + changed_values.numel() * target.element_size()
        gap_bytes = gaps.numel() * gaps.element_size() + changed_values.numel() * target.element_size()

        if encoding == "dense" or (encoding == "auto" and dense_bytes <= min(absolute_bytes, gap_bytes)):
            selected_encoding = "dense"
            selected_positions = None
            values = target
        elif encoding == "indices" or (encoding == "auto" and absolute_bytes <= gap_bytes):
            selected_encoding = f"indices_u{absolute_width}"
            selected_positions = absolute
            values = changed_values
        elif encoding in ("gap", "auto"):
            selected_encoding = f"gap_u{gap_width}"
            selected_positions = gaps
            values = changed_values
        else:
            raise ValueError(f"unsupported encoding: {encoding}")

        return cls(
            name=name,
            shape=tuple(target.shape),
            dtype=_dtype_name(target.dtype),
            encoding=selected_encoding,
            positions=selected_positions,
            values=values,
            target_hash=tensor_sha256(target),
            changed_elements=positions.numel(),
            numel=target.numel(),
        )

    def decoded_positions(self) -> torch.Tensor | None:
        if self.encoding == "dense":
            return None
        if self.positions is None:
            raise ValueError("sparse delta is missing positions")
        prefix, _, width_text = self.encoding.partition("_u")
        if prefix not in ("indices", "gap") or not width_text.isdigit():
            raise ValueError(f"unsupported encoded positions: {self.encoding}")
        width = int(width_text)
        if prefix == "gap":
            return decode_gap_positions(self.positions, width)
        if self.positions.dtype != _unsigned_dtype(width):
            raise ValueError("absolute position width and dtype disagree")
        return self.positions.to(torch.int64)

    def apply(self, base: torch.Tensor) -> torch.Tensor:
        base = _as_cpu_contiguous(base)
        if tuple(base.shape) != self.shape or _dtype_name(base.dtype) != self.dtype:
            raise ValueError(f"base tensor metadata does not match delta for {self.name}")
        if base.numel() != self.numel:
            raise ValueError(f"base tensor size does not match delta for {self.name}")

        if self.encoding == "dense":
            result = _as_cpu_contiguous(self.values)
            if tuple(result.shape) != self.shape or _dtype_name(result.dtype) != self.dtype:
                raise ValueError(f"dense payload metadata does not match delta for {self.name}")
        else:
            positions = self.decoded_positions()
            assert positions is not None
            if positions.numel() != self.changed_elements or self.values.numel() != self.changed_elements:
                raise ValueError(f"sparse payload count does not match delta for {self.name}")
            if positions.numel() and (
                positions[0] < 0 or positions[-1] >= self.numel or torch.any(positions[1:] <= positions[:-1])
            ):
                raise ValueError(f"sparse positions are invalid for {self.name}")
            if _dtype_name(self.values.dtype) != self.dtype:
                raise ValueError(f"sparse payload dtype does not match delta for {self.name}")
            result = base.clone()
            result.reshape(-1)[positions] = self.values.reshape(-1)

        if tensor_sha256(result) != self.target_hash:
            raise ValueError(f"target hash mismatch after applying delta for {self.name}")
        return result


def encode_part(entries: tuple[TensorDelta, ...]) -> bytes:
    ordered = tuple(sorted(entries, key=lambda entry: entry.name))
    if len({entry.name for entry in ordered}) != len(ordered):
        raise ValueError("part contains duplicate tensor names")

    tensors: dict[str, torch.Tensor] = {}
    descriptors: list[dict[str, object]] = []
    for index, entry in enumerate(ordered):
        value_key = f"e{index:06d}.values"
        position_key = None
        tensors[value_key] = _as_cpu_contiguous(entry.values)
        if entry.positions is not None:
            position_key = f"e{index:06d}.positions"
            tensors[position_key] = _as_cpu_contiguous(entry.positions)
        descriptors.append(
            {
                "name": entry.name,
                "shape": list(entry.shape),
                "dtype": entry.dtype,
                "encoding": entry.encoding,
                "position_key": position_key,
                "value_key": value_key,
                "target_hash": entry.target_hash,
                "changed_elements": entry.changed_elements,
                "numel": entry.numel,
            }
        )

    payload = save_safetensors(tensors, metadata={"format": "prime.delta.v1"})
    header = canonical_json(
        {
            "version": PART_VERSION,
            "entries": descriptors,
            "payload_size": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    prefix = PART_MAGIC + struct.pack(">I", len(header)) + header + payload
    return prefix + hashlib.sha256(prefix).digest()


def decode_part(
    body: bytes,
    *,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> tuple[TensorDelta, ...]:
    minimum_size = len(PART_MAGIC) + 4 + PART_DIGEST_BYTES
    if len(body) < minimum_size:
        raise ValueError("part is truncated")
    if len(body) > max_part_bytes:
        raise ValueError("part exceeds configured byte limit")
    if not body.startswith(PART_MAGIC):
        raise ValueError("part magic is invalid")
    prefix, expected_digest = body[:-PART_DIGEST_BYTES], body[-PART_DIGEST_BYTES:]
    if hashlib.sha256(prefix).digest() != expected_digest:
        raise ValueError("part checksum mismatch")

    header_size = struct.unpack(">I", body[len(PART_MAGIC) : len(PART_MAGIC) + 4])[0]
    if header_size > max_header_bytes:
        raise ValueError("part header exceeds configured byte limit")
    header_start = len(PART_MAGIC) + 4
    header_end = header_start + header_size
    if header_end > len(prefix):
        raise ValueError("part header is truncated")
    try:
        header = json.loads(body[header_start:header_end])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("part header is invalid JSON") from error
    if not isinstance(header, dict) or set(header) != {"version", "entries", "payload_size", "payload_sha256"}:
        raise ValueError("part header has unknown or missing fields")
    if header["version"] != PART_VERSION or not isinstance(header["entries"], list):
        raise ValueError("part version or entries are invalid")

    payload = body[header_end:-PART_DIGEST_BYTES]
    if header["payload_size"] != len(payload):
        raise ValueError("part payload size mismatch")
    if header["payload_sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("part payload checksum mismatch")
    try:
        tensors = load_safetensors(payload)
    except Exception as error:
        raise ValueError("part safetensors payload is invalid") from error

    entries: list[TensorDelta] = []
    used_keys: set[str] = set()
    for descriptor in header["entries"]:
        expected_fields = {
            "name",
            "shape",
            "dtype",
            "encoding",
            "position_key",
            "value_key",
            "target_hash",
            "changed_elements",
            "numel",
        }
        if not isinstance(descriptor, dict) or set(descriptor) != expected_fields:
            raise ValueError("part tensor descriptor is invalid")
        value_key = descriptor["value_key"]
        position_key = descriptor["position_key"]
        if not isinstance(value_key, str) or value_key not in tensors:
            raise ValueError("part tensor values are missing")
        used_keys.add(value_key)
        positions = None
        if position_key is not None:
            if not isinstance(position_key, str) or position_key not in tensors:
                raise ValueError("part tensor positions are missing")
            used_keys.add(position_key)
            positions = tensors[position_key]
        entry = TensorDelta(
            name=descriptor["name"],
            shape=tuple(descriptor["shape"]),
            dtype=descriptor["dtype"],
            encoding=descriptor["encoding"],
            positions=positions,
            values=tensors[value_key],
            target_hash=descriptor["target_hash"],
            changed_elements=descriptor["changed_elements"],
            numel=descriptor["numel"],
        )
        if not entry.name or entry.dtype not in _DTYPES:
            raise ValueError("part tensor descriptor has invalid name or dtype")
        entries.append(entry)
    if used_keys != set(tensors):
        raise ValueError("part contains unreferenced tensors")
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("part contains duplicate tensor names")
    return tuple(entries)
