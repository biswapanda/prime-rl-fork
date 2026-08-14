import importlib.util
from pathlib import Path

RANKS_PATH = Path(__file__).parents[3] / "src/prime_rl/inference/vllm/worker/ranks.py"
SPEC = importlib.util.spec_from_file_location("prime_rl_worker_ranks", RANKS_PATH)
assert SPEC and SPEC.loader
ranks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranks)


def test_engine_spans_produce_unique_static_collective_ranks():
    first = {
        ranks.global_inference_rank(
            rank_offset=0,
            data_parallel_index=0,
            data_parallel_size=1,
            worker_rank=tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=6,
            engine_world_size=2,
        )
        for tp_rank in range(2)
    }
    second = {
        ranks.global_inference_rank(
            rank_offset=2,
            data_parallel_index=dp_index,
            data_parallel_size=1,
            worker_rank=tp_rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            inference_world_size=6,
            engine_world_size=4,
        )
        for dp_index in range(2)
        for tp_rank in range(2)
    }

    assert first | second == set(range(6))
