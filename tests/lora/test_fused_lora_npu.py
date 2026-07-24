# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NPU-side sanity tests for the simplified ``fused_moe_lora`` family.

These tests exercise the three registered operators in
``vllm/lora/ops/triton_ops/fused_moe_lora_op_npu.py`` on Ascend NPU:

  1. ``fused_moe_lora``        — full shrink + expand fused op
  2. ``fused_moe_lora_shrink`` — shrink-only op (hidden → lora_a → rank)
  3. ``fused_moe_lora_expand`` — expand-only op (rank → lora_b → output)

The tests mirror the style of ``test_lora_expand_npu.py`` and use the
*naive* dispatch path (``sorted_token_ids=None``) that the NPU kernel
implements directly.
"""

import random
from threading import Lock

import pytest
import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_op_npu import (
    _LORA_PTR_DICT,
    fused_moe_lora,
    fused_moe_lora_expand,
    fused_moe_lora_shrink,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .utils import assert_close

# ---------------------------------------------------------------------------
# Device / triton availability
# ---------------------------------------------------------------------------
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:{0}"]

supports_triton = current_platform.is_cuda_alike() or current_platform.is_xpu()
if not supports_triton:
    try:
        import triton  # noqa: F401

        supports_triton = True
    except ImportError:
        supports_triton = False

pytestmark = pytest.mark.skipif(
    not supports_triton,
    reason="fused_moe_lora triton kernels require a triton-capable device "
    "(CUDA / XPU / Ascend NPU with triton support).",
)


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    """Ensure torch's default device is restored between tests."""
    pass


# Reuse the same lock pattern as test_punica_ops.py to avoid stale pointer
# cache issues across tests.
_dict_lock = Lock()

# ---------------------------------------------------------------------------
# Kernel config
# ---------------------------------------------------------------------------
SHRINK_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "NUM_WARPS": 4,
    "NUM_STAGES": 3,
    "SPLIT_K": 1,
}

EXPAND_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 16,
    "GROUP_SIZE_M": 1,
    "NUM_WARPS": 4,
    "NUM_STAGES": 3,
    "SPLIT_K": 1,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def assign_loras_block_aligned(
    num_tokens: int, block_size: int, max_loras: int
):
    """Assign a random LoRA id to each block of ``block_size`` tokens.

    The NPU kernel resolves one LoRA id per tile, so all tokens that fall
    into the same tile must share the same LoRA.  Calling code must pass
    ``block_size = BLOCK_SIZE_M // top_k_num`` for the relevant kernel.
    """
    assert block_size > 0 and max_loras > 0
    assert num_tokens % block_size == 0, (
        f"num_tokens={num_tokens} must be divisible by block_size={block_size}"
    )

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)
    for block_start in range(0, num_tokens, block_size):
        lora_id = random.randint(0, max_loras - 1)
        token_lora_mapping[block_start : block_start + block_size] = lora_id
    return token_lora_mapping


def assign_experts_block_aligned(
    num_tokens: int,
    num_experts: int,
    top_k_num: int,
    block_size_m: int,
):
    """Assign experts to tokens so that each kernel tile sees a single expert.

    The NPU kernel reads ``expert_ids[pid_m]`` once per tile, so all tokens
    that fall into the same tile must share the same expert.  For
    ``top_k_num > 1`` every row of ``topk_ids`` stores the same expert id
    (duplicated ``top_k_num`` times) so the reference implementation stays
    consistent with the tile-level lookup.
    """
    assert block_size_m % top_k_num == 0, (
        f"BLOCK_SIZE_M={block_size_m} must be divisible by top_k_num={top_k_num}"
    )
    tokens_per_tile = block_size_m // top_k_num
    num_tiles = (num_tokens * top_k_num + block_size_m - 1) // block_size_m

    topk_ids = torch.zeros(num_tokens, top_k_num, dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, top_k_num, dtype=torch.float32)
    expert_ids = torch.zeros(num_tiles, dtype=torch.int32)

    for tile in range(num_tiles):
        token_start = tile * tokens_per_tile
        token_end = min(token_start + tokens_per_tile, num_tokens)
        if token_start < num_tokens:
            expert = random.randint(0, num_experts - 1)
            expert_ids[tile] = expert
            topk_ids[token_start:token_end, :] = expert

    return topk_ids, topk_weights, expert_ids


def build_meta(max_loras: int):
    """Build ``adapter_enabled`` / ``num_active_loras`` for the naive path."""
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    num_active_loras = torch.tensor(
        [max_loras + 1], dtype=torch.int32, device="cpu"
    )
    return adapter_enabled, num_active_loras


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------
def ref_fused_moe_lora_shrink(
    hidden_states: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    top_k_num: int,
) -> torch.Tensor:
    """Reference for ``fused_moe_lora_shrink``: hidden @ lora_a.T per pair."""
    num_slices = len(lora_a_stacked)
    num_tokens, _ = hidden_states.shape
    rank = lora_a_stacked[0].shape[2]
    cache = torch.zeros(
        num_slices, num_tokens, top_k_num, rank, dtype=hidden_states.dtype
    )

    for i in range(num_tokens):
        lora_idx = int(token_lora_mapping[i].item())
        expert = int(topk_ids[i, 0].item())
        h = hidden_states[i].float()
        for s in range(num_slices):
            a = lora_a_stacked[s][lora_idx, expert].float()
            for x in range(top_k_num):
                cache[s, i, x] = (h @ a.T).to(hidden_states.dtype)
    return cache


def ref_fused_moe_lora_expand(
    a_intermediate_cache1: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    mul_routed_weight: bool,
    output_initial: torch.Tensor,
) -> torch.Tensor:
    """Reference for ``fused_moe_lora_expand`` with ``ADD_INPUTS=True``."""
    num_slices = len(lora_b_stacked)
    num_tokens = a_intermediate_cache1.shape[1]
    top_k_num = a_intermediate_cache1.shape[2]
    n_per_slice = lora_b_stacked[0].shape[2]
    output = output_initial.clone()

    for i in range(num_tokens):
        lora_idx = int(token_lora_mapping[i].item())
        expert = int(topk_ids[i, 0].item())
        for x in range(top_k_num):
            w = float(topk_weights[i, x].item()) if mul_routed_weight else 1.0
            for s in range(num_slices):
                b = lora_b_stacked[s][lora_idx, expert].float()
                a = a_intermediate_cache1[s, i, x].float()
                delta = (a @ b.T).to(output.dtype) * w
                output[
                    i, x, s * n_per_slice : (s + 1) * n_per_slice
                ] += delta
    return output


def ref_fused_moe_lora(
    hidden_states: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    """Reference for ``fused_moe_lora``: hidden @ A.T @ B.T per pair."""
    num_slices = len(lora_a_stacked)
    num_tokens = hidden_states.shape[0]
    top_k_num = topk_ids.shape[1]
    n_per_slice = lora_b_stacked[0].shape[2]
    output = torch.zeros(
        num_tokens, top_k_num, num_slices * n_per_slice, dtype=hidden_states.dtype
    )

    for i in range(num_tokens):
        lora_idx = int(token_lora_mapping[i].item())
        expert = int(topk_ids[i, 0].item())
        h = hidden_states[i].float()
        for x in range(top_k_num):
            w = float(topk_weights[i, x].item()) if mul_routed_weight else 1.0
            for s in range(num_slices):
                a = lora_a_stacked[s][lora_idx, expert].float()
                b = lora_b_stacked[s][lora_idx, expert].float()
                delta = (h @ a.T @ b.T).to(output.dtype) * w
                output[
                    i, x, s * n_per_slice : (s + 1) * n_per_slice
                ] = delta
    return output


# ---------------------------------------------------------------------------
# Kernel call wrappers (naive path)
# ---------------------------------------------------------------------------
def call_fused_moe_lora_shrink(
    hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    max_loras: int,
    a_intermediate_cache1: torch.Tensor,
) -> None:
    """Call the ``fused_moe_lora_shrink`` op via the naive dispatch path."""
    device = hidden_states.device
    M = hidden_states.shape[0]
    K = hidden_states.shape[1]
    N = a_intermediate_cache1.shape[-1]
    num_tokens_internal = M * top_k_num
    EM = num_tokens_internal
    num_slices = len(lora_a_stacked)
    num_experts = lora_a_stacked[0].shape[1]
    adapter_enabled, num_active_loras = build_meta(max_loras)

    with _dict_lock:
        _LORA_PTR_DICT.clear()
        fused_moe_lora_shrink(
            a_intermediate_cache1,
            hidden_states,
            lora_a_stacked,
            topk_weights,
            None,  # sorted_token_ids — naive path
            expert_ids,
            None,  # num_tokens_post_padded — naive path
            token_lora_mapping,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            N,
            M,
            EM,
            K,
            num_tokens_internal,
            num_experts,
            num_slices,
            SHRINK_CONFIG["BLOCK_SIZE_M"],
            SHRINK_CONFIG["BLOCK_SIZE_N"],
            SHRINK_CONFIG["BLOCK_SIZE_K"],
            SHRINK_CONFIG["GROUP_SIZE_M"],
            SHRINK_CONFIG["NUM_WARPS"],
            SHRINK_CONFIG["NUM_STAGES"],
            SHRINK_CONFIG["SPLIT_K"],
            num_active_loras,
            False,  # mul_routed_weight
            False,  # use_gdc
            False,  # use_tma
        )


def call_fused_moe_lora_expand(
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    max_loras: int,
    output: torch.Tensor,
    mul_routed_weight: bool = False,
) -> None:
    """Call the ``fused_moe_lora_expand`` op via the naive dispatch path."""
    device = a_intermediate_cache1.device
    M = a_intermediate_cache1.shape[1]
    K = a_intermediate_cache1.shape[-1]
    num_tokens_internal = M * top_k_num
    EM = num_tokens_internal
    num_slices = len(lora_b_stacked)
    num_experts = lora_b_stacked[0].shape[1]
    n_per_slice = lora_b_stacked[0].shape[2]
    adapter_enabled, num_active_loras = build_meta(max_loras)

    with _dict_lock:
        _LORA_PTR_DICT.clear()
        fused_moe_lora_expand(
            output,
            a_intermediate_cache1,
            lora_b_stacked,
            topk_weights,
            None,  # sorted_token_ids — naive path
            expert_ids,
            None,  # num_tokens_post_padded — naive path
            token_lora_mapping,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            n_per_slice,
            M,
            EM,
            K,
            num_tokens_internal,
            num_experts,
            num_slices,
            K,  # max_lora_rank
            n_per_slice,  # w1_output_dim_size
            EXPAND_CONFIG["BLOCK_SIZE_M"],
            EXPAND_CONFIG["BLOCK_SIZE_N"],
            EXPAND_CONFIG["BLOCK_SIZE_K"],
            EXPAND_CONFIG["GROUP_SIZE_M"],
            EXPAND_CONFIG["NUM_WARPS"],
            EXPAND_CONFIG["NUM_STAGES"],
            EXPAND_CONFIG["SPLIT_K"],
            num_active_loras,
            mul_routed_weight,
            0,  # offset
            False,  # use_gdc
            False,  # use_tma
        )


def call_fused_moe_lora(
    hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    max_loras: int,
    max_lora_rank: int,
    output: torch.Tensor,
    mul_routed_weight: bool = False,
) -> None:
    """Call the full ``fused_moe_lora`` op via the naive dispatch path."""
    adapter_enabled, num_active_loras = build_meta(max_loras)

    with _dict_lock:
        _LORA_PTR_DICT.clear()
        fused_moe_lora(
            output,
            hidden_states,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            None,  # sorted_token_ids — naive path
            expert_ids,
            None,  # num_tokens_post_padded — naive path
            token_lora_mapping,
            max_lora_rank,
            top_k_num,
            lora_ids,
            num_active_loras,
            adapter_enabled,
            SHRINK_CONFIG["BLOCK_SIZE_M"],
            SHRINK_CONFIG["BLOCK_SIZE_N"],
            SHRINK_CONFIG["BLOCK_SIZE_K"],
            SHRINK_CONFIG["GROUP_SIZE_M"],
            SHRINK_CONFIG["NUM_WARPS"],
            SHRINK_CONFIG["NUM_STAGES"],
            SHRINK_CONFIG["SPLIT_K"],
            EXPAND_CONFIG["BLOCK_SIZE_M"],
            EXPAND_CONFIG["BLOCK_SIZE_N"],
            EXPAND_CONFIG["BLOCK_SIZE_K"],
            EXPAND_CONFIG["GROUP_SIZE_M"],
            EXPAND_CONFIG["NUM_WARPS"],
            EXPAND_CONFIG["NUM_STAGES"],
            EXPAND_CONFIG["SPLIT_K"],
            mul_routed_weight,
            False,  # fully_sharded
            0,  # offset
            True,  # add_inputs
        )


# ---------------------------------------------------------------------------
# Shared check helpers
# ---------------------------------------------------------------------------
def _prepare_lora_ids(token_lora_mapping: torch.Tensor, max_loras: int):
    active_lora_ids = torch.unique(token_lora_mapping, sorted=True)
    lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    lora_ids[: active_lora_ids.size(0)].copy_(active_lora_ids)
    return lora_ids


DTYPES = [torch.float16, torch.bfloat16]
SEED = [0]


def check_fused_moe_lora_shrink(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_loras: int,
    K: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> None:
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)
    random.seed(seed)

    lora_block_size = SHRINK_CONFIG["BLOCK_SIZE_M"] // top_k_num
    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, SHRINK_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = assign_loras_block_aligned(
        num_tokens, lora_block_size, max_loras
    )
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    lora_a_stacked = [
        torch.rand(max_loras, num_experts, max_lora_rank, K, dtype=dtype)
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand(num_tokens, K, dtype=dtype)
    cache = torch.zeros(
        num_slices, num_tokens, top_k_num, max_lora_rank, dtype=dtype
    )

    expert_ids = expert_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    lora_ids = lora_ids.to(device)
    hidden_states = hidden_states.to(device)
    lora_a_stacked = [w.to(device) for w in lora_a_stacked]
    cache = cache.to(device)

    call_fused_moe_lora_shrink(
        hidden_states,
        lora_a_stacked,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        max_loras,
        cache,
    )

    ref = ref_fused_moe_lora_shrink(
        hidden_states.cpu(),
        token_lora_mapping.cpu(),
        topk_ids,
        [w.cpu() for w in lora_a_stacked],
        top_k_num,
    )
    assert_close(cache, ref.to(device))


def check_fused_moe_lora_expand(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_loras: int,
    N: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    mul_routed_weight: bool = False,
) -> None:
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)
    random.seed(seed)

    lora_block_size = EXPAND_CONFIG["BLOCK_SIZE_M"] // top_k_num
    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, EXPAND_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = assign_loras_block_aligned(
        num_tokens, lora_block_size, max_loras
    )
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    lora_b_stacked = [
        torch.rand(max_loras, num_experts, N // num_slices, max_lora_rank, dtype=dtype)
        for _ in range(num_slices)
    ]
    cache = torch.rand(
        num_slices, num_tokens, top_k_num, max_lora_rank, dtype=dtype
    )
    output = torch.rand(num_tokens, top_k_num, N, dtype=dtype)
    output_initial = output.clone()

    expert_ids = expert_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    lora_ids = lora_ids.to(device)
    cache = cache.to(device)
    lora_b_stacked = [w.to(device) for w in lora_b_stacked]
    output = output.to(device)

    call_fused_moe_lora_expand(
        cache,
        lora_b_stacked,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        max_loras,
        output,
        mul_routed_weight,
    )

    ref = ref_fused_moe_lora_expand(
        cache.cpu(),
        token_lora_mapping.cpu(),
        topk_ids,
        [w.cpu() for w in lora_b_stacked],
        topk_weights.cpu(),
        mul_routed_weight,
        output_initial.cpu(),
    )
    assert_close(output, ref.to(device))


def check_fused_moe_lora_full(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_loras: int,
    K: int,
    N: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    mul_routed_weight: bool = False,
) -> None:
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)
    random.seed(seed)

    lora_block_size = SHRINK_CONFIG["BLOCK_SIZE_M"] // top_k_num
    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, SHRINK_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = assign_loras_block_aligned(
        num_tokens, lora_block_size, max_loras
    )
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    lora_a_stacked = [
        torch.rand(max_loras, num_experts, max_lora_rank, K, dtype=dtype)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(max_loras, num_experts, N // num_slices, max_lora_rank, dtype=dtype)
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand(num_tokens, K, dtype=dtype)
    output = torch.zeros(num_tokens, top_k_num, N, dtype=dtype)

    expert_ids = expert_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    lora_ids = lora_ids.to(device)
    hidden_states = hidden_states.to(device)
    lora_a_stacked = [w.to(device) for w in lora_a_stacked]
    lora_b_stacked = [w.to(device) for w in lora_b_stacked]
    output = output.to(device)

    call_fused_moe_lora(
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        max_loras,
        max_lora_rank,
        output,
        mul_routed_weight,
    )

    ref = ref_fused_moe_lora(
        hidden_states.cpu(),
        token_lora_mapping.cpu(),
        topk_ids,
        [w.cpu() for w in lora_a_stacked],
        [w.cpu() for w in lora_b_stacked],
        topk_weights.cpu(),
        mul_routed_weight,
    )
    assert_close(output, ref.to(device))


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_shrink_smoke(device: str):
    """Smallest reasonable shrink config."""
    check_fused_moe_lora_shrink(
        num_tokens=16,
        top_k_num=1,
        num_experts=2,
        max_loras=4,
        K=128,
        max_lora_rank=16,
        num_slices=1,
        dtype=torch.float16,
        device=device,
        seed=0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_expand_smoke(device: str):
    """Smallest reasonable expand config."""
    check_fused_moe_lora_expand(
        num_tokens=16,
        top_k_num=1,
        num_experts=2,
        max_loras=4,
        N=128,
        max_lora_rank=16,
        num_slices=1,
        dtype=torch.float16,
        device=device,
        seed=0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_smoke(device: str):
    """Smallest reasonable full fused op config."""
    check_fused_moe_lora_full(
        num_tokens=16,
        top_k_num=1,
        num_experts=2,
        max_loras=4,
        K=128,
        N=128,
        max_lora_rank=16,
        num_slices=1,
        dtype=torch.float16,
        device=device,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Parameterized correctness tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", [16, 32])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [2, 8])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_shrink_param_variations(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """Vary token count, top-k, experts, rank, slices and dtype for shrink."""
    check_fused_moe_lora_shrink(
        num_tokens=num_tokens,
        top_k_num=top_k_num,
        num_experts=num_experts,
        max_loras=4,
        K=128,
        max_lora_rank=max_lora_rank,
        num_slices=num_slices,
        dtype=dtype,
        device=device,
        seed=seed,
    )


@pytest.mark.parametrize("num_tokens", [16, 32])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [2, 8])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_expand_param_variations(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """Vary token count, top-k, experts, rank, slices and dtype for expand."""
    check_fused_moe_lora_expand(
        num_tokens=num_tokens,
        top_k_num=top_k_num,
        num_experts=num_experts,
        max_loras=4,
        N=128,
        max_lora_rank=max_lora_rank,
        num_slices=num_slices,
        dtype=dtype,
        device=device,
        seed=seed,
    )


@pytest.mark.parametrize("num_tokens", [16, 32])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [2, 8])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_full_param_variations(
    num_tokens: int,
    top_k_num: int,
    num_experts: int,
    max_lora_rank: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """Vary token count, top-k, experts, rank, slices and dtype for full op."""
    check_fused_moe_lora_full(
        num_tokens=num_tokens,
        top_k_num=top_k_num,
        num_experts=num_experts,
        max_loras=4,
        K=128,
        N=128,
        max_lora_rank=max_lora_rank,
        num_slices=num_slices,
        dtype=dtype,
        device=device,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Routed weight tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mul_routed_weight", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_expand_routed_weight(
    mul_routed_weight: bool,
    dtype: torch.dtype,
    device: str,
):
    """Verify ``mul_routed_weight`` scaling in the expand kernel."""
    check_fused_moe_lora_expand(
        num_tokens=32,
        top_k_num=1,
        num_experts=2,
        max_loras=4,
        N=128,
        max_lora_rank=16,
        num_slices=1,
        dtype=dtype,
        device=device,
        seed=0,
        mul_routed_weight=mul_routed_weight,
    )


@pytest.mark.parametrize("mul_routed_weight", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_routed_weight(
    mul_routed_weight: bool,
    dtype: torch.dtype,
    device: str,
):
    """Verify ``mul_routed_weight`` scaling end-to-end."""
    check_fused_moe_lora_full(
        num_tokens=32,
        top_k_num=1,
        num_experts=2,
        max_loras=4,
        K=128,
        N=128,
        max_lora_rank=16,
        num_slices=1,
        dtype=dtype,
        device=device,
        seed=0,
        mul_routed_weight=mul_routed_weight,
    )


# ---------------------------------------------------------------------------
# No-LoRA early-exit tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_shrink_no_lora_early_exit(device: str):
    """When every token maps to lora_id -1 the cache must stay untouched."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 1, 2, 4
    K, max_lora_rank, num_slices = 128, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, SHRINK_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = torch.full((num_tokens,), -1, dtype=torch.int32)
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    lora_a_stacked = [
        torch.rand(max_loras, num_experts, max_lora_rank, K, dtype=dtype).to(device)
    ]
    hidden_states = torch.rand(num_tokens, K, dtype=dtype).to(device)
    cache = torch.zeros(
        num_slices, num_tokens, top_k_num, max_lora_rank, dtype=dtype
    ).to(device)
    snapshot = cache.clone()

    call_fused_moe_lora_shrink(
        hidden_states,
        lora_a_stacked,
        topk_weights.to(device),
        expert_ids.to(device),
        token_lora_mapping.to(device),
        top_k_num,
        lora_ids.to(device),
        max_loras,
        cache,
    )

    assert torch.equal(cache, snapshot), (
        "fused_moe_lora_shrink modified the cache despite no active LoRA"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_expand_no_lora_early_exit(device: str):
    """When every token maps to lora_id -1 the output must stay untouched."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 1, 2, 4
    N, max_lora_rank, num_slices = 128, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, EXPAND_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = torch.full((num_tokens,), -1, dtype=torch.int32)
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    cache = torch.rand(
        num_slices, num_tokens, top_k_num, max_lora_rank, dtype=dtype
    ).to(device)
    lora_b_stacked = [
        torch.rand(max_loras, num_experts, N, max_lora_rank, dtype=dtype).to(device)
    ]
    output = torch.rand(num_tokens, top_k_num, N, dtype=dtype).to(device)
    snapshot = output.clone()

    call_fused_moe_lora_expand(
        cache,
        lora_b_stacked,
        topk_weights.to(device),
        expert_ids.to(device),
        token_lora_mapping.to(device),
        top_k_num,
        lora_ids.to(device),
        max_loras,
        output,
        mul_routed_weight=False,
    )

    assert torch.equal(output, snapshot), (
        "fused_moe_lora_expand modified the output despite no active LoRA"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_no_lora_early_exit(device: str):
    """When every token maps to lora_id -1 the full op must leave output zero."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 1, 2, 4
    K, N, max_lora_rank, num_slices = 128, 128, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, expert_ids = assign_experts_block_aligned(
        num_tokens, num_experts, top_k_num, SHRINK_CONFIG["BLOCK_SIZE_M"]
    )
    token_lora_mapping = torch.full((num_tokens,), -1, dtype=torch.int32)
    lora_ids = _prepare_lora_ids(token_lora_mapping, max_loras)

    lora_a_stacked = [
        torch.rand(max_loras, num_experts, max_lora_rank, K, dtype=dtype).to(device)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(max_loras, num_experts, N, max_lora_rank, dtype=dtype).to(device)
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand(num_tokens, K, dtype=dtype).to(device)
    output = torch.zeros(num_tokens, top_k_num, N, dtype=dtype).to(device)
    snapshot = output.clone()

    call_fused_moe_lora(
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights.to(device),
        expert_ids.to(device),
        token_lora_mapping.to(device),
        top_k_num,
        lora_ids.to(device),
        max_loras,
        max_lora_rank,
        output,
        mul_routed_weight=False,
    )

    assert torch.equal(output, snapshot), (
        "fused_moe_lora modified the output despite no active LoRA"
    )
