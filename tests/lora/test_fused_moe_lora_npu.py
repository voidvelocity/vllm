# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NPU-side sanity tests for the triton ``fused_moe_lora`` family of kernels.

These tests verify that the three registered operators in
``vllm/lora/ops/triton_ops/fused_moe_lora_op.py`` can run normally on NPU
(Ascend) hardware and produce numerically correct results when compared
with PyTorch reference implementations:

  1. ``fused_moe_lora``        — the full shrink + expand fused op
  2. ``fused_moe_lora_shrink`` — the shrink-only op (hidden → lora_a → rank)
  3. ``fused_moe_lora_expand`` — the expand-only op (rank → lora_b → output)

fp8 variants are intentionally excluded.

NOTE: All tests use the *naive* dispatch path (``sorted_token_ids=None``)
to avoid depending on the ``moe_lora_align_block_size`` C++ op, which is not
registered on NPU. The naive path is natively supported by all three
kernels via the ``naive_block_assignment`` flag.
"""

import random
from threading import Lock

import pytest
import torch

from vllm.lora.ops.triton_ops import (
    fused_moe_lora,
    fused_moe_lora_expand,
    fused_moe_lora_shrink,
)
from vllm.lora.ops.triton_ops.fused_moe_lora_op import _LORA_PTR_DICT
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

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
# Small helpers (mirroring test_fused_moe_lora_kernel.py)
# ---------------------------------------------------------------------------
def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    """Split ``num_tokens`` into ``num_sequences`` sequences; each sequence
    gets a single random LoRA id applied to all its tokens."""
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences

    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)
    start = 0
    for seq_idx in range(num_sequences):
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)
        lora_id = random.randint(0, max_loras - 1)
        token_lora_mapping[start:end] = lora_id
        start = end
    return token_lora_mapping


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int):
    """For each token, pick ``top_k_num`` distinct experts with normalized
    random weights summing to 1."""
    assert top_k_num <= num_experts
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected
    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
    return expert_indices, expert_weights


def sample_data(num_tokens, num_sequences, max_loras, num_experts, top_k_num):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num
    )
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    active_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    lora_ids = torch.unique(token_lora_mapping, sorted=True)
    active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
    return topk_ids, topk_weights, token_lora_mapping, active_lora_ids


def _build_naive_meta(max_loras):
    """Build metadata for the naive dispatch path (sorted_token_ids=None).

    In naive mode the kernel uses ``expert_ids = topk_ids.reshape(-1)`` and
    does not require the ``moe_lora_align_block_size`` C++ op.
    """
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    # In naive mode grid_lora_dim is always 1 regardless of num_active_loras,
    # but we keep max_loras+1 for consistency with the kernel's adapter_enabled
    # lookup (index max_loras is the "no-lora" slot).
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")
    return adapter_enabled, num_active_loras


# ---------------------------------------------------------------------------
# Kernel config (matches the defaults used in test_fused_moe_lora_kernel.py)
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
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "NUM_WARPS": 4,
    "NUM_STAGES": 3,
    "SPLIT_K": 1,
}


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------
def use_torch_full(hidden_states, token_lora_mapping, topk_ids,
                   lora_a_stacked, lora_b_stacked, top_k_num, num_slices=1):
    """Reference for ``fused_moe_lora``: hidden @ lora_a.T @ lora_b.T per
    token-expert pair, with slices concatenated along the output dim."""
    outputs = []
    for i in range(hidden_states.shape[0]):
        slice_tensors = []
        for slice_id in range(num_slices):
            lora_idx = token_lora_mapping[i]
            expert_ids = topk_ids[i]
            lora_a = lora_a_stacked[slice_id][lora_idx][expert_ids]
            lora_b = lora_b_stacked[slice_id][lora_idx][expert_ids]
            tensors = [
                hidden_states[i] @ lora_a[x].T @ lora_b[x].T
                for x in range(top_k_num)
            ]
            slice_tensors.append(torch.stack(tensors, dim=0))
        outputs.append(torch.concat(slice_tensors, dim=-1))
    return torch.stack(outputs, dim=0)


def use_torch_shrink(hidden_states, token_lora_mapping, topk_ids,
                     lora_a_stacked, top_k_num, num_slices=1):
    """Reference for ``fused_moe_lora_shrink``: hidden @ lora_a.T per
    token-expert pair.

    Returns a tensor of shape (num_slices, num_tokens, top_k_num, max_lora_rank).
    """
    num_tokens = hidden_states.shape[0]
    max_lora_rank = lora_a_stacked[0].shape[2]
    output = torch.zeros(
        (num_slices, num_tokens, top_k_num, max_lora_rank),
        dtype=hidden_states.dtype,
    )
    for i in range(num_tokens):
        lora_idx = token_lora_mapping[i]
        expert_ids = topk_ids[i]
        for slice_id in range(num_slices):
            lora_a = lora_a_stacked[slice_id][lora_idx][expert_ids]
            for x in range(top_k_num):
                output[slice_id, i, x] = hidden_states[i] @ lora_a[x].T
    return output


def use_torch_expand(a_intermediate_cache1, token_lora_mapping, topk_ids,
                     lora_b_stacked, top_k_num, num_slices=1):
    """Reference for ``fused_moe_lora_expand``: cache @ lora_b.T per
    token-expert pair, with slices concatenated along the output dim.

    ``a_intermediate_cache1`` has shape (num_slices, num_tokens, top_k_num, max_lora_rank).
    Returns a tensor of shape (num_tokens, top_k_num, N * num_slices).
    """
    num_tokens = a_intermediate_cache1.shape[1]
    N = lora_b_stacked[0].shape[2]
    output = torch.zeros(
        (num_tokens, top_k_num, N * num_slices),
        dtype=a_intermediate_cache1.dtype,
    )
    for i in range(num_tokens):
        lora_idx = token_lora_mapping[i]
        expert_ids = topk_ids[i]
        for slice_id in range(num_slices):
            lora_b = lora_b_stacked[slice_id][lora_idx][expert_ids]
            for x in range(top_k_num):
                output[i, x, slice_id * N:(slice_id + 1) * N] = (
                    a_intermediate_cache1[slice_id, i, x] @ lora_b[x].T
                )
    return output


# ---------------------------------------------------------------------------
# Kernel call wrappers (naive path)
# ---------------------------------------------------------------------------
def call_fused_moe_lora(
    topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
    lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, output,
    max_loras, block_size, add_inputs=True,
):
    """Call the full ``fused_moe_lora`` op via the naive dispatch path."""
    adapter_enabled, num_active_loras = _build_naive_meta(max_loras)
    # naive path: expert_ids is the flattened topk_ids
    expert_ids = topk_ids.reshape(-1)

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
            False,  # mul_routed_weight
            False,  # fully_sharded
            0,      # offset
            add_inputs,
        )


def call_fused_moe_lora_shrink(
    topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
    lora_ids, lora_a_stacked, hidden_states, a_intermediate_cache1,
    max_loras, num_slices, block_size,
):
    """Call the ``fused_moe_lora_shrink`` op via the naive dispatch path."""
    adapter_enabled, num_active_loras = _build_naive_meta(max_loras)
    expert_ids = topk_ids.reshape(-1)

    device = hidden_states.device
    M = topk_weights.shape[0]          # num_tokens
    K = hidden_states.shape[1]         # hidden_size
    N = max_lora_rank                  # shrink output dim
    # In naive mode EM = num_tokens_internal * block_size_m, where
    # num_tokens_internal = M * top_k_num (matches _fused_moe_lora).
    num_tokens_internal = M * top_k_num
    EM = num_tokens_internal * block_size

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
            lora_a_stacked[0].shape[1],  # num_experts
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
        )


def call_fused_moe_lora_expand(
    topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
    lora_ids, lora_b_stacked, a_intermediate_cache1, output,
    max_loras, num_slices, N_per_slice, block_size,
):
    """Call the ``fused_moe_lora_expand`` op via the naive dispatch path."""
    adapter_enabled, num_active_loras = _build_naive_meta(max_loras)
    expert_ids = topk_ids.reshape(-1)

    device = a_intermediate_cache1.device
    M = topk_weights.shape[0]          # num_tokens
    K = max_lora_rank                  # expand input dim (rank)
    num_tokens_internal = M * top_k_num
    EM = num_tokens_internal * block_size

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
            N_per_slice,
            M,
            EM,
            K,
            num_tokens_internal,
            lora_b_stacked[0].shape[1],  # num_experts
            num_slices,
            max_lora_rank,
            N_per_slice,  # w1_output_dim_size
            EXPAND_CONFIG["BLOCK_SIZE_M"],
            EXPAND_CONFIG["BLOCK_SIZE_N"],
            EXPAND_CONFIG["BLOCK_SIZE_K"],
            EXPAND_CONFIG["GROUP_SIZE_M"],
            EXPAND_CONFIG["NUM_WARPS"],
            EXPAND_CONFIG["NUM_STAGES"],
            EXPAND_CONFIG["SPLIT_K"],
            num_active_loras,
            False,  # mul_routed_weight
            0,      # offset
        )


# ---------------------------------------------------------------------------
# Test parameter sets (kept small for NPU CI)
# ---------------------------------------------------------------------------
DTYPES = [torch.float16, torch.bfloat16]
SEED = [0]


# ===========================================================================
# Tests for fused_moe_lora (full op)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [16, 64])
@pytest.mark.parametrize("top_k_num", [2, 6])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [512])
@pytest.mark.parametrize("K", [1024])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_full(
    num_tokens, top_k_num, num_experts, max_loras, N, K,
    max_lora_rank, block_size, num_slices, dtype, device, seed,
):
    """Verify ``fused_moe_lora`` (full shrink+expand) on NPU via naive path."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)

    num_sequences = max(1, min(num_tokens, 8))
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N // num_slices, max_lora_rank), dtype=dtype)
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, output,
        max_loras, block_size,
    )

    ref = use_torch_full(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, lora_b_stacked, top_k_num, num_slices,
    )
    torch.testing.assert_close(output, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_single_token(device: str):
    """Smoke test: smallest reasonable config for the full op."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 4, 2, 8, 4
    N, K, max_lora_rank, block_size, num_slices = 256, 512, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 2, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, output,
        max_loras, block_size,
    )
    ref = use_torch_full(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, lora_b_stacked, top_k_num,
    )
    torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_add_inputs(add_inputs: bool, device: str):
    """Verify the ``add_inputs`` flag: when True the LoRA result is added to
    the existing output; when False the output is overwritten."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 2, 8, 4
    N, K, max_lora_rank, block_size, num_slices = 256, 512, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 4, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, output,
        max_loras, block_size, add_inputs=add_inputs,
    )
    ref = use_torch_full(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, lora_b_stacked, top_k_num,
    )
    torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Tests for fused_moe_lora_shrink (shrink only)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [16, 64])
@pytest.mark.parametrize("top_k_num", [2, 6])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("K", [1024])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_shrink(
    num_tokens, top_k_num, num_experts, max_loras, K,
    max_lora_rank, block_size, num_slices, dtype, device, seed,
):
    """Verify ``fused_moe_lora_shrink`` on NPU via naive path.

    The shrink op computes: a_intermediate_cache1 = hidden @ lora_a.T
    Output shape: (num_slices, num_tokens, top_k_num, max_lora_rank)
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)

    num_sequences = max(1, min(num_tokens, 8))
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    a_intermediate_cache1 = torch.zeros(
        (num_slices, num_tokens, top_k_num, max_lora_rank), dtype=dtype
    )
    call_fused_moe_lora_shrink(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, hidden_states, a_intermediate_cache1,
        max_loras, num_slices, block_size,
    )

    ref = use_torch_shrink(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, top_k_num, num_slices,
    )
    torch.testing.assert_close(a_intermediate_cache1, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_shrink_single_lora(device: str):
    """Smoke test: 1 lora, small config for shrink."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 4, 2, 8, 1
    K, max_lora_rank, block_size, num_slices = 512, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 2, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    a_intermediate_cache1 = torch.zeros(
        (num_slices, num_tokens, top_k_num, max_lora_rank), dtype=dtype
    )
    call_fused_moe_lora_shrink(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, hidden_states, a_intermediate_cache1,
        max_loras, num_slices, block_size,
    )
    ref = use_torch_shrink(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, top_k_num,
    )
    torch.testing.assert_close(a_intermediate_cache1, ref, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Tests for fused_moe_lora_expand (expand only)
# ===========================================================================
@pytest.mark.parametrize("num_tokens", [16, 64])
@pytest.mark.parametrize("top_k_num", [2, 6])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [512])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_expand(
    num_tokens, top_k_num, num_experts, max_loras, N,
    max_lora_rank, block_size, num_slices, dtype, device, seed,
):
    """Verify ``fused_moe_lora_expand`` on NPU via naive path.

    The expand op computes: output = a_intermediate_cache1 @ lora_b.T
    We feed a random intermediate cache as input so that only the expand
    kernel is under test.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)

    num_sequences = max(1, min(num_tokens, 8))
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    N_per_slice = N // num_slices
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N_per_slice, max_lora_rank), dtype=dtype)
        for _ in range(num_slices)
    ]
    # Build a random intermediate cache (same layout as the shrink output).
    a_intermediate_cache1 = torch.randn(
        (num_slices, num_tokens, top_k_num, max_lora_rank), dtype=dtype
    )

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora_expand(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_b_stacked, a_intermediate_cache1, output,
        max_loras, num_slices, N_per_slice, block_size,
    )

    ref = use_torch_expand(
        a_intermediate_cache1, token_lora_mapping, topk_ids,
        lora_b_stacked, top_k_num, num_slices,
    )
    torch.testing.assert_close(output, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_expand_single_lora(device: str):
    """Smoke test: 1 lora, small config for expand."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 4, 2, 8, 1
    N, max_lora_rank, block_size, num_slices = 256, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 2, max_loras, num_experts, top_k_num
    )
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    a_intermediate_cache1 = torch.randn(
        (num_slices, num_tokens, top_k_num, max_lora_rank), dtype=dtype
    )

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora_expand(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_b_stacked, a_intermediate_cache1, output,
        max_loras, num_slices, N, block_size,
    )
    ref = use_torch_expand(
        a_intermediate_cache1, token_lora_mapping, topk_ids,
        lora_b_stacked, top_k_num,
    )
    torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Edge case: max_loras boundary
# ===========================================================================
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_max_loras_boundary(device: str):
    """Exercise the boundary where active LoRAs equals max_loras."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 2, 8, 8
    N, K, max_lora_rank, block_size, num_slices = 256, 512, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 8, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, output,
        max_loras, block_size,
    )
    ref = use_torch_full(
        hidden_states, token_lora_mapping, topk_ids,
        lora_a_stacked, lora_b_stacked, top_k_num,
    )
    torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Determinism / reproducibility
# ===========================================================================
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_full_deterministic(device: str):
    """Two runs with the same inputs should produce identical outputs."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens, top_k_num, num_experts, max_loras = 16, 2, 8, 4
    N, K, max_lora_rank, block_size, num_slices = 256, 512, 16, 16, 1
    dtype = torch.float16

    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, 4, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
    ]
    lora_b_stacked = [
        torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    out1 = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    out2 = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, out1,
        max_loras, block_size,
    )
    call_fused_moe_lora(
        topk_ids, topk_weights, token_lora_mapping, max_lora_rank, top_k_num,
        lora_ids, lora_a_stacked, lora_b_stacked, hidden_states, out2,
        max_loras, block_size,
    )
    assert torch.equal(out1, out2), (
        "fused_moe_lora produced non-deterministic outputs on NPU"
    )
