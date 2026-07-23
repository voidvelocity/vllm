# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simplified MoE-LoRA Triton kernels for Ascend NPU.

The original CUDA/GPU-oriented implementation contains TMA, GDC, PDL,
small-batch fast paths and GPU occupancy heuristics which are not
applicable on Ascend NPUs.  This file keeps only the general two-kernel
shrink + expand path, using plain tl.load/tl.dot/tl.store, so that it
runs on NPU through the Triton backend.

Conventions:
  - LoRA stacked weights:
      lora_a_stacked[s]: (max_loras, num_experts, rank, K)
      lora_b_stacked[s]: (max_loras, num_experts, N, rank)
    where `s` indexes slices (e.g. gate/up for fused w13).
  - `N` is the output dimension of the expand step (N per slice).
  - `K` is the contraction dimension.
  - Hidden state x has shape (M, K), where M == num_tokens.
  - Top-k routing expands the M dimension to num_tokens * top_k_num pairs.
"""

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


_LORA_PTR_DICT: dict[tuple[int, ...], torch.Tensor] = {}


def _get_ptr(weight_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Cache a uint64 pointer table for the LoRA weight slices.

    The same trick is used in grouped GEMM Triton examples: the kernel
    receives a small tensor of device pointers and looks up the
    appropriate slice pointer at runtime.
    """
    key = tuple(t.data_ptr() for t in weight_list)
    ptr_tensor = _LORA_PTR_DICT.get(key, None)
    if ptr_tensor is None or ptr_tensor.device != device:
        ptr_tensor = torch.tensor(
            list(key), dtype=torch.uint64, device=device
        )
        _LORA_PTR_DICT[key] = ptr_tensor
    return ptr_tensor


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _fused_moe_lora_kernel(
    # ---- pointers ----
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    expert_ids_ptr,
    token_lora_mapping_ptr,
    lora_ids_ptr,
    adapter_enabled_ptr,
    # ---- dims ----
    N,
    K,
    EM,
    num_tokens,
    num_experts,
    top_k_num,
    max_loras,
    # ---- strides ----
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # ---- slice sizes ----
    slice_a_size,
    slice_c_size,
    num_slice_a,
    num_slice_c,
    # ---- constexpr ----
    token_mapping_factor: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """MoE-LoRA GEMM kernel (naive, SPLIT_K=1, sort_c=False).

    For each (slice, token_block, n_block) work item, resolve lora_id /
    expert_id from the naive 1D tables, load the A and B tiles, compute
    the GEMM, optionally multiply by topk weight and add to output.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Resolve lora_id from the token at (pid_m // top_k_num).
    token_idx = pid_m // top_k_num
    lora_id = tl.load(token_lora_mapping_ptr + token_idx)
    if lora_id < 0:
        return
    if lora_id >= max_loras:
        return
    enabled = tl.load(adapter_enabled_ptr + lora_id)
    if enabled == 0:
        return

    # Resolve expert_id (naive: 1D table indexed by pid_m).
    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id == -1:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    raw = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.where(raw < num_tokens, raw, num_tokens)
    token_mask = offs_token < num_tokens

    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // token_mapping_factor * stride_am
        + offs_k[None, :] * stride_ak
    )
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
    b_ptrs = (
        cur_b_ptr
        + lora_id * stride_bl
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        cur_k_offset = k * BLOCK_SIZE_K
        k_remaining = K - cur_k_offset
        k_mask = offs_k[None, :] < k_remaining
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask,
            other=0.0,
        )
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0.0
        )
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        cur_c_ptr
        + stride_cm * offs_token[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if ADD_INPUTS:
        prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, prev + accumulator, mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator, mask=c_mask)


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    num_active_loras: torch.Tensor,
    mul_routed_weight: bool = False,
) -> None:
    """Shrink step: x @ lora_a^T -> intermediate cache."""
    w1_lora_a_stacked = lora_a_stacked[0]

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
    )

    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        lora_ids,
        adapter_enabled,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids.numel(),
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(-2),
        a_intermediate_cache1.stride(-1),
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        MUL_ROUTED_WEIGHT=False,
        ADD_INPUTS=False,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    num_active_loras: torch.Tensor,
    mul_routed_weight: bool = False,
    offset: int = 0,
) -> None:
    """Expand step: intermediate cache @ lora_b^T -> output."""
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]
    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[-1]
    )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
    )

    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)

    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        out_view,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        lora_ids,
        adapter_enabled,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids.numel(),
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    add_inputs: bool = True,
) -> None:
    """Top-level entry point for MoE-LoRA.

    On NPU we always use the two-kernel shrink + expand path, with an
    optional all-reduce / all-gather in between for fully_sharded LoRA.
    """
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2
    assert expert_ids.dim() == 1
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]
    assert shrink_block_size_m == expand_block_size_m

    assert add_inputs, (
        "fused_moe_lora(add_inputs=False) is only supported on the "
        "fully_sharded=False fast path"
    )

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]
    EM = num_tokens

    intermediate_cache_shape = (
        num_slices,
        M,
        top_k_num,
        max_lora_rank,
    )
    a_intermediate_cache1 = torch.zeros(
        intermediate_cache_shape,
        dtype=output.dtype,
        device=device,
    )

    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        adapter_enabled,
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        num_active_loras,
        mul_routed_weight,
    )

    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )
            max_lora_rank = a_intermediate_cache1.shape[-1]

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        expert_ids,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        adapter_enabled,
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        max_lora_rank,
        w1_output_dim_size,
        expand_block_size_m,
        expand_block_size_n,
        expand_block_size_k,
        expand_group_size_m,
        expand_num_warps,
        expand_num_stages,
        num_active_loras,
        mul_routed_weight,
        offset,
    )


try:
    direct_register_custom_op(
        op_name="fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=None,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=None,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=None,
    )

    fused_moe_lora = torch.ops.vllm.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.vllm.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.vllm.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand



if __name__ == "__main__":
    device = torch.device("npu")

    # ---- Sub-test 1: single expert, top_k_num=2 ----
    # All pairs use expert 0; one M-tile covers all 8 pairs.
    torch.manual_seed(0)
    M, K, N, tk = 4, 64, 16, 2
    num_experts, max_loras, num_slices = 1, 2, 1
    EM = M * tk  # 8

    hidden = torch.randn(M, K, dtype=torch.float32, device=device)
    lora_a = torch.randn(
        max_loras, num_experts, N, K, dtype=torch.float32, device=device
    ).contiguous()
    topk_weights = torch.rand(M, tk, dtype=torch.float32, device=device)
    topk_ids = torch.zeros(M, tk, dtype=torch.int32, device=device)
    expert_ids = torch.tensor([0], dtype=torch.int32, device=device)
    token_lora_mapping = torch.zeros(M, dtype=torch.int32, device=device)
    lora_ids = torch.tensor([0], dtype=torch.int32, device=device)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)
    num_active_loras = torch.tensor([1], dtype=torch.int32, device="cpu")

    cache = torch.zeros(
        num_slices, M, tk, N, dtype=torch.float32, device=device
    )

    _LORA_PTR_DICT.clear()
    _fused_moe_lora_shrink(
        cache, hidden, [lora_a], topk_weights, expert_ids,
        token_lora_mapping, tk, lora_ids, adapter_enabled,
        device,
        N, M, EM, K, EM, num_experts, num_slices,
        16, 16, 64, 1, 4, 3, num_active_loras, False,
    )

    ref = torch.zeros(num_slices, M, tk, N, dtype=torch.float32)
    lora_a_cpu = lora_a.cpu()
    hidden_cpu = hidden.cpu()
    for i in range(M):
        for x in range(tk):
            ref[0, i, x] = hidden_cpu[i] @ lora_a_cpu[0, 0].T

    err1 = (cache.cpu() - ref).abs().max().item()
    print(f"  fused_moe_lora_shrink_triton (single-expert) max_err={err1:.2e}")

    # ---- Sub-test 2: multi-block, multi-expert, top_k_num=1 ----
    # Block 0 (tokens 0-15) → expert 0; block 1 (tokens 16-31) → expert 1.
    torch.manual_seed(1)
    M2, K2, N2, tk2 = 32, 64, 16, 1
    num_experts2, max_loras2, num_slices2 = 2, 2, 1
    EM2 = M2 * tk2  # 32

    hidden2 = torch.randn(M2, K2, dtype=torch.float32, device=device)
    lora_a2 = torch.randn(
        max_loras2, num_experts2, N2, K2, dtype=torch.float32, device=device
    ).contiguous()
    topk_ids2 = torch.zeros(M2, tk2, dtype=torch.int32, device=device)
    topk_ids2[M2 // 2:, 0] = 1
    expert_ids2 = torch.tensor([0, 1], dtype=torch.int32, device=device)
    token_lora_mapping2 = torch.zeros(M2, dtype=torch.int32, device=device)
    topk_weights2 = torch.rand(M2, tk2, dtype=torch.float32, device=device)
    lora_ids2 = torch.tensor([0], dtype=torch.int32, device=device)
    adapter_enabled2 = torch.ones(max_loras2 + 1, dtype=torch.int32, device=device)
    num_active_loras2 = torch.tensor([1], dtype=torch.int32, device="cpu")

    cache2 = torch.zeros(
        num_slices2, M2, tk2, N2, dtype=torch.float32, device=device
    )

    _LORA_PTR_DICT.clear()
    _fused_moe_lora_shrink(
        cache2, hidden2, [lora_a2], topk_weights2, expert_ids2,
        token_lora_mapping2, tk2, lora_ids2, adapter_enabled2,
        device,
        N2, M2, EM2, K2, EM2, num_experts2, num_slices2,
        16, 16, 64, 1, 4, 3, num_active_loras2, False,
    )

    ref2 = torch.zeros(num_slices2, M2, tk2, N2, dtype=torch.float32)
    lora_a2_cpu = lora_a2.cpu()
    hidden2_cpu = hidden2.cpu()
    for i in range(M2):
        eid = int(topk_ids2[i, 0].item())
        ref2[0, i, 0] = hidden2_cpu[i] @ lora_a2_cpu[0, eid].T

    err2 = (cache2.cpu() - ref2).abs().max().item()
    print(f"  fused_moe_lora_shrink_triton (multi-expert) max_err={err2:.2e}")
