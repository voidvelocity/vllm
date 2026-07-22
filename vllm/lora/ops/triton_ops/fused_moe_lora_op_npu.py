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


@triton.jit
def _get_lora_id(
    lora_ids,
    token_lora_mapping_ptr,
    lora_idx,
    pid_m,
    top_k_num,
    naive_block_assignment: tl.constexpr,
):
    """Return lora_id for the current block."""
    if naive_block_assignment:
        token_idx = pid_m // top_k_num
        return tl.load(token_lora_mapping_ptr + token_idx)
    else:
        return tl.load(lora_ids + lora_idx)


@triton.jit
def _get_expert_id(
    expert_ids_ptr,
    lora_id,
    pid_m,
    stride_el,
    max_loras,
    naive_block_assignment: tl.constexpr,
):
    """Return expert_id for the current block."""
    if naive_block_assignment:
        return tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        return tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)


@triton.jit
def _get_token_offs(
    sorted_token_ids_ptr,
    lora_id,
    pid_m,
    offs,
    stride_tl,
    max_loras,
    num_valid_tokens,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Return token offsets for each lane in the block."""
    if naive_block_assignment:
        raw = pid_m * BLOCK_SIZE_M + offs
        return tl.where(raw < num_valid_tokens, raw, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        token_ind = stride_tl * lora_id + offs_token_id
        return tl.load(
            sorted_token_ids_ptr + token_ind,
            token_ind < max_loras * stride_tl,
            num_valid_tokens,
        )


@triton.jit
def _get_c_ptrs(
    cur_c_ptr,
    lora_id,
    pid_m,
    offs,
    offs_token,
    offs_cn,
    stride_cm,
    stride_cn,
    EM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    sort_c: tl.constexpr,
):
    if sort_c:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        c_ptrs = (
            cur_c_ptr
            + lora_id * EM * stride_cm
            + stride_cm * offs_token_id[:, None]
            + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = (
            cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        )
    return c_ptrs


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


def _adjust_kernel_inputs(
    num_active_loras: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
):
    """Return (grid_lora_dim, stride_tl, stride_el) for the kernel grid."""
    if sorted_token_ids is None:
        return 1, 0, 0
    assert expert_ids.dim() == 2
    return int(num_active_loras.item()), sorted_token_ids.stride(0), expert_ids.stride(
        0)


@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _fused_moe_lora_kernel(
    # ---- pointers ----
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
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
    # ---- strides ----
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    # ---- slice sizes ----
    slice_a_size,
    slice_c_size,
    num_slice_a,
    num_slice_c,
    # ---- constexpr ----
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    sort_c: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """General MoE-LoRA GEMM kernel used for both shrink and expand.

    Inputs:
      a_ptr : 2-D tensor, shape depends on caller (x for shrink,
              intermediate cache for expand).
      b_ptr : pointer table of LoRA weight slices.
      c_ptr : output tensor.

    For each (slice, lora_id, expert_id, token_block, n_block) work item,
    load the corresponding A tile and B tile, compute the GEMM, optionally
    multiply by topk weight and add to existing output.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    pid_sk = pid // (num_pid_m * num_pid_n)

    # Resolve lora_id.
    lora_id = _get_lora_id(
        lora_ids_ptr,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id < 0:
        return
    if lora_id >= lora_ids_ptr.numel():
        return
    enabled = tl.load(adapter_enabled_ptr + lora_id)
    if enabled == 0:
        return

    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id.
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        lora_ids_ptr.numel(),
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        lora_ids_ptr.numel(),
        num_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    token_mask = offs_token < num_tokens

    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
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
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
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
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0.0
        )
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = _get_c_ptrs(
        cur_c_ptr,
        lora_id,
        pid_m,
        offs,
        offs_token,
        offs_cn,
        stride_cm,
        stride_cn,
        EM,
        BLOCK_SIZE_M,
        sort_c,
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
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
    split_k: int,
    num_active_loras: torch.Tensor,
    mul_routed_weight: bool = False,
) -> None:
    """Shrink step: x @ lora_a^T -> intermediate cache."""
    w1_lora_a_stacked = lora_a_stacked[0]

    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)
    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        grid_lora_dim,
    )

    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        lora_ids,
        adapter_enabled,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(-2),
        a_intermediate_cache1.stride(-1),
        stride_tl,
        stride_el,
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=False,
        ADD_INPUTS=False,
        sort_c=False,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
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
    split_k: int,
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

    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": 1,  # No split-K on expand path.
    }

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim,
    )

    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)

    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        lora_ids,
        adapter_enabled,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        stride_tl,
        stride_el,
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        sort_c=False,
        **expand_config,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
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
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
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
    if sorted_token_ids is None:
        assert expert_ids.dim() == 1
    else:
        assert num_tokens_post_padded is not None
        assert (
            sorted_token_ids.dim()
            == expert_ids.dim()
            == topk_weights.dim()
            == qcurr_hidden_states.dim()
            == 2
        )
        assert (
            sorted_token_ids.shape[0]
            == expert_ids.shape[0]
            == num_tokens_post_padded.shape[0]
        )
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
    EM = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens
    )

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
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
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
        shrink_split_k,
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
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
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
        expand_split_k,
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
