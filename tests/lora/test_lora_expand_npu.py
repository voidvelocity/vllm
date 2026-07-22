# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NPU-side sanity tests for the triton ``lora_expand`` kernel.

These tests verify that ``vllm.lora.ops.triton_ops.lora_expand`` can run
normally on NPU (Ascend) hardware and produce numerically correct results
when compared with the reference torch implementation
(``vllm.lora.ops.torch_ops.sgmv_expand_slice``).

The kernel itself lives in
``vllm/lora/ops/triton_ops/lora_expand_op.py``.
"""

from threading import Lock

import pytest
import torch

import vllm.lora.ops.torch_ops as torch_ops
import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import _LORA_B_PTR_DICT
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .utils import PunicaTensors, assert_close, generate_data_for_nslices

DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:{0}"]

# Triton kernels are only exercised on platforms that expose a triton backend.
# On NPU (Ascend) the triton backend is provided through vllm-ascend, so we
# skip when running on a host without an accelerator.
supports_triton = current_platform.is_cuda_alike() or current_platform.is_xpu()
if not supports_triton:
    try:
        import triton  # noqa: F401

        supports_triton = True
    except ImportError:
        supports_triton = False

pytestmark = pytest.mark.skipif(
    not supports_triton,
    reason="lora_expand triton kernel requires a triton-capable device "
    "(CUDA / XPU / Ascend NPU with triton support).",
)


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    """Ensure torch's default device is restored between tests."""
    pass


# Reuse the same lock pattern as test_punica_ops.py to avoid stale pointer
# cache issues across tests.
_dict_lock = Lock()


def _run_reference_expand(
    nslices: int,
    hidden_size: int,
    data: PunicaTensors,
    batches: int,
    add_inputs: bool,
) -> torch.Tensor:
    """Run the torch reference sgmv_expand_slice implementation.

    Each slice writes into a distinct ``[hidden_size * index]`` region of the
    output tensor, mirroring how the triton kernel lays out multi-slice
    outputs.
    """
    max_seq_length, token_nums = data.meta()
    for index in range(nslices):
        torch_ops.sgmv_expand_slice(
            data.inputs_tensor[index],
            data.lora_weights[index],
            data.ref_out_tensor,
            data.b_seq_start_loc,
            data.seq_len_tensor,
            data.prompt_lora_mapping,
            batches,
            max_seq_length,
            token_nums,
            slice_offset=hidden_size * index,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )
    return data.ref_out_tensor


def _run_triton_expand(
    data: PunicaTensors,
    num_loras: int,
    token_nums: int,
    add_inputs: bool,
) -> torch.Tensor:
    """Run the triton lora_expand kernel under test."""
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras,
        max_num_tokens=token_nums,
        device=DEVICE_TYPE,
    )
    lora_meta.prepare_tensors(data.token_lora_mapping)

    out_tensor = data.our_out_tensor.clone()
    with _dict_lock:
        # The LoRA pointer dict is keyed by tensor data_ptr(); clear it
        # between tests to avoid stale pointer lookups.
        _LORA_B_PTR_DICT.clear()
        triton_ops.lora_expand(
            data.inputs_tensor,
            data.lora_weights,
            out_tensor,
            *lora_meta.meta_args(token_nums=token_nums, specialize_active_lora=False),
            offset_start=0,
            add_inputs=add_inputs,
        )
    return out_tensor


def check_lora_expand_on_npu(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    add_inputs: bool,
) -> None:
    """
    Generate random inputs, run both the triton lora_expand kernel and the
    torch reference, and assert the outputs match.
    """
    set_random_seed(0)

    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
    )
    _, token_nums = data.meta()

    out_tensor = _run_triton_expand(data, num_loras, token_nums, add_inputs)
    ref_out_tensor = _run_reference_expand(
        nslices, hidden_size, data, batches, add_inputs
    )

    assert_close(out_tensor, ref_out_tensor)


# ---------------------------------------------------------------------------
# Test parameter sets
# ---------------------------------------------------------------------------
# Subset of hidden sizes that are commonly used by LoRA-supported models on
# NPU. Kept small to keep NPU CI runtime reasonable.
NPU_HIDDEN_SIZES = [128, 512, 1024, 2048, 4096]

NPU_TEST_PARAMS = {
    "hidden_sizes": NPU_HIDDEN_SIZES,
    "batches": [1, 4, 16],
    "num_loras": [1, 4, 8],
    "max_ranks": [8, 16, 32, 64],
}

DTYPES = [torch.float16, torch.bfloat16]
SEED = [0]


# ---------------------------------------------------------------------------
# Basic sanity tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_expand_no_lora_early_exit(device: str):
    """
    When ``no_lora_flag_cpu`` is True (all tokens map to lora_id -1), the
    kernel should early-exit without touching the output tensor.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    num_tokens = 16
    hidden_size = 128
    rank = 8
    dtype = torch.float16

    inputs = torch.rand((1, num_tokens, rank), dtype=dtype, device=device)
    lora_weights = [
        torch.rand((1, hidden_size, rank), dtype=dtype, device=device)
    ]
    output_tensor = torch.zeros(
        (num_tokens, hidden_size), dtype=dtype, device=device
    )
    snapshot = output_tensor.clone()

    lora_meta = LoRAKernelMeta.make(
        max_loras=1,
        max_num_tokens=num_tokens,
        device=DEVICE_TYPE,
    )
    # token_lora_mapping of all -1 triggers no_lora_flag_cpu=True
    token_lora_mapping = torch.full(
        (num_tokens,), -1, dtype=torch.int32, device=device
    )
    lora_meta.prepare_tensors(token_lora_mapping)

    with _dict_lock:
        _LORA_B_PTR_DICT.clear()
        triton_ops.lora_expand(
            inputs,
            lora_weights,
            output_tensor,
            *lora_meta.meta_args(token_nums=num_tokens, specialize_active_lora=False),
            offset_start=0,
            add_inputs=False,
        )

    assert torch.equal(output_tensor, snapshot), (
        "lora_expand modified output tensor despite no_lora_flag_cpu=True"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_expand_single_lora_single_token(device: str):
    """Smoke test: 1 lora, 1 token, smallest reasonable config."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    check_lora_expand_on_npu(
        batches=1,
        num_loras=1,
        rank=8,
        hidden_size=128,
        nslices=1,
        dtype=torch.float16,
        device=device,
        seq_length=1,
        add_inputs=True,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_expand_single_lora_multi_token(device: str):
    """1 lora, multiple tokens, nslices=1."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    check_lora_expand_on_npu(
        batches=4,
        num_loras=1,
        rank=16,
        hidden_size=512,
        nslices=1,
        dtype=torch.float16,
        device=device,
        seq_length=32,
        add_inputs=True,
    )


# ---------------------------------------------------------------------------
# Parameterized correctness tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batches", NPU_TEST_PARAMS["batches"])
@pytest.mark.parametrize("num_loras", NPU_TEST_PARAMS["num_loras"])
@pytest.mark.parametrize("rank", NPU_TEST_PARAMS["max_ranks"])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_expand_param_variations(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """
    Vary batches, num_loras, rank, nslices, dtype while keeping hidden_size
    fixed. Mirrors the structure of test_punica_ops.test_kernels.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)
    check_lora_expand_on_npu(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        seq_length=128,
        add_inputs=True,
    )


@pytest.mark.parametrize("batches", [4])
@pytest.mark.parametrize("num_loras", [4])
@pytest.mark.parametrize("rank", [32])
@pytest.mark.parametrize("hidden_size", NPU_TEST_PARAMS["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_expand_hidden_size_variations(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    """
    Vary hidden_size across common values; keep other params fixed.
    This is the NPU equivalent of test_kernels_hidden_size.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)
    check_lora_expand_on_npu(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        seq_length=128,
        add_inputs=True,
    )


# ---------------------------------------------------------------------------
# add_inputs tests (expand's counterpart to the shrink scaling factor)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_expand_add_inputs(add_inputs: bool, device: str):
    """
    Verify the add_inputs flag is correctly applied by the kernel: when True
    the LoRA output is accumulated onto the existing output tensor; when
    False the output tensor is overwritten.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)
    check_lora_expand_on_npu(
        batches=4,
        num_loras=4,
        rank=16,
        hidden_size=512,
        nslices=1,
        dtype=torch.float16,
        device=device,
        seq_length=16,
        add_inputs=add_inputs,
    )


# ---------------------------------------------------------------------------
# Edge case: larger num_loras (close to max_loras)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_expand_max_loras_boundary(device: str):
    """
    Exercise the boundary where the number of active LoRAs equals max_loras.
    This is the case the project memory flagged as historically problematic
    on NPU when graph capture forces num_active_loras=max_loras.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)
    check_lora_expand_on_npu(
        batches=8,
        num_loras=8,
        rank=16,
        hidden_size=512,
        nslices=1,
        dtype=torch.float16,
        device=device,
        seq_length=4,
        add_inputs=True,
    )


# ---------------------------------------------------------------------------
# Edge case: large batch to exercise the kernel with many tokens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_expand_large_batch(device: str):
    """
    Verify correctness on the NPU with a large batch (many tokens) to
    exercise the kernel's larger-grid launch path.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)
    check_lora_expand_on_npu(
        batches=128,
        num_loras=4,
        rank=32,
        hidden_size=1024,
        nslices=1,
        dtype=torch.float16,
        device=device,
        seq_length=2,
        add_inputs=True,
    )


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_expand_deterministic_across_runs(device: str):
    """Two runs with the same inputs should produce identical outputs."""
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(0)

    data: PunicaTensors = generate_data_for_nslices(
        batches=4,
        hidden_size=512,
        lora_nums=4,
        max_rank=16,
        seq_length=16,
        nslices=1,
        dtype=torch.float16,
        op_type="expand",
        device=device,
    )
    _, token_nums = data.meta()

    out1 = _run_triton_expand(
        data, num_loras=4, token_nums=token_nums, add_inputs=True
    )
    out2 = _run_triton_expand(
        data, num_loras=4, token_nums=token_nums, add_inputs=True
    )

    assert torch.equal(out1, out2), (
        "lora_expand produced non-deterministic outputs on NPU"
    )
