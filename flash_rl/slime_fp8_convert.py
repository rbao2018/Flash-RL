import re
import torch

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .qwen2 import convert_qwen2_to_hf
from .qwen3moe import convert_qwen3moe_to_hf

import triton
import triton.language as tl
from typing import Tuple

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    return (x + y - 1) // y


@triton.jit
def _blockwise_cast_to_fp8_triton(
    X,
    Y,
    S,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    x_s = _absmax / fp8_max
    s_inv = 1.0 / x_s
    y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)


def blockwise_cast_to_fp8_triton(x: torch.Tensor, block_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_M, BLOCK_N = 128, 128
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    _blockwise_cast_to_fp8_triton[grid](
        x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, fp8_min, fp8_max, **kwargs
    )
    return y, s

def ceildiv(a, b):
    return -(-a // b)


def quantize_param(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    assert quantization_config["quant_method"] == "fp8"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        return converted_named_params

    layer_idx, rest = match.groups()
    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


cached_tensors = {}


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
        # to compatible with sglang implementation
        if args.q_lora_rank is not None:
            old_converted_named_tensors = converted_named_tensors
            converted_named_tensors = []
            for converted_name, converted_param in old_converted_named_tensors:
                if "q_a_proj" in converted_name:
                    pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                    if pair_name in cached_tensors:
                        converted_named_tensors += [
                            (converted_name, converted_param),
                            (pair_name, cached_tensors[pair_name]),
                        ]
                        del cached_tensors[pair_name]
                    else:
                        cached_tensors[converted_name] = converted_param
                elif "kv_a_proj_with_mqa" in converted_name:
                    pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                    if pair_name in cached_tensors:
                        converted_named_tensors += [
                            (converted_name, converted_param),
                            (pair_name, cached_tensors[pair_name]),
                        ]
                        del cached_tensors[pair_name]
                    else:
                        cached_tensors[converted_name] = converted_param
                else:
                    converted_named_tensors.append((converted_name, converted_param))

    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not quantization_config:
        return converted_named_tensors

    return quantize_params(args, name, converted_named_tensors, quantization_config)