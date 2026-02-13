import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from packaging.version import Version
# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers greater than each input element
        that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible, otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device
    )
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def our_eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """标准的 Eager Attention 实现"""
    bsz, q_len, num_heads, head_dim = query_states.shape
    
    # 重塑维度进行矩阵乘法 [batch, heads, seq, dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.reshape(bsz, q_len, -1)  
def apply_rope(x, positions, max_wavelength=10000, dtype=torch.float32):
    """旋转位置编码 (RoPE) 的物理实现"""
    original_dtype = x.dtype
    d = x.shape[-1]
    d_half = d // 2
    device = x.device

    # 转为指定精度计算
    x_casted = x.to(dtype)
    positions_casted = positions.to(dtype)

    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=device)
    timescale = max_wavelength**freq_exponents
    
    # 计算旋转弧度
    radians = torch.einsum("bl,h->blh", positions_casted, 1.0 / timescale)
    radians = radians[..., None, :]  # [B, L, 1, D_half]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x_casted.split(d_half, dim=-1)
    
    # 旋转变换：[x1*cos - x2*sin, x2*cos + x1*sin]
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return res.to(original_dtype)