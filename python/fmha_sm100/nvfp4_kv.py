# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The NVFP4 KV cache page layout that fmha_sm100 reads."""

from typing import Tuple

import torch


def nvfp4_head_slot_views(
    k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the per-head K/V slots of an NVFP4 cache into data and block-scale views.

    The NVFP4 cache is uint8 ``[pages, 2 * Hkv, P, 72]`` with P=128 or 256: slot ``2 * h``
    is head ``h``'s K and slot ``2 * h + 1`` its V. A slot holds the head's packed E2M1
    data (P * 64 bytes) followed by its E4M3 block scales (P * 8 bytes).
    ``k`` and ``v`` are the K and V slots, ``cache[:, 0::2]`` and
    ``cache[:, 1::2]``; any other layout is rejected.

    Returns ``(k_data, k_scale, v_data, v_scale)``, ``[pages, Hkv, P, 64]`` and
    ``[pages, Hkv, P, 8]`` uint8 views.
    """
    if k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise TypeError(f"NVFP4 k/v must be torch.uint8, got {k.dtype} and {v.dtype}")
    if (k.ndim != 4 or k.shape[2] not in (128, 256) or k.shape[3] != 72
            or k.shape != v.shape):
        raise ValueError("NVFP4 k/v must be [pages, Hkv, P, 72] slot views with P=128/256, got "
                         f"{tuple(k.shape)} and {tuple(v.shape)}")
    pages, heads = k.shape[:2]
    page_size = k.shape[2]
    data_bytes, slot_bytes = page_size * 64, page_size * 72
    head_stride, page_stride = 2 * slot_bytes, k.stride(0)
    if (any(t.stride(2) != 72 or t.stride(3) != 1 for t in (k, v))
            or v.stride(0) != page_stride
            or v.data_ptr() != k.data_ptr() + slot_bytes
            or (heads > 1 and not k.stride(1) == v.stride(1) == head_stride)
            or page_stride < heads * head_stride):
        raise ValueError(
            "NVFP4 k/v must be the per-head K/V slots cache[:, 0::2] and cache[:, 1::2] of a "
            "[pages, 2 * Hkv, P, 72] cache, each head's K slot followed by its V slot")
    if page_stride % 16 or k.data_ptr() % 16:
        raise ValueError("NVFP4 cache pages must be 16-byte aligned")
    views = []
    for slots in (k, v):
        for width, offset in ((64, 0), (8, data_bytes)):
            views.append(slots.as_strided((pages, heads, page_size, width),
                                          (page_stride, head_stride, width, 1),
                                          slots.storage_offset() + offset))
    k_data, k_scale, v_data, v_scale = views
    return k_data, k_scale, v_data, v_scale
