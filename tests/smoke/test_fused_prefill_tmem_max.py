# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

from fmha_sm100.api import (
    fmha_index_score_sharded,
    fmha_index_score_sharded_plan,
    sparse_topk_merge_prefill_partials,
    sparse_topk_select_prefill,
)


@pytest.mark.parametrize("tp,page_size", [(2, 128), (4, 128), (2, 256), (4, 256)])
@torch.inference_mode()
def test_fused_prefill_tmem_max_special_values(tp, page_size):
    tokens, heads, history = 129, 4, 12160
    seq_len = history + tokens
    pages = (seq_len + page_size - 1) // page_size
    positions = torch.arange(pages * page_size)
    modes = positions // 128 % 5
    values = torch.zeros_like(positions, dtype=torch.float32)
    values[modes == 0] = -0.0
    values[modes == 2] = -1
    values[modes == 3] = float("nan")
    values[modes == 4] = -2
    values[(modes == 4) & ((positions // tp) % 2 == 0)] = float("nan")
    values[:128] = float("nan")
    values[seq_len // 128 * 128:] = -16
    cache = torch.zeros(pages, page_size, 128, device="cuda")
    cache[..., 0] = values.reshape(pages, page_size).cuda()
    cache = cache.to(torch.float8_e4m3fn)
    query = torch.zeros(tokens, heads, 128, device="cuda").to(cache.dtype)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None]
    plan = fmha_index_score_sharded_plan(
        torch.tensor([tokens], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32), heads,
        qo_offset=torch.tensor([history], dtype=torch.int32), device="cuda",
    )
    valid = (torch.arange(tokens, device="cuda", dtype=torch.int32) + history + 128) // 128
    scale = page_size // 128
    columns = ((seq_len + 127) // 128 + scale - 1) // scale
    partials = torch.empty(tokens, heads, 64, device="cuda")
    scores = torch.empty(tokens, heads, columns, device="cuda")
    actual = torch.empty(heads, tokens, 33, device="cuda", dtype=torch.int32)
    materialized = torch.empty_like(actual)
    force = dict(force_begin_blocks=1, force_end_blocks=1)
    for rank in range(tp):
        rows = page_size // tp
        offset = rank * rows // 128
        shard = (cache[:, rank::tp] if page_size == 128
                 else cache[:, rank * rows:(rank + 1) * rows]).contiguous()
        local_positions = (positions.reshape(pages, page_size)[:, rank::tp]
                           if page_size == 128 else
                           positions.reshape(pages, page_size)[:, rank * rows:(rank + 1) * rows])
        local_positions = local_positions.flatten()
        mapping = dict(index_scale=scale, index_offset=offset, num_valid_pages=valid)

        def fused():
            fmha_index_score_sharded(
                query, shard, table, plan, rank=rank, tp=tp,
                max_score=partials, sm_scale=1.0, compact=page_size == 256,
                fused_topk=True, **force,
            )
            sparse_topk_merge_prefill_partials(partials, actual, **mapping, **force)

        graph = None
        for sign in (1, -1, 1):
            q = torch.zeros(tokens, heads, 128, device="cuda")
            q[..., 0] = sign
            q[64, :, 0] = float("nan")
            query.copy_(q.to(query.dtype))
            fmha_index_score_sharded(
                query, shard, table, plan, rank=rank, tp=tp,
                max_score=scores, sm_scale=1.0, compact=page_size == 256,
            )
            sparse_topk_select_prefill(
                scores, valid, materialized, index_scale=scale, index_offset=offset, **force,
            )
            partials.fill_(float("nan"))
            actual.fill_(0x13131313)
            if graph is None:
                fused()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused()
            else:
                graph.replay()
            expected = torch.empty(heads, tokens, 33, dtype=torch.int32)
            expected[..., :16] = torch.full((16,), -float("inf")).view(torch.int32)
            expected[..., 16:32] = -1
            expected[..., 32] = 1
            for token in range(tokens):
                pos = history + token
                count = pos // 128 + 1
                block_scores = {}
                for block in range(offset, count, scale):
                    selected = local_positions[(local_positions // 128 == block)
                                               & (local_positions <= pos)]
                    finite = [float(v) * sign for v in values[selected]
                              if not math.isnan(float(v)) and token != 64]
                    block_scores[block] = max(finite, default=-float("inf"))
                forced = {i for i in block_scores if i == 0 or i == count - 1}
                chosen = sorted(sorted(block_scores, key=lambda i:
                                       (i not in forced, -block_scores[i], i))[:16])
                for slot, block in enumerate(chosen):
                    score = block_scores[block]
                    expected[:, token, slot] = torch.tensor(0.0 if score == 0 else score).view(torch.int32)
                    expected[:, token, 16 + slot] = block
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, materialized, rtol=0, atol=0)
