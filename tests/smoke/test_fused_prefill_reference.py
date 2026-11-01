# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import pytest
import torch

from fmha_sm100.api import (
    _get_num_cta,
    fmha_index_score_sharded,
    fmha_index_score_sharded_plan_device,
    sparse_topk_merge_prefill_partials,
)


@pytest.mark.parametrize("tp,page_size", [(2, 128), (4, 128), (2, 256), (4, 256)])
@pytest.mark.parametrize("force_begin,force_end", [(0, 1), (1, 1), (8, 8)])
@torch.inference_mode()
def test_fused_prefill_device_plan_reference(tp, page_size, force_begin, force_end):
    # Q selects one K component: expected maxima do not depend on either scorer.
    total_q, heads, max_seq_len = 129, 4, 4097
    pages = (max_seq_len + page_size - 1) // page_size
    query = torch.zeros(total_q, heads, 128, device="cuda")
    query[..., 0] = 1
    query = query.to(torch.float8_e4m3fn)
    cache = torch.zeros(pages, page_size, 128, device="cuda")
    block_ids = torch.arange(pages * page_size, device="cuda") // 128
    cache[..., 0] = -(block_ids.reshape(pages, page_size) % 7 + 1)
    # The last causal block must require forcing even with 17 local P256 blocks.
    cache[-1, :, 0] = -16
    cache = cache.to(query.dtype)
    # Both requests share all physical prefix pages.
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(2, 1)
    q_lens = torch.empty(2, device="cuda", dtype=torch.int32)
    k_lens = torch.empty_like(q_lens)
    prefixes = torch.empty_like(q_lens)
    q_offsets = torch.empty(3, device="cuda", dtype=torch.int32)
    k_offsets = torch.empty_like(q_offsets)
    valid = torch.empty(total_q, device="cuda", dtype=torch.int32)
    work_ranges = torch.empty(_get_num_cta(query.device), device="cuda", dtype=torch.int64)
    work_info = torch.empty(heads * 3, device="cuda", dtype=torch.int64)
    workspace = torch.empty(32 << 20, device="cuda", dtype=torch.uint8)
    partials = torch.empty(total_q, heads, 64, device="cuda")
    sentinel = 0x13131313
    storage = torch.full((heads, total_q + 2, 41), sentinel,
                         device="cuda", dtype=torch.int32)
    carrier = storage[:, 1:total_q + 1, :33]
    guards = torch.ones_like(storage, dtype=torch.bool)
    guards[:, 1:total_q + 1, :33] = False
    scale = page_size // 128
    force = dict(force_begin_blocks=force_begin, force_end_blocks=force_end)

    for rank in range(tp):
        rows = page_size // tp
        offset = rank * rows // 128
        shard = (cache[:, rank::tp] if page_size == 128
                 else cache[:, rank * rows:(rank + 1) * rows]).contiguous()

        def score():
            plan = fmha_index_score_sharded_plan_device(
                q_lens, k_lens, q_offsets, k_offsets, prefixes, heads,
                total_q=total_q, max_query_len=total_q, max_seq_len=max_seq_len,
                packed_work_range=work_ranges, packed_work_info=work_info,
                workspace=workspace,
            )
            fmha_index_score_sharded(
                query, shard, table, plan, rank=rank, tp=tp,
                max_score=partials, sm_scale=1.0, compact=page_size == 256,
                fused_topk=True, **force,
            )
        graph = None
        for lengths, starts in [([65, 64], [4032, 3968]),
                                ([1, 128], [4096, 3968]),
                                ([0, 129], [4096, 3968]),
                                ([1, 64], [4096, 3968]),
                                ([0, 0], [4096, 3968]),
                                ([65, 64], [4032, 3968])]:
            kv = [q + p for q, p in zip(lengths, starts)]
            active_q = sum(lengths)
            q_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
            k_lens.copy_(torch.tensor(kv, dtype=torch.int32))
            prefixes.copy_(torch.tensor(starts, dtype=torch.int32))
            q_offsets.copy_(torch.tensor([0, lengths[0], active_q], dtype=torch.int32))
            k_offsets.copy_(torch.tensor([0, kv[0], sum(kv)], dtype=torch.int32))
            positions = [p + i for q, p in zip(lengths, starts) for i in range(q)]
            valid.zero_()
            valid[:active_q].copy_(torch.tensor([p // 128 + 1 for p in positions], dtype=torch.int32))
            expected = torch.empty((heads, active_q, 33), dtype=torch.int32)
            expected[..., :16] = torch.full((16,), -float("inf")).view(torch.int32)
            expected[..., 16:32] = -1
            expected[..., 32] = 1
            for token, pos in enumerate(positions):
                count = pos // 128 + 1
                ids = list(range(offset, count, scale))
                scores = {}
                for block in ids:
                    first_local_token = (block * 128 + rank if page_size == 128
                                         else block // 2 * 256 + rank * rows)
                    block_score = -16.0 if block == max_seq_len // 128 else -float(block % 7 + 1)
                    scores[block] = (block_score
                                     if first_local_token <= pos else -float("inf"))
                forced = {i for i in ids if i < force_begin
                          or i >= max(0, count - force_end)}
                chosen = sorted(sorted(ids, key=lambda i:
                                       (i not in forced, -scores[i], i))[:16])
                for slot, block in enumerate(chosen):
                    expected[:, token, slot] = torch.tensor(scores[block]).view(torch.int32)
                    expected[:, token, 16 + slot] = block
            partials.fill_(float("nan"))
            storage.fill_(sentinel)
            if graph is None:
                score()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    score()
            else:
                graph.replay()
            # Padding is outside the scorer's defined outputs; only active rows
            # reach selection, matching the serving prefill slice.
            if active_q:
                sparse_topk_merge_prefill_partials(
                    partials[:active_q], carrier[:, :active_q],
                    index_scale=scale, index_offset=offset,
                    num_valid_pages=valid[:active_q], **force,
                )
            torch.testing.assert_close(carrier[:, :active_q].cpu(), expected, rtol=0, atol=0)
            assert torch.isnan(partials[active_q:]).all()
            assert (carrier[:, active_q:] == sentinel).all()
            assert torch.all(storage[guards] == sentinel)
