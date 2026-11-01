# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

import pytest
import torch

from fmha_sm100 import sparse_topk_merge_prefill_partials


@pytest.mark.parametrize("scale,offset", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("force_begin,force_end", [(0, 0), (2, 3)])
@torch.inference_mode()
def test_merge_partials_tails_ties_and_reuse(scale, offset, force_begin, force_end):
    lengths = [0, 1, 15, 16, 17, 65]
    heads, tokens = 2, len(lengths)
    partials = torch.empty((tokens, heads, 64), device="cuda")
    valid = torch.empty(tokens, device="cuda", dtype=torch.int32)
    sentinel = 0x13131313
    storage = torch.full((heads, tokens * 2 + 3, 41), sentinel,
                         device="cuda", dtype=torch.int32)
    carrier = storage[:, 2:2 + tokens * 2:2, :33]
    guard = torch.ones_like(storage, dtype=torch.bool)
    guard[:, 2:2 + tokens * 2:2, :33] = False

    def merge():
        sparse_topk_merge_prefill_partials(
            partials, carrier, index_scale=scale, index_offset=offset,
            num_valid_pages=valid, force_begin_blocks=force_begin,
            force_end_blocks=force_end,
        )

    graph = None
    for iteration in range(3):
        current = lengths if iteration % 2 == 0 else lengths[::-1]
        source = torch.empty((tokens, heads, 64), dtype=torch.float32)
        source[..., :32] = -float("inf")
        source[..., 32:].view(torch.int32).fill_(
            torch.iinfo(torch.int32).max if iteration == 0 else -1
        )
        expected = torch.empty((heads, tokens, 33), dtype=torch.int32)
        expected[..., :16] = torch.full((16,), -float("inf")).view(torch.int32)
        expected[..., 16:32] = -1
        expected[..., 32] = 1
        for token, count in enumerate(current):
            local_ids = list(range(max(0, (count - offset + scale - 1) // scale)))
            for head in range(heads):
                scores = {
                    i: ([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")]
                        [(i + head + iteration) % 6])
                    for i in local_ids
                }
                # The low-score forced tail must survive both local truncations.
                forced = {
                    i for i in local_ids
                    if i * scale + offset < force_begin
                    or i * scale + offset >= max(0, count - force_end)
                }
                for i in forced:
                    scores[i] = -float("inf")

                def order(i):
                    return (i not in forced, -scores[i], i)

                for stage in range(2):
                    selected = sorted(local_ids[stage::2], key=order)[:16]
                    selected = selected[::2] + selected[1::2][::-1]
                    for lane, i in enumerate(selected):
                        slot = stage * 16 + lane
                        source[token, head, slot] = scores[i]
                        source[token, head, 32:].view(torch.int32)[slot] = i
                chosen = sorted(sorted(local_ids, key=order)[:16])
                for slot, i in enumerate(chosen):
                    expected[head, token, slot] = torch.tensor(scores[i]).view(torch.int32)
                    expected[head, token, 16 + slot] = i * scale + offset

        partials.copy_(source)
        valid.copy_(torch.tensor(current, dtype=torch.int32))
        storage.fill_(sentinel)
        if graph is None:
            merge()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                merge()
        else:
            graph.replay()
        torch.testing.assert_close(carrier.cpu(), expected, rtol=0, atol=0)
        assert torch.all(storage[guard] == sentinel)


@torch.inference_mode()
def test_merge_partials_discards_nan_candidates():
    partials = torch.empty((1, 1, 64), device="cuda")
    partials[..., :32] = float("nan")
    partials[..., 32:] = torch.arange(32, device="cuda", dtype=torch.int32).view(torch.float32)
    carrier = torch.empty((1, 1, 33), device="cuda", dtype=torch.int32)
    sparse_topk_merge_prefill_partials(partials, carrier)
    result = carrier.cpu()
    assert torch.isneginf(result[..., :16].view(torch.float32)).all()
    assert (result[..., 16:32] == -1).all()
    assert result[..., 32].item() == 1


@pytest.mark.parametrize("tp,page_size", [(2, 128), (4, 128), (2, 256), (4, 256)])
@torch.inference_mode()
def test_fused_prefill_multisegment_chunk_reuse(tp, page_size):
    from fmha_sm100.api import (
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
        sparse_topk_select_prefill,
    )

    torch.manual_seed(72)
    q_lens, prefixes = [0, 257, 65, 1], [127, 2049, 127, 8192]
    offsets = [0]
    for q in q_lens:
        offsets.append(offsets[-1] + q)
    lengths = [q + p for q, p in zip(q_lens, prefixes)]
    total_q, heads, chunk = sum(q_lens), 4, 128
    pages = (max(lengths) + page_size - 1) // page_size
    compact = page_size == 256
    scale = 2 if compact else 1
    columns = ((max(lengths) + 127) // 128 + scale - 1) // scale
    query = torch.randn(total_q, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    cache = torch.randn(pages, page_size, 128, device="cuda").to(query.dtype)
    table = torch.stack([torch.randperm(pages, device="cuda") for _ in q_lens]).int()
    table[:, 0] = table[0, 0].clone()
    valid = torch.cat([
        (torch.arange(q, device="cuda", dtype=torch.int32) + p + 128) // 128
        for q, p in zip(q_lens, prefixes)
    ])
    plans = []
    for begin in range(0, total_q, chunk):
        end = min(begin + chunk, total_q)
        chunk_lens = [max(0, min(end, off + q) - max(begin, off))
                      for q, off in zip(q_lens, offsets)]
        starts = [p + min(q, max(0, begin - off))
                  for q, p, off in zip(q_lens, prefixes, offsets)]
        plan = fmha_index_score_sharded_plan(
            torch.tensor(chunk_lens, dtype=torch.int32),
            torch.tensor([s + q for s, q in zip(starts, chunk_lens)], dtype=torch.int32),
            heads, qo_offset=torch.tensor(starts, dtype=torch.int32), device="cuda",
        )
        plans.append((begin, end, plan))
    partials = torch.empty(chunk, heads, 64, device="cuda")
    scores = torch.empty(chunk, heads, columns, device="cuda")
    actual = torch.empty(heads, total_q, 33, device="cuda", dtype=torch.int32)
    expected = torch.empty_like(actual)
    force = dict(force_begin_blocks=2, force_end_blocks=3)
    for rank in range(tp):
        rows = page_size // tp
        shard = (cache[:, rank::tp] if page_size == 128
                 else cache[:, rank * rows:(rank + 1) * rows]).contiguous()
        mapping = dict(index_scale=scale, index_offset=rank * scale // tp)

        def score_chunks(fused):
            for begin, end, plan in plans:
                count = end - begin
                scratch = partials[:count] if fused else scores[:count]
                fmha_index_score_sharded(
                    query[begin:end], shard, table, plan, rank=rank, tp=tp,
                    max_score=scratch, sm_scale=1.0, compact=compact,
                    fused_topk=fused, **(force if fused else {}),
                )
                if fused:
                    sparse_topk_merge_prefill_partials(
                        scratch, actual[:, begin:end], num_valid_pages=valid[begin:end],
                        **mapping, **force,
                    )
                else:
                    sparse_topk_select_prefill(
                        scratch, valid[begin:end], expected[:, begin:end],
                        **mapping, **force,
                    )

        score_chunks(False)
        score_chunks(True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            score_chunks(True)
        for _ in range(2):
            query.copy_((-query.float()).to(query.dtype))
            score_chunks(False)
            partials.fill_(float("nan"))
            actual.fill_(0x13131313)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
