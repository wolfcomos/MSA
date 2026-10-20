"""2 or 4 ranks: the D3 route and the NCCL route merge to the same top-k as a dense torch oracle.

Run with `torchrun --nproc_per_node=2 -m pytest tests/icp/test_exchange.py`.
"""

import os

import pytest
import torch
import torch.distributed as dist

from fmha_sm100.icp import _build
from fmha_sm100.icp.exchange import _Exchange

pytestmark = pytest.mark.distributed
WORLD = int(os.environ.get("WORLD_SIZE", "2"))
HL = 4 // WORLD
EXTENTS = [64, 512, 2048]


def _inputs(tokens, columns, seed):
    """All ranks' score planes (unique scores), per-token live counts, forcing and planes."""
    g = torch.Generator().manual_seed(seed)
    hg = WORLD * HL
    scores = torch.randperm(WORLD * tokens * hg * columns, generator=g).float()
    scores = scores.view(WORLD, tokens, hg, columns) / 7
    nvalid = torch.randint(0, columns + 1, (tokens,), generator=g, dtype=torch.int32)
    active = torch.rand(tokens, generator=g) > 0.1
    has_forced = (nvalid > 0) & (torch.rand(tokens, generator=g) < 0.8)
    forced_column = torch.where(has_forced, nvalid - 1, -1).int()
    forced = torch.where(active & has_forced, forced_column, -1).int()
    n_ordinary = torch.where(forced >= 0, forced.clamp(max=15), 0).int()
    return scores, nvalid, forced_column, active, forced, n_ordinary


def _oracle(scores, nvalid, active, forced, n_ordinary, rank):
    """Block max over all fragments; the n_ordinary best non-forced ids + forced, ascending."""
    dense = scores.amax(0)[:, rank * HL:(rank + 1) * HL]
    out = torch.full((*dense.shape[:2], 16), -1, dtype=torch.int32)
    for t in (t for t in range(dense.shape[0]) if active[t]):
        for h in range(HL):
            row = dense[t, h, : nvalid[t]].clone()
            if forced[t] >= 0:
                row[forced[t]] = -torch.inf
            ids = row.argsort(descending=True)[: n_ordinary[t]].tolist()
            ids = sorted(ids + ([int(forced[t])] if forced[t] >= 0 else []))
            out[t, h, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return out


@pytest.mark.parametrize("tokens,columns", [(64, 1172), (37, 33), (512, 300), (2048, 64)])
def test_d3_and_nccl_routes_match_the_oracle(icp_group, cuda_device, tokens, columns):
    rank, dev = dist.get_rank(icp_group), cuda_device
    exchange = _Exchange(icp_group, rank=rank, heads_local=HL, extents=EXTENTS, device=dev)
    try:
        assert exchange.provenance["symm_backend"] == "CUDA"
        extent = next(e for e in EXTENTS if e >= tokens)
        select = _build._select_ext()
        for layer in range(7):  # wraps the 3 D3 slots twice
            scores, nvalid, forced_column, active, forced, n_ordinary = (
                x.to(dev) for x in _inputs(tokens, columns, seed=100 * layer + tokens))
            pad = extent - tokens  # rows past the live tokens are published inactive
            mine = torch.nn.functional.pad(scores[rank], (0, 0, 0, 0, 0, pad))
            nvalid_e, column_e = (torch.nn.functional.pad(x, (0, pad)) for x in (nvalid, forced_column))
            active_e = torch.nn.functional.pad(active, (0, pad))
            forced_e = torch.nn.functional.pad(forced, (0, pad), value=-1)
            ordinary_e = torch.nn.functional.pad(n_ordinary, (0, pad))
            d3 = torch.full((extent, HL, 16), 0x5A5A5A5A, dtype=torch.int32, device=dev)
            exchange.publish(mine, nvalid_e, column_e, active_e, layer=layer, token_offset=0)
            exchange.merge_decode(d3, layer=layer, forced=forced_e, n_ordinary=ordinary_e)
            candidates = torch.empty((extent, WORLD * HL, 16, 2), dtype=torch.float32, device=dev)
            select.select_prefill_candidates(mine, nvalid_e, column_e, active_e, candidates, 0, False,
                                             global_block_stride=1)
            nccl = torch.full_like(d3, 0x5A5A5A5A)
            exchange.merge_prefill(candidates.view(torch.int32), nccl, forced=forced_e,
                                   n_ordinary=ordinary_e)
            expected = _oracle(scores.cpu(), nvalid.cpu(), active.cpu(), forced.cpu(),
                               n_ordinary.cpu(), rank)
            ok = torch.equal(d3[:tokens].cpu(), expected) and torch.equal(nccl[:tokens].cpu(), expected)
            ok = ok and bool((d3[tokens:] == -1).all()) and bool((nccl[tokens:] == -1).all())
            flag = torch.tensor([int(ok)], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group)
            assert flag.item() == 1, (layer, rank)
        torch.cuda.synchronize(dev)
        assert int(exchange._status.item()) == 0 and int(exchange._k2_status.item()) == 0
    finally:
        exchange.close()


def _fragment_inputs(columns, seed):
    """Independent B128 oracle for contiguous rank fragments of P128 pages."""
    g = torch.Generator().manual_seed(seed)
    visible = [0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97,
               127, 128, 129, columns * 128 - 1, columns * 128]
    tokens, hg, fragment = len(visible), WORLD * HL, 128 // WORLD
    scores = torch.randint(-5, 6, (WORLD, tokens, hg, columns), generator=g).float()
    active = torch.tensor([v > 0 for v in visible])
    counts = torch.tensor([[v // 128 + (v % 128 > r * fragment) for v in visible]
                           for r in range(WORLD)], dtype=torch.int32)
    forced = torch.tensor([(v - 1) // 128 if v else -1 for v in visible], dtype=torch.int32)
    forced_columns = torch.full_like(counts, -1)
    ordinary = torch.where(forced >= 0, forced.clamp(max=15), 0).int()
    expected = torch.full((tokens, hg, 16), -1, dtype=torch.int32)
    for t, length in enumerate(visible):
        dense = torch.full((hg, (length + 127) // 128), -torch.inf)
        for r in range(WORLD):
            n = int(counts[r, t])
            scores[r, t, :, n:] = torch.nan
            if 0 <= forced[t] < n:
                forced_columns[r, t] = forced[t]
                scores[r, t, :, forced_columns[r, t]] = torch.nan
            for c in range(n):
                if c != forced[t]:
                    dense[:, c] = torch.maximum(dense[:, c], scores[r, t, :, c])
        if not length:
            continue
        for h in range(hg):
            ids = sorted((b for b in range(dense.shape[1]) if b != forced[t]),
                         key=lambda b: (-float(dense[h, b]), b))[:int(ordinary[t])]
            ids = sorted(ids + [int(forced[t])])
            expected[t, h, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return scores, counts, forced_columns, active, forced, ordinary, expected


@pytest.mark.parametrize("columns", [33, 129])
def test_rank_fragments_forcing_ties_and_graph_slots(icp_group, cuda_device, columns):
    rank, dev, extent = dist.get_rank(icp_group), cuda_device, 64
    exchange = _Exchange(icp_group, rank=rank, heads_local=HL, extents=[extent], device=dev)
    select = _build._select_ext()
    try:
        for layer in range(6):
            scores, counts, columns_forced, active, forced, ordinary, expected = _fragment_inputs(columns, layer)
            tokens = len(active)
            pad = extent - tokens
            mine = torch.nn.functional.pad(scores[rank], (0, 0, 0, 0, 0, pad)).to(dev)
            nvalid = torch.nn.functional.pad(counts[rank], (0, pad)).to(dev)
            local_forced = torch.nn.functional.pad(columns_forced[rank], (0, pad), value=-1).to(dev)
            active = torch.nn.functional.pad(active, (0, pad)).to(dev)
            forced = torch.nn.functional.pad(forced, (0, pad), value=-1).to(dev)
            ordinary = torch.nn.functional.pad(ordinary, (0, pad)).to(dev)
            out = torch.empty((extent, HL, 16), dtype=torch.int32, device=dev)
            candidates = torch.empty((extent, WORLD * HL, 16, 2), dtype=torch.float32, device=dev)
            exchange.publish(mine, nvalid, local_forced, active, layer=layer,
                             token_offset=0)
            exchange.merge_decode(out, layer=layer, forced=forced, n_ordinary=ordinary)
            reference = expected[:, rank * HL:(rank + 1) * HL]
            ok = torch.equal(out[:tokens].cpu(), reference) and bool((out[tokens:] == -1).all())
            select.select_prefill_candidates(mine, nvalid, local_forced, active, candidates,
                                             0, False, global_block_stride=1)
            exchange.merge_prefill(candidates.view(torch.int32), out, forced=forced,
                                   n_ordinary=ordinary)
            ok &= torch.equal(out[:tokens].cpu(), reference) and bool((out[tokens:] == -1).all())
            flag = torch.tensor(int(ok), device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group)
            assert flag.item() == 1, (layer, rank)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer in range(3):
                exchange.publish(mine, nvalid, local_forced, active, layer=layer,
                                 token_offset=0)
                exchange.merge_decode(out, layer=layer, forced=forced, n_ordinary=ordinary)
        for _ in range(4):
            out.fill_(-77)
            graph.replay()
            ok = torch.equal(out[:tokens].cpu(), reference) and bool((out[tokens:] == -1).all())
            flag = torch.tensor(int(ok), device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group)
            assert flag.item() == 1, rank
        assert int(exchange._status.item()) == 0 and int(exchange._k2_status.item()) == 0
    finally:
        exchange.close()
