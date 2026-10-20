"""GPU: the ICP prefill selector, OnlyScoreIcp and the CuTe decode scan against torch oracles."""

import math
import os

import pytest
import torch

from fmha_sm100 import api
from fmha_sm100.icp import _build
from fmha_sm100.icp.scorer.decode import icp_decode_score as scorer

pytestmark = pytest.mark.gpu
PAGE, DIM, HEADS = 128, 128, 4
PROFILES = [(world, rank) for world in (2, 4) for rank in range(world)]
NEG_INF = -8388608  # float32 -inf bits


def _word(x):
    return torch.tensor([0.0 if x == 0 else x]).view(torch.int32).item()


def _select_oracle(scores, counts, forced, active):
    """C4 rows: per head the 16 best (score desc, id asc) live non-forced columns."""
    out = torch.empty((*scores.shape[:2], 16, 2), dtype=torch.int32)
    out[..., 0], out[..., 1] = NEG_INF, -1
    for t in (t for t, on in enumerate(active) if on):
        for h in range(scores.shape[1]):
            row = [(s, c) for c, s in enumerate(scores[t, h, : counts[t]].tolist()) if c != forced[t]]
            for k, (s, c) in enumerate(sorted(row, key=lambda p: (-p[0], p[1]))[:16]):
                out[t, h, k, 0], out[t, h, k, 1] = _word(s), c
    return out


@pytest.mark.parametrize("columns", [1, 32, 33, 128, 129, 8192])
def test_prefill_selector_every_kernel_width(cuda_device, columns):
    """Both sides of the tiny-warp (32), four-warp (128) and wide (8192) dispatch edges."""
    counts = [columns, columns, columns - 1, min(17, columns), 1, 0, (1 << 31) - 1]
    forced = [-1, columns // 2, -1, 0, -1, -1, (1 << 31) - 1]
    forced = [f if f < n else -1 for f, n in zip(forced, counts)]
    active = [True] * 6 + [False]  # padding keeps arbitrary metadata
    col = torch.arange(columns)
    scores = torch.full((len(counts), HEADS, columns), math.nan)
    for t in range(6):
        live = slice(0, counts[t])
        scores[t, 0, live] = (((col * 37 + t * 11) % 67).float() - 33)[live]
        scores[t, 1, live] = -math.inf
        scores[t, 2, live] = torch.where(col % 2 == 0, 0.0, -0.0)[live]  # signed-zero ties
        scores[t, 3, live] = (0x3F800000 + (col * 73 + t * 17) % 257).int().view(torch.float32)[live]
        if forced[t] >= 0:
            scores[t, :, forced[t]] = math.nan  # never read
    if columns > 1:
        scores[0, :, columns - 1] = 1e30  # a truncated row cannot pass
    expected = _select_oracle(scores, counts, forced, active)
    dev = cuda_device
    out = torch.full((len(counts), HEADS, 16, 2), 0x5A5A5A5A, dtype=torch.int32, device=dev)
    _build._select_ext().select_prefill_candidates(
        scores.to(dev), torch.tensor(counts, dtype=torch.int32, device=dev),
        torch.tensor(forced, dtype=torch.int32, device=dev), torch.tensor(active, device=dev),
        out.view(torch.float32), 0, False, global_block_stride=1)
    assert torch.equal(out.cpu(), expected)


def _requests(query_lens, histories, rank, seed, world_size, padded, page_size=128):
    """Shared logical FP8 keys, contiguous rank fragments and an independent causal oracle."""
    frag = page_size // world_size
    g = torch.Generator().manual_seed(seed)
    seqs = [h + q for q, h in zip(query_lens, histories)]
    blocks = [-(-s // page_size) for s in seqs]
    cols, pages = max(blocks), sum(blocks) + 3
    keys = (torch.randn(pages, page_size, DIM, generator=g) * 0.7).to(torch.float8_e4m3fn)
    perm = torch.randperm(pages, generator=g).int()
    table = torch.full((len(seqs), cols + 2), -(1 << 30), dtype=torch.int32)
    q = (torch.randn(sum(query_lens), HEADS, DIM, generator=g) * 0.7).to(torch.float8_e4m3fn)
    pos = torch.cat([torch.arange(n) + h for n, h in zip(query_lens, histories)])
    score = torch.full((q.shape[0], HEADS, cols), -math.inf)
    valid = torch.zeros_like(score, dtype=torch.bool)
    page_start, row_start = 0, 0
    for b, (n, count, length) in enumerate(zip(query_lens, blocks, seqs)):
        table[b, :count] = perm[page_start:page_start + count]
        local = keys[table[b, :count].long(), rank * frag:(rank + 1) * frag].float()
        dots = torch.einsum("thd,cpd->thcp", q[row_start:row_start + n].float(), local)
        key_pos = torch.arange(count)[:, None] * page_size + rank * frag + torch.arange(frag)
        visible = (key_pos[None] <= pos[row_start:row_start + n, None, None]) & (key_pos[None] < length)
        score[row_start:row_start + n, :, :count] = torch.where(visible[:, None], dots, -math.inf).amax(-1)
        valid[row_start:row_start + n, :, :count] = visible.any(-1)[:, None]
        row_start += n
        page_start += count
    pitch = frag * DIM + (256 if padded else 0)
    storage = torch.full((pages * pitch,), 0x7F, dtype=torch.uint8)
    _, local = _fragment_view(storage, "cpu", world_size, padded, page_size)
    local.copy_(keys[:, rank * frag:(rank + 1) * frag])
    return q, storage, table, pos, seqs, score, valid


def _fragment_view(storage, device, world_size, padded, page_size=128):
    frag = page_size // world_size
    pitch, offset = frag * DIM + (256 if padded else 0), 128 if padded else 0
    storage = storage.to(device)
    return storage, storage.view(torch.float8_e4m3fn).as_strided(
        (storage.numel() // pitch, frag, DIM), (pitch, DIM, 1), offset)


def _write_device_plan(store, query_lens, seqs):
    """Host model of the writer's plan CTAs: every (tile, head, request) once, cheapest CTA."""
    n, ctas = len(query_lens), store.num_ctas
    qo_off = [0, *torch.tensor(query_lens).cumsum(0).tolist()]
    kv_off = [0, *torch.tensor(seqs).cumsum(0).tolist()]
    causal = [s - q for q, s in zip(query_lens, seqs)]
    rows = [qo_off, kv_off, query_lens, seqs, causal]
    for r, values in enumerate(rows):
        store.segments[0, r, : len(values)] = torch.tensor(values, dtype=torch.int32)
    cost, buckets = [165] * ctas, [[] for _ in range(ctas)]
    for qt in range(max(-(-q // 128) for q in query_lens) - 1, -1, -1):
        for b in (b for b in range(n) if qt < -(-query_lens[b] // 128)):
            iters = -(-max(0, min((qt + 1) * 128 + causal[b], seqs[b])) // 256)
            for h in range(store.num_heads):
                cta = min(range(ctas), key=lambda c: (cost[c], c))
                buckets[cta].append((qt << 32) | (h << 16) | b)
                cost[cta] += 43 * iters + 110
    ranges, info = [], []
    for items in buckets:
        ranges.append(len(info) | ((len(info) + len(items)) << 32))
        info += items
    store.work[0, :ctas] = torch.tensor(ranges)
    store.work[0, ctas:ctas + len(info)] = torch.tensor(info)


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("world_size,rank", PROFILES)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("query_lens,histories", [
    ([300, 64, 65, 1, 129], [1000, 0, 0, 5000, 200]),  # mixed; rank 1 has no Q64 fragment
    ([1, 2, 3, 1, 1, 1], [70000, 129, 31, 63, 95, 0]),  # non-uniform decode: the single-warpgroup variant
])
def test_onlyscoreicp_on_a_writer_model_device_plan(cuda_device, world_size, rank, padded, query_lens, histories, page_size):
    q, storage, table, _, seqs, score_ref, valid_ref = _requests(query_lens, histories, rank, 7, world_size, padded, page_size)
    cols, dev = score_ref.shape[-1], cuda_device
    store = api._IcpDevicePlanStore(device=dev, num_slots=1, max_segments=len(seqs),
                                    num_heads=HEADS, max_chunk_tokens=q.shape[0])
    _write_device_plan(store, query_lens, seqs)
    table = table.to(dev)
    plan = api._fmha_sm100_icp_plan(
        torch.tensor(query_lens, dtype=torch.int32), HEADS, icp_rank=rank, block_table=table,
        block_table_row_stride=table.stride(0), block_table_row_begin=0,
        device_plan=store.slot(0), max_k_tiles=cols,
        page_size=page_size // world_size, world_size=world_size)
    _storage, keys = _fragment_view(storage, dev, world_size, padded, page_size)
    score = torch.full((q.shape[0], HEADS, cols + 3), math.nan, device=dev)
    valid = torch.full_like(score, 201, dtype=torch.uint8)
    api._fmha_sm100_icp(q.to(dev), keys.unsqueeze(1), plan, max_score=score[..., :cols],
                        valid_score=valid[..., :cols], sm_scale=DIM ** -0.5,
                        qo_offset=store.segments[0, 4, : len(seqs)])
    score, valid = score.cpu(), valid.cpu()
    assert torch.equal(valid[..., :cols].bool(), valid_ref)
    assert bool(torch.isneginf(score[..., :cols][~valid_ref]).all())
    torch.testing.assert_close(score[..., :cols][valid_ref], score_ref[valid_ref], rtol=2e-5, atol=1e-4)
    assert bool(torch.isnan(score[..., cols:]).all()) and bool((valid[..., cols:] == 201).all())


@pytest.fixture(scope="module")
def decode_scan(cuda_device):
    arch = torch.cuda.get_device_capability(cuda_device)
    # The export targets the device arch; the DSL reads it at first import.
    os.environ.setdefault("CUTE_DSL_ARCH", f"sm_{arch[0]}{arch[1]}a")

    def get(query_len, rank, world_size, page_size):
        if not scorer._aot_path(query_len, rank, arch, page_size, world_size).is_file():
            scorer._export_aot_scan(query_len, rank, arch, page_size, world_size)
        return scorer._compile_scan(query_len, rank, cuda_device.index, arch, page_size, world_size)

    return get


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("world_size,rank", PROFILES)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
def test_cute_decode_scan_matches_torch(cuda_device, decode_scan, query_len, world_size, rank, padded, page_size):
    seqs = [1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256, 257, 4000]
    query_lens = [query_len] * len(seqs)
    histories = [max(0, s - query_len) for s in seqs]
    seqs = [h + query_len for h in histories]
    q, storage, table, pos, _, score_ref, valid_ref = _requests(query_lens, histories, rank, 3, world_size, padded, page_size)
    dev, tokens, cols = cuda_device, q.shape[0], score_ref.shape[-1]
    _storage, keys = _fragment_view(storage, dev, world_size, padded, page_size)
    qsl = torch.arange(0, tokens + 1, query_len, dtype=torch.int32, device=dev)
    active = torch.ones(tokens, dtype=torch.uint8, device=dev)
    active[-1] = 0  # an inactive row: undefined output, never read
    q, table, pos = q.to(dev), table.to(dev), pos.to(dev)
    seqs_device = torch.tensor(seqs, dtype=torch.int32, device=dev)
    split = (len(seqs) // 2) * query_len
    scan = decode_scan(query_len, rank, world_size, page_size)
    for t0, t1 in ((0, split), (split, tokens)):
        score = torch.full((t1 - t0, HEADS, cols), math.nan, device=dev)
        valid = torch.zeros_like(score, dtype=torch.uint8)

        def launch():
            scan(q, keys, table, score, seqs_device, qsl, pos, active, valid,
                 t0, t1 - t0, t0 // query_len, (t1 - t0) // query_len, 128)

        launch()
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        live = valid_ref[t0:t1].clone()
        live[(active[t0:t1] == 0).cpu()] = False
        for _ in range(3):
            score.fill_(math.nan)
            valid.zero_()
            graph.replay()
            assert bool((valid.cpu()[live] == 1).all())
            torch.testing.assert_close(score.cpu()[live], score_ref[t0:t1][live], rtol=2e-5, atol=1e-4)


@pytest.mark.distributed
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("phase", ["decode", "nonuniform_decode", "prefill"])
def test_indexer_pipeline_matches_global_dense_blocks(icp_group, cuda_device, phase, page_size):
    """Actual plan/select wiring; metadata is a CPU writer oracle, not the native writer."""
    import torch.distributed as dist

    from fmha_sm100.icp import IcpIndexer

    world_size = dist.get_world_size(icp_group)
    fragment, heads_local = page_size // world_size, HEADS // world_size
    rank, dev = dist.get_rank(icp_group), cuda_device
    lengths = [4, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256, 257, 4099]
    query_lens = [4] * len(lengths)
    if phase == "nonuniform_decode":
        query_lens = [i % 4 + 1 for i in range(len(lengths))]
    if phase == "prefill":
        query_lens = [65] + [1] * (len(lengths) - 2) + [4]
        lengths[0] = 65
    histories = [n - q for n, q in zip(lengths, query_lens)]
    counts = [math.ceil(n / page_size) for n in lengths]
    pages, rows, cap = sum(counts) + 3, sum(query_lens), 128
    gen = torch.Generator(device="cpu").manual_seed(142)
    # Small FP8 integers make QK exact in both FP16-MMA and FP32 references,
    # and deliberately exercise score ties with deterministic block-ID order.
    all_keys = torch.randint(-3, 4, (pages, page_size, DIM), generator=gen).to(torch.float8_e4m3fn)
    query = torch.randint(-3, 4, (rows, 4, 128), generator=gen).to(torch.float8_e4m3fn)
    permutation = torch.randperm(pages, generator=gen).int()
    table = torch.full((len(lengths), max(counts) + 2), -(1 << 30), dtype=torch.int32)
    expected = torch.full((cap, 4, 16), -1, dtype=torch.int32)
    positions, page_offset, row_offset = [], 0, 0
    for b, (length, qlen, history, npage) in enumerate(zip(lengths, query_lens, histories, counts)):
        table[b, :npage] = permutation[page_offset:page_offset + npage]
        # The oracle sees complete physical pages and reduces contiguous B128
        # blocks; it does not use rank-fragment scores or candidate mappings.
        dense_keys = all_keys[table[b, :npage].long()].float().reshape(-1, 128)
        dots = torch.einsum("thd,kd->thk", query[row_offset:row_offset + qlen].float(), dense_keys)
        for t in range(qlen):
            position = history + t
            positions.append(position)
            forced = position // 128
            dots[t, :, position + 1:] = -math.inf
            block_scores = dots[t].reshape(4, npage * (page_size // PAGE), PAGE).amax(-1)
            for h in range(4):
                best = sorted(range(forced), key=lambda block: (-float(block_scores[h, block]), block))[:15]
                ids = sorted(best + [forced])
                expected[row_offset + t, h, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
        page_offset += npage
        row_offset += qlen

    keys = all_keys[:, rank * fragment:(rank + 1) * fragment].contiguous().to(dev)
    query, table = query.to(dev), table.to(dev)
    positions = torch.tensor(positions, dtype=torch.int64)
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    seq = torch.tensor(lengths, dtype=torch.int32)
    out = torch.full((cap, heads_local, 16), -77, dtype=torch.int32, device=dev)
    indexer = IcpIndexer(group=icp_group, rank=rank, topk_buffer=out, num_sparse_layers=3,
                         index_head_dim=128, max_model_len=max(lengths),
                         max_num_batched_tokens=cap, max_num_seqs=len(lengths),
                         cudagraph_capture_sizes=[4, 8, 16, 32, 64, 128], page_size=page_size)
    try:
        indexer.allocate(max_decode_query_len=4)
        step = indexer.plan(block_table=table, query_start_loc=qsl.to(dev),
                            query_start_loc_cpu=qsl, seq_lens=seq.to(dev),
                            seq_lens_cpu_upper_bound=seq,
                            is_prefilling=torch.full((len(lengths),), phase == "prefill"),
                            num_reqs=len(lengths), num_tokens=rows)
        forced = (positions // 128).int()
        visible = positions + 1
        local_count = (visible // page_size + (visible % page_size > rank * fragment)).int()
        block_stride, block_begin = page_size // PAGE, rank * fragment // PAGE
        local_forced = torch.where((forced % block_stride == block_begin) &
                                   (forced // block_stride < local_count), forced // block_stride, -1)
        for target, values in ((indexer._positions, positions), (indexer._nvalid, local_count),
                               (indexer._forced_column, local_forced), (indexer._forced, forced),
                               (indexer._global_nvalid, forced + 1),
                               (indexer._n_ordinary, forced.clamp(max=15))):
            target[:rows].copy_(values)
        indexer._active[:rows] = True
        if phase != "decode":
            _write_device_plan(indexer._store, query_lens, lengths)
            indexer._qo_offset[0, :len(lengths)].copy_(torch.tensor(histories, dtype=torch.int32))

        def check():
            actual = out[:step.token_capacity].cpu()
            reference = expected[:step.token_capacity, rank * heads_local:(rank + 1) * heads_local]
            flag = torch.tensor(int(torch.equal(actual, reference)), device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group)
            assert flag.item() == 1, (phase, rank)

        for layer in range(3):
            indexer.select(step, query, keys, layer=layer, scale=128 ** -0.5)
            check()
        if phase != "prefill":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for layer in range(3):
                    indexer.select(step, query, keys, layer=layer, scale=128 ** -0.5)
            for _ in range(3):
                out.fill_(-77)
                graph.replay()
                check()
        assert int(indexer._exchange._status.item()) == 0
        assert int(indexer._exchange._k2_status.item()) == 0
    finally:
        indexer.close()
