"""CPU: IcpIndexer planning and launch order over the real device plan store and planner."""

from types import SimpleNamespace

import pytest
import torch

from fmha_sm100 import api
from fmha_sm100.icp import IcpIndexer, indexer as ix
from fmha_sm100.icp.scorer.decode import icp_decode_score

HL, NB, CAP = 2, 128, 2048  # local heads; max_model_len 16384 / page 128; batched tokens


class _Exchange:
    def __init__(self, group, *, rank, heads_local, extents, device, control_group=None):
        self.extents, self.calls, self.provenance = extents, [], {"kernel": "d3_fused"}
        self.geometry = []
        self.control_group = control_group

    def publish(self, scores, nvalid, forced_column, active, *, layer, token_offset,
                block_begin=0, block_stride=1):
        self.calls.append(("publish", layer, token_offset, scores.shape[0]))
        self.geometry.append((block_begin, block_stride))

    def merge_decode(self, out, *, layer, forced, n_ordinary):
        self.calls.append(("merge_decode", layer, out.shape[0]))

    def merge_prefill(self, candidates, out, *, forced, n_ordinary):
        self.calls.append(("merge_prefill", candidates.dtype, candidates.shape[0], out.shape[0]))

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def make(monkeypatch):
    calls = []
    monkeypatch.setattr(ix, "_Exchange", _Exchange)
    # Small score budgets: a 128-row decode chunk, so windows and D3 padding both appear.
    monkeypatch.setattr(ix, "_DECODE_SCORE_BYTES", 1 << 18)
    monkeypatch.setattr(ix, "_PREFILL_SCORE_BYTES", 8 << 20)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(icp_decode_score, "_compile_scan", lambda q, r, d, a, page_size=128, world_size=2: (
        lambda *args: calls.append(("cute", q, args[9:13]))))
    monkeypatch.setattr(ix._build, "_select_ext", lambda: SimpleNamespace(
        select_prefill_candidates=lambda *a, **k: calls.append(("select", a[0].shape, a[5:], k))))
    monkeypatch.setattr(api, "get_fmha_variant", lambda *key: SimpleNamespace(
        run=lambda *args: calls.append(("fmha", key, args))))
    monkeypatch.setattr(api, "_get_num_cta", lambda device: 8)
    workspace = torch.empty(32 << 20, dtype=torch.uint8)
    monkeypatch.setattr(api, "_alloc_workspace_buf", lambda *args: workspace)

    def build(layers=5, capture=(1, 2, 4, 8, 16, 32, 64), page_size=128, world_size=2, rank=1, control_group=None):
        monkeypatch.setattr(ix.dist, "get_world_size", lambda group: world_size)
        monkeypatch.setattr(ix.dist, "get_rank", lambda group: rank)
        indexer = IcpIndexer(group=None, rank=rank, topk_buffer=torch.zeros(CAP, 4 // world_size, 16, dtype=torch.int32),
                             num_sparse_layers=layers, index_head_dim=128, max_model_len=NB * 128,
                             max_num_batched_tokens=CAP, max_num_seqs=64,
                             cudagraph_capture_sizes=capture, page_size=page_size, control_group=control_group)
        indexer.allocate(max_decode_query_len=4)
        calls.clear()
        return indexer, calls

    return build


def _plan(indexer, query_lens, seq_lens, prefill, num_tokens=None, table=None):
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    return indexer.plan(
        block_table=torch.zeros(64, NB, dtype=torch.int32) if table is None else table,
        query_start_loc=qsl, query_start_loc_cpu=qsl, seq_lens=seq, seq_lens_cpu_upper_bound=seq,
        is_prefilling=torch.tensor(prefill), num_reqs=len(query_lens),
        num_tokens=qsl[-1].item() if num_tokens is None else num_tokens)


def _select(indexer, step, layer=4):
    tokens = step._num_tokens
    fp8 = torch.float8_e4m3fn
    indexer.select(step, torch.zeros(tokens, 2 * HL * 128, dtype=fp8),
                   torch.zeros(10, indexer._fragment, 128, dtype=fp8),
                   layer=layer, scale=0.5)
    return indexer._exchange.calls


def test_ladders_extents_and_slot_rule(make):
    assert ix._ladder(16) == [4, 6, 8, 12, 16]
    assert ix._ladder(100, ix._align4) == [4, 8, 12, 16, 24, 32, 48, 64, 96, 100]
    indexer, _ = make(capture=(1, 3, 24, 4096))
    assert indexer._decode_caps == [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048]
    assert indexer._prefill_caps == ix._ladder(CAP, ix._align4)
    assert indexer._exchange.extents == sorted({*indexer._decode_caps, *indexer._prefill_caps})
    assert indexer._decode_chunk == 128 and indexer._columns == ix._ladder(NB)
    with pytest.raises(ValueError, match="repeat a D3 slot"):
        make(layers=4)


def test_uniform_decode_is_cute_windows_and_d3_covers_the_extent(make):
    indexer, calls = make()
    # 200 live Q1 requests padded to 300 tokens: windows skip the padding, D3 publishes it.
    step = _plan(indexer, [1] * 200, [1000] * 200, [False] * 200, num_tokens=300)
    assert step.token_capacity == 512 and step.writer_args["icp_chunk_width"] == 128
    assert not any(k.startswith("icp_plan_") for k in step.writer_args)
    assert step.writer_args["icp_qo_offsets"].shape[0] == 4
    published = _select(indexer, step)
    assert [c[1:] for c in calls] == [(1, (0, 128, 0, 128)), (1, (128, 128, 128, 72))]
    rows = [(c[2], c[2] + c[3]) for c in published if c[0] == "publish"]
    assert sorted(rows) == [(0, 128), (128, 256), (256, 384), (384, 512)]
    assert published[-1] == ("merge_decode", 4, 512)


def test_nonuniform_decode_scores_with_onlyscoreicp_on_the_device_plan(make):
    indexer, calls = make()
    step = _plan(indexer, [1, 2, 1], [300, 129, 5], [False] * 3)
    store = indexer._store
    assert step.writer_args["icp_plan_segments"] is store.segments
    assert step.writer_args["icp_plan_num_ctas"] == 8 and step.token_capacity == 4
    published = _select(indexer, step)
    (_, key, args), = calls
    assert key == api._icp_scorer_variants()[0][1]  # max_qo_len 2 <= 64: single warpgroup
    assert args[1].shape == (4, 4, 128) and args[2].shape == (10, 1, 64, 128)
    assert args[4].data_ptr() == store.segments[0, 2].data_ptr() and args[4].shape == (3,)
    assert args[29] == NB and args[-3:] == (0, 1, 2)  # decode scores every column
    assert args[28].shape == (4, 4, NB) and args[11] == 0.5
    assert published == [("publish", 4, 0, 4), ("merge_decode", 4, 4)]


def test_prefill_scores_the_rung_covering_the_live_extent(make):
    indexer, calls = make()
    step = _plan(indexer, [100, 1], [100, 500], [True, False])
    width = indexer._waves[4]  # 500 tokens -> 4 live blocks -> rung 4
    assert step.writer_args["icp_chunk_width"] == width and step.token_capacity == 128
    published = _select(indexer, step, layer=3)
    (fmha, select) = calls
    assert fmha[2][29] == 4 and fmha[2][-3] == 0 and fmha[2][4].shape == (2,)
    assert select[1] == (128, 4, 4) and select[2] == (0, False)
    assert select[3] == {"global_block_stride": 1}
    assert published == [("merge_prefill", torch.int32, 128, 128)]


def test_capture_padding_beyond_live_requests_is_never_scored(make):
    indexer, calls = make()
    step = _plan(indexer, [2, 2], [300, 300], [False] * 2, num_tokens=8)
    assert step.token_capacity == 8
    _select(indexer, step)
    assert calls == [("cute", 2, (0, 8, 0, 2))]


def test_block_table_identity_is_latched(make):
    indexer, _ = make()
    first = torch.zeros(64, NB, dtype=torch.int32)
    _plan(indexer, [1], [10], [False], table=first)
    _plan(indexer, [1], [10], [False], table=first)
    with pytest.raises(RuntimeError, match="moved"):
        _plan(indexer, [1], [10], [False], table=torch.zeros(64, NB, dtype=torch.int32))


@pytest.mark.parametrize("world_size,rank", [(w, r) for w in (2, 4) for r in range(w)])
def test_p128_plan_keeps_global_block_ids_and_uses_rank_fragments(make, world_size, rank):
    control = object()
    indexer, calls = make(world_size=world_size, rank=rank, control_group=control)
    assert (indexer._world, indexer._hl, indexer._hg, indexer._fragment) == (
        world_size, 4 // world_size, 4, 128 // world_size)
    assert indexer._exchange.control_group is control
    step = _plan(indexer, [1, 2, 1], [300, 129, 5], [False] * 3)
    _select(indexer, step)
    (_, key, args), = calls
    assert key == api._icp_scorer_variants(128 // world_size)[0][1]
    assert args[2].shape == (10, 1, 128 // world_size, 128)
    assert args[28].shape == (4, 4, NB)
    assert args[-2:] == (rank, world_size)
    assert indexer._exchange.geometry == [(0, 1)]


@pytest.mark.parametrize("world_size", [2, 4])
def test_p128_prefill_candidates_keep_global_block_ids(make, world_size):
    indexer, calls = make(world_size=world_size)
    step = _plan(indexer, [100, 1], [100, 1000], [True, False])
    _select(indexer, step)
    fmha, select = calls
    assert fmha[2][29] == 8
    assert select[2] == (0, False)
    assert select[3] == {"global_block_stride": 1}


@pytest.mark.parametrize("world_size", [2, 4])
def test_other_physical_page_sizes_are_rejected(make, world_size):
    with pytest.raises(ValueError, match="P128/P256 index pages"):
        make(page_size=512, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("page_size", [128, 256])
def test_prewarm_contains_only_indexer_artifacts(monkeypatch, world_size, page_size):
    from fmha_sm100.icp import prewarm

    monkeypatch.setattr(prewarm._build, "arch", lambda: "103a")
    artifacts = prewarm._artifacts(None, page_size=page_size, world_size=world_size)
    assert len(artifacts) == len(prewarm._build.EXTENSIONS) + 2 + 4 * world_size
    labels = [name for name, _ in artifacts]
    assert len(set(labels)) == len(labels)
    assert all(name.startswith(("icp:", "fmha:OnlyScoreIcp", "decode:")) for name in labels)
    for q in range(1, 5):
        for rank in range(world_size):
            assert "decode:" + icp_decode_score._aot_function_name(q, rank, page_size, world_size) in labels


@pytest.mark.parametrize("fragment", [32, 64, 128])
def test_prefill_tiles_fit_rank_pages(fragment):
    from fmha_sm100 import jit

    for _, args in api._icp_scorer_variants(fragment):
        _, params = jit._variant_key_from_runtime(*args)
        stage_columns = int(params["tile_kv"].removeprefix("_")) // 2
        assert stage_columns == fragment
        assert not jit._FMHA_SM100_IMPOSSIBLE(params)


@pytest.mark.parametrize("world_size,rank", [(w, r) for w in (2, 4) for r in range(w)])
def test_p256_columns_map_to_global_b128_ids(make, world_size, rank):
    indexer, calls = make(page_size=256, world_size=world_size, rank=rank)
    begin = rank * (256 // world_size) // 128
    assert indexer._max_blocks == NB // 2
    assert indexer._fragment == 256 // world_size
    table = torch.zeros(64, NB, dtype=torch.int32)
    step = _plan(indexer, [1, 2, 1], [300, 129, 5], [False] * 3, table=table)
    _select(indexer, step)
    (_, key, args), = calls
    assert key == api._icp_scorer_variants(256 // world_size)[0][1]
    assert args[2].shape == (10, 1, 256 // world_size, 128)
    assert args[28].shape == (4, 4, NB // 2)
    assert indexer._exchange.geometry == [(begin, 2)]
    calls.clear()
    step = _plan(indexer, [100, 1], [100, 1000], [True, False], table=table)
    _select(indexer, step)
    _, select = calls
    assert select[2] == (begin, False)
    assert select[3] == {"global_block_stride": 2}


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("page_size", [128, 256])
def test_compound_main_views_alias_payload_and_scales(page_size, world_size):
    from fmha_sm100.icp.prewarm import _head_slot_views, _page_bytes

    slot, pitch = _page_bytes(page_size, world_size)
    storage = torch.randint(0, 256, (3, pitch), dtype=torch.uint8,
                            generator=torch.Generator().manual_seed(7))
    kd, ks, vd, vs = _head_slot_views(storage, page_size, world_size)
    heads = 4 // world_size
    for view, base, width in ((kd, 0, 64), (ks, page_size * 64, 8),
                               (vd, slot, 64), (vs, slot + page_size * 64, 8)):
        assert view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        assert view.shape == (3, heads, page_size, width)
        for page in range(3):
            for head in range(heads):
                offset = head * 2 * slot + base
                expected = storage[page, offset:offset + page_size * width].view(page_size, width)
                assert torch.equal(view[page, head], expected)


def test_main_attention_prewarm_is_separate():
    from fmha_sm100.icp import prewarm

    labels = [label for label, _ in prewarm._main_attention_artifacts(None, 256, 4)]
    assert labels[-2:] == ["nvfp4:prefill[bf16]", "nvfp4:prefill[fp8]"]
    assert all(label.startswith("nvfp4:decode:") for label in labels[:-2])
    assert any("split-KV reduction" in label for label in labels)
    assert any("nvfp4_v1" in label for label in labels)
