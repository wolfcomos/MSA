"""Indexer context parallelism: one plan per step; score, select, exchange per layer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .. import api
from . import _build
from .exchange import _SLOTS, _Exchange
from .scorer.decode import icp_decode_score

# Logical index pages and sparse ranking blocks both cover 128 tokens.
_PAGE = 128
# Inner MSA tile; the planner pins it to 128 whenever icp_c > 1.
_QO_TILE = 128
# Decode fp32 score + uint8 validity scratch budget (r13 sizing).
_DECODE_SCORE_BYTES = 32 << 20
_MAX_DECODE_CHUNK = 1024
# Retained prefill score + validity arena, once per indexer (not per layer).
_PREFILL_SCORE_BYTES = 1 << 30
_MAX_PREFILL_WAVE = 16384
# r13's fixed CuTe decode split.
_CUTE_SPLIT_K = 128
_MAX_CUTE_QUERY_LEN = 4

_logger = logging.getLogger(__name__)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _align4(n: int) -> int:
    # Selected-ID rows are int4-aligned in the shared top-k buffer.
    return _cdiv(n, 4) * 4


def _ladder(top: int, align: Callable[[int], int] = int) -> list[int]:
    """``top`` plus the rungs ``o`` and ``1.5 o`` of every octave ``o = 4 * 2^k <= top``."""
    rungs, octave = {top}, 4
    while octave <= top:
        rungs.update(r for r in (align(octave), align(octave + octave // 2)) if r <= top)
        octave *= 2
    return sorted(rungs)


def _wave(columns: int, heads: int, tokens: int, budget: int, max_wave: int) -> int:
    """Query rows of ``[wave, heads, columns]`` fp32 + uint8 that fit ``budget``."""
    tiles = max(1, (budget // (heads * columns * 5)) // _QO_TILE)
    return max(_QO_TILE, min(tiles * _QO_TILE, max_wave, _cdiv(tokens, _QO_TILE) * _QO_TILE))


def _uniform_decode_query_len(qsl: torch.Tensor) -> int:
    """The shared query length of a pure-decode step the CuTe scorer takes, else 0."""
    if int(qsl[0]) != 0:
        return 0
    lengths = qsl[1:] - qsl[:-1]
    live = lengths[lengths > 0]
    # V2 packs live requests before zero-length capture padding.
    if not live.numel() or not torch.equal(lengths[: live.numel()], live):
        return 0
    query_len = int(live[0])
    uniform = query_len <= _MAX_CUTE_QUERY_LEN and bool((live == query_len).all())
    return query_len if uniform else 0


@dataclass(frozen=True)
class IcpStep:
    """One forward's plan for every sparse layer: the writer's ICP kwargs and top-k extent."""

    #: Exact ICP kwargs of vLLM's fused KV writer on the first sparse layer.
    writer_args: dict
    #: Rows of ``topk_buffer`` that :meth:`IcpIndexer.select` writes.
    token_capacity: int
    _num_tokens: int
    _windows: tuple[tuple[Callable, Callable], ...]
    _padding: tuple[Callable, ...]
    _merge: Callable[[int], None]


class IcpIndexer:
    """One per model: ICP sparse indexer; owns plan store, score arena, D3 window, carriers."""

    #: D3 build facts for the startup log; ``kernel`` is ``"d3_fused"``.
    provenance: dict

    def __init__(self, *, group, rank: int, topk_buffer: torch.Tensor, num_sparse_layers: int,
                 index_head_dim: int, max_model_len: int, max_num_batched_tokens: int,
                 max_num_seqs: int, cudagraph_capture_sizes, page_size: int = 128,
                 control_group=None) -> None:
        """Collective (D3 rendezvous); ``topk_buffer`` is the shared ``[T_cap, H_local, 16]``."""
        # With no ACK, a sweep whose wrap repeats a slot would race silently.
        if num_sparse_layers % _SLOTS == 1:
            raise ValueError(f"ICP: {num_sparse_layers} sparse layers repeat a D3 slot "
                             f"across the wrap of {_SLOTS} slots")
        self._world = dist.get_world_size(group)
        if self._world not in (2, 4) or page_size != _PAGE:
            raise ValueError("ICP requires TP2/TP4 and 128-token index pages")
        if rank != dist.get_rank(group):
            raise ValueError("ICP rank must be the rank within its process group")
        self._page_size = page_size
        self._fragment = page_size // self._world
        self._rank, self._topk, self._head_dim = rank, topk_buffer, index_head_dim
        self._hl = topk_buffer.shape[1]
        self._hg = self._world * self._hl
        if self._hg != 4:
            raise ValueError("ICP decode scoring requires four global index heads")
        self._max_num_seqs = max_num_seqs
        self._max_blocks = _cdiv(max_model_len, page_size)
        cap = _align4(max_num_batched_tokens)
        # Rank-identical exchange extents: capture sizes + powers of two (decode), octaves (prefill).
        decode, eager = {_align4(s) for s in cudagraph_capture_sizes or () if 0 < s <= cap}, 4
        while eager < cap:
            decode.add(eager)
            eager *= 2
        self._decode_caps = sorted({*decode, cap})
        self._prefill_caps = _ladder(cap, _align4)
        self._exchange = _Exchange(
            group, rank=rank, heads_local=self._hl, device=topk_buffer.device,
            extents=sorted({*self._decode_caps, *self._prefill_caps}),
            control_group=control_group)
        self.provenance = self._exchange.provenance
        self._allocated = False

    def allocate(self, *, max_decode_query_len: int) -> None:
        """Allocate and load everything the steps use (idempotent; before graph capture)."""
        # Each metadata builder brings its own block table; capture follows the last one.
        self._table = None
        if self._allocated:
            return
        self._allocated = True
        device, hg, nb = self._topk.device, self._hg, self._max_blocks
        tokens = self._decode_caps[-1]
        self._columns = _ladder(nb)
        self._waves = {c: _wave(c, hg, tokens, _PREFILL_SCORE_BYTES, _MAX_PREFILL_WAVE)
                       for c in self._columns}
        # The decode chunk is a startup constant of every decode graph.
        self._decode_chunk = chunk = _wave(nb, hg, tokens, _DECODE_SCORE_BYTES,
                                           _MAX_DECODE_CHUNK)
        waves = set(self._waves.values())
        # Rounded so the last chunk's candidate slice is whole for every width.
        cap_padded = max(_cdiv(tokens, w) * w for w in {chunk, *waves})
        # One device plan slot per chunk position either band can reach.
        num_slots = max(_cdiv(tokens, chunk), _cdiv(tokens, min(waves)))
        pool_chunk = max(chunk, *waves)
        reqs_per_slot = min(pool_chunk, self._max_num_seqs)
        # Unsplit direct-table plans, derived on device by the KV writer.
        self._store = api._IcpDevicePlanStore(
            device=device, num_slots=num_slots, max_segments=reqs_per_slot,
            num_heads=hg, max_chunk_tokens=pool_chunk)

        # One score/validity arena; the decode plane is its prefix.
        plane = (chunk, hg, nb)
        arena = max(chunk * hg * nb, *(w * hg * c for c, w in self._waves.items()))
        score_arena = torch.empty(arena, dtype=torch.float32, device=device)
        valid_arena = torch.empty(arena, dtype=torch.uint8, device=device)

        def views(shape):
            n = shape[0] * shape[1] * shape[2]
            return score_arena[:n].view(shape), valid_arena[:n].view(shape)

        # CuTe ABI2 leaves tails undefined; selection reads exact live prefixes.
        self._scores, self._valid = views(plane)
        self._prefill_views = {c: views((w, hg, c)) for c, w in self._waves.items()}
        self._candidates = torch.empty((cap_padded, hg, 16, 2), dtype=torch.float32,
                                       device=device)

        def full(dtype, fill=0):
            return torch.full((cap_padded,), fill, dtype=dtype, device=device)

        self._positions = full(torch.int64)
        self._active = full(torch.bool)
        # Same bytes retyped once: the CuTe ABI takes uint8, the selector bool.
        self._active_bytes = self._active.view(torch.uint8)
        self._nvalid = full(torch.int32)
        self._global_nvalid = full(torch.int32)
        self._forced_column = full(torch.int32, -1)
        self._forced = full(torch.int32, -1)
        self._n_ordinary = full(torch.int32)
        # Slot-major causal offsets: row `slot` is at a startup-constant address.
        self._qo_offset = torch.zeros((num_slots, reqs_per_slot), dtype=torch.int32,
                                      device=device)
        # `(data_ptr, row_stride)` of the block table, latched on the first step.
        self._table = None

        capability = tuple(torch.cuda.get_device_capability(device.index))
        self._scans = {
            q: icp_decode_score._compile_scan(
                q, self._rank, device.index, capability, self._page_size, self._world)
            for q in range(1, min(_MAX_CUTE_QUERY_LEN, max_decode_query_len) + 1)
        }
        self._select = _build._select_ext()
        for _, args in api._icp_scorer_variants(self._fragment):
            api.get_fmha_variant(*args)
        _logger.info(
            "ICP_INDEXER rank=%d decode_chunk=%d prefill_waves=%s plan_slots=%d "
            "arena=%.1fMiB", self._rank, chunk, sorted(waves), num_slots, arena * 5 / 2**20)

    def plan(self, *, block_table, query_start_loc, query_start_loc_cpu, seq_lens,
             seq_lens_cpu_upper_bound, is_prefilling, num_reqs: int,
             num_tokens: int) -> IcpStep:
        """Plan one forward for every sparse layer; host only: no device read, no allocation."""
        nb = self._max_blocks
        qsl = query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
        q_begin, q_end = qsl[:-1], qsl[1:]
        live_tokens = int(q_end[-1]) if num_reqs > 0 else 0
        # The phase decides the transport, so it must be identical on every rank.
        has_prefill = bool(is_prefilling[:num_reqs].any().item())
        caps = self._prefill_caps if has_prefill else self._decode_caps
        cap = next(c for c in caps if num_tokens <= c)
        query_len = 0 if has_prefill else _uniform_decode_query_len(qsl)
        scan = self._scans.get(query_len)
        kv_ub = seq_lens_cpu_upper_bound[:num_reqs].to(torch.int64)

        row_stride = block_table.stride(0)
        # Captured graphs bake the table's address: a moved table would be read stale.
        identity = (block_table.data_ptr(), row_stride)
        if self._table is None:
            self._table = identity
        elif self._table != identity:
            raise RuntimeError("ICP: the block table moved; captured graphs would be stale")

        if has_prefill:
            kv_max = max(1, int(kv_ub.max()) if num_reqs else 1)
            # The smallest column rung covering the host-bounded live extent.
            columns = next(c for c in self._columns if c >= min(_cdiv(kv_max, self._page_size), nb))
            width = self._waves[columns]
            scores, valid = self._prefill_views[columns]
        else:
            width = self._decode_chunk
            scores, valid = self._scores, self._valid

        windows, has_fmha = [], False
        for slot, t0 in enumerate(range(0, num_tokens, width)):
            t1, rows = min(t0 + width, num_tokens), min(t0 + width, cap) - t0
            if scan is not None:
                r0 = t0 // query_len
                if r0 >= num_reqs:
                    continue  # capture padding only: stays invalid candidates
                count = min(num_tokens - t0, width)
                r1 = min(_cdiv(t0 + count, query_len), num_reqs)
                windows.append((t0, rows, self._cute(
                    scan, block_table, query_start_loc, seq_lens, num_tokens,
                    t0, count, r0, r1 - r0), self._publish(t0, rows)))
                continue
            # FMHA OnlyScoreIcp window (r13), unsplit.
            lo, hi = q_begin.clamp(t0, t1), q_end.clamp(t0, t1)
            sel = (hi - lo > 0).nonzero(as_tuple=True)[0]
            if not sel.numel():
                continue
            has_fmha = True
            last_position = int((kv_ub[sel] - q_end[sel] + hi[sel] - 1).max())
            chunk_live_blocks = min(_cdiv(last_position + 1, self._page_size), nb)
            score_extent = chunk_live_blocks if has_prefill else nb
            plan = api._fmha_sm100_icp_plan(
                (hi[sel] - lo[sel]).to(torch.int32),
                self._hg,
                icp_rank=self._rank,
                block_table=block_table,
                block_table_row_stride=row_stride,
                block_table_row_begin=int(sel[0]),
                device_plan=self._store.slot(slot),
                max_k_tiles=score_extent,
                page_size=self._fragment,
                world_size=self._world,
            )
            live = min(t1, live_tokens) - t0
            score = self._fmha(plan, t0, live, scores[:live, :, :score_extent],
                               valid[:live, :, :score_extent],
                               self._qo_offset[slot, : sel.numel()])
            if has_prefill:
                pick = self._select_prefill(scores[:rows], t0, rows)
            else:
                pick = self._publish(t0, rows)
            windows.append((t0, rows, score, pick))

        # D3: every row of the extent is published; uncovered rows publish (-inf, -1).
        padding = []
        if not has_prefill:
            at, gaps = 0, []
            for begin, end in sorted((w[0], w[0] + w[1]) for w in windows):
                if begin > at:
                    gaps.append((at, begin))
                at = max(at, end)
            if at < cap:
                gaps.append((at, cap))
            for gap_begin, gap_end in gaps:
                for p0 in range(gap_begin, gap_end, width):
                    padding.append(self._publish(p0, min(width, gap_end - p0)))

        out, forced, n_ordinary = self._topk[:cap], self._forced[:cap], self._n_ordinary[:cap]
        if has_prefill:
            # `view`, not a conversion: both carrier words are bitcast.
            candidates = self._candidates[:cap].view(torch.int32)

            def merge(layer: int) -> None:
                self._exchange.merge_prefill(candidates, out, forced=forced,
                                             n_ordinary=n_ordinary)
        else:

            def merge(layer: int) -> None:
                self._exchange.merge_decode(out, layer=layer, forced=forced,
                                            n_ordinary=n_ordinary)

        writer_args = {
            "write_icp_metadata": True,
            "icp_query_start_loc": query_start_loc,
            "icp_seq_lens": seq_lens,
            "icp_num_reqs": num_reqs,
            "icp_positions": self._positions[:cap],
            "icp_active": self._active[:cap],
            "icp_local_nvalid": self._nvalid[:cap],
            "icp_local_forced": self._forced_column[:cap],
            "icp_global_nvalid": self._global_nvalid[:cap],
            "icp_forced": forced,
            "icp_n_ordinary": n_ordinary,
            "icp_candidates": self._candidates[:cap],
            "icp_qo_offsets": self._qo_offset[: _cdiv(cap, width)],
            "icp_chunk_width": width,
        }
        if has_fmha:
            writer_args.update(self._store.writer_kwargs())
        return IcpStep(writer_args, cap, num_tokens, tuple(w[2:] for w in windows),
                       tuple(padding), merge)

    def select(self, step: IcpStep, index_q: torch.Tensor, index_k: torch.Tensor, *,
               layer: int, scale: float) -> None:
        """Score, select and exchange one sparse layer into ``topk_buffer[:token_capacity]``."""
        q = index_q[: step._num_tokens].view(step._num_tokens, self._hg, self._head_dim)
        # Preserve the caller's index-page stride, including sliced cache views.
        k_pages = index_k.unsqueeze(1)
        for score, pick in step._windows:
            score(q, index_k, k_pages, scale)
            pick(layer)
        for publish in step._padding:
            publish(layer)
        # ONE exchange per layer, after every window.
        step._merge(layer)

    def close(self) -> None:
        """Release the D3 window; collective, once on every rank."""
        self._exchange.close()

    def _cute(self, scan, block_table, query_start_loc, seq_lens, num_tokens, token_begin,
              token_count, request_begin, request_count) -> Callable:
        scores, valid = self._scores[:token_count], self._valid[:token_count]
        positions = self._positions[:num_tokens]
        active = self._active_bytes[:num_tokens]

        def score(q, k, k_pages, scale) -> None:
            scan(q, k, block_table, scores, seq_lens, query_start_loc, positions, active,
                 valid, token_begin, token_count, request_begin, request_count,
                 _CUTE_SPLIT_K)

        return score

    def _fmha(self, plan, row_begin, rows, max_score, valid_score, qo_offset) -> Callable:
        def score(q, k, k_pages, scale) -> None:
            api._fmha_sm100_icp(q[row_begin : row_begin + rows], k_pages, plan,
                                max_score=max_score, valid_score=valid_score,
                                sm_scale=scale, qo_offset=qo_offset)

        return score

    def _publish(self, begin: int, rows: int) -> Callable[[int], None]:
        args = (self._scores[:rows], self._nvalid[begin : begin + rows],
                self._forced_column[begin : begin + rows], self._active[begin : begin + rows])

        def publish(layer: int) -> None:
            self._exchange.publish(*args, layer=layer, token_offset=begin)

        return publish

    def _select_prefill(self, scores, begin: int, rows: int) -> Callable[[int], None]:
        end = begin + rows
        args = (scores, self._nvalid[begin:end], self._forced_column[begin:end],
                self._active[begin:end], self._candidates[begin:end], 0, False)

        def select(layer: int) -> None:
            # Unit stride from block 0, no PDL: prefill is eager, never captured.
            self._select.select_prefill_candidates(*args, global_block_stride=1)

        return select
