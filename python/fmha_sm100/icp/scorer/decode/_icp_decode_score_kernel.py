# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2/TP4 P128 index-fragment decode scorer (FP8 E4M3, H4, D128).

Adapted from vLLM IndexDecodeScoreKernel at 9de0daa25f47; mma.sync rather than
tcgen05 because the score GEMM's tiny N favours CTA occupancy over a deep pipeline.
"""

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Boolean,
    Float8E4M3FN,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    cute,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T, dsl_user_op

_CUTE_TO_PTX_DTYPE = {
    BFloat16: "bf16",
    Float16: "f16",
    Float8E4M3FN: "e4m3",
    Float32: "f32",
}

# https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
EVICT_FIRST = Int64(0x12F0000000000000)


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError

    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )

    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)
    else:
        raise ValueError


@dsl_user_op
def mma_sync(a, b, c: cute.Tensor, *, loc=None, ip=None):
    a_ty = _CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = _CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = _CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    # Recast expects tensor-backed fragments; materialize SSA fragments here so
    # callsites can pass converted FP8 fragments directly.
    if isinstance(a, cute.TensorSSA):
        a_ = cute.make_rmem_tensor_like(a)
        a_.store(a, loc=loc, ip=ip)
        a = a_
    if isinstance(b, cute.TensorSSA):
        b_ = cute.make_rmem_tensor_like(b)
        b_.store(b, loc=loc, ip=ip)
        b = b_

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


@dsl_user_op
def fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@cute.jit
def _fp8_to_f16_mma_fragments(src: cute.Tensor):
    src_elems = cute.size(src)
    src_u32 = cute.recast_tensor(src, Uint32)
    src_f16 = cute.make_rmem_tensor(src_elems, Float16)
    src_f16_u32 = cute.recast_tensor(src_f16, Uint32)
    # This packed conversion is faster and emits fewer SASS instructions than
    # src.load().to(Float16).
    for i in cutlass.range_constexpr(src_elems // 4):
        converted = fp8x4_to_fp16x4(src_u32[i])
        src_f16_u32[i * 2] = converted[0]
        src_f16_u32[i * 2 + 1] = converted[1]
    lower = cute.make_rmem_tensor(src_elems // 2, Float16)
    upper = cute.make_rmem_tensor(src_elems // 2, Float16)

    # FP8 ldmatrix gives four consecutive values along K. Split each group
    # into the lower two and upper two values for two FP16 MMA k-fragments.
    for i in cutlass.range_constexpr(src_elems // 2):
        lower[i] = src_f16[(i // 2) * 4 + i % 2]
        upper[i] = src_f16[(i // 2) * 4 + 2 + i % 2]
    return lower, upper


class IndexDecodeScoreKernel:
    BAR_MMA = 1
    num_stages = 2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        num_heads: int,
        max_decode_query_len: int,
        split_k: int,
        head_dim: int = 128,
        *, query_len: int, rank: int,
        num_stages: int = 2,
        page_size: int = 128,
        world_size: int = 2,
    ):
        self.dtype = dtype
        self.num_heads = num_heads
        self.max_decode_query_len = max_decode_query_len
        self.split_k = split_k
        self.head_dim = head_dim
        self.query_len = query_len
        self.rank = rank
        assert world_size in (2, 4) and page_size == 128
        assert 0 <= rank < world_size
        self.page_size = page_size
        self.BLOCK_K = page_size // world_size
        self.NUM_COMPUTE_WARPS = self.BLOCK_K // 32
        # K fragments in flight per CTA; decode scoring is latency bound.
        self.num_stages = num_stages

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # full invocation [tokens, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [bs, max_pages]
        score: cute.Tensor,  # [window_rows, num_heads, max_pages]
        seq_lens: cute.Tensor,  # exact device lengths
        query_start_loc: cute.Tensor,
        positions: cute.Tensor,
        active_rows: cute.Tensor,  # uint8 liveness bytes; nonzero is live
        valid: cute.Tensor,
        # The token window and request range are RUNTIME launch scalars, so one
        # compile serves every window; a captured graph records its own values.
        token_begin: Int32,
        token_count: Int32,
        request_begin: Int32,
        request_count: Int32,
        split_count: Int32,
        stream: CUstream,
    ):
        dtype = self.dtype
        num_heads = self.num_heads
        head_dim = self.head_dim
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = num_heads * MAX_DQL
        assert BLOCK_Q <= 32

        grid = (request_count, split_count, 1)
        block = (32 * (self.NUM_COMPUTE_WARPS + 1), 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // dtype.width

        sQ_layout = cute.make_layout(
            (MAX_DQL, num_heads, (elems, head_dim // elems)),
            stride=(elems, MAX_DQL * elems, (1, BLOCK_Q * elems)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, elems)),
            sQ_layout,
            cta_tiler=(MAX_DQL, num_heads, head_dim),
        )

        sK_layout = cute.make_layout(
            (1, BLOCK_K, (elems, head_dim // elems), num_stages),
            stride=(0, elems, (1, BLOCK_K * elems), BLOCK_K * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, elems)),
            sK_layout,
            cta_tiler=(1, BLOCK_K, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            score,
            seq_lens,
            query_start_loc, positions, active_rows, valid, gQ.shape[0],
            token_begin, token_count, request_begin,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,
        score: cute.Tensor,
        seq_lens: cute.Tensor,
        query_start_loc: cute.Tensor, positions: cute.Tensor,
        active_rows: cute.Tensor, valid: cute.Tensor, total_tokens,
        token_begin: Int32, token_count: Int32, request_begin: Int32,
    ):
        request_slot, split_id, _ = cute.arch.block_idx()
        batch_id = request_slot + request_begin
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        NUM_HEADS = self.num_heads
        MAX_DQL = self.max_decode_query_len
        BLOCK_Q = NUM_HEADS * MAX_DQL
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim
        dtype = self.dtype
        MMA_N = 8
        num_stages = self.num_stages
        Q_TILES = cute.ceil_div(BLOCK_Q, MMA_N)
        EPI_Q = Q_TILES * MMA_N

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            dtype,
            K_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=K_tma.smem_layout.inner,
        )[0, None, None, None]
        # Own Q tile: K stage 0 may hold a pre-wait fragment.
        sQ_tma = smem.allocate_tensor(
            dtype,
            Q_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=Q_tma.smem_layout.inner,
        )
        # TMA sees Q as (query, head, dim), while ldmatrix consumes a
        # flattened Q column mode. The target profile keeps the rank-2 view even
        # for degenerate shapes like DQL1.
        q_tma_elems = 128 * 8 // dtype.width
        sQ = cute.coalesce(
            cute.group_modes(sQ_tma, 0, 2),
            target_profile=(BLOCK_Q, (q_tma_elems, head_dim // q_tma_elems)),
        )
        epi_buffer = smem.allocate_tensor(Float32, cute.make_layout((EPI_Q, self.NUM_COMPUTE_WARPS)))

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        q_full_mbar = smem.allocate_array(Int64, 1)

        # Step inputs written before the forward (GPU-produced under Model
        # Runner V2); safe to read before the PDL wait only because this
        # scorer's primary, the fused ICP producer, is a non-PDL launch.
        seqlen = seq_lens[batch_id]
        qb = query_start_loc[batch_id]
        qe = query_start_loc[batch_id + 1]

        n_pre = Int32(0)
        pre_end = Int32(0)
        # Blocks [0, n_old) end before this request's first new token, so
        # this step's writer does not touch their index rows.
        n_old = (seqlen - (qe - qb)) // self.page_size
        if split_id < n_old:
            n_pre = (n_old - split_id + split_k - 1) // split_k
            if n_pre > num_stages:  # noqa: PLR1730
                n_pre = Int32(num_stages)
            pre_end = split_id + (n_pre - 1) * split_k + 1
        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 32 * self.NUM_COMPUTE_WARPS)
                cute.arch.mbarrier_init(q_full_mbar, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == self.NUM_COMPUTE_WARPS:
            cpasync.prefetch_descriptor(Q_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
        cute.arch.sync_threads()
        if warp_id == self.NUM_COMPUTE_WARPS:
            for i in cutlass.range_constexpr(num_stages):
                if i < n_pre:
                    pre_page = block_table[batch_id, split_id + i * split_k]
                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * (dtype.width // 8)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            tma_full_mbar + i, K_size
                        )
                    simple_tma_copy(
                        K_tma.atom,
                        K_tma.tma_tensor[pre_page, None, None],
                        sK[None, None, i],
                        tma_full_mbar + i,
                        cache_policy=EVICT_FIRST,
                    )
        # Every global read of this step's producer output and every store
        # follow the wait; the selector may launch once all CTAs run.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        query_positions = cute.make_rmem_tensor(MAX_DQL, Int64)
        query_active = cute.make_rmem_tensor(MAX_DQL, Boolean)
        query_positions.fill(-1)
        query_active.fill(False)
        max_position = Int64(-1)
        # Every CTA participant uses the same exact bound, including the producer.
        for u in cutlass.range_constexpr(MAX_DQL):
            t = qb + u
            if u < self.query_len and t >= 0 and t < qe and t < total_tokens:
                if t >= token_begin and t < token_begin + token_count:
                    # Uint8 byte plane (a torch bool viewed without a copy).
                    if active_rows[t] != 0:
                        p = positions[t]
                        query_positions[u] = p
                        query_active[u] = True
                        max_position = p if p > max_position else max_position
        visible_end = Int64(seqlen) if seqlen < max_position + 1 else max_position + 1
        visible_end = visible_end if visible_end > 0 else Int64(0)
        num_blocks = Int32(visible_end // self.page_size
                          + Int64(visible_end % self.page_size > self.rank * BLOCK_K))
        # Pre-issued fragments are always consumed; their stores stay masked
        # by the exact causal bound below.
        if num_blocks < pre_end:  # noqa: PLR1730
            num_blocks = pre_end

        if split_id < num_blocks:
            if warp_id == self.NUM_COMPUTE_WARPS:
                # TMA warp
                tma_stage = 0
                tma_parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset(
                        (qb, 0, 0),
                        Q_tma.tma_tensor,
                    ),
                    tiler=(MAX_DQL, NUM_HEADS, head_dim),
                    coord=(0, 0, 0),
                )
                Q_size = BLOCK_Q * head_dim * (dtype.width // 8)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full_mbar, Q_size)
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ_tma, q_full_mbar)
                # Continue after the n_pre pre-wait fragments.
                tma_stage = n_pre % num_stages
                tma_parity = Int32(1) - n_pre // num_stages

                for block_id in range(split_id + n_pre * split_k, num_blocks, split_k):
                    page_id = block_table[batch_id, block_id]
                    gK_tile = K_tma.tma_tensor[page_id, None, None]
                    k_mbar = tma_full_mbar + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * (dtype.width // 8)
                        cute.arch.mbarrier_arrive_and_expect_tx(k_mbar, K_size)
                    simple_tma_copy(
                        K_tma.atom,
                        gK_tile,
                        sK[None, None, tma_stage],
                        k_mbar,
                        cache_policy=EVICT_FIRST,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

            else:
                # MMA warps
                # each warp handles K[32, head_dim] @ Q[BLOCK_Q, head_dim].T
                sK_warp = cute.local_tile(
                    sK, (32, head_dim, num_stages), (warp_id, 0, 0)
                )

                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                # Pre-compute ldmatrix address.
                # sK loads a [16 x 16B] tile:
                #   ((16, (16B, 2), 1), (32 / 16, head_dim / 32B, num_stages))
                # sQ loads an [8 x 32B] tile:
                #   ((8, (16B, 4)), (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)), 1)
                )
                sQ_ldsm = cute.zipped_divide(sQ, (MMA_N, cute.make_layout((elems, 4))))

                # sK: (16B, (32 / 16, head_dim / 32B, num_stages))
                # sQ: (16B, (BLOCK_Q / MMA_N, head_dim / 64B))
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), head_dim // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor((elems, 2, head_dim // MMA_K), dtype)
                rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)

                if warp_id == 0:
                    cute.arch.mbarrier_wait(q_full_mbar, 0)
                cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=32 * self.NUM_COMPUTE_WARPS)
                for q in cutlass.range_constexpr(Q_TILES):
                    cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                tma_stage = 0
                tma_parity = 0

                # sm100 doesn't have native mma.sync.f8. ptxas lowers mma.sync.f8
                # to F2FP.F16.E4M3 + HMMA; doing the conversion explicitly gives
                # better codegen while keeping the two FP16 k-fragments visible.
                if cutlass.const_expr(dtype is Float8E4M3FN):
                    rQ_f16 = cute.make_rmem_tensor(
                        (4, head_dim // MMA_K, Q_TILES, 2), Float16
                    )
                    q_lower, q_upper = _fp8_to_f16_mma_fragments(rQ)
                    rQ_f16[None, None, None, 0].store(q_lower.load())
                    rQ_f16[None, None, None, 1].store(q_upper.load())

                for block_id in range(split_id, num_blocks, split_k):
                    rC.fill(0.0)

                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=32 * self.NUM_COMPUTE_WARPS)

                    for k in cutlass.range_constexpr(head_dim // MMA_K):
                        cute.copy(
                            ldsm_atom,
                            sK_ldsm[None, (None, k, tma_stage)],
                            rK[None, None, k],
                        )
                        for m in cutlass.range_constexpr(2):
                            if cutlass.const_expr(dtype is Float8E4M3FN):
                                rK_lower, rK_upper = _fp8_to_f16_mma_fragments(
                                    rK[None, m, k]
                                )
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK_lower,
                                        rQ_f16[None, k, n, 0],
                                        rC[None, m, n],
                                    )
                                    rC[None, m, n] = mma_sync(
                                        rK_upper,
                                        rQ_f16[None, k, n, 1],
                                        rC[None, m, n],
                                    )
                            else:
                                for n in cutlass.range_constexpr(Q_TILES):
                                    rC[None, m, n] = mma_sync(
                                        rK[None, m, k],
                                        rQ[(None, k % 2), k // 2, n],
                                        rC[None, m, n],
                                    )

                    # Order generic shared-memory reads before a producer
                    # reuses this slot through the TMA async proxy.
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    k_start = block_id * self.page_size + self.rank * BLOCK_K + warp_id * 32

                    # The exact causal mask, including suppression of NaNs in
                    # future K rows.
                    for q in cutlass.range_constexpr(Q_TILES):
                        for i in cutlass.range_constexpr(4):
                            for j in cutlass.range_constexpr(2):
                                col = q * 8 + (lane_id % 4) * 2 + j
                                q_local_pos = col % MAX_DQL
                                q_pos = query_positions[q_local_pos]
                                k_pos = k_start + i * 8 + lane_id // 4
                                rC[q * 8 + i * 2 + j] = (
                                    rC[q * 8 + i * 2 + j]
                                    if query_active[q_local_pos] and q_pos >= k_pos and k_pos < seqlen
                                    else float("-inf")
                                )

                    for q in cutlass.range_constexpr(Q_TILES):
                        # thread-reduction along BLOCK_K dim
                        rScore = cute.make_rmem_tensor(2, Float32)
                        rScore.fill(float("-inf"))
                        for i in cutlass.range_constexpr(4):
                            rScore[0] = cute.arch.fmax(rScore[0], rC[i * 2 + 0 + q * 8])
                            rScore[1] = cute.arch.fmax(rScore[1], rC[i * 2 + 1 + q * 8])

                        # warp-reduction among lanes 0,4,8,12,...
                        for i in cutlass.range_constexpr(3):
                            offset = 4 << i
                            other0 = cute.arch.shuffle_sync_bfly(
                                rScore[0], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            other1 = cute.arch.shuffle_sync_bfly(
                                rScore[1], offset=offset, mask=-1, mask_and_clamp=31
                            )
                            rScore[0] = cute.arch.fmax(rScore[0], other0)
                            rScore[1] = cute.arch.fmax(rScore[1], other1)

                        # Each scratch column is one compute warp of this block.
                        if lane_id * 2 < MMA_N:
                            epi_buffer[q * MMA_N + lane_id * 2 + 0, warp_id] = rScore[0]
                            epi_buffer[q * MMA_N + lane_id * 2 + 1, warp_id] = rScore[1]
                    cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=32 * self.NUM_COMPUTE_WARPS)

                    head_id = lane_id // MAX_DQL
                    q_local_pos = lane_id - head_id * MAX_DQL
                    valid_q = head_id < NUM_HEADS and query_active[q_local_pos]
                    if warp_id == 0 and lane_id < BLOCK_Q and valid_q:
                        final_score = epi_buffer[lane_id, 0]
                        for i in cutlass.range_constexpr(1, self.NUM_COMPUTE_WARPS):
                            final_score = cute.arch.fmax(
                                final_score, epi_buffer[lane_id, i]
                            )

                        t = qb + q_local_pos - token_begin
                        first_key = block_id * self.page_size + self.rank * BLOCK_K
                        if first_key <= query_positions[q_local_pos] and first_key < seqlen:
                            score[t, head_id, block_id] = final_score
                            valid[t, head_id, block_id] = Uint8(1)

                    tma_stage = (tma_stage + 1) % self.num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1
