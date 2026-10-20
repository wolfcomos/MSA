# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Prewarmed FP8 decode scoring for TP2/TP4, with independent P128 index pages.

One AOT scan per ``(query_len, rank)`` serves uniform Q1-Q4 decode; the token window,
request range and split are launch scalars. Call order: ``scan(index_q, index_k,
block_table, score_out, seq_lens, query_start_loc, positions, active_rows, valid_out,
token_begin, token_count, request_begin, request_count, split_k)``. For request i and
active row t, with U = max(0, min(seq_lens[i], positions[t] + 1)), columns
[0, U // P + (U % P > rank * (P // world_size))) of all four heads get scores and valid=1; every
other cell is undefined and the selector must not read it.

K contains contiguous rank-owned token fragments with a dynamic page stride.
``positions`` and ``active_rows`` (uint8, nonzero = live) must have
exactly ``index_q``'s row count: the compiled signature binds all three to one symbol.
"""

from functools import cache

#: K fragments in flight per CTA.
_NUM_STAGES = 4


@cache
def _compile_scan(query_len, rank, device_index, architecture, page_size=128, world_size=2):
    """The prewarmed scan of one profile, loaded once per device; never compiles."""
    path = _aot_path(query_len, rank, architecture, page_size, world_size)
    if not path.is_file():
        raise RuntimeError(f"ICP decode scan {path} is missing; run the ICP prewarm build")
    from cutlass import runtime

    module = runtime.load_module(str(path), enable_tvm_ffi=True)
    return getattr(module, _aot_function_name(query_len, rank, page_size, world_size))


def _trace_scan(query_len, rank, page_size=128, world_size=2):
    """The native scan's fake signature and kernel object, ready for cute.compile."""
    from cutlass import Float8E4M3FN, Float32, Int32, Int64, Uint8, cute
    from quack.compile_utils import make_fake_tensor

    from ._icp_decode_score_kernel import IndexDecodeScoreKernel

    total_tokens, output_rows, output_columns = (
        cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    )
    # Only the innermost stride is static; page pitch and THK strides stay dynamic.
    q = make_fake_tensor(
        Float8E4M3FN, (total_tokens, 4, 128), divisibility=16, leading_dim=-1
    )
    k = make_fake_tensor(
        Float8E4M3FN, (cute.sym_int64(), page_size // world_size, 128), divisibility=16, leading_dim=-1
    )
    table = make_fake_tensor(
        Int32, (cute.sym_int64(), cute.sym_int64()), divisibility=1, leading_dim=-1
    )
    shape = (output_rows, 4, output_columns)
    score = make_fake_tensor(Float32, shape, divisibility=1, leading_dim=-1)
    valid = make_fake_tensor(Uint8, shape, divisibility=1, leading_dim=-1)
    lengths = make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=-1)
    offsets = make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=-1)
    positions = make_fake_tensor(Int64, (total_tokens,), divisibility=1, leading_dim=-1)
    # Uint8, not Boolean: quack <= 0.6.2 floors a 1-bit element's alignment to 0.
    active = make_fake_tensor(Uint8, (total_tokens,), divisibility=1, leading_dim=-1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = IndexDecodeScoreKernel(
        # A Q1/H4 tile still needs eight initialized columns for ldmatrix/MMA.
        Float8E4M3FN, 4, 2 if query_len <= 2 else 4, 0, 128,
        query_len=query_len,
        rank=rank,
        num_stages=_NUM_STAGES,
        page_size=page_size,
        world_size=world_size,
    )
    # Dynamic launch scalars (token/request window, split_k); these values are placeholders.
    window = (Int32(0), Int32(1), Int32(0), Int32(1), Int32(1))
    return kernel, (q, k, table, score, lengths, offsets, positions, active, valid,
                    *window, stream)


def _aot_function_name(query_len, rank, page_size=128, world_size=2):
    """The AOT object's exported symbol for one profile."""
    if page_size != 128 or world_size not in (2, 4) or not 0 <= rank < world_size:
        raise ValueError("ICP decode requires TP2/TP4 ranks and 128-token index pages")
    suffix = "" if world_size == 2 else f"_w{world_size}"
    return f"icp_decode_scan_q{int(query_len)}_r{int(rank)}_s{_NUM_STAGES}_pdl{suffix}"


def _aot_path(query_len, rank, arch, page_size=128, world_size=2):
    """Where the prewarm build keeps one profile's ``.so`` for ``arch``."""
    from ... import _build

    name = _aot_function_name(query_len, rank, page_size, world_size)
    return _build.component_dir("decode", _build.normalize_arch(arch)) / f"{name}.so"


def _export_aot_scan(query_len, rank, arch, page_size=128, world_size=2):
    """Compile one profile for ``arch`` without a device into its ``.so``; needs CUTE_DSL_ARCH=arch."""
    import os
    import subprocess
    import tempfile

    from ... import _build

    arch = _build.normalize_arch(arch)
    dsl_arch = os.environ.get("CUTE_DSL_ARCH", "")
    if _build.normalize_arch(dsl_arch or "0") != arch:
        raise RuntimeError(f"CUTE_DSL_ARCH={dsl_arch!r} but the artifact is keyed sm_{arch}")
    from cutlass import cute, runtime

    target = _aot_path(query_len, rank, arch, page_size, world_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = _aot_function_name(query_len, rank, page_size, world_size)
    kernel, args = _trace_scan(query_len, rank, page_size, world_size)
    scan = cute.compile(kernel, *args, options="--enable-tvm-ffi")
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        obj = os.path.join(tmp, f"{name}.o")
        lib = os.path.join(tmp, f"{name}.so")
        scan.export_to_c(obj, function_name=name)
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-shared", "-o", lib, obj,
                        *runtime.find_runtime_libraries(enable_tvm_ffi=True)],
                       check=True, capture_output=True, text=True)
        os.replace(lib, target)
    return target
