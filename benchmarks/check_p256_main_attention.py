#!/usr/bin/env python
"""Paired physical-page validation; logical sparse blocks stay B128.

The two cache layouts contain identical NVFP4 bytes and TopK lists. The P128
kernel output is the numerical oracle for the P256 address translation. This
is operator validation, not model or request-level correctness.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from fmha_sm100.decode_q8kv4 import interleave_v_scales, plan_decode, run_decode
from fmha_sm100.nvfp4_kv import nvfp4_head_slot_views


def fixture(lengths, q_lengths, *, seed=7, heads=2, page_sizes=(128, 256), world_size=2):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = random.Random(seed)
    counts = [math.ceil(n / 128) for n in lengths]
    blocks = max(counts)
    shape = (len(lengths) * blocks, heads, 128)
    sources = [torch.randint(0, 256, (*shape, 64), dtype=torch.uint8,
                             device=device, generator=gen) for _ in range(2)]
    scales = [torch.randint(0x18, 0x39, (*shape, 8), dtype=torch.uint8,
                            device=device, generator=gen) for _ in range(2)]
    scales[1] = interleave_v_scales(scales[1])
    layouts = {}
    for page_size in page_sizes:
        ratio = page_size // 128
        per_request = math.ceil(blocks / ratio)
        total_pages = len(lengths) * per_request + 5
        page_gen = torch.Generator(device=device).manual_seed(seed + page_size)
        permutation = torch.randperm(total_pages, generator=page_gen, device=device)
        table = permutation[:len(lengths) * per_request].view(len(lengths), -1)
        # Match the compound cache: an index region follows the main KV slots.
        page_stride = 2 * heads * page_size * 72 + (page_size // world_size) * 128
        backing = torch.full((total_pages * page_stride + 256,), 0xE7,
                             dtype=torch.uint8, device=device)
        slots = backing.as_strided((total_pages, 2 * heads, page_size, 72),
                                   (page_stride, page_size * 72, 72, 1), 256)
        kd, ks, vd, vs = nvfp4_head_slot_views(slots[:, 0::2], slots[:, 1::2])
        for half in range(ratio):
            logical = torch.arange(half, blocks, ratio, device=device)
            dest = table[:, logical // ratio].reshape(-1)
            src = (torch.arange(len(lengths), device=device)[:, None] * blocks
                   + logical[None, :]).reshape(-1)
            for target, values in ((kd, sources[0]), (vd, sources[1]),
                                   (ks, scales[0]), (vs, scales[1])):
                target[:, :, half * 128:(half + 1) * 128].index_copy_(
                    0, dest, values.index_select(0, src))
        layouts[page_size] = dict(
            cache=(kd, vd), sf=(ks.view(torch.float8_e4m3fn),
                               vs.view(torch.float8_e4m3fn)),
            table=table.to(torch.int32), backing=backing,
            indptr=torch.arange(len(lengths) + 1, device=device,
                                 dtype=torch.int32) * per_request,
        )
    lists = []
    for length, q_length in zip(lengths, q_lengths, strict=True):
        for pos in range(length - q_length, length):
            current = pos // 128
            row = []
            for _ in range(heads):
                history = sorted(rng.sample(range(current), min(15, current)))
                ids = history + [current]
                row.append(ids + [-1] * (16 - len(ids)))
            lists.append(row)
    topk = torch.tensor(lists, device=device, dtype=torch.int32)
    q_gen = torch.Generator(device=device).manual_seed(seed + 1)
    q = torch.randn((sum(q_lengths), heads * 16, 128), generator=q_gen,
                    device=device).mul_(0.5)
    cuq = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()],
                       dtype=torch.int32, device=device)
    cuk = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                       dtype=torch.int32, device=device)
    canonical_digest = {name: tensor_digest(tensor) for name, tensor in (
        ("k_nvfp4", sources[0]), ("v_nvfp4", sources[1]),
        ("k_block_scale", scales[0]), ("v_block_scale_interleaved", scales[1]),
    )}
    return layouts, q, topk, cuq, cuk, sum(counts), canonical_digest


def capture_call(fn, inner_calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_calls):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def case_id(kind, lengths, q_lengths):
    config = dict(kind=kind, lengths=lengths, q_lengths=q_lengths)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return f"{kind}_b{len(lengths)}_k{max(lengths)}_q{sum(q_lengths)}_{digest}"


def tensor_digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    # Bound host staging even for multi-gigabyte long-context cache fixtures.
    for begin in range(0, data.numel(), 32 * 1024 * 1024):
        digest.update(data[begin:begin + 32 * 1024 * 1024].cpu().numpy())
    return digest.hexdigest()


def source_info():
    import fmha_sm100

    package = Path(fmha_sm100.__file__).resolve().parent
    files = ["nvfp4_kv.py", "decode_q8kv4/interface.py", "cute/interface.py",
             "cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py"]
    files += [str(p.relative_to(package)) for p in sorted((package / "decode_q8kv4/csrc").rglob("*"))
              if p.is_file()]
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files}
    info = dict(package=str(package), file_sha256=hashes)
    for name, git_args in (("head", ["rev-parse", "HEAD"]),
                           ("diff", ["diff", "--binary", "HEAD"])):
        proc = subprocess.run(["git", "-C", str(package), *git_args],
                              capture_output=True, check=False)
        if proc.returncode == 0:
            info[name if name == "head" else "diff_sha256"] = (
                proc.stdout.decode().strip() if name == "head"
                else hashlib.sha256(proc.stdout).hexdigest())
    return info


def validate_case(kind, lengths, q_lengths, *, args):
    torch.cuda.reset_peak_memory_stats()
    layouts, q, topk, cuq, cuk, rows, canonical_digest = fixture(
        lengths, q_lengths, seed=args.seed, page_sizes=args.page_sizes,
        heads=4 // args.world_size, world_size=args.world_size)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    unit = torch.ones((1,), device="cuda", dtype=torch.float32)
    if kind == "decode":
        assert len(set(q_lengths)) == 1
        q = q.to(torch.float8_e4m3fn)
        plan = plan_decode(batch_size=len(lengths), q_len_per_req=q_lengths[0],
                           topk=16, device=q.device, num_q_heads=q.shape[1],
                           num_kv_heads=4 // args.world_size, num_kv_splits=1, split_mode="streamk",
                           block_scale_shift=3)

        def call(layout):
            return run_decode(plan, q, layout["cache"], kv_cache_sf=layout["sf"],
                              seq_lens=seq, kv_indices=layout["table"].reshape(-1),
                              kv_indptr=layout["indptr"], topk_indices=topk,
                              kv_global_scale=(unit, unit))
    else:
        from fmha_sm100.sparse import build_k2q_csr, sparse_atten_nvfp4_kv_func

        q = q.to(torch.bfloat16)
        row_ptr, indices, schedule = build_k2q_csr(
            topk.transpose(0, 1), cuq, cuk, 128, total_k=sum(lengths),
            max_seqlen_k=max(lengths), max_seqlen_q=max(q_lengths),
            total_rows=rows, qhead_per_kv=16, return_schedule=True)

        def call(layout):
            kd, vd = layout["cache"]
            ks, vs = (x.view(torch.uint8) for x in layout["sf"])
            return sparse_atten_nvfp4_kv_func(
                q, kd, vd, ks, vs, unit, unit, row_ptr, indices, 16,
                cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                max_seqlen_q=max(q_lengths), max_seqlen_k=max(lengths),
                blk_kv=128, causal=True, softmax_scale=128**-0.5,
                page_table=layout["table"], seqused_k=seq,
                schedule=schedule, kv_layout="vllm")

    inputs_digest = dict(q=tensor_digest(q), topk=tensor_digest(topk), **canonical_digest)
    outputs = {p: call(layout) for p, layout in layouts.items()}
    torch.cuda.synchronize()
    baseline = outputs[args.page_sizes[0]]
    assert torch.isfinite(baseline).all(), "baseline produced non-finite output"
    identity = case_id(kind, lengths, q_lengths)
    if args.reference:
        reference = torch.load(Path(args.reference) / f"{identity}.pt", weights_only=True)
        assert reference["page_size"] == 128, "reference must be a P128 run"
        assert reference["input_digest"] == inputs_digest, "reference fixture mismatch"
        assert torch.equal(baseline.cpu(), reference["output"]), "original Fan P128 output mismatch"
    for p, output in outputs.items():
        assert torch.equal(baseline, output), (
            f"{kind} P{p} mismatch: {(baseline.float() - output.float()).abs().max().item()}")
    if args.save_reference:
        target = Path(args.save_reference)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(dict(input_digest=inputs_digest, output=baseline.cpu(), page_size=128,
                        label=args.label), target / f"{identity}.pt")
    graphs = {}
    captured_outputs = {}
    for p, layout in layouts.items():
        graphs[p], captured = capture_call(lambda: call(layout), args.inner_calls)
        captured_outputs[p] = captured
        assert torch.equal(captured, baseline), f"P{p} capture mismatch"
        for _ in range(args.warmup_replays):
            graphs[p].replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, baseline), f"P{p} graph replay mismatch"
    samples = {p: [] for p in args.page_sizes}
    orders = []
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in args.page_sizes]
    for r in range(args.rounds):
        order = args.page_sizes if r % 2 == 0 else args.page_sizes[::-1]
        orders.append(order)
        for p, (begin, end) in zip(order, events, strict=True):
            begin.record()
            for _ in range(args.replays):
                graphs[p].replay()
            end.record()
            end.synchronize()
            samples[p].append(begin.elapsed_time(end) * 1000 / (args.replays * args.inner_calls))
            assert torch.equal(captured_outputs[p], baseline), f"P{p} timed replay mismatch"
    result = dict(case_id=identity, kind=kind, lengths=lengths, q_lengths=q_lengths,
                  batch=len(lengths), query_rows=sum(q_lengths), seed=args.seed,
                  decode_plan=(dict(num_kv_splits=1, split_mode="streamk", block_scale_shift=3)
                               if kind == "decode" else None),
                  input_digest=inputs_digest, exact_equal=True, graph_equal=True,
                  original_reference_equal=bool(args.reference), samples_us=samples,
                  median_us={p: statistics.median(v) for p, v in samples.items()},
                  round_order=orders, inner_calls=args.inner_calls, replays=args.replays,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if len(args.page_sizes) == 2:
        a, b = args.page_sizes
        result["paired_speedups"] = [x / y for x, y in zip(samples[a], samples[b], strict=True)]
        result["median_paired_speedup"] = statistics.median(result["paired_speedups"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--formal", action="store_true",
                        help="160Ki/8Ki decode sweep and bounded chunked-prefill cases")
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[128, 256], choices=(128, 256))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--contexts", type=int, nargs="+", default=[163840, 172032])
    parser.add_argument("--q-len", type=int, default=4)
    parser.add_argument("--kind", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--inner-calls", type=int, default=8)
    parser.add_argument("--replays", type=int, default=40)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference", help="original Fan P128 outputs saved by a separate process")
    parser.add_argument("--save-reference", help="save exact outputs and fixture digests for later comparison")
    parser.add_argument("--label", default="experimental")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    if len(args.page_sizes) != len(set(args.page_sizes)):
        parser.error("page sizes must be distinct")
    if min(args.rounds, args.inner_calls, args.replays, args.warmup_replays) < 1:
        parser.error("timing counts must be positive")
    if args.smoke_only and args.formal:
        parser.error("choose --smoke-only or --formal")
    if args.save_reference and args.page_sizes != [128]:
        parser.error("--save-reference requires --page-sizes 128")
    torch.cuda.set_device(args.device)
    cases = [
        ("decode", [129, 257, 513, 1100], [4] * 4),
        ("prefill", [131, 260, 777], [8, 9, 129]),
    ]
    if args.formal:
        cases = [("decode", [context] * batch, [args.q_len] * batch)
                 for context in args.contexts for batch in args.batches]
        # Prefill chunk rows are not request concurrency or the full 160Ki prompt.
        cases += [("prefill", [context] * batch, [chunk] * batch)
                  for context in args.contexts for batch, chunk in ((1, 512), (1, 2048), (4, 512))]
    elif not args.smoke_only:
        cases += [
            ("decode", [163840, 172032], [4, 4]),
            ("prefill", [163840], [512]),
        ]
    cases = [case for case in cases if (args.kind == "all" or case[0] == args.kind)
             and args.case_filter in case_id(*case)]
    if not cases:
        parser.error("no cases matched")
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, timestamp=time.time(), label=args.label,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source=source_info(), args=vars(args),
                  env={name: os.environ.get(name) for name in (
                      "CUDA_VISIBLE_DEVICES", "TORCH_EXTENSIONS_DIR", "MINFER_FMHA_CACHE_DIR",
                      "MM_SPARSE_ATTN_AOT_CACHE")},
                  scope=f"one TP{args.world_size} rank's main attention, Hkv{4 // args.world_size}/Hq{64 // args.world_size}; fixed synthetic NVFP4 cache and B128 TopK16; no model, indexer, or request-level generation",
                  length_convention="160Ki = 163840 input tokens; 8Ki = 8192 output tokens; contexts sample start/end only",
                  timing="CUDA-event per call inside an unrolled CUDA Graph; warmed repeated inputs; AB/BA rounds; prefill schedule built before timing",
                  cases=[])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print(f"START {case}", flush=True)
        result["cases"].append(validate_case(*case, args=args))
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result["cases"][-1]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
