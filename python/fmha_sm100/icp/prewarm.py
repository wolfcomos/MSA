"""Load or compile every kernel used by the TP2/TP4 P128/P256 indexer.

``python -m <pkg>.icp.prewarm build --arch 107a`` compiles them into the image;
``python -m <pkg>.icp.prewarm verify`` loads them with ICP_RUNTIME_JIT=fail.
``--include-main-attention`` also warms the compound-cache NVFP4 consumers.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from functools import partial

from . import _build


# Production config: four global KV heads, D128, GQA16, top-16 B128, FP8 scale shift 3.
_HEAD_DIM, _GQA, _TOPK, _BLK, _SCALE_SHIFT = 128, 16, 16, 128, 3


def _page_bytes(page_size, world_size=2):
    slot_bytes = page_size * (_HEAD_DIM // 2 + _HEAD_DIM // 16)
    return slot_bytes, 2 * (4 // world_size) * slot_bytes + (page_size // world_size) * _HEAD_DIM


def _call(module: str, attr: str, *args, **kwargs):
    return getattr(importlib.import_module(module, __package__), attr)(*args, **kwargs)


def _head_slot_views(storage, page_size=128, world_size=2):
    from ..nvfp4_kv import nvfp4_head_slot_views

    # Strided off the backing storage so every view keeps the compound page stride.
    slot_bytes, compound_bytes = _page_bytes(page_size, world_size)
    slots = storage.as_strided(
        (storage.shape[0], 2 * (4 // world_size), page_size, slot_bytes // page_size),
        (compound_bytes, slot_bytes, slot_bytes // page_size, 1),
        storage.storage_offset())
    return nvfp4_head_slot_views(slots[:, 0::2], slots[:, 1::2])


def _nvfp4_prefill(device, page_size=128, world_size=2):
    """One serving-form prefill (CSR schedule, caller-owned out): k2q builder + CuTe AOT."""
    import torch

    from ..sparse import build_k2q_csr, sparse_atten_nvfp4_kv_func

    if device is None:
        raise RuntimeError("the NVFP4 prefill needs a CUDA device")
    # Host draws: vLLM constructs the model under a CUDA default device.
    g = torch.Generator(device="cpu").manual_seed(0)
    q_lens, k_lens = (200, 64), (2000, 600)
    pages = [-(-k // page_size) for k in k_lens]
    table = torch.zeros((len(k_lens), max(pages)), dtype=torch.int32)
    perm = torch.randperm(sum(pages) + 3, generator=g, device="cpu")
    for b, n in enumerate(pages):
        table[b, :n] = perm[sum(pages[:b]):sum(pages[:b]) + n].to(torch.int32)
    heads_kv = 4 // world_size
    q2k = torch.full((sum(q_lens), heads_kv, _TOPK), -1, dtype=torch.int32)
    row = 0
    for q_len, k_len in zip(q_lens, k_lens, strict=True):
        for t in range(q_len):
            visible = -(-(k_len - q_len + t + 1) // _BLK)
            pick = torch.randperm(visible, generator=g, device="cpu")[:_TOPK].sort().values
            q2k[row + t, :, :pick.numel()] = pick.to(torch.int32)
        row += q_len
    heads = heads_kv * _GQA
    q = torch.randn((sum(q_lens), heads, _HEAD_DIM), generator=g, device="cpu").to(torch.bfloat16)
    _, compound_bytes = _page_bytes(page_size, world_size)
    storage = torch.randint(0, 256, (sum(pages) + 3, compound_bytes), generator=g,
                            device="cpu", dtype=torch.uint8)
    for scales in _head_slot_views(storage, page_size, world_size)[1::2]:
        scales.fill_(0x38)
    cu_q, cu_k = (torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
                  for lens in (q_lens, k_lens))
    q, q2k, table, cu_q, cu_k = (t.to(device) for t in (q, q2k, table, cu_q, cu_k))
    k, k_sf, v, v_sf = _head_slot_views(storage.to(device), page_size, world_size)
    rowptr, qidx, schedule = build_k2q_csr(
        q2k.transpose(0, 1), cu_q, cu_k, _BLK, total_k=0, max_seqlen_k=max(k_lens),
        max_seqlen_q=max(q_lens), total_rows=sum(-(-k // _BLK) for k in k_lens), qhead_per_kv=_GQA,
        return_schedule=True)
    # Decode's DLPack bridge normalizes the singleton head stride in place.
    # Warm both metadata signatures; the addresses are identical for Hkv=1.
    for normalize_head_stride in ((False, True) if heads_kv == 1 else (False,)):
        if normalize_head_stride:
            k = k.as_strided(k.shape, (k.stride(0), page_size * 64, 64, 1))
            v = v.as_strided(v.shape, (v.stride(0), page_size * 64, 64, 1))
        sparse_atten_nvfp4_kv_func(
            q, k, v, k_sf, v_sf,
            torch.tensor([0.75], dtype=torch.float32, device=device),
            torch.tensor([1.25], dtype=torch.float32, device=device),
            rowptr, qidx, topK=_TOPK, blk_kv=_BLK, causal=True, softmax_scale=_HEAD_DIM ** -0.5,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens), page_table=table,
            seqused_k=torch.tensor(k_lens, dtype=torch.int32, device=device), schedule=schedule,
            out=torch.empty_like(q), kv_layout="vllm")
    torch.cuda.synchronize(device)


def _scorer_variant(index: int, page_size: int, world_size: int):
    from .. import api, jit

    return jit.get_fmha_variant(*api._icp_scorer_variants(page_size // world_size)[index][1])


def _decode_scan(scorer, query_len: int, rank: int, arch: str, device, page_size: int, world_size: int):
    if not scorer._aot_path(query_len, rank, arch, page_size, world_size).is_file():
        _build.refuse_jit("decode", scorer._aot_function_name(query_len, rank, page_size, world_size))
        # The DSL must target the keyed arch; CLI build sets it before CUTLASS is imported.
        os.environ.setdefault("CUTE_DSL_ARCH", f"sm_{_build.normalize_arch(arch)}")
        scorer._export_aot_scan(query_len, rank, arch, page_size, world_size)
    scorer._compile_scan(query_len, rank, getattr(device, "index", None) or 0, arch, page_size, world_size)


def _artifacts(device, page_size=128, world_size=2) -> list[tuple[str, partial]]:
    from .scorer.decode import icp_decode_score as scorer

    return [
        *((f"icp:{name}", partial(_build._load_extension, name)) for name in _build.EXTENSIONS),
        *((f"fmha:OnlyScoreIcp[{i}]", partial(_scorer_variant, i, page_size, world_size)) for i in range(2)),
        *((f"decode:{scorer._aot_function_name(q, r, page_size, world_size)}",
           partial(_decode_scan, scorer, q, r, _build.arch(), device, page_size, world_size))
          for q in (1, 2, 3, 4) for r in range(world_size)),
    ]


def _main_attention_artifacts(device, page_size=128, world_size=2) -> list[tuple[str, partial]]:
    """Optional compound-cache consumers; independent index caches do not need these."""
    q8 = {"topk": _TOPK, "gqa_ratio": _GQA, "device": device, "block_scale_shift": _SCALE_SHIFT}
    return [
        ("q8kv4:cpp_ext", partial(_call, "..decode_q8kv4.interface", "_get_cpp")),
        ("q8kv4:plan", partial(_call, "..decode_q8kv4.jit", "get_plan_fn", device=device)),
        *((f"q8kv4:forward[split_kv={split}]",
           partial(_call, "..decode_q8kv4.jit", "get_fmha_fwd_variant", split_kv=split, **q8))
          for split in (False, True)),
        ("nvfp4:prefill", partial(_nvfp4_prefill, device, page_size, world_size)),
    ]


def _run(policy: str | None, page_size=128, world_size=2, include_main_attention=False) -> tuple[int, list[str]]:
    import torch

    if world_size not in (2, 4) or page_size not in (128, 256):
        raise ValueError("ICP prewarm requires TP2/TP4 and P128/P256 index pages")

    saved = os.environ.pop("ICP_RUNTIME_JIT", None)
    if policy:
        os.environ["ICP_RUNTIME_JIT"] = policy
    try:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        artifacts, problems = _artifacts(device, page_size, world_size), []
        if include_main_attention:
            artifacts += _main_attention_artifacts(device, page_size, world_size)
        for label, load in artifacts:
            try:
                if device is None:
                    load()
                else:
                    with torch.cuda.device(device):
                        load()
            except Exception as exc:  # noqa: BLE001 - every artifact is reported
                problems.append(f"{label}: {exc}")
        return len(artifacts), problems
    finally:
        os.environ.pop("ICP_RUNTIME_JIT", None)
        if saved is not None:
            os.environ["ICP_RUNTIME_JIT"] = saved


def warm(*, page_size=128, world_size=2, include_main_attention=False) -> None:
    """Warm the indexer and optionally compound main attention; raise on failure."""
    problems = _run(None, page_size, world_size, include_main_attention)[1]
    if problems:
        raise RuntimeError("ICP prewarm failed:\n" + "\n".join(problems))


def check(*, page_size=128, world_size=2, include_main_attention=False) -> list[str]:
    """Load the selected artifacts with ICP_RUNTIME_JIT=fail; return any failures."""
    return _run("fail", page_size, world_size, include_main_attention)[1]


def _main(argv=None) -> int:
    """CLI: ``build --arch A`` warms with JIT allowed; ``verify`` checks with JIT refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--arch", required=True)
    build.add_argument("--page-size", type=int, choices=(128, 256), default=128)
    build.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    build.add_argument("--include-main-attention", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--arch")
    verify.add_argument("--page-size", type=int, choices=(128, 256), default=128)
    verify.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    verify.add_argument("--include-main-attention", action="store_true")
    verify.add_argument("--policy", choices=("fail",), default="fail")
    args = parser.parse_args(argv)
    arch = (args.arch or _build.arch()).strip().lower().removeprefix("sm_")
    if args.command == "build":
        if "cutlass" in sys.modules and os.environ.get("CUTE_DSL_ARCH") != f"sm_{arch}":
            parser.error("CUTLASS DSL was imported before CUTE_DSL_ARCH was set")
        # Every compiler targets the requested arch, with or without a device.
        os.environ.update(ICP_KERNEL_ARCH=arch, CUTE_DSL_ARCH=f"sm_{arch}")
        if args.include_main_attention:
            os.environ["FMHA_SM100_DECODE_Q8KV4_ARCH"] = arch
    elif arch != _build.arch():
        parser.error(f"--arch {args.arch} but this process targets sm_{_build.arch()}")
    count, problems = _run(None if args.command == "build" else "fail", args.page_size, args.world_size,
                           args.include_main_attention)
    for problem in problems:
        print(f"ICP_PREWARM_PROBLEM {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"ICP_PREWARM_{'BUILT' if args.command == 'build' else 'VERIFIED'} "
          f"sm_{arch} {count} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
