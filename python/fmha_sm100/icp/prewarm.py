"""Load or compile every kernel used by the TP2/TP4 P128 indexer.

``python -m <pkg>.icp.prewarm build --arch 107a`` compiles them into the image;
``python -m <pkg>.icp.prewarm verify`` loads them with ICP_RUNTIME_JIT=fail.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

from . import _build


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


def _run(policy: str | None, page_size=128, world_size=2) -> tuple[int, list[str]]:
    import torch

    if world_size not in (2, 4) or page_size != 128:
        raise ValueError("ICP prewarm requires TP2/TP4 and 128-token index pages")

    saved = os.environ.pop("ICP_RUNTIME_JIT", None)
    if policy:
        os.environ["ICP_RUNTIME_JIT"] = policy
    try:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        artifacts, problems = _artifacts(device, page_size, world_size), []
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


def warm(*, page_size=128, world_size=2) -> None:
    """Load every indexer artifact on the current device, compiling what is missing; raise on failure."""
    problems = _run(None, page_size, world_size)[1]
    if problems:
        raise RuntimeError("ICP prewarm failed:\n" + "\n".join(problems))


def check(*, page_size=128, world_size=2) -> list[str]:
    """Load every indexer artifact with ICP_RUNTIME_JIT=fail; empty iff nothing would compile."""
    return _run("fail", page_size, world_size)[1]


def _main(argv=None) -> int:
    """CLI: ``build --arch A`` warms with JIT allowed; ``verify`` checks with JIT refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--arch", required=True)
    build.add_argument("--page-size", type=int, choices=(128,), default=128)
    build.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    verify = sub.add_parser("verify")
    verify.add_argument("--arch")
    verify.add_argument("--page-size", type=int, choices=(128,), default=128)
    verify.add_argument("--world-size", type=int, choices=(2, 4), default=2)
    verify.add_argument("--policy", choices=("fail",), default="fail")
    args = parser.parse_args(argv)
    arch = (args.arch or _build.arch()).strip().lower().removeprefix("sm_")
    if args.command == "build":
        if "cutlass" in sys.modules and os.environ.get("CUTE_DSL_ARCH") != f"sm_{arch}":
            parser.error("CUTLASS DSL was imported before CUTE_DSL_ARCH was set")
        # Every compiler targets the requested arch, with or without a device.
        os.environ.update(ICP_KERNEL_ARCH=arch, CUTE_DSL_ARCH=f"sm_{arch}")
    elif arch != _build.arch():
        parser.error(f"--arch {args.arch} but this process targets sm_{_build.arch()}")
    count, problems = _run(None if args.command == "build" else "fail", args.page_size, args.world_size)
    for problem in problems:
        print(f"ICP_PREWARM_PROBLEM {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"ICP_PREWARM_{'BUILT' if args.command == 'build' else 'VERIFIED'} "
          f"sm_{arch} {count} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
