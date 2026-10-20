# Indexer context parallelism in MSA

`fmha_sm100.icp` is the TP2/TP4 indexer-context-parallel (ICP) sparse indexer over an
independent index-K cache: each rank scores its contiguous 64-row (TP2) or 32-row (TP4)
fragment of every 128-token page, selects local top-16 candidates and exchanges them with
the other ranks. The engine owns the KV writer, per-step live metadata and OnlyScoreIcp
device plans, config admission, graph capture and lifetime.

Optional compound pages combine NVFP4 main KV and FP8 index K at TP2/TP4 with physical
P128/P256 pages. Each rank owns `page_size / TP` contiguous index rows; ranking blocks
remain B128. Pass `page_size=256` to `IcpIndexer` for P256 storage. Index views retain the
compound page stride without a pack copy.

## Public API

```python
from fmha_sm100.icp import IcpIndexer, IcpStep

indexer = IcpIndexer(group=..., rank=..., topk_buffer=..., num_sparse_layers=...,
                     index_head_dim=..., max_model_len=..., max_num_batched_tokens=...,
                     max_num_seqs=..., cudagraph_capture_sizes=...)  # collective
indexer.allocate(max_decode_query_len=...)     # once, before graph capture
step = indexer.plan(...)                       # per forward, host only
prepare_metadata(step.writer_args)             # engine fills live metadata before forward/replay
indexer.select(step, index_q, index_k, layer=i, scale=s)  # per sparse layer
indexer.close()                                # collective, at shutdown
```

Uniform decode with query length at most 4 scores with the CuTe decode scan; every other
step scores with the OnlyScoreIcp FMHA variant. Pure-decode steps exchange through the fused
symmetric-memory route (D3); steps with prefill rows use pack + `all_to_all_single` + K2.
D3 needs the CUDA symmetric-memory backend: leave `TORCH_SYMMMEM` and `NVSHMEM_MAX_TEAMS`
unset. The device-plan layout the engine fills is ABI 4.

## Prewarm

Build every indexer kernel into the image and verify it in a fresh process. Match
`--world-size` to the serving TP size (2 or 4):

```bash
export ICP_CACHE_ROOT=/opt/msa-icp-cache
python -m fmha_sm100.icp.prewarm build --arch 107a --world-size 4
python -m fmha_sm100.icp.prewarm verify --arch 107a --world-size 4   # prints ICP_PREWARM_VERIFIED
```

`prewarm.warm()` and `prewarm.check()` run the same loads in-process; `check()`
sets `ICP_RUNTIME_JIT=fail` and returns one line per artifact that would compile.
ICP extensions and decode scans are keyed by a source digest under `ICP_CACHE_ROOT`; the
OnlyScoreIcp FMHA variants use the usual MSA cache. Main-attention kernels are excluded by
default. For P256 compound caches, add `--page-size 256 --include-main-attention` to both
commands to include generic NVFP4 decode and BF16/FP8-query prefill; use `--page-size 128`
for P128. In-process, pass the matching `page_size`, `world_size` and
`include_main_attention=True` to `warm()` or `check()`.
