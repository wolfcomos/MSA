# Fused indexer prefill prototype

Reference implementation for FlashInfer CAKE issue #8170. This branch contains
the scorer and local candidate reduction, not a standalone serving backend.
The caller still owns cache writes, candidate exchange and cross-rank owner merge.

## Run the baseline

Use a Linux CUDA environment satisfying the repository requirements, with a
CUDA-enabled PyTorch installation and an SM100 or SM103 GPU. The SM103 TMEM
load-max path requires a toolchain exposing the corresponding CUTLASS feature;
other supported builds retain the ordinary load/reduction path.

```bash
git clone --recursive --branch prototype/fused-prefill-top16-cake \
  https://github.com/wolfcomos/MSA.git msa-fused-prefill
cd msa-fused-prefill
python -m pip install -e . pytest
python -m pytest -q \
  tests/smoke/test_index_score_sharded.py \
  tests/smoke/test_fused_prefill_topk.py \
  tests/smoke/test_fused_prefill_reference.py \
  tests/smoke/test_fused_prefill_tmem_max.py
```

To run just parametrizations exercising P256, append `-k 256`. Run the full
command as well to cover the page-independent candidate-merge tests.

These tests simulate each TP rank on one GPU and check its local results.
They do not test NCCL exchange, distributed rank synchronization, or full-model
accuracy. Full-model validation is a separate integration gate.

## P256 contract

The physical page has 256 tokens; ranking blocks remain B128. Index K is
contiguously sharded within each physical page:

| TP | Rows per rank per page | Local column to global B128 ID |
| --- | --- | --- |
| 2 | 128 | `2 * column + rank` |
| 4 | 64 | `2 * column + rank // 2` |

Call `fmha_index_score_sharded` with `tp`, `rank`, `compact=True`,
`fused_topk=True`, and FP32 scratch `[query_tokens, index_heads, 64]`.
The physical page table is unchanged. Pass `index_scale=2` and
`index_offset=rank * 2 // tp` to `sparse_topk_merge_prefill_partials`.
`num_valid_pages` and forced-block counts use global **B128** units, not P256
physical-page units. Pass the same forced-block policy to both operations.

The scorer retains two disjoint stage-local Top16 lists. The partial merge
writes the existing int32 carrier `[index_heads, query_tokens, 33]`:
16 bit-cast FP32 scores, 16 global block IDs, and a ready field. Invalid IDs are
`-1`. Selected IDs are sorted for downstream merging.

For TP4/P256, ranks 0/1 contribute to the same even B128 block and ranks 2/3
to the same odd block. The owner merge must reduce duplicate block IDs by max
before global selection. TP2/P256 still requires head-owner exchange even
though each B128 block is wholly local to one rank.

Reuse bounded scratch across query chunks, writing each chunk's candidates
into a different slice of the full-step candidate buffer. Exchange once after
the chunks finish. Do not concatenate different queries into one Top16 list.

`test_fused_prefill_multisegment_chunk_reuse` is the executable example for
chunking and buffer reuse. It compares against materialized scoring/selection
and replays a CUDA Graph after changing Q. The independent-reference and TMEM
tests cover forced tails, padding, ties and special values.

P256 indexer support does not by itself establish support for a model's main
attention cache layout. Check both physical main-KV and index-K page shapes
in the serving integration; an accepted `--block-size 256` flag is insufficient.
