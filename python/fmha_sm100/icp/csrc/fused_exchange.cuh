// D3: the decode selector publishes straight into the owner's symmetric window;
// a separate, early-launched merge kernel only polls and merges.
//
// Window per slot: u64 [W_source][T_cap][H_local][16][2], u64 = (gen << 32) | word.
// Generations are per row: the selector CTA for (t, h) alone owns pub_gen[t][h]
// and the merge warp for (t, hl) alone owns exp_gen[t][hl]; both advance once
// per launch pair, so they agree without either reading the other's counter,
// which lets the merge launch before the selector finishes.
//
// Ordering: selector (PDL wait on the scorer, then early trigger) -> merge (no
// wait at start, the tags carry the dependency; early trigger; optional wait
// at the end so its completion implies the selector's).
#pragma once

#include <cstdint>

#include "local_candidates.cuh"
#include "merge_topk.cuh"

namespace icp {
namespace fused {

constexpr int kMaxWorld = 8;
constexpr int kMergeWarps = 8;
constexpr int kMergeThreads = kMergeWarps * 32;
constexpr int kMergeMinBlocksPerSm = 4;

constexpr int32_t kStatusTransport = 4;
static_assert((kStatusTransport & (kStatusNaN | kStatusRowMeta)) == 0,
              "the transport bit must not alias a K2 status bit");

constexpr int kFlagPdl = 1;           // launch as a programmatic dependent
constexpr int kFlagEarlyTrigger = 2;  // trigger dependents at kernel start
constexpr int kFlagEndWait = 4;       // merge: wait on the selector at exit
constexpr int kFlagsKnown = kFlagPdl | kFlagEarlyTrigger | kFlagEndWait;
// Never set by the host; the merge keeps their branches so its codegen matches 3e7fb69.
constexpr int kFlagNoAcquire = 16;
constexpr int kFlagClassicMerge = 32;

struct PeerWindows {
  int32_t* buf[kMaxWorld];
};

__device__ __forceinline__ uint64_t ld_relaxed_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_relaxed_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Generation 0 is what a zeroed window holds, so it is never issued.
__device__ __forceinline__ uint32_t next_generation(uint32_t prev) {
  const uint32_t next = prev + 1u;
  return next == 0u ? 1u : next;
}

// The bits store_local_candidate_keyonly would write, tagged with the generation.
struct PublishSink {
  uint64_t* dst;  // this row's 16 records in the owner's window; null = drop
  uint64_t tag;   // generation << 32
  float* mirror;  // always null; kept so the publisher's codegen matches 3e7fb69
  __device__ __forceinline__ void store(int slot, uint64_t key) const {
    const uint32_t gid = key ? ~static_cast<uint32_t>(key) : 0xffffffffu;
    const float score =
        key ? score_from_sortable(static_cast<uint32_t>(key >> 32)) : -CUDART_INF_F;
    if (dst != nullptr) {
      st_relaxed_sys_u64(dst + 2 * slot, tag | __float_as_uint(score));
      st_relaxed_sys_u64(dst + 2 * slot + 1, tag | gid);
    }
    if (mirror != nullptr) store_local_candidate_keyonly(mirror, slot, key);
  }
};

// One CTA per (token, head) row of this chunk; `pub_gen` is this slot's [T_cap][H_group] plane.
__global__ void __launch_bounds__(kFullRowThreads) fused_select_publish_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    int heads, int blocks, int scan_block_begin, int global_block_stride,
    const __grid_constant__ PeerWindows peers, uint32_t* __restrict__ pub_gen,
    // Always null; dropping it raises registers 98 -> 111 vs 3e7fb69 (sm_90a).
    const int32_t* __restrict__ status, float* __restrict__ mirror, int rank,
    int heads_local, int T_cap,
    int token_offset, long long slot_words, int slot, int flags) {
  __shared__ FullRowSelectSmem sm;
  __shared__ uint32_t s_seq;
  __shared__ int s_dead;
  const int row = blockIdx.x;
  const int token = row / heads;
  const int h = row - token * heads;
  const long t = static_cast<long>(token_offset) + token;
  uint32_t* const counter = pub_gen + t * heads + h;
  // Neither word is written by the scorer, so both loads overlap the wait.
  uint32_t prev = 0u;
  int dead = 0;
  if (threadIdx.x == 0) {
    prev = *counter;
    dead = (*reinterpret_cast<const volatile int32_t*>(status) & kStatusTransport) != 0;
  }
  local_pdl_wait(flags & kFlagPdl);
  if (flags & kFlagEarlyTrigger) local_pdl_complete(true);
  if (threadIdx.x == 0) {
    s_seq = next_generation(prev);
    s_dead = dead;
  }
  __syncthreads();
  const int p = h / heads_local;
  const int hl = h - p * heads_local;
  uint64_t* dst = nullptr;
  if (!s_dead) {
    dst = reinterpret_cast<uint64_t*>(peers.buf[p] + slot * slot_words) +
          2L * (((static_cast<long>(rank) * T_cap + t) * heads_local + hl) * kTopK);
  }
  const PublishSink sink{
      dst, static_cast<uint64_t>(s_seq) << 32,
      mirror != nullptr ? mirror + static_cast<int64_t>(row) * kTopK * 2 : nullptr};
  full_row_select_row(scores, nvalid, forced_col, active, row, heads, blocks,
                      scan_block_begin, global_block_stride, sink, sm);
  if (threadIdx.x == 0 && !s_dead) *counter = s_seq;
  if (!(flags & kFlagEarlyTrigger) && (flags & kFlagPdl)) local_pdl_complete(true);
}

// Poll + merge, one warp per (token, local head) row. `mybuf` is this rank's
// window at this slot; `exp_gen` this slot's [T_cap][H_local] plane. One
// candidate per lane for W <= 2, two for W = 4.
template <int KPT>
__global__ __launch_bounds__(kMergeThreads, kMergeMinBlocksPerSm) void fused_merge_kernel(
    const uint64_t* __restrict__ mybuf, int32_t* __restrict__ out,
    const int32_t* __restrict__ forced, const int32_t* __restrict__ n_ordinary,
    uint32_t* __restrict__ exp_gen, int32_t* __restrict__ status, int world,
    int T, int T_cap, int Hl, int tile_tok, int ntiles, int flags,
    long long spin_cycles) {
  constexpr int K = kTopK;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int j0 = (int)(((long long)ntiles * blockIdx.x) / gridDim.x);
  const int j1 = (int)(((long long)ntiles * (blockIdx.x + 1)) / gridDim.x);

  // No wait here: every input is tagged (the window) or older than the selector.
  if (flags & kFlagEarlyTrigger) local_pdl_complete(true);

  __shared__ int s_dead;
  if (tid == 0)
    s_dead = (*reinterpret_cast<volatile int32_t*>(status) & kStatusTransport) != 0;
  __syncthreads();
  if (s_dead) {
    const long r0 = (long)j0 * tile_tok * Hl * K;
    const long t1 = (long)j1 * tile_tok < T ? (long)j1 * tile_tok : T;
    for (long i = r0 + tid; i < t1 * Hl * K; i += kMergeThreads) out[i] = -1;
    if (flags & kFlagEndWait) local_pdl_wait(true);
    return;
  }

  const int n = world * K;
  bool lam_bad = false;  // warp-uniform and sticky: one timeout per warp
  for (int j = j0; j < j1; ++j) {
    const int t0 = j * tile_tok;
    const int ntok = (T - t0) < tile_tok ? (T - t0) : tile_tok;
    for (int w = warp; w < ntok * Hl; w += kMergeWarps) {
      const int lt = w / Hl;
      const int hl = w - lt * Hl;
      const int t = t0 + lt;
      uint32_t* const counter = exp_gen + (long)t * Hl + hl;
      uint32_t prev = 0u;
      int32_t f = kNoForcedBlock;
      int q = kTopK;
      if (lane == 0) {
        prev = *counter;
        if (forced != nullptr) {
          f = forced[t];
          q = n_ordinary[t];
        }
      }
      const uint32_t seq = next_generation(__shfl_sync(kFullMask, prev, 0));
      f = __shfl_sync(kFullMask, f, 0);
      q = __shfl_sync(kFullMask, q, 0);
      int32_t err = kStatusOk;
      if (forced != nullptr) {
        if (q != ordinary_quota(f)) err |= kStatusRowMeta;
      }

      uint64_t key[KPT];
      int32_t gid[KPT];
      bool nan_seen = false;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        // Same (lane, i) -> (source, k) map as load_candidates_c4.
        const int idx = lane + i * 32;
        const bool act = idx < n;
        const int c = act ? idx / K : 0;
        const int k = act ? idx - c * K : 0;
        const uint64_t* p2 = mybuf + ((((long)c * T_cap + t) * Hl + hl) * K + k) * 2;
        uint32_t sw = 0u;
        uint32_t gw = 0xffffffffu;
        if (!lam_bad && !(flags & kFlagNoAcquire)) {
          const long long q0 = clock64();
          for (;;) {
            bool ready = true;
            if (act) {
              const uint64_t w0 = ld_relaxed_sys_u64(p2);
              const uint64_t w1 = ld_relaxed_sys_u64(p2 + 1);
              // Exact compare: a later generation is refused, not accepted.
              ready = ((uint32_t)(w0 >> 32) == seq) && ((uint32_t)(w1 >> 32) == seq);
              if (ready) {
                sw = (uint32_t)w0;
                gw = (uint32_t)w1;
              }
            }
            if (__all_sync(kFullMask, ready)) break;
            if (__any_sync(kFullMask, (clock64() - q0) > spin_cycles)) {
              lam_bad = true;
              break;
            }
          }
        } else if (!lam_bad && act) {
          sw = (uint32_t)ld_relaxed_sys_u64(p2);
          gw = (uint32_t)ld_relaxed_sys_u64(p2 + 1);
        }
        const int32_t g = (act && !lam_bad) ? (int32_t)gw : -1;
        const float score = __uint_as_float(sw);
        nan_seen = nan_seen || (g >= 0 && score != score);
        key[i] = (act && !lam_bad) ? canonical_key(score, g) : 0ull;
        gid[i] = g;
      }
      if (lam_bad) err |= kStatusTransport;
      if (__any_sync(kFullMask, nan_seen)) err |= kStatusNaN;
      if (err != kStatusOk) {
        f = kNoForcedBlock;
        q = 0;
        if (lane == 0) atomicOr(status, err);
      }
      int32_t* const row_out = out + (long)(t * Hl + hl) * kTopK;
      if constexpr (KPT == 1) {
        if (!(flags & kFlagClassicMerge)) {
          warp_merge_topk16_net(key[0], gid[0], lane, f, q, row_out);
        } else {
          warp_merge_topk16<KPT>(key, gid, lane, f, q, row_out);
        }
      } else {
        warp_merge_topk16<KPT>(key, gid, lane, f, q, row_out);
      }
      if (lane == 0) *counter = seq;
    }
  }
  // Completion of this grid then implies the selector's, independent of PDL transitivity.
  if (flags & kFlagEndWait) local_pdl_wait(true);
}

__host__ __device__ inline int merge_tile_tokens(int Hl) {
  const int tt = kMergeWarps / Hl;
  return tt < 1 ? 1 : tt;
}

// int32 words per slot.
__host__ __device__ inline long long slot_words(long long T_cap, long long Hg) {
  return T_cap * Hg * kTopK * 4;
}

}  // namespace fused
}  // namespace icp
