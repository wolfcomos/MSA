// K2: the NCCL-route merge. Source-major C4 receive carrier
// cand [W, Qchunk, H_local, 16, 2] int32 (score bits, id; both bitcast)
// -> out [Qchunk, H_local, 16] int32 ids ascending with a -1 tail (C6).
// One warp per (query, local head) row. `forced`/`n_ordinary` are per-token C3
// planes (n_ordinary is redundant with forced on purpose: it is checked against
// C3's formula); `status` is the required C5 failure word, only ORed into.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "merge_topk.cuh"

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

template <int KPT>
__global__ void k2_merge_kernel(const int32_t* __restrict__ cand,
                                int32_t* __restrict__ out,
                                const int32_t* __restrict__ forced,
                                const int32_t* __restrict__ n_ordinary,
                                int32_t* __restrict__ status, int W, int Qchunk,
                                int K, int Hl) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= Qchunk * Hl) return;

  const int t = row / Hl;
  const int hl = row - t * Hl;

  // Warp-uniform row metadata: forcing is indexed by the token only.
  int32_t f = icp::kNoForcedBlock;
  int q = icp::kTopK;
  int32_t err = icp::kStatusOk;
  if (forced != nullptr) {
    f = forced[t];
    q = n_ordinary[t];
    if (q != icp::ordinary_quota(f)) err |= icp::kStatusRowMeta;
  }

  uint64_t key[KPT];
  int32_t gid[KPT];
  bool nan_seen = false;
  icp::load_candidates_c4<KPT>(cand, W, Qchunk, Hl, K, t, hl, lane, key, gid,
                               nan_seen);
  // C5: a NaN on any valid record of this row fails the invocation.
  if (__any_sync(icp::kFullMask, nan_seen)) err |= icp::kStatusNaN;

  if (err != icp::kStatusOk) {
    // A failed row emits all -1 through the merge itself; an early return
    // here split loads from their consumers and cost registers.
    f = icp::kNoForcedBlock;
    q = 0;
    if (lane == 0) atomicOr(status, err);
  }

  icp::warp_merge_topk16<KPT>(key, gid, lane, f, q,
                              out + (long)row * icp::kTopK);
}

}  // namespace

void k2_merge(at::Tensor cand, at::Tensor out, int64_t head_offset,
              int64_t world, int64_t rank,
              c10::optional<at::Tensor> forced,
              c10::optional<at::Tensor> n_ordinary, at::Tensor status) {
  TORCH_CHECK(cand.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  // A float32 `cand` is the pre-C4 gathered tensor (axis 2 = H_group): refuse it.
  TORCH_CHECK(cand.scalar_type() == at::kInt,
              "cand must be the int32 C4 carrier [W,Qchunk,H_local,16,2]; got ",
              cand.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
  TORCH_CHECK(cand.is_contiguous() && out.is_contiguous(), "must be contiguous");
  TORCH_CHECK(cand.dim() == 5, "cand must be [W, Qchunk, H_local, K, 2]");
  TORCH_CHECK(out.dim() == 3, "out must be [Qchunk, H_local, 16]");

  const int C = cand.size(0);          // W, the SOURCE axis of the C4 carrier
  const int T = cand.size(1);          // Qchunk
  const int Hl_cand = cand.size(2);    // H_LOCAL, not H_group
  const int K = cand.size(3);
  TORCH_CHECK(cand.size(4) == 2, "last dim must be 2 (score bits, id)");
  TORCH_CHECK(K == icp::kTopK, "K must be 16");
  TORCH_CHECK(out.size(0) == T, "out query count must match cand");
  TORCH_CHECK(out.size(2) == icp::kTopK, "out last dim must be 16");
  const int Hl = out.size(1);
  TORCH_CHECK(world == C, "world must be the carrier's source axis W; got ",
              world, " for W=", C);
  TORCH_CHECK(rank >= 0 && rank < world, "rank outside [0, world), got ", rank);
  TORCH_CHECK(Hl_cand == Hl,
              "C4 receive carrier axis 2 must be H_local, matching out.size(1); "
              "got carrier ", Hl_cand, " vs out ", Hl);
  // C7 pins head_offset even though the source-major carrier no longer needs it.
  TORCH_CHECK(head_offset == rank * Hl,
              "C7 pins head_offset = icp_rank * H_local; got ", head_offset);
  TORCH_CHECK(C == 1 || C == 2 || C == 4, "the merge supports W in {1, 2, 4}");

  TORCH_CHECK(status.is_cuda(), "status must be a CUDA tensor");
  TORCH_CHECK(status.scalar_type() == at::kInt, "status must be int32");
  TORCH_CHECK(status.is_contiguous() && status.numel() >= 1,
              "status must be a contiguous int32 tensor with >= 1 element");

  // Both C3 planes or neither.
  TORCH_CHECK(forced.has_value() == n_ordinary.has_value(),
              "forced and n_ordinary must be supplied together (C3): got "
              "forced=", forced.has_value(), " n_ordinary=",
              n_ordinary.has_value());
  const int32_t* fp = nullptr;
  const int32_t* np = nullptr;
  if (forced.has_value()) {
    const at::Tensor& ft = forced.value();
    const at::Tensor& nt = n_ordinary.value();
    TORCH_CHECK(ft.is_cuda() && nt.is_cuda(), "forced/n_ordinary must be CUDA");
    TORCH_CHECK(ft.scalar_type() == at::kInt && nt.scalar_type() == at::kInt,
                "forced/n_ordinary must be int32");
    TORCH_CHECK(ft.is_contiguous() && nt.is_contiguous(),
                "forced/n_ordinary must be contiguous");
    TORCH_CHECK(ft.numel() == T && nt.numel() == T,
                "forced/n_ordinary are per-token planes of length T=", T,
                "; got ", ft.numel(), " and ", nt.numel());
    fp = ft.data_ptr<int32_t>();
    np = nt.data_ptr<int32_t>();
  }

  const int rows = T * Hl;
  if (rows == 0) return;
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto stream = c10::cuda::getCurrentCUDAStream();

  const int32_t* cp = cand.data_ptr<int32_t>();
  int32_t* op = out.data_ptr<int32_t>();
  int32_t* sp = status.data_ptr<int32_t>();
  if (C <= 2) {
    k2_merge_kernel<1><<<blocks, kThreads, 0, stream>>>(cp, op, fp, np, sp, C, T, K, Hl);
  } else {
    k2_merge_kernel<2><<<blocks, kThreads, 0, stream>>>(cp, op, fp, np, sp, C, T, K, Hl);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k2_merge", &k2_merge,
        "ICP candidate merge (refined-icp-v1.k2.2 host ABI on the "
        "refined-icp-v1.C4 carrier): [W,Qchunk,Hl,16,2] int32 -> "
        "[Qchunk,Hl,16] int32, with the C3 reserved forced slot and the C5 NaN "
        "failure word",
        // No defaults: C7, C3 and C5 are statements only the caller can make.
        pybind11::arg("cand"), pybind11::arg("out"),
        pybind11::arg("head_offset"), pybind11::arg("world"),
        pybind11::arg("rank"), pybind11::arg("forced"),
        pybind11::arg("n_ordinary"), pybind11::arg("status"));
}
