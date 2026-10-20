// Host side of the D3 fused decode exchange (see fused_exchange.cuh).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "fused_exchange.cuh"

namespace {

namespace F = icp::fused;

int cdiv(long a, long b) { return (int)((a + b - 1) / b); }

// The merge's resident-CTA cap per device and candidates per lane.
template <int KPT>
int resident_ctas_for() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  static std::mutex mu;
  static std::vector<int> cache;
  std::lock_guard<std::mutex> lock(mu);
  if ((int)cache.size() <= dev) cache.resize(dev + 1, 0);
  if (cache[dev] == 0) {
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, F::fused_merge_kernel<KPT>, F::kMergeThreads, 0));
    TORCH_CHECK(per_sm >= 1, "fused merge cannot be resident on device ", dev);
    cache[dev] = sms * per_sm;
  }
  return cache[dev];
}

int resident_ctas(int64_t world) {
  return world <= 2 ? resident_ctas_for<1>() : resident_ctas_for<2>();
}

void check_world(int64_t world) {
  TORCH_CHECK(world == 1 || world == 2 || world == 4,
              "fused: world must be 1, 2 or 4");
}

F::PeerWindows peer_windows(const std::vector<int64_t>& ptrs, int64_t world) {
  check_world(world);
  TORCH_CHECK((int64_t)ptrs.size() == world, "fused: need one window pointer per rank");
  F::PeerWindows peers{};
  for (int i = 0; i < (int)world; ++i) {
    TORCH_CHECK(ptrs[i] != 0, "fused: null window ", i);
    TORCH_CHECK((ptrs[i] % 16) == 0, "fused: window ", i, " is not 16-byte aligned");
    peers.buf[i] = reinterpret_cast<int32_t*>(ptrs[i]);
  }
  return peers;
}

void check_int32(const at::Tensor& t, const at::Device& device, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, "fused: ", name,
              " must be on the scores' device");
  TORCH_CHECK(t.scalar_type() == at::kInt, "fused: ", name, " must be int32");
  TORCH_CHECK(t.is_contiguous(), "fused: ", name, " must be contiguous");
}

void set_pdl(cudaLaunchConfig_t& cfg, cudaLaunchAttribute* attr, int64_t flags) {
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = (flags & F::kFlagPdl) ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
}

}  // namespace

std::vector<int64_t> fused_plan(int64_t T, int64_t T_cap, int64_t Hl,
                                int64_t world, int64_t max_ctas) {
  TORCH_CHECK(Hl >= 1 && T >= 0 && T_cap >= T, "fused_plan: bad geometry");
  const int tt = F::merge_tile_tokens((int)Hl);
  const int ntiles = cdiv(T, tt);
  check_world(world);
  const int cap = max_ctas > 0 ? (int)max_ctas : resident_ctas(world);
  const int grid = ntiles < cap ? ntiles : cap;
  return {tt, ntiles, grid, cap};
}

int64_t fused_slot_words(int64_t T_cap, int64_t Hg) { return F::slot_words(T_cap, Hg); }

// Selector for one decode chunk: rows [token_offset, token_offset + T_e) of
// the invocation, published into slot `slot` of every owner's window.
void fused_select_publish(at::Tensor scores, at::Tensor nvalid, at::Tensor forced_col,
                          at::Tensor active, std::vector<int64_t> peer_buf_ptrs,
                          at::Tensor pub_gen, at::Tensor status, int64_t rank,
                          int64_t world, int64_t heads_local, int64_t token_offset,
                          int64_t slot, int64_t slots, int64_t tokens_capacity,
                          int64_t slot_capacity_words, int64_t scan_block_begin,
                          int64_t global_block_stride, int64_t flags) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat &&
                  scores.is_contiguous() && scores.dim() == 3,
              "fused: scores must be contiguous CUDA fp32 [T, H_group, N]");
  const auto device = scores.device();
  const int64_t Te = scores.size(0);
  const int64_t Hg = scores.size(1);
  const int64_t N = scores.size(2);
  TORCH_CHECK(N >= 1 && N <= icp::kMaxRowBlocks,
              "fused: the full-row selector covers N <= ", icp::kMaxRowBlocks, " columns; got ", N);
  TORCH_CHECK(Hg == world * heads_local, "fused: H_group must be world * H_local (C7)");
  TORCH_CHECK(heads_local >= 1 && heads_local <= 16 && 16 % heads_local == 0,
              "fused: H_local must be in {1, 2, 4, 8, 16}");
  TORCH_CHECK(rank >= 0 && rank < world, "fused: rank outside [0, world)");
  TORCH_CHECK(slots >= 1 && slot >= 0 && slot < slots, "fused: slot outside [0, slots)");
  TORCH_CHECK(flags >= 0 && (flags & ~(int64_t)F::kFlagsKnown) == 0,
              "fused: unknown flags ", flags);
  TORCH_CHECK(token_offset >= 0 && token_offset + Te <= tokens_capacity,
              "fused: rows [", token_offset, ", ", token_offset + Te,
              ") exceed the window capacity ", tokens_capacity);
  TORCH_CHECK(slot_capacity_words >= F::slot_words(tokens_capacity, Hg),
              "fused: an undersized window is a silent overrun of a PEER's allocation");
  TORCH_CHECK(scan_block_begin >= 0 &&
                  scan_block_begin + global_block_stride * (N - 1) <= INT32_MAX,
              "fused: global block ids exceed int32");
  check_int32(nvalid, device, "nvalid");
  check_int32(forced_col, device, "forced_col");
  TORCH_CHECK(nvalid.numel() == Te && forced_col.numel() == Te,
              "fused: nvalid/forced_col are per-token planes of length T");
  TORCH_CHECK(active.is_cuda() && active.device() == device &&
                  active.scalar_type() == at::kBool && active.is_contiguous() &&
                  active.numel() == Te,
              "fused: active must be a contiguous bool [T] on the scores' device");
  check_int32(pub_gen, device, "pub_gen");
  TORCH_CHECK(pub_gen.numel() >= slots * tokens_capacity * Hg,
              "fused: pub_gen must hold slots * T_cap * H_group counters");
  check_int32(status, device, "status");
  const F::PeerWindows peers = peer_windows(peer_buf_ptrs, world);
  if (Te == 0) return;

  const c10::cuda::CUDAGuard guard(device);
  if (flags & F::kFlagPdl) {
    TORCH_CHECK(at::cuda::getDeviceProperties(device.index())->major >= 9,
                "PDL requires compute capability >= 9.0");
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(Te * Hg));
  cfg.blockDim = dim3(icp::kFullRowThreads);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  set_pdl(cfg, attr, flags);
  uint32_t* gp = reinterpret_cast<uint32_t*>(pub_gen.data_ptr<int32_t>()) +
                 slot * tokens_capacity * Hg;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, F::fused_select_publish_kernel, scores.data_ptr<float>(),
      nvalid.data_ptr<int32_t>(), forced_col.data_ptr<int32_t>(),
      active.data_ptr<bool>(), (int)Hg, (int)N, (int)scan_block_begin,
      (int)global_block_stride, peers, gp, status.data_ptr<int32_t>(), nullptr,
      (int)rank, (int)heads_local, (int)tokens_capacity, (int)token_offset,
      (long long)slot_capacity_words, (int)slot, (int)flags));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Poll this rank's own window and merge rows [0, T) into `out`.
void fused_merge(at::Tensor out, int64_t own_buf_ptr, at::Tensor exp_gen,
                 at::Tensor status, c10::optional<at::Tensor> forced,
                 c10::optional<at::Tensor> n_ordinary, int64_t world,
                 int64_t slot, int64_t slots, int64_t tokens_capacity,
                 int64_t slot_capacity_words, int64_t max_ctas,
                 int64_t spin_cycles, int64_t flags) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.is_contiguous() &&
                  out.dim() == 3 && out.size(2) == icp::kTopK,
              "fused: out must be contiguous int32 [T, H_local, 16]");
  const auto device = out.device();
  const int64_t T = out.size(0);
  const int64_t Hl = out.size(1);
  const int64_t Hg = world * Hl;
  check_world(world);
  TORCH_CHECK(Hl >= 1 && Hl <= 16 && 16 % Hl == 0, "fused: H_local must be in {1, 2, 4, 8, 16}");
  TORCH_CHECK(slots >= 1 && slot >= 0 && slot < slots, "fused: slot outside [0, slots)");
  TORCH_CHECK(flags >= 0 && (flags & ~(int64_t)F::kFlagsKnown) == 0,
              "fused: unknown flags ", flags);
  TORCH_CHECK(spin_cycles > 0, "fused: spin_cycles must be positive");
  TORCH_CHECK(T <= tokens_capacity, "fused: extent ", T, " exceeds the window capacity ",
              tokens_capacity);
  TORCH_CHECK(slot_capacity_words >= F::slot_words(tokens_capacity, Hg),
              "fused: undersized window");
  TORCH_CHECK(own_buf_ptr != 0 && own_buf_ptr % 16 == 0, "fused: bad own window pointer");
  check_int32(exp_gen, device, "exp_gen");
  TORCH_CHECK(exp_gen.numel() >= slots * tokens_capacity * Hl,
              "fused: exp_gen must hold slots * T_cap * H_local counters");
  check_int32(status, device, "status");
  TORCH_CHECK(forced.has_value() == n_ordinary.has_value(),
              "fused: forced and n_ordinary must be supplied together (C3)");
  const int32_t* fp = nullptr;
  const int32_t* np = nullptr;
  if (forced.has_value()) {
    check_int32(forced.value(), device, "forced");
    check_int32(n_ordinary.value(), device, "n_ordinary");
    TORCH_CHECK(forced.value().numel() == T && n_ordinary.value().numel() == T,
                "fused: forced/n_ordinary are per-token planes of length T");
    fp = forced.value().data_ptr<int32_t>();
    np = n_ordinary.value().data_ptr<int32_t>();
  }
  if (T == 0) return;

  const c10::cuda::CUDAGuard guard(device);
  const int tt = F::merge_tile_tokens((int)Hl);
  const int ntiles = cdiv(T, tt);
  const int cap = max_ctas > 0 ? (int)max_ctas : resident_ctas(world);
  const int grid = ntiles < cap ? ntiles : cap;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(F::kMergeThreads);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  set_pdl(cfg, attr, flags);
  const uint64_t* mybuf = reinterpret_cast<const uint64_t*>(
      reinterpret_cast<const int32_t*>(own_buf_ptr) + slot * slot_capacity_words);
  uint32_t* gp = reinterpret_cast<uint32_t*>(exp_gen.data_ptr<int32_t>()) +
                 slot * tokens_capacity * Hl;
  int32_t* const outp = out.data_ptr<int32_t>();
  int32_t* const statp = status.data_ptr<int32_t>();

  if (world <= 2) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, F::fused_merge_kernel<1>, mybuf, outp, fp, np,
                                      gp, statp, (int)world, (int)T, (int)tokens_capacity,
                                      (int)Hl, tt, ntiles, (int)flags,
                                      (long long)spin_cycles));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, F::fused_merge_kernel<2>, mybuf, outp, fp, np,
                                      gp, statp, (int)world, (int)T, (int)tokens_capacity,
                                      (int)Hl, tt, ntiles, (int)flags,
                                      (long long)spin_cycles));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_select_publish", &fused_select_publish,
        "Full-row decode selection that publishes tagged records into the "
        "owners' symmetric windows (D3)",
        pybind11::arg("scores"), pybind11::arg("nvalid"), pybind11::arg("forced_col"),
        pybind11::arg("active"), pybind11::arg("peer_buf_ptrs"), pybind11::arg("pub_gen"),
        pybind11::arg("status"), pybind11::arg("rank"),
        pybind11::arg("world"), pybind11::arg("heads_local"),
        pybind11::arg("token_offset"), pybind11::arg("slot"), pybind11::arg("slots"),
        pybind11::arg("tokens_capacity"), pybind11::arg("slot_capacity_words"),
        pybind11::arg("scan_block_begin"), pybind11::arg("global_block_stride"),
        pybind11::arg("flags"));
  m.def("fused_merge", &fused_merge,
        "Poll this rank's window and merge (K2 semantics); no publish",
        pybind11::arg("out"), pybind11::arg("own_buf_ptr"), pybind11::arg("exp_gen"),
        pybind11::arg("status"), pybind11::arg("forced"), pybind11::arg("n_ordinary"),
        pybind11::arg("world"), pybind11::arg("slot"), pybind11::arg("slots"),
        pybind11::arg("tokens_capacity"), pybind11::arg("slot_capacity_words"),
        pybind11::arg("max_ctas"), pybind11::arg("spin_cycles"), pybind11::arg("flags"));
  m.def("fused_plan", &fused_plan, "(tile_tok, ntiles, grid, resident_cap)",
        pybind11::arg("T"), pybind11::arg("T_cap"), pybind11::arg("H_local"),
        pybind11::arg("world"), pybind11::arg("max_ctas") = 0);
  m.def("fused_slot_words", &fused_slot_words, "int32 words per slot (K5T layout)",
        pybind11::arg("T_cap"), pybind11::arg("H_group"));
  m.attr("fused_abi_version") = "refined-icp-v1.d3.1";
  m.attr("fused_status_transport") = static_cast<int>(F::kStatusTransport);
  m.attr("fused_flag_pdl") = F::kFlagPdl;
  m.attr("fused_flag_early_trigger") = F::kFlagEarlyTrigger;
  m.attr("fused_flag_end_wait") = F::kFlagEndWait;
}
