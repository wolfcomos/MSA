"""Candidate transport: D3 for pure decode, pack + all_to_all + K2 merge otherwise."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist

from . import _build

# D3 has no ACK: slot rotation is its only reuse protection.
_SLOTS = 3
# ~1 s at 2 GHz: a mismatched call sequence becomes a reported error.
_SPIN_CYCLES = 2_000_000_000
_TOPK = 16


@contextmanager
def _local_init(group):
    """Agree on local failures before entering a peer-memory collective."""
    if group is None:
        yield
        return
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    ok = torch.tensor([int(error is None)], dtype=torch.int32, device="cpu")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    if not ok.item():
        raise RuntimeError(f"ICP local initialization failed: {error!r}") from error


class _Exchange:
    """Model-scoped transport for every sparse layer; allocates nothing per call."""

    def __init__(self, group, *, rank: int, heads_local: int, extents: list[int],
                 device: torch.device, control_group=None) -> None:
        with _local_init(control_group):
            import torch.distributed._symmetric_memory as symm_mem

            self.group, self.rank, self.world, self.hl = group, rank, dist.get_world_size(group), heads_local
            if self.world not in (2, 4) or rank != dist.get_rank(group):
                raise ValueError("ICP exchange requires a TP2/TP4 group and its group-local rank")
            self.device = device
            self.T = extents[-1]
            hg = self.world * heads_local
            self.mod = m = _build._fused()
            self._pack, self._k2 = _build._pack(), _build._k2()
            self._select_flags = int(m.fused_flag_pdl) | int(m.fused_flag_early_trigger)
            # The merge waits at its end so the next layer's publish cannot overrun it.
            self._merge_flags = self._select_flags | int(m.fused_flag_end_wait)
            tile_tok, _, _, resident_cap = (int(v) for v in m.fused_plan(
                self.T, self.T, heads_local, self.world, 0))
            self.slot_words = int(m.fused_slot_words(self.T, hg))
            index = device.index
            backend = symm_mem.get_backend(device)
            if backend not in (None, "CUDA"):
                raise RuntimeError(f"symmetric memory backend is {backend!r}; D3 needs CUDA. "
                                   "Leave TORCH_SYMMMEM and NVSHMEM_MAX_TEAMS unset.")
            self._data = symm_mem.empty(self.slot_words * _SLOTS, dtype=torch.int32, device=index)
            self._pub_gen = torch.zeros((_SLOTS, self.T, hg), dtype=torch.int32, device=device)
            self._exp_gen = torch.zeros((_SLOTS, self.T, heads_local), dtype=torch.int32,
                                        device=device)
            self._status = torch.zeros(1, dtype=torch.int32, device=device)
            self._k2_status = torch.zeros(1, dtype=torch.int32, device=device)
            # Extents are identical on every rank, including capture padding.
            self._carriers = {
                e: tuple(torch.empty((self.world, e, heads_local, _TOPK, 2), dtype=torch.int32,
                                     device=device) for _ in range(2))
                for e in extents
            }
            self._data.zero_()
            torch.cuda.synchronize(index)
            self.provenance = {
                "kernel": "d3_fused", "abi": str(m.fused_abi_version), "world": self.world,
                "T_cap": self.T, "H_group": hg, "H_local": heads_local, "slots": _SLOTS,
                "use_pdl": 1, "tile_tok": tile_tok, "resident_cap": resident_cap,
                "slot_words": self.slot_words,
                "symm_mib": f"{self._data.numel() * 4 / 2 ** 20:.2f}",
                "symm_backend": backend or "CUDA", "arch": _build.arch(),
            }
        self._handle = symm_mem.rendezvous(self._data, group=group)
        with _local_init(control_group):
            self._ptrs = [int(p) for p in self._handle.buffer_ptrs]
        self._handle.barrier()
        torch.cuda.synchronize(index)

    def publish(self, scores, nvalid, forced_column, active, *, layer: int,
                token_offset: int) -> None:
        """Select decode rows ``[token_offset, +len(scores))`` into the owners' windows."""
        self.mod.fused_select_publish(
            scores, nvalid, forced_column, active, self._ptrs, self._pub_gen, self._status,
            self.rank, self.world, self.hl, token_offset,
            layer % _SLOTS, _SLOTS, self.T, self.slot_words,
            0, 1, self._select_flags)

    def merge_decode(self, out, *, layer: int, forced, n_ordinary) -> None:
        """Poll + merge this layer's published rows into ``out``."""
        self.mod.fused_merge(
            out, self._ptrs[self.rank], self._exp_gen, self._status,
            forced, n_ordinary, self.world, layer % _SLOTS,
            _SLOTS, self.T, self.slot_words, 0,
            _SPIN_CYCLES, self._merge_flags)

    def merge_prefill(self, candidates, out, *, forced, n_ordinary) -> None:
        """Pack, all_to_all and K2-merge ``candidates`` ``[cap, H_group, 16, 2]`` into ``out``."""
        send, recv = self._carriers[candidates.shape[0]]
        self._pack.pack_send_carrier(candidates, send, self.world)
        dist.all_to_all_single(recv, send, group=self.group)
        self._k2.k2_merge(recv, out, self.rank * self.hl, self.world, self.rank, forced,
                          n_ordinary, self._k2_status)

    def close(self) -> None:
        """Drain, barrier, drain; collective."""
        torch.cuda.synchronize(self.device)
        self._handle.barrier()
        torch.cuda.synchronize(self.device)
        self._handle = self._data = None
        self._ptrs = []
