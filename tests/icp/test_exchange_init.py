"""CPU process-group regression for failures before peer-memory collectives."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _initialization_failure(rank, rendezvous, failure):
    import torch.distributed._symmetric_memory as symm_mem

    from fmha_sm100.icp import exchange

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    entered_rendezvous = False

    def fail():
        raise RuntimeError("injected rank-local failure")

    class Handle:
        @property
        def buffer_ptrs(self):
            if rank == 1 and failure == "pointer_setup":
                fail()
            return [0, 0]

        def barrier(self):
            dist.barrier()

    def peer_rendezvous(*args, **kwargs):
        nonlocal entered_rendezvous
        entered_rendezvous = True
        dist.barrier()
        return Handle()

    def allocation(*args, **kwargs):
        if rank == 1 and failure == "allocation":
            fail()
        kwargs["device"] = "cpu"
        return torch.empty(*args, **kwargs)

    module = SimpleNamespace(
        fused_flag_pdl=1, fused_flag_early_trigger=2, fused_flag_end_wait=4,
        fused_plan=lambda *args: (1, 0, 0, 1),
        fused_slot_words=lambda *args: 64, fused_abi_version=1,
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(exchange._build, "_fused",
                          fail if rank == 1 and failure == "compile" else lambda: module)
            patch.setattr(exchange._build, "_pack", lambda: module)
            patch.setattr(exchange._build, "_k2", lambda: module)
            patch.setattr(exchange._build, "arch", lambda: "cpu-test")
            patch.setattr(symm_mem, "empty", allocation)
            patch.setattr(symm_mem, "get_backend", lambda device: "CUDA")
            patch.setattr(symm_mem, "rendezvous", peer_rendezvous)
            patch.setattr(torch.cuda, "synchronize", lambda device=None: None)
            with pytest.raises(RuntimeError, match="ICP local initialization failed"):
                exchange._Exchange(
                    dist.group.WORLD, rank=rank, heads_local=2, extents=[4],
                    device=torch.device("cpu"), control_group=dist.group.WORLD,
                )
            assert entered_rendezvous == (failure == "pointer_setup")
        # Every rank must leave the failed constructor, rather than wait on a peer.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["compile", "allocation", "pointer_setup"])
def test_rank_local_failure_reaches_every_rank(tmp_path, failure):
    mp.spawn(
        _initialization_failure,
        args=((tmp_path / "rendezvous").as_uri(), failure),
        nprocs=2,
    )
