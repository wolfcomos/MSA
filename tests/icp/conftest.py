"""Markers and fixtures for the ICP tests: `gpu` needs CUDA, `distributed` needs 2 or 4 ranks."""

import os

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and builds ICP kernels")
    config.addinivalue_line("markers", "distributed: needs 2 or 4 ranks under torchrun")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords or "distributed" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cuda_device():
    """This rank's device, set before any collective initialises."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="session")
def icp_group(cuda_device):
    """The NCCL world group under torchrun with two or four ranks."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4):
        pytest.skip("needs 2 or 4 ranks under torchrun")
    if os.environ.get("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"):
        pytest.fail("unset TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES: it disables NVLS multicast")
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=cuda_device)
    yield dist.group.WORLD
    dist.barrier()
