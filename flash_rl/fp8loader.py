from vllm.device_allocator.cumem import CuMemAllocator
from contextlib import contextmanager
import torch
from torch.cuda.memory import MemPoolContext
from torch._C import (
_cuda_beginAllocateToPool,
_cuda_endAllocateCurrentStreamToPool,
)

@contextmanager
def disable_mem_pool():
    assert MemPoolContext.active_pool() == CuMemAllocator.get_instance().allocator_and_pools["weights"][0]
    pool = MemPoolContext.active_pool()
    ctx = MemPoolContext(None)
    device_index = torch.cuda.current_device()
    _cuda_endAllocateCurrentStreamToPool(device_index, pool.id)
    try:
        yield
    finally:
        _cuda_beginAllocateToPool(device_index, pool.id)
        del ctx