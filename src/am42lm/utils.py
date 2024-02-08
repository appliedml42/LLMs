import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist


def bfloat_support():
    return (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
