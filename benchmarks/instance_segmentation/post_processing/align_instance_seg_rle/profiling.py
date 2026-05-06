from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx


def nvtx_range_if_cuda(message: str, device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return nvtx.range(message)
    return nullcontext()
