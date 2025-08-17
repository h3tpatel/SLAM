# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope as _kernels  # run `python setup.py install`
    _HAS_KERNELS = True
except Exception:  # pragma: no cover - fall back to pure PyTorch
    try:
        from . import curope as _kernels  # run `python setup.py build_ext --inplace`
        _HAS_KERNELS = True
    except Exception:  # pragma: no cover
        _kernels = None
        _HAS_KERNELS = False


def _rope_2d_torch(tokens: torch.Tensor, positions: torch.Tensor, base: float, F0: float) -> None:
    """Pure PyTorch implementation used when CUDA kernels are unavailable."""
    B, N, H, D4 = tokens.shape
    D = D4 // 4
    device = tokens.device
    dtype = tokens.dtype
    freqs = base ** (torch.arange(D, device=device, dtype=dtype) / float(D))
    for axis in range(2):
        p = positions[:, :, axis].to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        ang = F0 * p / freqs.view(1, 1, 1, D)
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        u = tokens[..., axis * 2 * D : axis * 2 * D + D]
        v = tokens[..., axis * 2 * D + D : axis * 2 * D + 2 * D]
        tokens[..., axis * 2 * D : axis * 2 * D + D] = u * cos - v * sin
        tokens[..., axis * 2 * D + D : axis * 2 * D + 2 * D] = v * cos + u * sin


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        if _HAS_KERNELS and tokens.is_cuda():
            _kernels.rope_2d(tokens, positions, base, F0)
        else:
            _rope_2d_torch(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        if _HAS_KERNELS and grad_res.is_cuda():
            _kernels.rope_2d(grad_res, positions, base, -F0)
        else:
            _rope_2d_torch(grad_res, positions, base, -F0)
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions): 
        cuRoPE2D_func.apply(tokens.transpose(1, 2), positions, self.base, self.F0)
        return tokens
