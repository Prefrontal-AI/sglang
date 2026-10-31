"""MLA decode with heads split over the prefill-CP group.

Prefill CP runs attention TP 1, so every rank holds every MLA head and a decode
token would run all of them on every rank. Instead, while a decode forward runs,
the attention module is given this rank's view - its block of heads, slices of
q_b_proj, w_kc, w_vc and o_proj - so every existing absorbed path, with DCP's Q
head gather and reduce-scatter among them, runs as it does on a TP rank, and the
partial output projection is all-reduced over the group.
"""

import copy
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from torch import nn

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.cp.base import is_cp_enabled
from sglang.srt.runtime_context import get_parallel


def is_mla_decode_head_split() -> bool:
    parallel = get_parallel()
    return (
        is_cp_enabled()
        and parallel.attn_tp_size == 1
        and parallel.attn_cp_size > 1
        and parallel.attn_cp_size == parallel.tp_size
    )


class _BlockFp8Slice(nn.Module):
    """A block-fp8 linear on given weight and scale slices, called as the layer it slices."""

    def __init__(
        self,
        *,
        source: nn.Module,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        all_reduce: bool,
    ) -> None:
        super().__init__()
        self.quant_method = source.quant_method
        self.weight = weight
        self.weight_scale_inv = weight_scale_inv
        self.all_reduce = all_reduce

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = self.quant_method.w8a8_block_fp8_linear(
            input=x,
            weight=self.weight,
            block_size=self.quant_method.weight_block_size,
            weight_scale=self.weight_scale_inv,
            input_scale=None,
            bias=None,
        )
        if self.all_reduce:
            out = tensor_model_parallel_all_reduce(out)
        return out, None


class MLADecodeHeadView:
    """This rank's heads of one MLA layer, swapped in for the length of a decode forward."""

    def __init__(self, attn: nn.Module) -> None:
        parallel = get_parallel()
        num_heads = attn.num_local_heads // parallel.attn_cp_size
        first = parallel.attn_cp_rank * num_heads
        heads = slice(first, first + num_heads)
        q_rows = slice(first * attn.qk_head_dim, (first + num_heads) * attn.qk_head_dim)
        o_cols = slice(first * attn.v_head_dim, (first + num_heads) * attn.v_head_dim)
        block_n, block_k = attn.q_b_proj.quant_method.weight_block_size
        # Slices must fall on the fp8 weights' scale blocks
        if q_rows.start % block_n or o_cols.start % block_k:
            raise ValueError(
                f"MLA decode head split needs {num_heads} heads to cover whole "
                f"{block_n}x{block_k} scale blocks"
            )

        self.attrs = {
            "num_local_heads": num_heads,
            "w_kc": attn.w_kc[heads],
            "w_vc": attn.w_vc[heads],
            "q_b_proj": _BlockFp8Slice(
                source=attn.q_b_proj,
                weight=attn.q_b_proj.weight[q_rows],
                weight_scale_inv=attn.q_b_proj.weight_scale_inv[
                    q_rows.start // block_n : q_rows.stop // block_n
                ],
                all_reduce=False,
            ),
            # [hidden, local heads * v_head_dim]: this rank's columns, copied contiguous
            "o_proj": _BlockFp8Slice(
                source=attn.o_proj,
                weight=attn.o_proj.weight[:, o_cols].contiguous(),
                weight_scale_inv=attn.o_proj.weight_scale_inv[
                    :, o_cols.start // block_k : o_cols.stop // block_k
                ].contiguous(),
                all_reduce=True,
            ),
            "attn_mqa": copy.copy(attn.attn_mqa),
            "decode_heads_split": True,
        }
        self.attrs["attn_mqa"].tp_q_head_num = num_heads

    @contextmanager
    def active(self, attn: nn.Module) -> Iterator[None]:
        # Submodules live in _modules; anything else goes in the instance dict,
        # which attribute lookup reads before nn.Module's _parameters
        modules = attn._modules
        missing = object()
        saved = {}
        for name, value in self.attrs.items():
            store = modules if name in modules else attn.__dict__
            saved[name] = (store, store.get(name, missing))
            store[name] = value
        try:
            yield
        finally:
            for name, (store, value) in saved.items():
                if value is missing:
                    del store[name]
                else:
                    store[name] = value


def attach_mla_decode_head_view(attn: nn.Module) -> Optional[MLADecodeHeadView]:
    """Build the view for a loaded MLA layer when decode splits its heads, else None."""
    if not is_mla_decode_head_split():
        return None
    return MLADecodeHeadView(attn)
