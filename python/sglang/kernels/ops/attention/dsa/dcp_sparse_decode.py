"""Split-KV sparse MLA decode for decode context parallel (DCP) on CUDA.

Under DCP every rank holds 1/dcp of the KV rows (global slot ``s`` lives on rank
``s % dcp`` at local row ``s // dcp``) but receives the whole sparse index list of
each query row. The single-CTA ``sparse_attention_fwd_kernel_v1`` walks the whole
list serially and just masks the entries other ranks own, so a decode step pays the
full list width on one CTA per row.

This module replaces it with three CUDA-graph-safe launches (static grids, no host
syncs, device-side lengths):

1. ``_compact_kernel``: per row, keep only the entries this rank owns, convert them
   to local rows (``slot // dcp``) and pack them at the front of a ``[rows, width]``
   buffer, together with the per-row count. The worst case (one rank owns every
   entry) is still correct; only the tail past the count is never read.
2. ``_partial_kernel``: grid ``(rows, num_splits, head_blocks)``. Split ``s`` walks
   key tiles ``s, s + num_splits, ...`` below ``ceil(count / BLOCK_K)`` and writes an
   fp32 partial output and a base-2 LSE (``-inf`` for a split with no valid key).
3. ``_combine_kernel``: merges the splits with the usual LSE weights and writes the
   bf16 output and the final base-2 LSE that ``cp_lse_ag_out_rs_mla`` merges across
   ranks. A row with no owned entry yields output 0 and LSE ``-inf``, the same
   contract as the ``return_lse`` TileLang kernel.

LSE convention (identical to ``sparse_attention_fwd_kernel_v1``): base 2 of the
scaled logits, ``log2(sum_j 2^(s_j * sm_scale * log2e - m)) + m``.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_NEG_BIG = tl.constexpr(-1.0e30)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _compact_kernel(
    idx_ptr,  # [R, W] int32, global slots, -1 = masked
    out_ptr,  # [R, W] int32, local rows of the owned entries packed at the front
    cnt_ptr,  # [R] int32
    W,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_W)
    in_range = col < W
    slot = tl.load(idx_ptr + row.to(tl.int64) * W + col, mask=in_range, other=-1)
    owned = (slot >= 0) & ((slot % DCP) == RANK)
    flag = owned.to(tl.int32)
    pos = tl.cumsum(flag, axis=0) - 1
    tl.store(out_ptr + row.to(tl.int64) * W + pos, slot // DCP, mask=owned)
    tl.store(cnt_ptr + row, tl.sum(flag, axis=0))


@triton.jit
def _partial_kernel(
    q_ptr,  # [R, H, D] bf16
    kv_ptr,  # [slots, D] bf16 (row stride stride_kv)
    idx_ptr,  # [R, W] int32 compacted local rows
    cnt_ptr,  # [R] int32
    po_ptr,  # [R, S, H, D] fp32
    plse_ptr,  # [R, S, H] fp32
    qk_scale,  # sm_scale * log2(e)
    W,
    stride_qr,
    stride_qh,
    stride_kv,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    hb = tl.program_id(2)

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h < H
    d = tl.arange(0, D)
    k_off = tl.arange(0, BLOCK_K)

    q = tl.load(
        q_ptr + row.to(tl.int64) * stride_qr + h[:, None] * stride_qh + d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    n = tl.load(cnt_ptr + row)
    n_tiles = tl.cdiv(n, BLOCK_K)

    m_i = tl.full((BLOCK_H,), _NEG_BIG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    for j in range(split, n_tiles, S):
        k_pos = j * BLOCK_K + k_off
        valid = k_pos < n
        slot = tl.load(idx_ptr + row.to(tl.int64) * W + k_pos, mask=valid, other=0)
        kv = tl.load(
            kv_ptr + slot.to(tl.int64)[:, None] * stride_kv + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(kv)) * qk_scale
        s = tl.where(valid[None, :], s, _NEG_BIG)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    has = l_i > 0.0
    o = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, tl.log2(tl.where(has, l_i, 1.0)) + m_i, float("-inf"))

    base = (row.to(tl.int64) * S + split) * H
    tl.store(po_ptr + (base + h)[:, None] * D + d[None, :], o, mask=h_mask[:, None])
    tl.store(plse_ptr + base + h, lse, mask=h_mask)


@triton.jit
def _combine_kernel(
    po_ptr,  # [R, S, H, D] fp32
    plse_ptr,  # [R, S, H] fp32
    out_ptr,  # [R, H, D] bf16
    lse_ptr,  # [R, H] fp32
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    S_P2: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h < H
    sp = tl.arange(0, S_P2)
    s_mask = sp < S
    d = tl.arange(0, D)

    lse = tl.load(
        plse_ptr + (row.to(tl.int64) * S + sp)[None, :] * H + h[:, None],
        mask=h_mask[:, None] & s_mask[None, :],
        other=float("-inf"),
    )  # [BLOCK_H, S_P2]
    m = tl.max(lse, axis=1)
    live = m > float("-inf")
    m_safe = tl.where(live, m, 0.0)
    w = tl.exp2(lse - m_safe[:, None])  # exp2(-inf) = 0
    wsum = tl.sum(w, axis=1)
    wsum_safe = tl.where(live, wsum, 1.0)

    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    for s in tl.static_range(S):
        # pick split s's weight column out of w without a gather
        w_s = tl.sum(tl.where(sp[None, :] == s, w, 0.0), axis=1)
        po = tl.load(
            po_ptr + ((row.to(tl.int64) * S + s) * H + h)[:, None] * D + d[None, :],
            mask=h_mask[:, None],
            other=0.0,
        )
        acc += po * w_s[:, None]
    out = tl.where(live[:, None], acc / wsum_safe[:, None], 0.0)
    tl.store(
        out_ptr + (row.to(tl.int64) * H + h)[:, None] * D + d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None],
    )
    tl.store(
        lse_ptr + row.to(tl.int64) * H + h,
        tl.where(live, m_safe + tl.log2(wsum_safe), float("-inf")),
        mask=h_mask,
    )


def dcp_compact_indices(
    indices: torch.Tensor, dcp_size: int, dcp_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Owned entries of each row as local rows, packed at the front, plus the counts."""
    if indices.dim() != 2 or indices.dtype != torch.int32:
        raise ValueError(
            f"DCP indices must be a 2D int32 tensor, got dim={indices.dim()} "
            f"dtype={indices.dtype}"
        )
    indices = indices.contiguous()
    rows, width = indices.shape
    out = torch.empty_like(indices)
    cnt = torch.empty(rows, dtype=torch.int32, device=indices.device)
    block_w = max(triton.next_power_of_2(width), 128)
    _compact_kernel[(rows,)](
        indices,
        out,
        cnt,
        width,
        DCP=dcp_size,
        RANK=dcp_rank,
        BLOCK_W=block_w,
        num_warps=8 if block_w >= 2048 else 4,
    )
    return out, cnt


@functools.lru_cache(maxsize=None)
def _max_block_h(device_index: int) -> int:
    # On Blackwell a 64-head tile of the 512-wide latent faults with a misaligned address
    # (and two stages of it overflow shared memory); 32 heads fit and run as fast (B300:
    # 19.8 us for 2 rows of 4612 keys, error 3e-4 against the reference)
    return 32 if torch.cuda.get_device_capability(device_index)[0] >= 10 else 64


def pick_split_config(
    rows: int, width: int, dcp_size: int, heads: int, device: torch.device
) -> Tuple[int, int]:
    """(BLOCK_K, num_splits) from static shapes only (CUDA-graph safe).

    The expected owned length is ``width / dcp``; the split count is capped by the
    number of key tiles that length fills and by what keeps ~2 CTAs per SM busy.
    A row that owns more than expected still comes out right (each split loops over
    its strided tiles), only slower.
    """
    # Measured on H200 at 2048-4612 wide lists: 64-key tiles and 6-12 splits are the flat optimum.
    block_k = 64
    head_blocks = triton.cdiv(heads, 64)
    expected = triton.cdiv(width, dcp_size) * 5 // 4 + 1
    tiles = max(1, triton.cdiv(expected, block_k))
    target = max(1, (2 * _sm_count(device.index or 0)) // max(1, rows * head_blocks))
    return block_k, max(1, min(tiles, target, 12))


def dcp_sparse_decode_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    dcp_size: int,
    dcp_rank: int,
    *,
    block_k: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode over this rank's shard of the KV rows.

    q: [R, H, D] bf16; kv: [local_rows, (1,) D] bf16, the rank's own rows;
    indices: [R, W] int32 GLOBAL slots, -1 = masked.
    Returns (out [R, H, D] bf16, lse [R, H] fp32, base 2).
    """
    if q.dim() != 3 or indices.dim() != 2 or indices.dtype != torch.int32:
        raise ValueError(
            "DCP sparse decode needs q [R, H, D] and int32 indices [R, W], got "
            f"q.dim()={q.dim()} indices.dim()={indices.dim()} "
            f"indices.dtype={indices.dtype}"
        )
    rows, heads, dim = q.shape
    if dim != triton.next_power_of_2(dim):
        raise NotImplementedError(
            f"DCP sparse decode needs a power-of-2 head dim, got {dim}"
        )
    kv2 = kv.reshape(kv.shape[0], -1)
    if kv2.shape[1] != dim or kv2.stride(1) != 1 or kv2.dtype != q.dtype:
        raise ValueError(
            f"DCP sparse decode needs kv rows of dim {dim} with unit stride and "
            f"dtype {q.dtype}, got dim={kv2.shape[1]} stride={kv2.stride(1)} "
            f"dtype={kv2.dtype}"
        )
    width = indices.shape[1]

    auto_bk, auto_s = pick_split_config(rows, width, dcp_size, heads, q.device)
    block_k = block_k or auto_bk
    num_splits = num_splits or auto_s

    compact, cnt = dcp_compact_indices(indices, dcp_size, dcp_rank)
    max_block_h = _max_block_h(q.device.index or 0)
    block_h = (
        max_block_h if heads >= max_block_h else max(16, triton.next_power_of_2(heads))
    )
    head_blocks = triton.cdiv(heads, block_h)

    partial_o = torch.empty(
        (rows, num_splits, heads, dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (rows, num_splits, heads), dtype=torch.float32, device=q.device
    )
    _partial_kernel[(rows, num_splits, head_blocks)](
        q,
        kv2,
        compact,
        cnt,
        partial_o,
        partial_lse,
        sm_scale * _LOG2E,
        width,
        q.stride(0),
        q.stride(1),
        kv2.stride(0),
        H=heads,
        D=dim,
        S=num_splits,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=2,
    )
    out = torch.empty((rows, heads, dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    combine_bh = min(block_h, 16)
    _combine_kernel[(rows, triton.cdiv(heads, combine_bh))](
        partial_o,
        partial_lse,
        out,
        lse,
        H=heads,
        D=dim,
        S=num_splits,
        S_P2=max(2, triton.next_power_of_2(num_splits)),
        BLOCK_H=combine_bh,
        num_warps=4,
    )
    return out, lse
