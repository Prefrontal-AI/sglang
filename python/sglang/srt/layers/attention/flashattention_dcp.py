"""GQA/MHA extend under decode context parallelism (--dcp-size > 1) for the FA3 backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mha, get_dcp_lens
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Query rows (tokens x heads) of a DCP prefix pass from which it runs unsplit:
# 4096 rows are 32-64 FA3 M tiles, past which splitting the KV only buys L2
# locality at the price of a partial-output buffer that grows with the prefix.
_DCP_PREFIX_UNSPLIT_MIN_ROWS = 4096


def dcp_prefix_metadata(
    backend: FlashAttentionBackend, forward_batch: ForwardBatch, metadata
) -> None:
    """This rank's shard of every request's prefix, as an FA3 page table.

    Under GQA/MHA DCP the prefix token at absolute position p lives only on
    rank p % dcp_size, at local slot req_to_token[req, p] // dcp_size. The
    rank's shard of a prefix of length L is positions dcp_rank, dcp_rank +
    dcp_size, ... < L, in order; position order does not matter for the
    non-causal prefix pass, only the set does.
    """
    if metadata.dcp_has_prefix is not None:
        return
    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
    metadata.dcp_has_prefix = bool(prefix_lens_cpu is not None and any(prefix_lens_cpu))
    if not metadata.dcp_has_prefix:
        return
    bs = forward_batch.batch_size
    prefix_lens = forward_batch.extend_prefix_lens[:bs]
    local_lens = get_dcp_lens(prefix_lens, backend.dcp_size, backend.dcp_rank)
    max_local = max(
        (int(n) - backend.dcp_rank + backend.dcp_size - 1) // backend.dcp_size
        for n in prefix_lens_cpu[:bs]
    )
    max_local = max(max_local, 1)
    pos = (
        torch.arange(max_local, dtype=torch.int64, device=backend.device)
        * backend.dcp_size
        + backend.dcp_rank
    )
    pos = pos.clamp_max(backend.req_to_token.shape[1] - 1)
    rows = backend.req_to_token[forward_batch.req_pool_indices[:bs]]
    metadata.dcp_prefix_page_table = (
        torch.index_select(rows, 1, pos) // backend.dcp_size
    ).to(torch.int32)
    metadata.dcp_prefix_lens_int32 = local_lens.to(torch.int32)


def set_kv_buffer_dcp(
    backend: FlashAttentionBackend, forward_batch: ForwardBatch, layer, k, v
) -> None:
    # Same owner rule and local slot as TritonAttnBackend._set_kv_buffer, so
    # a triton decode backend reads back exactly what this extend wrote.
    loc = forward_batch.out_cache_loc // backend.dcp_size
    if (
        forward_batch.positions is not None
        and forward_batch.positions.numel() == loc.numel()
    ):
        dcp_kv_mask = forward_batch.positions % backend.dcp_size == backend.dcp_rank
    else:
        dcp_kv_mask = forward_batch.dcp_kv_mask
    backend.token_to_kv_pool.set_kv_buffer(
        layer, loc, k, v, layer.k_scale, layer.v_scale, dcp_kv_mask=dcp_kv_mask
    )


def forward_extend_dcp(
    backend: FlashAttentionBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool,
    *,
    sinks,
    score_mod,
    rel_bias,
    q_rope,
    k_descale,
    v_descale,
) -> torch.Tensor:
    """GQA/MHA extend under decode context parallelism (--dcp-size > 1).

    Mirrors TritonAttnBackend._forward_extend_dcp with FA3 kernels:
      1. write the chunk's K/V for the tokens this rank owns, at loc // dcp;
      2. current chunk: this rank's Q heads against the full chunk K/V
         (every rank computes the whole chunk's K/V), causal, with LSE;
      3. prefix: all-gather Q over the DCP group (all heads), attend
         non-causally to this rank's prefix shard, merge the per-rank
         partials by LSE and keep this rank's heads;
      4. merge prefix and chunk by LSE.
    """
    if (
        sinks is not None
        or score_mod is not None
        or rel_bias is not None
        or q_rope is not None
        or k_descale is not None
        or v_descale is not None
    ):
        raise NotImplementedError(
            "FA3 decode-context-parallel extend supports plain "
            "causal GQA/MHA attention only"
        )
    if (
        layer.is_cross_attention
        or layer.attn_type
        in (AttentionType.ENCODER_ONLY, AttentionType.DECODER_BIDIRECTIONAL)
        or (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
        or layer.logit_cap not in (None, 0, 0.0)
    ):
        raise NotImplementedError(
            "FA3 decode-context-parallel extend supports causal decoder "
            "self-attention without sliding window or softcap only"
        )
    if backend.kv_cache_dtype_str != "auto":
        raise NotImplementedError(
            "FA3 decode-context-parallel extend requires a bf16/fp16 KV cache"
        )
    if (
        is_cp_active(forward_batch)
        or forward_batch.mha_return_lse
        or not forward_batch.forward_mode.is_extend()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        raise NotImplementedError(
            "FA3 decode-context-parallel extend supports plain prefill "
            f"only (forward_mode={forward_batch.forward_mode})"
        )
    assert k is not None and v is not None
    if save_kv_cache:
        set_kv_buffer_dcp(backend, forward_batch, layer, k, v)

    metadata = backend.forward_metadata
    group = get_parallel().dcp_group
    head_dim = layer.head_dim
    q_local = q.contiguous().view(-1, layer.tp_q_head_num, head_dim)
    k = k.view(-1, layer.tp_k_head_num, head_dim)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    # Current chunk: this rank's Q shard only needs the KV head(s) it maps
    # to. The pool (and k, v) keep every head of the DCP group's TP slice.
    if layer.tp_k_head_num > 1:
        kv_start = group.rank_in_group * layer.tp_k_head_num // group.world_size
        kv_end = max(
            (group.rank_in_group + 1) * layer.tp_k_head_num // group.world_size,
            kv_start + 1,
        )
        k_cur, v_cur = k[:, kv_start:kv_end], v[:, kv_start:kv_end]
    else:
        k_cur, v_cur = k, v
    cur_out, cur_lse, *_ = flash_attn_varlen_func(
        q=q_local,
        k=k_cur.contiguous(),
        v=v_cur.contiguous(),
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_q,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=metadata.max_seq_len_q,
        softmax_scale=layer.scaling,
        causal=True,
        return_softmax_lse=True,
        ver=backend.fa_impl_ver,
    )

    dcp_prefix_metadata(backend, forward_batch, metadata)
    if not metadata.dcp_has_prefix:
        return cur_out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Prefix: every rank holds a strided 1/dcp_size of it for all of the
    # group's KV heads. All-gathered Q keeps the global head order, so FA3's
    # GQA mapping (q head h -> kv head h // group) matches the pool layout.
    q_all = group.all_gather(q_local, dim=1).contiguous()
    key_cache, value_cache = backend.get_paged_mha_kv_cache(layer)
    # FA3's SplitKV heuristic splits a non-causal pass once one KV head of the
    # prefix outgrows its L2 budget, and allocates an fp32 partial output of
    # num_splits x tokens x heads x head_dim for it, which grows with the prefix.
    # Keep the pass unsplit once the queries alone give the SMs enough tiles;
    # a short extend still lets FA3 split, where the partials are small.
    num_splits = (
        1
        if q_all.shape[0] * q_all.shape[1] >= _DCP_PREFIX_UNSPLIT_MIN_ROWS
        else backend.num_splits
    )
    pre_out, pre_lse, *_ = flash_attn_with_kvcache(
        q=q_all,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=metadata.dcp_prefix_page_table,
        cache_seqlens=metadata.dcp_prefix_lens_int32,
        cu_seqlens_q=metadata.cu_seqlens_q,
        max_seqlen_q=metadata.max_seq_len_q,
        softmax_scale=layer.scaling,
        causal=False,
        return_softmax_lse=True,
        num_splits=num_splits,
        ver=backend.fa_impl_ver,
    )
    # FA3 varlen LSE is [heads, tokens]; the DCP merge wants [tokens, heads].
    pre_out, pre_lse = cp_lse_ag_out_rs_mha(
        pre_out.float(), pre_lse.transpose(0, 1).float(), group, return_lse=True
    )
    cur_lse = cur_lse.transpose(0, 1).float()
    final_lse = torch.logaddexp(pre_lse, cur_lse)
    pre_scale = torch.nan_to_num(
        torch.exp(pre_lse - final_lse), nan=0.0, posinf=0.0, neginf=0.0
    ).unsqueeze(-1)
    cur_scale = torch.nan_to_num(
        torch.exp(cur_lse - final_lse), nan=0.0, posinf=0.0, neginf=0.0
    ).unsqueeze(-1)
    out = pre_out * pre_scale + cur_out.float() * cur_scale
    return out.to(q.dtype).reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
