# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Backend-side DCP helpers for DSA: index remaps, KV gathers and the split decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.dsa.dcp_sparse_gather import gather_selected_prefix
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def dcp_global_to_local_kv_indices(page_table: torch.Tensor) -> torch.Tensor:
    # TODO(kpham-sgl): Fuse the index conversion and masking into one GPU kernel.
    parallel = get_parallel()
    owned = (page_table >= 0) & (
        page_table % parallel.attn_dcp_size == parallel.attn_dcp_rank
    )
    return torch.where(owned, page_table // parallel.attn_dcp_size, -1)


def dcp_remap_extend_kv(
    token_to_kv_pool: KVCache,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathered KV and the lists remapped onto it, for sparse attention under DCP."""
    if forward_batch.forward_mode.is_target_verify():
        return kv_cache, dcp_global_to_local_kv_indices(page_table_1)
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return kv_cache, page_table_1
    assert k is not None
    # NOTE(kpham-sgl): Map RAGGED offsets into gathered KV without reordering KV.
    kv_indices = dcp_rank_major_kv_rows(forward_batch)
    if get_parallel().dcp_prefill_sparse_gather:
        return dcp_gather_selected_kv(
            token_to_kv_pool,
            layer,
            forward_batch,
            k,
            page_table_1,
            kv_indices,
            kv_cache.dtype,
        )
    kv_cache = dcp_gather_extend_kv(token_to_kv_pool, layer, forward_batch, k)
    page_table_1 = torch.where(
        page_table_1 >= 0,
        kv_indices[page_table_1.clamp_min(0)],
        -1,
    )
    return kv_cache, page_table_1


def dcp_gather_extend_kv(
    token_to_kv_pool: KVCache,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k: torch.Tensor,
) -> torch.Tensor:
    # The prefix lands rank-major, [rank][request][row], straight from the
    # all-gather; dcp_rank_major_kv_rows maps positions onto that order.
    # This avoids a token-order reshuffle that needs several prefix-sized
    # temporaries per layer.
    metadata = forward_batch.attn_dcp_metadata
    kv_buffer = metadata.dcp_kv_buffer
    prefix_len = metadata.dcp_extend_prefix_lens_sum
    if prefix_len > 0:
        cache_k_nope, cache_k_rope = token_to_kv_pool.get_mla_kv_buffer(
            layer, metadata.dcp_local_prefix_kv_indices
        )
        local_rows = (
            cache_k_nope
            if cache_k_rope is None
            else torch.cat([cache_k_nope, cache_k_rope], dim=-1)
        ).to(kv_buffer.dtype)
        gathered = kv_buffer[:prefix_len]
        # pynccl has no fp8 dtype; the all-gather is a byte copy
        if kv_buffer.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            gathered, local_rows = (
                gathered.view(torch.uint8),
                local_rows.view(torch.uint8),
            )
        get_parallel().dcp_group.all_gather_into_tensor(gathered, local_rows)
    kv_buffer[prefix_len:] = k.view(k.shape[0], 1, -1)
    return kv_buffer


def dcp_gather_selected_kv(
    token_to_kv_pool: KVCache,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k: torch.Tensor,
    page_table_1: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix rows the lists name plus the chunk's rows, and the lists rewritten onto them."""
    metadata = forward_batch.attn_dcp_metadata
    prefix_len = metadata.dcp_extend_prefix_lens_sum
    chunk_kv = k.view(k.shape[0], 1, -1)
    if prefix_len == 0:
        return chunk_kv.to(kv_dtype), torch.where(
            page_table_1 >= 0, kv_indices[page_table_1.clamp_min(0)], -1
        )

    def read_rows(ids: torch.Tensor) -> torch.Tensor:
        nope, rope = token_to_kv_pool.get_mla_kv_buffer(layer, ids)
        return nope if rope is None else torch.cat([nope, rope], dim=-1)

    parallel = get_parallel()
    return gather_selected_prefix(
        page_table_1,
        kv_indices,
        prefix_len,
        chunk_kv,
        kv_dtype,
        metadata.dcp_local_prefix_kv_indices,
        read_rows,
        parallel.dcp_group,
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
    )


def dcp_rank_major_kv_rows(forward_batch: ForwardBatch) -> torch.Tensor:
    """Row in dcp_gather_extend_kv's buffer of every [prefix, extend] position."""
    metadata = forward_batch.attn_dcp_metadata
    if metadata.dcp_rank_major_kv_rows is not None:
        return metadata.dcp_rank_major_kv_rows
    dcp_size = get_parallel().attn_dcp_size
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu)
    extend_lens = list(forward_batch.extend_seq_lens_cpu)
    device = metadata.dcp_local_prefix_kv_indices.device
    local_total = sum(prefix_lens) // dcp_size
    local_start, extend_start = 0, sum(prefix_lens)
    parts = []
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        # The planner's strided owner split needs dcp_size-aligned prefixes
        assert prefix_len % dcp_size == 0, (prefix_len, dcp_size)
        pos = torch.arange(prefix_len, device=device, dtype=torch.int32)
        parts.append((pos % dcp_size) * local_total + local_start + pos // dcp_size)
        parts.append(
            torch.arange(
                extend_start,
                extend_start + extend_len,
                device=device,
                dtype=torch.int32,
            )
        )
        local_start += prefix_len // dcp_size
        extend_start += extend_len
    metadata.dcp_rank_major_kv_rows = torch.cat(parts)
    return metadata.dcp_rank_major_kv_rows


def dcp_split_decode(
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    global_page_table: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.dsa.dcp_sparse_decode import (
        dcp_sparse_decode_fwd,
    )

    parallel = get_parallel()
    return dcp_sparse_decode_fwd(
        q=q_all,
        kv=kv_cache,
        indices=global_page_table,
        sm_scale=sm_scale,
        dcp_size=parallel.attn_dcp_size,
        dcp_rank=parallel.attn_dcp_rank,
    )
