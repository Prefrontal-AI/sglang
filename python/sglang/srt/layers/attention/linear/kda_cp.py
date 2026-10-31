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

"""Backend-side helpers for KDA prefill context parallelism.

Companion of the op-layer CP state pre-process
(``sglang.kernels.ops.attention.fla.chunk_delta_h_cp``): this module owns the
serving-side pieces the KDA attention backend needs to run one prefill
forward on a contiguous per-sequence token shard —

* ``KDACPPrefillMetadata`` / ``build_kda_cp_prefill_metadata``: the shard
  layout for one forward (compacted local cu_seqlens, global sequence
  mapping, and every rank's shard length table, all derived purely from the
  global extend lengths so each rank computes identical tables with no
  communication).
* ``exchange_kda_conv_halo``: the causal-conv counterpart of the SSM state
  hand-off. Each rank all-gathers only the last ``conv_width - 1`` tokens of
  its shard per sequence; every rank then assembles locally (a) its own halo
  window — the tokens immediately preceding its shard, spanning as many
  earlier ranks (and the carried pool window) as needed — and (b) the global
  tail window of the whole step, which every rank writes into its pool
  replica. Zero extra communication beyond the one small all-gather,
  mirroring the full-chain merge of the SSM pre-process.

The conv kernel itself is untouched: the halo is handed to
``causal_conv1d_fn`` as a scratch ``conv_states`` tensor (per-shard windows),
so its in-place final-window writeback lands on scratch and the pool is
updated only with the globally-correct tail.
"""

from typing import List, Optional, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.fla.chunk_delta_h_cp import (
    LinearAttnCPContext,
    build_cp_shard_layout,
)


def host_to_device(values: List, dtype: torch.dtype, device) -> torch.Tensor:
    """Small host list -> device tensor without a stream sync.

    ``torch.tensor(list, device="cuda")`` is a blocking pageable copy, i.e. a
    ``cudaStreamSynchronize``; a pinned source with ``non_blocking`` is a plain
    async H2D copy (the caching host allocator keeps the buffer alive until the
    copy has run).
    """
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


class KDACPPrefillMetadata(msgspec.Struct, frozen=True):
    """Shard layout of one CP prefill forward on one rank.

    All fields are pure functions of (global extend lengths, world_size,
    rank), so every rank derives consistent tables without communication.
    """

    world_size: int
    rank: int
    num_global_seqs: int
    # Compacted local layout (zero-length shards dropped; see the op layer's
    # empty-shard contract): local cu_seqlens and the kept sequences' global
    # ids, as device tensors ready for the kernels.
    query_start_loc: object  # int32 [N_local + 1]
    local_seq_ids: object  # int32 [N_local]
    local_seq_ids_list: List[int]
    # Host copy of ``query_start_loc`` (the layout is derived on the host, so
    # reading it back from the device would be a pointless stream sync).
    query_start_loc_list: List[int]
    # shard_lens[r][n] = number of tokens of global sequence n on rank r.
    shard_lens: List[List[int]]

    @property
    def num_local_seqs(self) -> int:
        return len(self.local_seq_ids_list)

    def to_cp_context(self, group) -> LinearAttnCPContext:
        return LinearAttnCPContext(
            world_size=self.world_size,
            rank=self.rank,
            group=group,
            num_global_seqs=self.num_global_seqs,
            local_seq_ids=self.local_seq_ids,
        )


def build_kda_cp_prefill_metadata(
    extend_seq_lens: List[int],
    world_size: int,
    rank: int,
    device: torch.device,
) -> KDACPPrefillMetadata:
    """Derive the per-rank shard layout from the global extend lengths."""
    cu_vals = [0]
    for n in extend_seq_lens:
        cu_vals.append(cu_vals[-1] + int(n))
    local_cu, _ranges, local_seq_ids = build_cp_shard_layout(
        cu_vals, world_size=world_size, rank=rank
    )
    shard_lens = [
        [
            (seq_len * (r + 1)) // world_size - (seq_len * r) // world_size
            for seq_len in extend_seq_lens
        ]
        for r in range(world_size)
    ]
    return KDACPPrefillMetadata(
        world_size=world_size,
        rank=rank,
        num_global_seqs=len(extend_seq_lens),
        query_start_loc=host_to_device(local_cu, torch.int32, device),
        local_seq_ids=host_to_device(local_seq_ids, torch.int32, device),
        local_seq_ids_list=local_seq_ids,
        query_start_loc_list=list(local_cu),
        shard_lens=shard_lens,
    )


def _collect_local_tails(
    conv_input: torch.Tensor,
    metadata: KDACPPrefillMetadata,
    window: int,
) -> torch.Tensor:
    """Last ``min(window, shard_len)`` tokens per kept sequence, right-aligned.

    conv_input is token-major ``[T_local, dim]``. Returns
    ``[N_global, window, dim]`` with zeros for absent sequences and in the
    front padding of short shards.
    """
    dim = conv_input.shape[1]
    tails = conv_input.new_zeros(metadata.num_global_seqs, window, dim)
    cu = metadata.query_start_loc_list
    for local_idx, global_idx in enumerate(metadata.local_seq_ids_list):
        end = cu[local_idx + 1]
        take = min(window, end - cu[local_idx])
        tails[global_idx, window - take :] = conv_input[end - take : end]
    return tails


def _roll_window(
    window_buf: torch.Tensor, valid: int, tail: torch.Tensor, take: int
) -> Tuple[torch.Tensor, int]:
    """Append ``take`` right-aligned tokens of ``tail`` to a rolling window."""
    if take <= 0:
        return window_buf, valid
    window = window_buf.shape[0]
    if take >= window:
        return tail[-window:], window
    return (
        torch.cat([window_buf[take:], tail[-take:]], dim=0),
        min(window, valid + take),
    )


def exchange_kda_conv_halo(
    conv_input: torch.Tensor,
    metadata: KDACPPrefillMetadata,
    prior_conv_windows: torch.Tensor,
    has_prior: Optional[torch.Tensor],
    group,
    has_prior_list: Optional[List[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One all-gather of shard tails -> local halo + global tail windows.

    Args:
        conv_input: token-major ``[T_local, dim]`` conv input (mixed qkv after
            the input projection, before the causal conv).
        prior_conv_windows: ``[N_global, window, dim]`` — the pool conv
            windows carried from the previous chunked-prefill step (content
            is ignored for sequences with ``has_prior == False``).
        has_prior: ``[N_global]`` bool (extend_prefix_lens > 0). Only read
            (one device sync) when ``has_prior_list`` is not given.
        has_prior_list: the same flags as a host list (from the scheduler's
            ``extend_prefix_lens_cpu``); lets the exchange run sync-free.

    Returns:
        halo_windows: ``[N_local, window, dim]`` — scratch conv states for
            this rank's ``causal_conv1d_fn`` call (token-major, right-aligned).
        halo_has_initial: ``[N_local]`` bool for the conv kernel.
        global_tails: ``[N_global, window, dim]`` — the end-of-step conv
            windows every rank writes into its pool replica (identical on all
            ranks by construction).
    """
    n_global, window, dim = prior_conv_windows.shape
    assert n_global == metadata.num_global_seqs, (
        f"prior_conv_windows has {n_global} seqs, metadata has {metadata.num_global_seqs}"
    )

    tails = _collect_local_tails(conv_input, metadata, window)
    ag_tails = tails.new_empty(metadata.world_size, *tails.shape)
    dist.all_gather_into_tensor(ag_tails, tails, group=group)

    if has_prior_list is None:
        has_prior_list = has_prior.tolist()
    halo_windows = conv_input.new_zeros(metadata.num_local_seqs, window, dim)
    halo_valid = [0] * metadata.num_local_seqs
    global_tails = conv_input.new_zeros(n_global, window, dim)
    local_pos = {g: i for i, g in enumerate(metadata.local_seq_ids_list)}

    for n in range(n_global):
        if has_prior_list[n]:
            window_buf, valid = prior_conv_windows[n], window
        else:
            window_buf, valid = conv_input.new_zeros(window, dim), 0
        for r in range(metadata.world_size):
            if r == metadata.rank and n in local_pos:
                halo_windows[local_pos[n]] = window_buf
                halo_valid[local_pos[n]] = valid
            window_buf, valid = _roll_window(
                window_buf,
                valid,
                ag_tails[r, n],
                min(window, metadata.shard_lens[r][n]),
            )
        global_tails[n] = window_buf

    halo_has_initial = host_to_device(
        [v > 0 for v in halo_valid], torch.bool, conv_input.device
    )
    return halo_windows, halo_has_initial, global_tails
