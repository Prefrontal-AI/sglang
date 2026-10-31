"""KDA decode with heads split over the prefill-CP group.

Prefill CP runs attention TP 1, so every rank of the group holds every KDA
head's weights and state. A decode token has no sequence to split, and running
all heads on every rank reads the full projection weights each step. Instead
each rank updates only its own contiguous block of heads, from slices of the
weights and state it already holds, and the partial output projections are
all-reduced across the group.

A rank updates only its own heads' state, so the others go stale on it; that is
exact while a request only decodes, and is why this needs the radix cache off.
"""

from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.cp.base import is_cp_enabled
from sglang.srt.runtime_context import get_memory, get_parallel


class KDAHeadShard(msgspec.Struct, frozen=True):
    # First head this rank decodes; its count is the shard layer's num_v_heads
    head_start: int
    # Channels of each of q, k and v over all heads: the conv weight and state
    # stack [q | k | v] along this axis
    segment_dim: int


def is_kda_decode_head_split() -> bool:
    parallel = get_parallel()
    return (
        is_cp_enabled()
        and parallel.attn_tp_size == 1
        and parallel.attn_cp_size > 1
        and parallel.attn_cp_size == parallel.tp_size
        and get_memory().disable_radix_cache
    )


def head_shard_conv_update(
    *,
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
    shard: KDAHeadShard,
    local_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """mixed_qkv [tokens, 3 * local_dim] -> the conv outputs; conv_states [slots, dim, width - 1]."""
    first_channel = shard.head_start * head_dim
    outputs = []
    for segment in range(3):
        channels = slice(
            segment * shard.segment_dim + first_channel,
            segment * shard.segment_dim + first_channel + local_dim,
        )
        outputs.append(
            causal_conv1d_update(
                mixed_qkv[:, segment * local_dim : (segment + 1) * local_dim],
                conv_states[:, channels],
                conv_weights[channels],
                None if conv_bias is None else conv_bias[channels],
                activation="silu",
                conv_state_indices=cache_indices,
            )
        )
    return torch.cat(outputs, dim=-1)
