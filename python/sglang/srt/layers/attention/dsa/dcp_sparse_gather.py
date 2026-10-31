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

"""Sparse prefix-KV gather for DSA prefill under decode context parallel (DCP).

Under DCP the prefix KV of a prefill chunk is sharded over the group (slot ``s`` lives
on rank ``s % size``). The dense path all-gathers every prefix row, per layer and per
chunk, although a sparse layer reads only the rows its selected lists name. This module
gathers just those rows:

1. every rank marks the prefix rows its queries select in a byte mask,
2. one all-reduce of the mask gives every rank the same union,
3. every rank reads its owned rows of the union and one all-gather of equal-sized
   pieces assembles them, owner-major,
4. the lists are remapped onto the compact buffer, ``[gathered rows | chunk rows]``.

The attention kernel then runs unchanged on the same lists in the same order, so its
output is bit-identical to the dense path's.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _mark_selected_kernel(
    page_table_ptr,  # [n] int32 flat ragged positions, -1 padding
    kv_rows_ptr,  # [n_positions] int32 rank-major row of every position
    mask_ptr,  # [n_prefix_rows] uint8
    n_elems,
    n_prefix_rows,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_elems
    pos = tl.load(page_table_ptr + offs, mask=in_range, other=-1)
    valid = in_range & (pos >= 0)
    row = tl.load(kv_rows_ptr + pos, mask=valid, other=0)
    selected = valid & (row < n_prefix_rows)
    tl.store(mask_ptr + row, tl.full((BLOCK,), 1, tl.uint8), mask=selected)


@triton.jit
def _remap_kernel(
    page_table_ptr,  # [n] int32
    kv_rows_ptr,  # [n_positions] int32
    table_ptr,  # [n_prefix_rows] int32 compact row of every selected prefix row
    out_ptr,  # [n] int32
    n_elems,
    n_prefix_rows,
    chunk_base,  # compact row of the chunk's first row
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n_elems
    pos = tl.load(page_table_ptr + offs, mask=in_range, other=-1)
    valid = in_range & (pos >= 0)
    row = tl.load(kv_rows_ptr + pos, mask=valid, other=0)
    in_prefix = valid & (row < n_prefix_rows)
    compact = tl.load(table_ptr + row, mask=in_prefix, other=0)
    new = tl.where(in_prefix, compact, row - n_prefix_rows + chunk_base)
    new = tl.where(valid, new, -1)
    tl.store(out_ptr + offs, new, mask=in_range)


def mark_selected_rows(
    page_table_1: torch.Tensor, kv_rows: torch.Tensor, n_prefix_rows: int
) -> torch.Tensor:
    """Byte mask over the prefix's rank-major rows, 1 where a list names the row."""
    mask = torch.zeros(n_prefix_rows, dtype=torch.uint8, device=page_table_1.device)
    page_table_1 = page_table_1.contiguous()
    n = page_table_1.numel()
    block = 1024
    _mark_selected_kernel[(triton.cdiv(n, block),)](
        page_table_1, kv_rows, mask, n, n_prefix_rows, BLOCK=block
    )
    return mask


def remap_lists(
    page_table_1: torch.Tensor,
    kv_rows: torch.Tensor,
    table: torch.Tensor,
    n_prefix_rows: int,
    chunk_base: int,
) -> torch.Tensor:
    """Rewrite lists of ragged positions as rows of the compact buffer."""
    page_table_1 = page_table_1.contiguous()
    out = torch.empty_like(page_table_1)
    n = page_table_1.numel()
    block = 1024
    _remap_kernel[(triton.cdiv(n, block),)](
        page_table_1, kv_rows, table, out, n, n_prefix_rows, chunk_base, BLOCK=block
    )
    return out


def _union(group, mask: torch.Tensor) -> torch.Tensor:
    # Straight to NCCL: the group's custom all-reduce has no uint8 kernel and JIT-builds one that fails
    dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=group.device_group)
    return mask


def gather_selected_prefix(
    page_table_1: torch.Tensor,
    kv_rows: torch.Tensor,
    n_prefix_rows: int,
    chunk_kv: torch.Tensor,
    kv_dtype: torch.dtype,
    local_prefix_ids: torch.Tensor,
    read_rows,
    group,
    dcp_size: int,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact buffer [owner-major selected prefix rows, padded to the largest owner | chunk rows],
    and the lists rewritten as rows of it. kv_rows numbers positions owner-major
    (owner * rows_per_owner + local row), chunk rows after the prefix."""
    device = page_table_1.device
    rows_per_owner = n_prefix_rows // dcp_size
    union = _union(group, mark_selected_rows(page_table_1, kv_rows, n_prefix_rows))
    # Inclusive running count per owner: its last column is how many rows the owner holds
    running = torch.cumsum(
        (union > 0).view(dcp_size, rows_per_owner), dim=1, dtype=torch.int32
    )
    # The one host read of the layer: the equal share every owner contributes
    per_owner = int(running[:, -1].max())
    chunk_rows = chunk_kv.shape[0]
    if per_owner == 0:
        return chunk_kv.to(kv_dtype), remap_lists(
            page_table_1, kv_rows, running.view(-1), n_prefix_rows, 0
        )
    # The k-th selected row of this owner is where the running count first exceeds k
    nth = torch.arange(per_owner, device=device, dtype=torch.int32)
    mine = torch.searchsorted(running[dcp_rank], nth, right=True).clamp_(
        max=rows_per_owner - 1
    )
    local = read_rows(local_prefix_ids[mine]).to(kv_dtype)
    gathered_rows = dcp_size * per_owner
    buffer = torch.empty(
        (gathered_rows + chunk_rows, *chunk_kv.shape[1:]),
        dtype=kv_dtype,
        device=device,
    )
    gathered = buffer[:gathered_rows]
    # pynccl has no fp8 dtype; the all-gather is a byte copy
    if buffer.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        group.all_gather_into_tensor(
            gathered.view(torch.uint8), local.view(torch.uint8)
        )
    else:
        group.all_gather_into_tensor(gathered, local)
    buffer[gathered_rows:] = chunk_kv
    # Compact row of every selected prefix row: its rank among its owner's, past earlier owners' shares
    running += (
        torch.arange(dcp_size, device=device, dtype=torch.int32) * per_owner - 1
    )[:, None]
    return buffer, remap_lists(
        page_table_1, kv_rows, running.view(-1), n_prefix_rows, gathered_rows
    )
