"""
Equivalence test for the DSA prefill sparse prefix gather under DCP
(``--dcp-prefill-sparse-gather``).

Every rank owns the prefix rows ``pos % N == rank`` and selects its own lists. The dense
path all-gathers the whole prefix and maps the lists through the rank-major row table;
the sparse path gathers only the union of the rows the lists name. Both feed the same
lists, in the same order, to the same TileLang kernel, so the outputs are compared with
``atol=0``: any difference is a bug in the gather or the remap, not rounding.

The file doubles as the torchrun worker, like test_reduce_scatter_along_dim.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.layers.attention.dsa.dcp_sparse_gather import gather_selected_prefix
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

HEADS = 64
KV_WIDTH = 512
LIST_WIDTH = 4672  # 4612 padded to the kernel's 64-entry blocks


@pytest.mark.parametrize("nproc", [2, 8])
def test_dcp_sparse_prefill_gather(nproc: int) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", __file__],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    return local_rank, world_size, device, coord


def make_lists(rank, queries, prefixes, chunk_starts, chunk_lens, width, device):
    """Lists like LSA's: sink, runs of 8 prefix rows, a window of the request's own chunk, -1 padding."""
    generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
    rows = []
    for request, (prefix, start, length) in enumerate(
        zip(prefixes, chunk_starts, chunk_lens)
    ):
        base = start  # flat position of this request's row 0
        for q in range(queries[request]):
            entries = list(range(4)) if prefix >= 4 else []
            runs = torch.randint(
                0, max(prefix // 8, 1), (min(prefix // 8, 400),), generator=generator
            )
            for r in runs.tolist():
                entries += [r * 8 + i for i in range(8) if r * 8 + i < prefix]
            position = int(torch.randint(0, length, (1,), generator=generator))
            lo = max(0, position - 64)
            entries += list(range(prefix + lo, prefix + position + 1))
            entries = sorted(set(entries))[:width]
            entries = [base + e for e in entries]
            rows.append(entries + [-1] * (width - len(entries)))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def reference_kv_rows(prefixes, chunk_lens, n, device):
    """Row of every [prefix, chunk] position of every request in the rank-major buffer."""
    local_total = sum(prefixes) // n
    local_start, chunk_start, parts = 0, sum(prefixes), []
    for prefix, length in zip(prefixes, chunk_lens):
        pos = torch.arange(prefix, device=device, dtype=torch.int32)
        parts.append((pos % n) * local_total + local_start + pos // n)
        parts.append(
            torch.arange(
                chunk_start, chunk_start + length, device=device, dtype=torch.int32
            )
        )
        local_start += prefix // n
        chunk_start += length
    return torch.cat(parts)


def run_case(
    name,
    rank,
    n,
    device,
    coord,
    prefixes,
    chunk_lens,
    queries_per_rank,
    width=LIST_WIDTH,
):
    from sglang.srt.layers.attention.dsa.dcp_backend_ops import dcp_rank_major_kv_rows

    torch.manual_seed(7)
    local_total = sum(prefixes) // n
    chunk_total = sum(chunk_lens)
    # Global pool: what rank o holds at row i is prefix position (pos % n == o, pos // n == i)
    pool_rows = torch.randn(local_total * n, KV_WIDTH, device=device).bfloat16()
    chunk_kv = torch.randn(chunk_total, KV_WIDTH, device=device).bfloat16()
    # Each rank's pool is a shuffled slab, read through ids - as the translated pool read ids are
    perm = torch.randperm(local_total, device=device, generator=None)
    slab = torch.empty(local_total, KV_WIDTH, device=device, dtype=torch.bfloat16)
    slab[perm] = pool_rows[rank * local_total : (rank + 1) * local_total]
    local_ids = perm.to(torch.int32)

    def read_rows(ids):
        return slab[ids.long()].unsqueeze(1)

    flat_starts, offset = [], 0
    for prefix, length in zip(prefixes, chunk_lens):
        flat_starts.append(offset)
        offset += prefix + length
    lists = make_lists(
        rank,
        [queries_per_rank] * len(prefixes),
        prefixes,
        flat_starts,
        chunk_lens,
        width,
        device,
    )

    fb = SimpleNamespace(
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens_cpu=chunk_lens,
        attn_dcp_metadata=SimpleNamespace(
            dcp_rank_major_kv_rows=None,
            dcp_local_prefix_kv_indices=local_ids,
            dcp_extend_prefix_lens_sum=sum(prefixes),
        ),
    )
    with mock.patch(
        "sglang.srt.layers.attention.dsa.dcp_backend_ops.get_parallel",
        return_value=SimpleNamespace(attn_dcp_size=n),
    ):
        kv_rows = dcp_rank_major_kv_rows(fb)
    expected_rows = reference_kv_rows(prefixes, chunk_lens, n, device)
    assert torch.equal(kv_rows, expected_rows), f"{name}: rank-major rows differ"

    # Dense path: all-gather every prefix row owner-major, chunk rows after
    dense = torch.empty(
        sum(prefixes) + chunk_total, 1, KV_WIDTH, device=device, dtype=torch.bfloat16
    )
    coord.all_gather_into_tensor(dense[: sum(prefixes)], read_rows(local_ids))
    dense[sum(prefixes) :] = chunk_kv.unsqueeze(1)
    dense_lists = torch.where(lists >= 0, kv_rows[lists.clamp_min(0)], -1)

    sparse, sparse_lists = gather_selected_prefix(
        lists,
        kv_rows,
        sum(prefixes),
        chunk_kv.unsqueeze(1),
        torch.bfloat16,
        local_ids,
        read_rows,
        coord,
        n,
        rank,
    )

    valid = lists >= 0
    assert torch.equal(sparse_lists >= 0, valid), f"{name}: validity differs"
    assert torch.equal(
        sparse[sparse_lists[valid].long()], dense[dense_lists[valid].long()]
    ), f"{name}: gathered rows differ"
    assert sparse.shape[0] <= dense.shape[0], f"{name}: buffer grew"

    kernel_checked = False
    try:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    except ImportError:
        tilelang_sparse_fwd = None
    if tilelang_sparse_fwd is not None:
        q = torch.randn(lists.shape[0], HEADS, KV_WIDTH, device=device).bfloat16()
        out_dense = tilelang_sparse_fwd(
            q=q,
            kv=dense,
            indices=dense_lists.unsqueeze(1),
            sm_scale=0.0625,
            d_v=KV_WIDTH,
        )
        out_sparse = tilelang_sparse_fwd(
            q=q,
            kv=sparse,
            indices=sparse_lists.unsqueeze(1),
            sm_scale=0.0625,
            d_v=KV_WIDTH,
        )
        assert torch.equal(out_dense, out_sparse), (
            f"{name}: outputs differ, max abs "
            f"{(out_dense.float() - out_sparse.float()).abs().max().item()}"
        )
        kernel_checked = True
    if rank == 0:
        print(
            f"{name}: ok, prefix rows {sum(prefixes)}, gathered {sparse.shape[0] - chunk_total}, "
            f"kernel {'bitwise equal' if kernel_checked else 'skipped'}",
            flush=True,
        )


def worker_main() -> None:
    rank, n, device, coord = init_distributed()
    with torch.inference_mode():
        run_case("one request", rank, n, device, coord, [n * 2048], [n * 64], 8)
        run_case(
            "two requests",
            rank,
            n,
            device,
            coord,
            [n * 1024, n * 512],
            [n * 64, n * 32],
            6,
        )
        run_case("small prefix", rank, n, device, coord, [n * 8], [n * 16], 4)
        run_case(
            "narrow lists", rank, n, device, coord, [n * 256], [n * 16], 4, width=64
        )
    dist.barrier()
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
