"""Before/after microbenchmark of DCP sparse decode (CUDA graph replay, per call).

before = dcp_global_to_local_kv_indices + 64-pad + sparse_attention_fwd_kernel_v1(return_lse)
after  = dcp_sparse_decode_fwd (compact + split-KV partial + combine)
"""

import argparse

import torch

from sglang.kernels.ops.attention.dsa.dcp_sparse_decode import dcp_sparse_decode_fwd
from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

H, D = 64, 256


def graph_time(fn, iters=200):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters // 10):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def before(q, kv, idx, dcp, rank, sm):
    owned = (idx >= 0) & (idx % dcp == rank)
    local = torch.where(owned, idx // dcp, -1)
    pad = (-local.shape[1]) % 64
    if pad:
        local = torch.cat((local, local.new_full((local.shape[0], pad), -1)), dim=-1)
    out, lse = tilelang_sparse_fwd(
        q, kv.view(-1, 1, D), local.unsqueeze(1), sm, d_v=D, return_lse=True
    )
    return out, lse.squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", type=int, default=8)
    ap.add_argument("--local-rows", type=int, default=1_300_000)
    args = ap.parse_args()
    dcp, rank, sm = args.dcp, 3, 256**-0.5
    gen = torch.Generator(device="cuda").manual_seed(0)
    kv = (
        torch.randn(args.local_rows, D, device="cuda", generator=gen) * 0.5
    ).bfloat16()
    print(f"{'width':>6} {'bs':>3} {'before_us':>10} {'after_us':>9} {'speedup':>8}")
    for width in (2048, 2564, 4612):
        for bs in (1, 2, 16):
            q = (torch.randn(bs, H, D, device="cuda", generator=gen) * 0.5).bfloat16()
            idx = torch.randint(
                0,
                args.local_rows * dcp,
                (bs, width),
                device="cuda",
                generator=gen,
                dtype=torch.int32,
            )
            tb = graph_time(lambda: before(q, kv, idx, dcp, rank, sm))
            ta = graph_time(lambda: dcp_sparse_decode_fwd(q, kv, idx, sm, dcp, rank))
            print(
                f"{width:>6} {bs:>3} {tb:>10.1f} {ta:>9.1f} {tb / ta:>7.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
