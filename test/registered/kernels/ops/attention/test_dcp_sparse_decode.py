"""DCP split-KV sparse decode vs a torch reference and the v1 TileLang kernel.

Tolerance: output max-abs error <= 2e-2 against the fp32 torch reference (bf16 P and
bf16 output rounding, same as v1, whose own error is checked alongside); LSE (base 2)
max-abs error <= 1e-3; empty rows must be exactly 0 / -inf.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

H, D = 64, 256
DCP = 8


def reference(q, kv_local, idx, sm_scale, dcp, rank):
    """fp32 torch: softmax over the owned entries of each row; (out, lse base 2)."""
    R = q.shape[0]
    out = torch.zeros(R, q.shape[1], q.shape[2], dtype=torch.float32, device=q.device)
    lse = torch.full((R, q.shape[1]), float("-inf"), device=q.device)
    for r in range(R):
        row = idx[r]
        own = (row >= 0) & (row % dcp == rank)
        loc = (row[own] // dcp).long()
        if loc.numel() == 0:
            continue
        k = kv_local[loc].float()
        s = torch.einsum("hd,kd->hk", q[r].float(), k) * sm_scale * 1.4426950408889634
        m = s.max(dim=1, keepdim=True).values
        p = torch.exp2(s - m)
        l = p.sum(dim=1, keepdim=True)
        out[r] = (p / l) @ k
        lse[r] = (torch.log2(l) + m).squeeze(1)
    return out, lse


def make_case(rows, width, local_rows, gen, kind):
    dev = "cuda"
    q = (torch.randn(rows, H, D, device=dev, generator=gen) * 0.5).bfloat16()
    kv = (torch.randn(local_rows, D, device=dev, generator=gen) * 0.5).bfloat16()
    hi = local_rows * DCP
    idx = torch.randint(
        0, hi, (rows, width), device=dev, generator=gen, dtype=torch.int32
    )
    # dedupe is not required by the kernels; random collisions are fine
    if kind == "mask":  # sprinkled -1
        idx = torch.where(
            torch.rand(rows, width, device=dev, generator=gen) < 0.3, -1, idx
        )
    elif kind == "empty":
        idx[:] = -1
    elif kind == "single":
        idx[:] = -1
        idx[:, width // 2] = (
            torch.randint(0, local_rows, (rows,), device=dev, generator=gen).int() * DCP
        )
    elif kind == "one_rank":  # skew: every entry owned by rank 0
        idx = (idx // DCP) * DCP
    elif kind == "mixed":  # some empty rows, some full
        idx[::2] = -1
    return q, kv, idx.contiguous()


class TestDcpSparseDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _check(self, rows, width, rank, kind, seed=0):
        from sglang.kernels.ops.attention.dsa.dcp_sparse_decode import (
            dcp_sparse_decode_fwd,
        )

        gen = torch.Generator(device="cuda").manual_seed(seed)
        q, kv, idx = make_case(rows, width, 5000, gen, kind)
        sm = 256**-0.5
        ref_o, ref_l = reference(q, kv, idx, sm, DCP, rank)
        out, lse = dcp_sparse_decode_fwd(q, kv, idx, sm, DCP, rank)
        empty = torch.isinf(ref_l)
        self.assertTrue(torch.isinf(lse[empty]).all() and (lse[empty] < 0).all())
        self.assertEqual(out[empty].abs().max().item() if empty.any() else 0.0, 0.0)
        self.assertEqual(torch.isnan(out).any().item(), False)
        self.assertLess((out.float() - ref_o).abs().max().item(), 2e-2)
        if (~empty).any():
            self.assertLess((lse[~empty] - ref_l[~empty]).abs().max().item(), 1e-3)

    def test_widths_and_batches(self):
        for width in (2048, 2564, 4612, 4616):
            for rows in (1, 2, 16):
                for kind in ("random", "mask"):
                    with self.subTest(width=width, rows=rows, kind=kind):
                        self._check(rows, width, rank=3, kind=kind)

    def test_glm_latent_width(self):
        # GLM-5.3-Flash reads a 512-wide latent, the shape whose 64-head tile faulted on Blackwell
        global D
        width_before = D
        D = 512
        try:
            for rows in (1, 2):
                with self.subTest(rows=rows):
                    self._check(rows, 4612, rank=3, kind="random")
        finally:
            D = width_before

    def test_edge_rows(self):
        for kind in ("empty", "single", "one_rank", "mixed"):
            for rank in (0, 5):
                with self.subTest(kind=kind, rank=rank):
                    self._check(4, 4612, rank=rank, kind=kind)

    def test_forced_splits_agree(self):
        from sglang.kernels.ops.attention.dsa.dcp_sparse_decode import (
            dcp_sparse_decode_fwd,
        )

        gen = torch.Generator(device="cuda").manual_seed(1)
        q, kv, idx = make_case(2, 4612, 5000, gen, "mask")
        base = dcp_sparse_decode_fwd(
            q, kv, idx, 0.0625, DCP, 2, block_k=32, num_splits=1
        )
        for bk, s in ((32, 3), (32, 16), (64, 5)):
            o, l = dcp_sparse_decode_fwd(
                q, kv, idx, 0.0625, DCP, 2, block_k=bk, num_splits=s
            )
            self.assertLess((o.float() - base[0].float()).abs().max().item(), 1e-2)
            self.assertLess((l - base[1]).abs().max().item(), 1e-4)

    def test_matches_v1_tilelang(self):
        from sglang.kernels.ops.attention.dsa.dcp_sparse_decode import (
            dcp_sparse_decode_fwd,
        )
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        gen = torch.Generator(device="cuda").manual_seed(2)
        for rank in (0, 7):
            q, kv, idx = make_case(2, 4612, 5000, gen, "mask")
            local = torch.where((idx >= 0) & (idx % DCP == rank), idx // DCP, -1).int()
            pad = (-local.shape[1]) % 64
            local = torch.nn.functional.pad(local, (0, pad), value=-1)
            o1, l1 = tilelang_sparse_fwd(
                q, kv.view(-1, 1, D), local.unsqueeze(1), 0.0625, d_v=D, return_lse=True
            )
            o2, l2 = dcp_sparse_decode_fwd(q, kv, idx, 0.0625, DCP, rank)
            d_o = (o1.float() - o2.float()).abs().max().item()
            d_l = (l1.squeeze(0) - l2).abs()
            rel_o = d_o / o1.float().abs().max().item()
            fin = torch.isfinite(l1.squeeze(0))
            rel_l = (d_l[fin] / l1.squeeze(0)[fin].abs().clamp_min(1.0)).max().item()
            print(
                f"[vs v1, rank {rank}] out max abs {d_o:.3e} rel {rel_o:.3e}; "
                f"lse max abs {d_l[fin].max().item():.3e} rel {rel_l:.3e}"
            )
            self.assertLess(d_o, 1e-2)
            self.assertLess(rel_o, 1e-2)
            self.assertLess(d_l[fin].max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
