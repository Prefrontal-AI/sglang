import types
import unittest

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# GLM-5.3-Flash under CP8: NoPE MLA, 64 heads a rank, latent dim 512
DIM, HEADS, DCP_SIZE = 512, 64, 8
CHUNK_LEN = 1024
SM_SCALE = DIM**-0.5
# Native top-k 2048, LSA list width 4612
WIDTHS = (2048, 4612)
PREFIX_LENS = (0, 8192)


def make_backend(kpool: int) -> types.SimpleNamespace:
    """Stand in for the backend instance, so the real kernel wrappers run."""
    return types.SimpleNamespace(
        device_sm_major=torch.cuda.get_device_capability()[0],
        _sink_pad_cache={},
        num_splits=0,
        dsa_index_kpool=kpool,
    )


def rank_major_rows(prefix_len: int, chunk_len: int) -> torch.Tensor:
    """Row of every [prefix, chunk] position in the gathered buffer, as the DCP all-gather lays it out."""
    local_total = prefix_len // DCP_SIZE
    pos = torch.arange(prefix_len, dtype=torch.int64)
    prefix_rows = (pos % DCP_SIZE) * local_total + pos // DCP_SIZE
    chunk_rows = torch.arange(prefix_len, prefix_len + chunk_len, dtype=torch.int64)
    return torch.cat([prefix_rows, chunk_rows]).to(torch.int32)


def make_lists(prefix_len: int, rows: torch.Tensor, width: int) -> torch.Tensor:
    """Causal top-k lists of positions, valid entries first and -1 padded."""
    lists = torch.full((len(rows), width), -1, dtype=torch.int32)
    for i, row in enumerate(rows.tolist()):
        visible = prefix_len + row + 1
        count = min(width, visible)
        lists[i, :count] = torch.randperm(visible)[:count].to(torch.int32)
    return lists


def tilelang_reference(q, kv, lists):
    padding = (-lists.shape[-1]) % 64
    lists = torch.nn.functional.pad(lists, (0, padding), value=-1)
    out = tilelang_sparse_fwd(
        q=q, kv=kv, indices=lists.unsqueeze(1), sm_scale=SM_SCALE, d_v=DIM
    )
    return out.squeeze(0)


class TestDSADCPPrefillKernels(CustomTestCase):
    def check_kernel(self, kernel: str, kpool: int):
        torch.manual_seed(7)
        device = "cuda"
        backend = make_backend(kpool)
        worst = 0.0
        for prefix_len in PREFIX_LENS:
            total = prefix_len + CHUNK_LEN
            kv_positions = torch.randn(total, 1, DIM, device=device) * 2
            kv_positions = kv_positions.to(torch.bfloat16)
            rows = rank_major_rows(prefix_len, CHUNK_LEN).to(device)
            gathered = torch.empty_like(kv_positions)
            gathered[rows.long()] = kv_positions
            rows_per_rank = CHUNK_LEN // DCP_SIZE
            for width in WIDTHS:
                for rank in range(DCP_SIZE):
                    with self.subTest(
                        kernel=kernel,
                        kpool=kpool,
                        prefix_len=prefix_len,
                        width=width,
                        rank=rank,
                    ):
                        # The rank's contiguous slice of the chunk
                        chunk_rows = torch.arange(
                            rank * rows_per_rank, (rank + 1) * rows_per_rank
                        )
                        positions = make_lists(prefix_len, chunk_rows, width).to(device)
                        q = (
                            torch.randn(rows_per_rank, HEADS, DIM, device=device) * 2
                        ).to(torch.bfloat16)
                        # The remap the DCP block applies
                        remapped = torch.where(
                            positions >= 0, rows[positions.clamp_min(0).long()], -1
                        ).to(torch.int32)

                        expected = tilelang_reference(q, gathered, remapped)
                        # The remap itself: gathered buffer + remapped lists = natural order + original lists
                        natural = tilelang_reference(q, kv_positions, positions)
                        torch.testing.assert_close(expected, natural, atol=0, rtol=0)

                        if kernel == "flashmla_sparse":
                            actual = DeepseekSparseAttnBackend._forward_flashmla_sparse(
                                backend,
                                q_all=q,
                                kv_cache=gathered,
                                v_head_dim=DIM,
                                page_table_1=remapped,
                                sm_scale=SM_SCALE,
                                topk_length=None,
                            )
                        else:
                            counts = (remapped >= 0).sum(-1).to(torch.int32)
                            actual = DeepseekSparseAttnBackend._forward_fa3(
                                backend,
                                q_rope=None,
                                kv_cache=gathered,
                                v_head_dim=DIM,
                                q_nope=q,
                                page_table=remapped,
                                cache_seqlens=counts,
                                cu_seqlens_q=torch.arange(
                                    rows_per_rank + 1,
                                    device=device,
                                    dtype=torch.int32,
                                ),
                                cu_seqlens_k=compute_cu_seqlens(counts),
                                max_seqlen_q=1,
                                sm_scale=SM_SCALE,
                                logit_cap=0.0,
                                page_size=1,
                            )
                        self.assertTrue(torch.isfinite(actual).all())
                        error = (actual.float() - expected.float()).abs().max().item()
                        worst = max(worst, error)
                        print(
                            f"{kernel} kpool={kpool} prefix={prefix_len} width={width} "
                            f"rank={rank} max_abs_err={error:.5f}"
                        )
                        torch.testing.assert_close(
                            actual.float(), expected.float(), atol=0.025, rtol=0.025
                        )
        print(f"{kernel} kpool={kpool} worst max_abs_err={worst:.5f}")

    def test_flashmla_sparse_prefill_matches_tilelang(self):
        self.check_kernel("flashmla_sparse", kpool=1)

    def test_fa3_prefill_matches_tilelang(self):
        self.check_kernel("fa3", kpool=1)

    def test_fa3_prefill_with_kpool_matches_tilelang(self):
        self.check_kernel("fa3", kpool=4)


if __name__ == "__main__":
    unittest.main()
