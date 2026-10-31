"""CUDA regression for the DSA index-K prefix gather past 2**31 bytes of pool."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    BLOCK_SIZE_K,
    INDEX_HEAD_DIM,
    gather_index_k_scale_prefix_into,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# One page holds BLOCK_SIZE_K fp8 rows of INDEX_HEAD_DIM bytes, then one fp32 scale per row
BUF_NUMEL_PER_PAGE = BLOCK_SIZE_K * (INDEX_HEAD_DIM + 4)
# First page whose byte offset no longer fits in int32
FIRST_WIDE_PAGE = (2**31 - 1) // BUF_NUMEL_PER_PAGE + 1


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDsaKpoolGatherPrefixInt64(CustomTestCase):
    def _gather(self, pages: list[int], dtype: torch.dtype) -> None:
        device = "cuda"
        num_pages = max(pages) + 1
        buf = torch.zeros(
            num_pages, BUF_NUMEL_PER_PAGE, dtype=torch.uint8, device=device
        )
        k_bytes = BLOCK_SIZE_K * INDEX_HEAD_DIM
        for i, page in enumerate(pages):
            k = torch.randint(0, 256, (k_bytes,), dtype=torch.uint8, device=device)
            scale = torch.arange(
                BLOCK_SIZE_K, dtype=torch.float32, device=device
            ) + 1000.0 * (i + 1)
            buf[page, :k_bytes] = k
            buf[page, k_bytes:] = scale.view(torch.uint8)

        seq_len = len(pages) * BLOCK_SIZE_K
        page_indices = torch.tensor(pages, dtype=dtype, device=device)
        k_out = torch.empty(seq_len, INDEX_HEAD_DIM, dtype=torch.uint8, device=device)
        scale_out = torch.empty(seq_len, dtype=torch.float32, device=device)
        pool = SimpleNamespace(index_kernel_page_size=BLOCK_SIZE_K)
        gather_index_k_scale_prefix_into(
            pool, buf, page_indices, seq_len, k_out, scale_out
        )
        torch.cuda.synchronize()

        rows = buf[pages]
        expected_k = rows[:, :k_bytes].reshape(seq_len, INDEX_HEAD_DIM)
        expected_scale = (
            rows[:, k_bytes:].contiguous().view(torch.float32).reshape(seq_len)
        )
        torch.testing.assert_close(k_out, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(scale_out, expected_scale, rtol=0, atol=0)

    def test_int32_page_indices_past_two_gib(self):
        # The served page table is int32; pages past FIRST_WIDE_PAGE sit beyond 2**31 bytes
        self._gather(
            [3, FIRST_WIDE_PAGE - 1, FIRST_WIDE_PAGE, FIRST_WIDE_PAGE + 40], torch.int32
        )

    def test_int64_page_indices_past_two_gib(self):
        self._gather([FIRST_WIDE_PAGE + 40, 7], torch.int64)

    def test_small_pool_unchanged(self):
        self._gather([0, 5, 2], torch.int32)


if __name__ == "__main__":
    unittest.main()
