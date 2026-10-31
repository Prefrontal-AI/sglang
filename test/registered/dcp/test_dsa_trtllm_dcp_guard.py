"""The DSA trtllm prefill refuses decode context parallel rather than read wrong rows.

Under DCP the trtllm branch returned before the gather that maps the selection's
global slots onto gathered rows, so it read the rank's own pool with global slots:
wrong rows below the pool size, an illegal address past it (GLM-5.3-Flash CP8 on a
B300 crashed at 1.28-1.31M tokens).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="default")


class TestDsaTrtllmDcpGuard(CustomTestCase):
    def _refuse(self, dcp_size, dsa_impl):
        backend = SimpleNamespace(dcp_size=dcp_size)
        DeepseekSparseAttnBackendKPoolMixin._refuse_trtllm_under_dcp(backend, dsa_impl)

    def test_trtllm_under_dcp_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "tilelang"):
            self._refuse(8, "trtllm")

    def test_other_reads_and_no_dcp_pass(self):
        self._refuse(1, "trtllm")
        for dsa_impl in ("tilelang", "flashmla_sparse", "fa3"):
            self._refuse(8, dsa_impl)


if __name__ == "__main__":
    unittest.main()
