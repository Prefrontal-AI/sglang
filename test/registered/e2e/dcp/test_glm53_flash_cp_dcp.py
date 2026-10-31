"""GLM-5.3-Flash with prefill context parallel and decode context parallel in one server.

The layout the fork serves 10M-token prompts with: attention CP 8 on contiguous spans, DCP 8, EP 8, attention TP 1,
radix cache off (which also turns on the KDA and MLA decode head splits). Small chunks make a modest prompt cross many
chunks, so needles land on chunk and span boundaries, and concurrent requests make the CP batches ragged.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=2400, stage="extra-b", runner_config="8-gpu-h200")

CHUNK = 8192
FILLER = "This document contains ordinary background information.\n"


class TestGLM53FlashPrefillCPAndDCP(GSM8KMixin, CustomTestCase):
    gsm8k_num_examples = 200
    gsm8k_score_threshold = 0.90

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.model = try_cached_model("zai-org/GLM-5.3-Flash")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--tp-size",
                "8",
                "--enable-prefill-cp",
                "--cp-strategy",
                "contiguous",
                "--attn-cp-size",
                "8",
                "--dcp-size",
                "8",
                "--dcp-comm-backend",
                "ag_rs",
                "--ep-size",
                "8",
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--kv-cache-dtype",
                "bfloat16",
                "--moe-runner-backend",
                "deep_gemm",
                "--disable-radix-cache",
                "--context-length",
                "131072",
                "--chunked-prefill-size",
                str(CHUNK),
                "--max-running-requests",
                "8",
                "--cuda-graph-max-bs-decode",
                "8",
            ],
            env={"SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _needle_ids(self, code: str, lines: int, at: float) -> list[int]:
        """Return a chat prompt of `lines` filler lines with the passcode line placed at fraction `at` of them."""
        before = int(lines * at)
        text = FILLER * before + f"The secret passcode is {code}.\n" + FILLER * (lines - before)
        text += "\nWhat is the secret passcode? Reply with only the six digits."
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )

    def _ask(self, ids: list[int]) -> str:
        response = requests.post(
            self.base_url + "/generate",
            json={"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 2048}},
            timeout=1200,
        )
        response.raise_for_status()
        return response.json()["text"]

    def test_needles_across_chunks_and_spans(self):
        # About 7 tokens a line => about 63k tokens, 8 chunks of 8 spans each
        lines = 9000
        for k, at in enumerate([0.0, 0.0625, 0.125, 0.5, 0.875, 1.0]):
            code = f"{k + 1}48213"
            ids = self._needle_ids(code, lines, at)
            self.assertGreater(len(ids), 6 * CHUNK)
            with self.subTest(position=at):
                self.assertIn(code, self._ask(ids))

    def test_concurrent_requests_of_different_lengths(self):
        # Ragged batches: each request's chunk splits into its own spans
        lengths = [900, 3100, 5200, 7000]
        cases = [
            (f"{k + 1}90317", lines, 0.3 + 0.1 * k) for k, lines in enumerate(lengths)
        ]
        with ThreadPoolExecutor(max_workers=len(cases)) as executor:
            texts = list(executor.map(lambda case: self._ask(self._needle_ids(*case)), cases))
        for (code, _, _), text in zip(cases, texts):
            self.assertIn(code, text)


if __name__ == "__main__":
    unittest.main()
