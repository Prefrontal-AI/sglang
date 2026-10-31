"""Send long prompts to one server, greedy, and keep the prompt logprobs of the last window (manual E2E).

Run it once against the context-parallel server and once against a non-CP baseline, then compare the two outputs
with parity_compare.py. For each length the prompt is the first <length> tokens of the text; the server returns,
for the last --window prompt positions, the logprob of the actual next token and its top-1 token, then
--new-tokens greedy tokens.

With --window equal to the chunked-prefill size, the window is the whole last chunk, so every CP rank's span of it
is compared, after every earlier chunk has gone through the CP and DCP paths.

Usage:
    python3 parity_client.py --base-url http://127.0.0.1:30000 --text-file book.txt \
        --lengths 32768 131072 1048576 --out cp.json
"""

import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer


def read_text(path: str) -> str:
    """Return the prompt text: a plain text file, or the first turn of a ShareGPT-style json list."""
    with open(path) as handle:
        if not path.endswith(".json"):
            return handle.read()
        return json.load(handle)[0]["conversations"][0]["value"]


def generate(base_url: str, ids: list[int], window: int, new_tokens: int) -> dict:
    """Return the server's response to one greedy request with prompt logprobs over the last window."""
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": new_tokens},
        "return_logprob": True,
        "logprob_start_len": max(0, len(ids) - window),
        "top_logprobs_num": 1,
    }
    request = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--text-file", required=True, help="Text long enough for the largest length")
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--window", type=int, default=32768)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_ids = tokenizer(read_text(args.text_file), add_special_tokens=False)["input_ids"]
    results = {}
    for length in args.lengths:
        if len(all_ids) < length:
            raise SystemExit(f"the text has {len(all_ids)} tokens, fewer than {length}")
        start = time.time()
        meta = generate(args.base_url, all_ids[:length], args.window, args.new_tokens)["meta_info"]
        results[str(length)] = {
            "seconds": round(time.time() - start, 1),
            "input_logprobs": [entry[0] for entry in meta["input_token_logprobs"]],
            "input_ids": [entry[1] for entry in meta["input_token_logprobs"]],
            "input_top1": [entries[0][1] if entries else None for entries in meta["input_top_logprobs"]],
            "output_ids": [entry[1] for entry in meta["output_token_logprobs"]],
            "output_logprobs": [entry[0] for entry in meta["output_token_logprobs"]],
        }
        print(f"{length}: {results[str(length)]['seconds']} s", flush=True)
        with open(args.out, "w") as handle:
            json.dump(results, handle)


if __name__ == "__main__":
    main()
