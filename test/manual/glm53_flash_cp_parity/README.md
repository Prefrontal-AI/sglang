# GLM-5.3-Flash context-parallel parity

Checks that the one-server context-parallel layout (prefill CP 8 + DCP 8 + EP 8) gives the same prompt logprobs and
greedy tokens as tensor parallel 8, at contexts up to 2M tokens.
CP re-chunks the prefill and changes reduction orders, so bf16 results are not bit-identical; the comparison is against
a noise floor - the same TP 8 server re-chunked - rather than against zero.

## Run

Three servers, one after another or on three nodes, all with these shared flags:

```bash
--tp 8 --ep-size 8 --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
--kv-cache-dtype bfloat16 --moe-runner-backend deep_gemm --disable-radix-cache --context-length 2200000
```

| server | extra flags |
|---|---|
| CP | `--enable-prefill-cp --cp-strategy contiguous --attn-cp-size 8 --dcp-size 8 --dcp-comm-backend ag_rs --chunked-prefill-size 32768` |
| TP 8, the baseline | `--chunked-prefill-size 32768` |
| TP 8, the noise floor | `--chunked-prefill-size 16384` |

Against each, with a text file of at least 2M tokens:

```bash
python3 parity_client.py --base-url http://127.0.0.1:30000 --text-file book.txt \
    --lengths 32768 131072 1048576 2097152 --out cp.json
```

Then:

```bash
python3 parity_compare.py tp8.json cp.json "CP"
python3 parity_compare.py tp8.json tp8_c16k.json "noise floor"
```

The window is the last 32,768 prompt positions, the whole last chunk, so every CP rank's span of it is compared after
every earlier chunk has gone through CP and DCP. Pass: CP's mean and p99 distances within the noise floor's, top-1
agreement equal, greedy tokens equal.

## Result at 57fa9a55ef

H200 node, Gutenberg text, 2026-10-02.

| length | CP: mean / p99 abs dlogprob | CP: top-1 | CP: greedy | noise floor: mean / p99 | noise floor: top-1 | noise floor: greedy |
|---|---|---|---|---|---|---|
| 32k | 0.0028 / 0.0064 | 99.94% | 64/64 | 0.0025 / 0.0056 | 99.94% | 64/64 |
| 128k | 0.0030 / 0.0228 | 99.87% | 64/64 | 0.0031 / 0.0215 | 99.88% | 64/64 |
| 1M | 0.0018 / 0.0074 | 99.94% | 64/64 | 0.0022 / 0.0091 | 99.94% | 64/64 |
| 2M | 0.0401 / 0.8566 | 98.66% | 64/64 | 0.0405 / 0.8526 | 98.69% | 64/64 |

At 2M, past the model's 1,048,576-token context, both distances grow together: the model is sensitive to chunking
there, and CP adds nothing beyond it.
