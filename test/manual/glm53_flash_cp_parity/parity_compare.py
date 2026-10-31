"""Compare two parity_client.py outputs (manual E2E): prompt logprobs, top-1 agreement and greedy tokens, per length.

    python3 parity_compare.py <baseline json> <candidate json> [label]

Run it twice: CP against the baseline, and a second baseline (the same server re-chunked) against the first,
which is the noise floor the CP deltas are read against.
"""

import json
import statistics
import sys

reference = json.load(open(sys.argv[1]))
candidate = json.load(open(sys.argv[2]))
label = sys.argv[3] if len(sys.argv) > 3 else "candidate"

print(f"{label} against {sys.argv[1]}")
print(f"{'length':>9} {'positions':>9} {'mean |dlp|':>11} {'p99 |dlp|':>10} {'max |dlp|':>10} {'top-1 agree':>12} {'greedy prefix':>14}")
for length in sorted(set(reference) & set(candidate), key=int):
    a, b = reference[length], candidate[length]
    if a["input_ids"] != b["input_ids"]:
        print(f"{length:>9} prompt tokens differ - not comparable")
        continue
    # The first position of a window has no logprob
    pairs = [(x, y) for x, y in zip(a["input_logprobs"], b["input_logprobs"]) if x is not None and y is not None]
    deltas = sorted(abs(x - y) for x, y in pairs)
    top1 = [x == y for x, y in zip(a["input_top1"], b["input_top1"]) if x is not None and y is not None]
    prefix = 0
    for x, y in zip(a["output_ids"], b["output_ids"]):
        if x != y:
            break
        prefix += 1
    p99 = deltas[int(0.99 * (len(deltas) - 1))]
    print(f"{length:>9} {len(deltas):>9} {statistics.mean(deltas):>11.4f} {p99:>10.4f} {deltas[-1]:>10.4f} "
          f"{100 * sum(top1) / len(top1):>11.2f}% {prefix:>7}/{len(a['output_ids'])}")
