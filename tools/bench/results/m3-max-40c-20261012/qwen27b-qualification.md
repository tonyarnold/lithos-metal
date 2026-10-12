# Qwen3.8-27B-NVFP4 qualification on the 40-core M3 Max

Validated on 2026-10-12 on the 64 GB M3 Max / macOS 27.0.1 used for the
[backend qualification](../../../../monolith/backends/metal/m3_max_40c/README.md).
Rerun after merging upstream at `08d9d53067eab538e3ad8435a6a5b50b80088554`, with the scoped
device-selection test fixes uncommitted. Earlier raw qualification evidence is
preserved separately in `build/m3-max-40c-validation/qwen27b/`.
The [compact evidence](qwen27b-summary.json) records hardware, versions, immutable
checkpoint hashes, pack formats and per-case checks.

## Checkpoints and runtime

- Target: `nvidia/Qwen3.8-27B-NVFP4` at `482ca0f3832238542f8f5295dde86b5f22711d80`.
- Automatic head: `LithosAI/Qwen3.8-27B-DSpark-NVFP4` at `b169bc4c8dc805a9e6178fa82f6a7b84b169a457`.
- All six downloaded weight shards matched their published SHA-256 hashes.
- Shared Apple9 shaders, default ICB encoding, measured profile, no fusion recipes.
- DSpark selected through the existing catalog, without a `--draft` override:
  seven proposals plus one target anchor. The head's native NVFP4 packing was retained.
- DSpark served at the default 32768-token capacity. The target-only control
  used 32774 to align its allocation and reuse the identical target pack:
  the speculative setup reserves six lookahead positions beyond the user capacity.

The normal CLI entry point was executed with a temporary observer that recorded
returned generations. It did not alter generation settings, kernels, tokens or
sampling. Metal shader validation was enabled in both phases. Raw server logs,
token/acceptance traces, full request results and the observer/runner scripts are
retained in `build/m3-max-40c-validation/` (ignored by Git).

## Results

Both CLI phases passed warmup and all HTTP checks. All four prompts returned
the exact same raw greedy token sequence with DSpark and target-only decoding.
Repeated requests matched their first request's raw tokens, visible text and usage.
The streamed 2K-prefix request matched the non-streaming tokens, text and usage,
including its terminal SSE event. Chat Completions, Responses and Messages
returned successful responses; the factual answers contained Paris.

| Case | Prompt tokens | Output budget | Returned tokens | DSpark cached tokens, first / repeat | Target-only raw token comparison |
| --- | --- | --- | --- | --- | --- |
| `fact` | 21 | 64 | 8 | 0 / 0 | Equal |
| `code` | 35 | 96 | 29 | 0 / 0 | Equal |
| `prefix` | 2082 | 64 | 8 | 0 / 2059 | Equal |
| `boundary_32k` | 32752 | 16 | 8 | 0 / 32729 | Equal |

The 32K request's 32752-token prompt plus 16-token output budget reaches the
advertised capacity. Its first request had no prefix-cache hit; its repeat
reused 32729 tokens. Both returned identical
tokens and text. This exercises long-context prefill, target/draft state restore,
and the 32K request-admission/allocation path under shader validation. The factual
answer stops at EOS before consuming the entire requested output budget.

Verification logs recorded accepted drafts and rejected rounds. The code prompt's
first request accepted 5, 7, 7 and 7 proposals across four verification rounds,
exercising both acceptance and rollback with the actual trained head.
Both owned servers stopped after validation.

## Reproduction

After the development setup and with the two pinned checkpoints in the local
download cache, the DSpark phase uses the normal CLI with no draft override:

```bash
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  .venv/bin/python -m monolith.serve \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --draft-revision b169bc4c8dc805a9e6178fa82f6a7b84b169a457 \
  --download-dir /private/tmp/lithos-m3-max-qwen27b-hub --local-files-only \
  --pack /private/tmp/lithos-m3-max-qwen27b-packs \
  --max-context 32768 --keep-warm 0 --port 18091
```

For the control, omit `--draft-revision`, add `--no-draft`, and use
`--max-context 32774`. The preserved runner constructs the prompts, checks
token and response equivalence, and stops its own servers.

## Limits

These checks qualify greedy inference and in-memory prefix reuse for the pinned
pair and the tested prompts, including a repetitive synthetic long context.
The target-only comparison uses the same engine; no independent full-model
numerical oracle or broad language-quality benchmark was run. Sampling,
disk-prefix persistence, other checkpoints and larger capacities remain unqualified.
Recorded elapsed times include shader validation and are not production
performance claims.
