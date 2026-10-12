# Apple M3 Max, 40 GPU cores

Registers `Apple M3 Max` / `Apple9` / 40 cores with independent configuration
and tuning-cache identity. Uses shared Metal shaders and the default ICB
runtime. Automatic mixer fusion and model-specific recipes remain disabled.
The 30-core M3 Max's context cap is not inherited.

## Kernel calibration, 2026-10-12

Measured on a Mac15,8 MacBook Pro with 64 GB unified memory, macOS 27.0.1
(26A434), Xcode 27.0 (27A266a), macOS SDK 27.0 and Apple clang 21.0.0.
Python 3.13.5, NumPy 2.5.4 and nanobind 3.1.0 were used. The machine was on
AC power with automatic power mode. Metal reports a 51.84 GiB recommended
working set and a 38.88 GiB maximum buffer.

`tools/profile_writer.py` measured the 17408 x 5120 projection with NVFP4,
FP8-E4M3, affine INT4 and BF16, at T = 1, 2, 4, 8 and matrix tiles of
8, 16, 32 rows. Each timing streams approximately 2 GiB of packs and takes
the minimum of three repetitions. Attention was measured at 1024 and 4096
context with T = 1 and 4, 32 query heads, four KV heads and head dimension 256.
Shader validation was disabled for timing; the GEMV and matrix-tile harnesses
checked outputs against independent CPU oracles. No oracle failures or unavailable candidates
were reported. The writer completed in 43 seconds.

| Profile setting | Selected value |
| --- | --- |
| Lane order | `interleaved16` |
| Scale placement | `block` |
| Threadgroups per core | 2 |
| Attention | `auto` (v3 at the measured points) |
| Apple9 matrix tile | On from T = 4 for NVFP4/FP8; T = 8 for INT4/BF16 |
| Automatic mixer fusion | Off |
| Sibling order / command-buffer budget | `either` / 16 ms; retained defaults |

The full measurements, decision explanations and previous engine block are
stored in [config.json](config.json). The matrix tile uses the shared Apple9
implementation; this is not M5 neural accelerator hardware. The nominal
400 GB/s is [Apple's specification](https://support.apple.com/en-au/117736),
not measured model throughput.

All 17 hardware probes completed. The unmodified
[probe output](../../../../probes/results/Apple-M3-Max_NAc_macOS27.0.1_20261012-165202.txt)
retains `NAc` because the shell wrapper did not read the core-count property.
The compiled probes independently detected 40 cores through IOKit; no
18-core fallback warning occurred. No probe or build failures were reported.
These probes do not automatically select fusion or overlap policies.

## Correctness and serving qualification

The reference checkpoint is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. Its weight-file SHA-256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The oracle environment uses PyTorch 2.14.1 and Transformers 5.17.0.
Transformers 5.19.0 failed two existing contract oracle tests because its
rotary-embedding constructor removed the `device` argument; 5.17.0 matches
the checked-in golden metadata and passes the contract tier. No shared code
or dependency declaration was changed for this compatibility issue.

| Check | Local result |
| --- | --- |
| CMake/Ninja native build | Passed |
| Wheel build, backend/config inclusion, device selection and bundled shader lookup | Passed |
| Contract tier | 1150 passed, 1 optional MLX comparison skipped |
| Kernel/runtime tier, Metal shader validation on | 1193 passed, 461 skipped |
| CPU layer/model oracles on real reference weights | 5 passed |
| Real-model GPU goldens, long prefill, sampling, fast math and speculative rollback, shader validation on | 6 passed, 1 optional MLX conversion skipped |
| Actual serving CLI, shader validation on, 4096-token capacity | Warmup, repeated greedy requests, 395 reused prefix tokens, SSE text/usage equivalence and Chat Completions/Responses/Messages passed |
| Repository hygiene, Python compilation and source/doc whitespace checks | Passed; raw probe output retained verbatim |

GPU skips cover Apple10-only task fusion and packed operands, unsupported tile
geometries, resident-prefill cases requiring speculative recipes, and optional
MLX comparisons. The default ICB path passed the tested workloads. CLI requests
used a short reference question and a repeated system prefix; the configured
4096-token capacity was not filled by a maximum-length request. Shader-validation
timings are correctness evidence, not performance measurements.

The [qualification summary](../../../../tools/bench/results/m3-max-40c-20261012/summary.json)
records versions, test counts and CLI checks. Detailed local logs and the CLI
smoke script are retained under `build/m3-max-40c-validation/` (ignored by Git).

## Large target and automatic DSpark qualification

The pinned `nvidia/Qwen3.8-27B-NVFP4` target and catalog-selected
`LithosAI/Qwen3.8-27B-DSpark-NVFP4` head passed actual CLI serving at the default
32768-token capacity on this 64 GB machine, with Metal shader validation enabled.
No draft override, fusion recipe or encoding-policy change was used. All six
weight shards matched their published SHA-256 hashes.

Fact, code, 2082-token prefix and 32752-token prompt cases returned identical
raw greedy tokens with and without DSpark. Repeated requests matched their
first request's tokens, text and usage. The 2K-prefix SSE request matched its
non-streaming result. Chat Completions, Responses and Messages passed in both
phases. Real draft acceptance and rejection/rollback were exercised.

The long prompt plus its 16-token requested output budget reaches 32K; its
answer stopped at EOS before exhausting that budget. The first long request
had no prefix-cache hit; its repeat reused 32729 tokens in both
phases with identical output. Both owned servers stopped after the checks.

The [large-model report](../../../../tools/bench/results/m3-max-40c-20261012/qwen27b-qualification.md)
records immutable revisions, source hashes, per-case comparisons and reproduction
commands. The target-only control uses the same engine; it is not an independent
full-model numerical oracle. Sampling and disk-prefix persistence remain unqualified
for this pair. Recorded timings include shader validation.

## Post-merge requalification

Rerun at upstream base `08d9d53067eab538e3ad8435a6a5b50b80088554` with scoped,
uncommitted device-selection test fixes: contracts passed (1165 passed, one
optional MLX skip); affected backend/GPU tests passed (103 passed, 171 Apple10-only
skips). No missing-profile skips remain in those GPU tests. The pinned 27B target,
automatic DSpark head, API adapters, SSE and 32K prefix reuse passed again, with
raw-token equality against both the fresh target-only control and the earlier run.
The [large-model report](../../../../tools/bench/results/m3-max-40c-20261012/qwen27b-qualification.md)
records this rerun. The full kernel/runtime and reference-model results above
remain the earlier qualification at base `9f6c23d`.

## Reproduction of reference and kernel checks

After the development setup in `CONTRIBUTING.md`:

```bash
uv pip install --python .venv/bin/python nanobind 'transformers==5.17.0'
cmake -S . -B build -G Ninja -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --parallel 4
GPU_CORES=40 ./probes/run_all.sh
OMP_NUM_THREADS=8 .venv/bin/python tools/profile_writer.py
.venv/bin/python -m pytest tests/contract
OMP_NUM_THREADS=8 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  .venv/bin/python -m pytest tests/kernels tests/runtime -ra
OMP_NUM_THREADS=8 MONOLITH_MODELS=/path/to/models \
  MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  .venv/bin/python -m pytest tests/layers/test_layers_vs_hf.py \
  tests/models/qwen3_5 -ra
.venv/bin/python -m monolith.serve --model /path/to/models/Qwen3.5-0.8B \
  --no-draft --local-files-only --pack /tmp/lithos-m3-max-serve-pack \
  --max-context 4096 --keep-warm 0 --port 18090
```

## Limits

`validation: measured` qualifies the leaf-kernel profile. The separate
large-model checks above qualify the pinned pair and tested greedy prompts at
32K on 64 GB. Other checkpoints, larger capacities, fused layers and end-to-end
speedups are not qualified. The working-set comparison in
`hosts_target_model` is a weight-capacity estimate, not a large-model test.
Per-model recipe and kernel-choice tables are not shipped by this backend.
