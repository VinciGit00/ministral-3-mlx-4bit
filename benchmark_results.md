# Benchmark results

This report records the reproducible smoke benchmark for the converted
checkpoint. It is a functional regression check, not a general accuracy
benchmark.

## Environment

- Hardware: Apple Mac mini M4, 16 GB unified memory
- OS: macOS 26.4
- Python: 3.13
- `mlx`: 0.32.2
- `mlx-vlm`: 0.7.1
- `mlx-lm`: 0.31.3
- Quantization: 4-bit affine, group size 64
- Sampler: default greedy sampler
- Maximum generation length: 96 tokens

## Command

```bash
python benchmark_mlx.py \
  --model artifacts/Ministral-3-3B-Instruct-2512-4bit \
  --max-tokens 96
```

## Functional checks

| Case | Expected check | Result |
| --- | --- | --- |
| Arithmetic: `17 × 6` | `102` | PASS |
| Science: why ice floats | `less dense` | PASS |
| Italian capital | `roma` | PASS |
| JSON instruction | `status` and `ok` | PASS |
| **Summary** | 4 fixed keyword checks | **4/4 passed** |

Observed outputs included `102`, an explanation that ice is less dense when
frozen, `Roma`, and a JSON object containing `status: ok`.

## Runtime measurements

The separate single-prompt runtime smoke test measured approximately:

| Measurement | Result |
| --- | ---: |
| Converted artifact size | 2.6 GiB |
| Prompt processing | 522.1 tokens/s |
| Generation | 46.9 tokens/s |
| Peak memory reported by MLX | 2.458 GiB |

Token-per-second and memory values vary with hardware, OS, runtime versions,
prompt length, and cache state. The functional checks are intentionally small
and do not establish factuality, safety, or BF16 quality parity.
