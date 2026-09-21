---
license: apache-2.0
language:
  - en
  - fr
  - es
  - de
  - it
  - pt
  - nl
  - zh
  - ja
  - ko
  - ar
library_name: mlx-vlm
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - mlx-lm
  - ministral3
  - mistral3
  - multimodal
  - image-text-to-text
  - 4-bit-quantized
  - apple-silicon
base_model: mistralai/Ministral-3-3B-Instruct-2512-BF16
base_model_relation: quantized
---

# Ministral 3 3B Instruct 2512 — MLX 4-bit

This repository contains a community MLX conversion and 4-bit quantization of
[`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16).
It is intended for local inference on Apple Silicon with
[`mlx-lm`](https://github.com/ml-explore/mlx-lm) for text and
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) for images.

Official Mistral AI references: [Ministral 3 3B documentation](https://docs.mistral.ai/models/ministral-3-3b-25-12)
and the [Introducing Mistral 3 announcement](https://mistral.ai/news/mistral-3/).

The reproducible source repository is
[`VinciGit00/ministral-3-mlx-4bit`](https://github.com/VinciGit00/ministral-3-mlx-4bit).

This is a format conversion and quantization, not a fine-tune, merge, or new
training run. The original model weights, tokenizer, vision encoder, chat
template, and model behaviour are inherited from Mistral AI; this repository
changes the storage and inference representation.

## Model details

| Property | Value |
| --- | --- |
| Base checkpoint | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| Variant | Instruct, multimodal (text + image) |
| Language model size | 3.4B parameters |
| Vision encoder | 0.4B parameters |
| Quantization | MLX affine quantization, 4-bit, group size 64 |
| Effective precision reported by converter | 5.756 bits/weight |
| Base-model context | 256K tokens; practical use depends on memory and KV cache |
| License | Apache 2.0 |
| Tested hardware | Original conversion: Mac mini M4, 16 GB; paired accuracy/runtime: M1 Pro, 16 GB |

## Intended use

Use this checkpoint for local experimentation, prototyping, image captioning,
multilingual assistance, extraction, and other interactive workloads where an
Apple Silicon MLX runtime is useful. Consult the [upstream model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
for the full capability, limitation, and intended-use guidance.

This derivative is not a safety audit or a production-readiness claim. Do not
assume that 4-bit outputs have the same quality as BF16 without testing your
own workload.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U mlx-lm mlx-vlm
```

Python 3.10+ and an Apple Silicon Mac are recommended. The repository's
[`requirements.txt`](requirements.txt) records the minimum package families
used for the conversion and smoke test.

## Quickstart: text

From this repository's root, run against the uploaded checkpoint directory:

```bash
mlx_lm.generate \
  --model ./Ministral-3-3B-Instruct-2512-4bit \
  --prompt "Explain quantization in three sentences." \
  --max-tokens 120 \
  --temp 0
```

Python usage:

```python
from mlx_lm import generate, load

model, tokenizer = load("./Ministral-3-3B-Instruct-2512-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain quantization briefly."}],
    add_generation_prompt=True,
)
from mlx_lm.sample_utils import make_sampler

print(generate(model, tokenizer, prompt=prompt, max_tokens=120, sampler=make_sampler(temp=0)))
```

## Quickstart: image understanding

For an image prompt, use `mlx-vlm` and provide a local image:

```bash
mlx_vlm.generate \
  --model ./Ministral-3-3B-Instruct-2512-4bit \
  --image ./example.jpg \
  --prompt "Describe this image briefly." \
  --max-tokens 120 \
  --temperature 0
```

Text-only generation is documented with `mlx-lm`; the current `mlx-vlm`
text-only dispatch path can construct an image processor even when no image is
supplied. The optional `special_tokens_map.json` is omitted from this
conversion because current Transformers releases can parse it incompatibly for
the Mistral tokenizer.

## Quantization and provenance

The checkpoint was created with:

```bash
mlx_vlm.convert \
  --hf-path mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --mlx-path ./Ministral-3-3B-Instruct-2512-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

The source is the official Mistral AI BF16 Safetensors checkpoint. The Ollama
distribution/GGUF is not used as an input. The reproducible source and exact
environment specification are maintained in the course repository:

- [conversion notebook](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/01_ministral_3_mlx_quantization.ipynb)
- [repeatable benchmark](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/benchmark_mlx.py)
- [requirements](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/requirements.txt)
- [tested version lock](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/requirements-lock.txt)

No training data was added and no parameter fine-tuning was performed.

## Historical single-prompt runtime benchmark

This is a **local runtime smoke benchmark**, not a quality benchmark. It was
run after conversion on a Mac mini M4 with 16 GB unified memory using
`mlx==0.32.2`, `mlx-vlm==0.7.1`, `mlx-lm==0.31.3`, Python 3.13, and greedy
decoding (`temperature=0`). The fixed English prompt used 538 prompt tokens
and generated 116 tokens.

| Measurement | Result |
| --- | ---: |
| Converted artifact size | 2.6 GiB |
| Prompt processing | 522.1 tokens/s |
| Generation | 46.9 tokens/s |
| Peak memory reported by MLX | 2.458 GiB |
| Quality evaluation at conversion time | Smoke only; measured paired GSM8K results below |

These values depend on hardware, software versions, prompt length, and cache
state. They must not be compared with published BF16 scores as accuracy
results. The upstream model card contains Mistral AI's evaluation tables; this
derivative has not independently reproduced them.

### Functional smoke benchmark

The same checkpoint was then loaded with `mlx-lm` and tested greedily on four
short, fixed prompts. A case passes when its expected answer keyword(s) appear
in the generated text. This is a deliberately small regression check for
loading, arithmetic, multilingual output, and constrained JSON—not a
scientific accuracy estimate.

| Case | Expected check | Result |
| --- | --- | --- |
| Arithmetic: `17 × 6` | contains `102` | PASS |
| Science: why ice floats | contains `less dense` | PASS |
| Italian capital | contains `Roma` | PASS |
| JSON instruction | contains `status` and `ok` | PASS |
| **Overall** | 4 fixed checks | **4/4 passed** |

The run generated the correct arithmetic answer (`102`), a scientifically
appropriate explanation that frozen water is less dense, the Italian answer
`Roma`, and the requested JSON fields. Re-run the check after changing the
quantization settings or runtime version; keyword checks should not be treated
as a substitute for a held-out benchmark.

The benchmark is repeatable from the checkpoint directory:

```bash
python benchmark_mlx.py --model ./Ministral-3-3B-Instruct-2512-4bit --max-tokens 96
```

For the single-model smoke run, it exits with status `0` only when all four
fixed keyword checks pass. It uses an explicit greedy sampler, fixed prompts,
and a fixed maximum generation length. Existing local checkpoints require no
network access for this smoke test.

For the same software versions used for the published numbers, install
`requirements-lock.txt` from the linked source repository before running the
benchmark. Reproducibility means keeping the model revision, conversion
arguments, benchmark code, and dependency versions together; it does not mean
that token-per-second values will be identical on different Apple Silicon
chips.

### BF16 baseline comparison

Measured on **2026-09-21, Apple M1 Pro, 16 GB unified memory**, macOS 26.6.2,
Python 3.13.7, using the versions in `requirements-lock.txt`. Both models use
an explicit greedy sampler and the same `fix_mistral_regex=True` correction.
Runtime uses the same four fixed smoke prompts with `max_tokens=96`; the
separate accuracy panel uses 100 GSM8K test problems with `max_tokens=512`.

| Variant | Mean inference latency | Mean output speed | Functional checks | GSM8K accuracy |
| --- | ---: | ---: | ---: | ---: |
| MLX BF16 baseline | 2.60 s | 8.51 tok/s | 4/4 | 87/100 (87%) |
| MLX 4-bit (this model) | 1.83 s | 11.19 tok/s | 4/4 | 87/100 (87%) |

Mean inference latency decreased by 29.5% and measured output throughput
increased by 1.31× in this four-prompt run. Latency covers the generation
call (prefill and decoding), excluding model loading and prompt formatting.
Generated lengths differ; these are workload-dependent observations, not peak
hardware throughput estimates. Runtime and accuracy use different workloads.

![BF16 versus MLX 4-bit: runtime and measured GSM8K accuracy](benchmark_artifacts/benchmark_comparison.png)

- Current runtime: [`benchmark_results.json`](benchmark_artifacts/benchmark_results.json)
- Accuracy table and uncertainty: [`accuracy_results.md`](benchmark_artifacts/accuracy/accuracy_results.md)
- Raw answers, scores, revisions, weight checksums, and environment: [`accuracy_results.json`](benchmark_artifacts/accuracy/accuracy_results.json)
- Historical M4 comparison: [archived runtime results](benchmark_artifacts/historical_m4/benchmark_results.json)

Re-run both workloads:

```bash
python benchmark_mlx.py \
  --baseline-model ./artifacts/Ministral-3-3B-Instruct-2512-bf16-mlx \
  --model ./artifacts/Ministral-3-3B-Instruct-2512-4bit \
  --max-tokens 96 --accuracy-samples 100 --accuracy-max-tokens 512 --seed 42 \
  --output-dir benchmark_artifacts
```

This writes runtime results, evaluates accuracy, and includes the measured
accuracy as the third panel of the same `benchmark_comparison.png` chart.
`--smoke-only` skips accuracy and labels that panel as not measured for that
run. A failed smoke check produces a non-zero exit code; accuracy is reported
without an arbitrary pass threshold.

### Accuracy: source precision versus quantized

[`benchmark_accuracy.py`](benchmark_accuracy.py) evaluates the unquantized MLX
source and its 4-bit derivative on the **same GSM8K test examples**. The source
precision for this conversion is **BF16**, not FP32. This zero-shot math test
is separate from the four keyword smoke checks above.

**Execution status (2026-09-21): measured.** Both checkpoints completed all
100 test examples on the same M1 Pro. BF16 scored **87/100 (87%)** and MLX
4-bit scored **87/100 (87%)**: **0.0 percentage points** difference.

The 95% Wilson interval is **79.02%–92.24%** for each score. The paired
bootstrap 95% interval for the difference is **−7.0 to +7.0 percentage points**.
There were **7 regressions and 7 improvements**, so the equal totals do not
mean identical answers. Invalid final answers counted as incorrect: **7 BF16,
4 quantized**. This small sample does not establish general quality parity.

The baseline source revision was `b6d637bef2393152b3da2b2fde72eecdee30557e`
of `mistralai/Ministral-3-3B-Instruct-2512-BF16`; the published 4-bit revision
was `a90c19bd1c02e796e897d26f7d9685d1904a3584` of
`vinci00/ministral-3-mlx-4bit`. Tokenizer files, chat templates, and architecture
were checked for equality; the baseline Safetensors contain BF16 weights.

The downloaded and converted weights are removed locally after evaluation at
the owner's request. Results and provenance remain checked in; download the
recorded revisions again before rerunning the commands below.

Protocol:

- Dataset: [OpenAI GSM8K](https://github.com/openai/grade-school-math), `test`
  split, pinned revision `3101c7d5072418e28b9008a6636bde82a006892c`.
  The downloaded JSONL is verified against a fixed SHA-256 digest.
- Default sample: 100 of 1,319 test problems selected without replacement with
  Python `random.Random(42)`, then sorted by original row index. No training,
  calibration, few-shot examples, or prompt tuning uses this test split.
- Both models receive the same question and instruction, their saved chat
  template, the same `fix_mistral_regex=True` tokenizer correction, greedy decoding (`temperature=0`), and 512 maximum output tokens.
  Use checkpoints from the same source revision and identical tokenizer/chat
  templates; conversion metadata alone cannot establish identical provenance.
- The prompt requests reasoning followed by `#### <number>`. Scoring compares
  the complete final numeric field, or an entirely numeric response, after
  decimal/sign/thousands-separator normalization. Numbers merely mentioned in
  reasoning do not pass. Missing, malformed, or incorrect final answers score zero, including those cut off before a valid answer.
- Outputs include correct/total, exact-match accuracy, 95% Wilson intervals,
  **4-bit minus baseline accuracy in percentage points**, a paired percentile
  bootstrap interval (2,000 resamples), and IDs of regressions/improvements.
  A zero or tied result is retained, not used to redesign the test.

Prepare an unquantized MLX baseline from the **same local upstream snapshot**
used for the quantized model, with sufficient disk space and unified memory:

```bash
mlx_vlm.convert \
  --hf-path /path/to/original-BF16-snapshot \
  --mlx-path ./artifacts/Ministral-3-3B-Instruct-2512-bf16-mlx \
  --dtype bfloat16
```

Do not use dequantized 4-bit weights as the baseline. Apply the notebook's
optional `special_tokens_map.json` compatibility workaround to both artifacts
if needed. The evaluator loads and releases models sequentially.

Run accuracy alone from this module (or the standalone repository root):

```bash
pip install -r requirements-lock.txt
python benchmark_accuracy.py \
  --baseline-model ./artifacts/Ministral-3-3B-Instruct-2512-bf16-mlx \
  --model ./artifacts/Ministral-3-3B-Instruct-2512-4bit \
  --samples 100 --max-tokens 512 --seed 42 \
  --output-dir benchmark_artifacts/accuracy
```

The first run downloads the pinned test data. Subsequent runs reuse its verified
cache. Use `--samples 1319` for the entire test split; BF16 inference may be slow
on a 16 GB machine. In the combined `benchmark_mlx.py` command, the equivalent
options are `--accuracy-samples`, `--accuracy-max-tokens`, and `--seed`.

The run writes `accuracy_results.json` (settings, dataset/sample identity,
checkpoint configs, versions, raw answers and per-example scores),
`accuracy_results.md` (table and paired comparison), and
`accuracy_comparison.png` (accuracy with uncertainty bars) under the selected
output directory. These files are produced only by an actual completed run.

This small math sample is not a general accuracy, multilingual, or vision
assessment. Pretraining contamination is unknown; small samples have wide
uncertainty. Even a zero-width paired bootstrap interval when all paired
outcomes tie does not establish quality parity. This protocol differs from
published leaderboard protocols. **MLX scores must not be attributed to the
separate Ollama/GGUF Q4_K_M model.**

CPU-only validation (no model weights required):

```bash
python -m unittest discover -s . -p 'test_benchmark_accuracy.py'
```

## Limitations and safety

- Quantization can change accuracy, calibration, tool use, and multimodal
  behaviour. Validate on representative prompts before relying on the model.
- The measured run tested short text generation only. It did not establish
  factuality, robustness, bias, refusal quality, long-context quality, or image
  understanding quality.
- A 256K context limit is an architectural capability, not a promise that a
  16 GB Mac can use 256K tokens efficiently. KV-cache memory can dominate.
- The model may produce incorrect, biased, unsafe, or culturally insensitive
  outputs. Apply the upstream safety guidance and add application-level
  filtering, access control, and human review where appropriate.
- This is not an official Mistral AI release. Report conversion/runtime issues
  here and model-behaviour issues to the upstream project.

## License and attribution

The upstream Ministral 3 checkpoint is released under the
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). Preserve
the upstream attribution and model card when redistributing this derivative.
The conversion runtimes are maintained by the
[MLX community](https://github.com/ml-explore/mlx-lm) and
[Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

## Evidence gaps

This card reports the measured conversion and historical runtime smoke test.
The paired GSM8K evaluator produced the measured 100-example scores above.
Broader text, vision, safety, and Ollama/GGUF quality comparisons remain
unmeasured; the tied score does not establish general BF16 parity.
