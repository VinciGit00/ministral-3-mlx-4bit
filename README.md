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
| Tested hardware | Apple Silicon Mac mini M4, 16 GB unified memory |

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

Run from the directory containing the converted checkpoint:

```bash
mlx_lm.generate \
  --model . \
  --prompt "Explain quantization in three sentences." \
  --max-tokens 120 \
  --temp 0
```

Python usage:

```python
from mlx_lm import generate, load

model, tokenizer = load(".")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain quantization briefly."}],
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=120, temp=0.0))
```

## Quickstart: image understanding

For an image prompt, use `mlx-vlm` and provide a local image:

```bash
mlx_vlm.generate \
  --model . \
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
- [recorded benchmark results](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/benchmark_results.md)
- [requirements](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/requirements.txt)
- [tested version lock](https://github.com/VinciGit00/ministral-3-mlx-4bit/blob/main/requirements-lock.txt)

No training data was added and no parameter fine-tuning was performed.

## Runtime benchmark

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
| Quality evaluation | Not performed for this conversion |

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
python benchmark_mlx.py --model . --max-tokens 96
```

It exits with status `0` only when all four fixed keyword checks pass. It uses
the runtime's default greedy sampler, fixed prompts, fixed maximum generation
length, and no network access.

For the same software versions used for the published numbers, install
`requirements-lock.txt` from the linked source repository before running the
benchmark. Reproducibility means keeping the model revision, conversion
arguments, benchmark code, and dependency versions together; it does not mean
that token-per-second values will be identical on different Apple Silicon
chips.

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

## Citation

Please cite Mistral AI's upstream model and paper when using this checkpoint:

```bibtex
@misc{ministral3,
  title  = {Ministral 3},
  author = {Mistral AI},
  year   = {2025},
  url    = {https://huggingface.co/collections/mistralai/ministral-3}
}
```

For the MLX runtime, also cite the relevant [MLX-LM](https://github.com/ml-explore/mlx-lm)
and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) projects.

## Evidence gaps

This card reports the measured conversion and local runtime smoke test. It does
not claim a new accuracy benchmark, safety audit, or parity with BF16. A future
release should add task-specific held-out evaluations, including text and
image tasks, before making quality comparisons.
