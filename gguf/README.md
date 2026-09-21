---
license: apache-2.0
language:
  - en
  - it
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - ollama
  - llama-cpp
  - ministral3
  - mistral3
  - 4-bit-quantized
  - apple-silicon
base_model: mistralai/Ministral-3-3B-Instruct-2512-BF16
base_model_relation: quantized
---

# Ministral 3 3B Instruct 2512 — GGUF Q4_K_M

This repository contains a community GGUF conversion and `Q4_K_M`
quantization of [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
for Ollama and llama.cpp-compatible runtimes.

This is a text-generation distribution. It is a separate GGUF quantization
from the MLX 4-bit artifact; it was created from the same official BF16
checkpoint, not by converting MLX weights back to GGUF.

## Files

| File | Description |
| --- | --- |
| `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2.1 GB GGUF Q4_K_M model |
| `Modelfile` | Ollama import recipe |
| `README.md` | This model card |

## Use with Ollama

Download the GGUF and `Modelfile` into the same directory, then run:

```bash
ollama create ministral-3-3b-mlx -f Modelfile
ollama run ministral-3-3b-mlx
```

The published model can also be referenced from the Hub using the Ollama
Hugging Face integration when supported by the installed Ollama version:

```bash
ollama run hf.co/vinci00/ministral-3-3b-gguf-q4
```

## Conversion provenance

The source checkpoint was the official Mistral AI BF16 Safetensors model. The
conversion used the current [`llama.cpp` converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
and `llama-quantize` with `Q4_K_M`. The reproducible MLX conversion notebook,
benchmark, and source card are maintained at
[`vinci00/ministral-3-mlx-4bit`](https://huggingface.co/vinci00/ministral-3-mlx-4bit)
and in the [course repository](https://github.com/VinciGit00/LLM-crash-course/tree/main/20-mlx-quantization).

## Validation

The same four fixed functional prompts used for the MLX artifact passed when
the resulting model was loaded through Ollama on an Apple M4 Mac mini. This is
a small smoke test, not a quality, safety, or task benchmark.

## Accuracy before and after quantization

**BF16 versus GGUF Q4_K_M accuracy has not been measured.** The four smoke
checks above do not estimate task accuracy or establish parity with BF16.

The [source repository](https://github.com/VinciGit00/ministral-3-mlx-4bit)
now includes `benchmark_accuracy.py`: a paired GSM8K test evaluation for the
**MLX** unquantized and 4-bit checkpoints, with exact-match scoring, accuracy
changes in percentage points, uncertainty intervals, and chart output. Its
MLX inference backend does not accept this GGUF artifact or call Ollama.

A valid GGUF comparison still requires evaluating the original unquantized
source and Q4_K_M on identical held-out questions, prompts, decoding settings,
and scoring, preferably within the same runtime. Record both runtime and
precision if comparing across runtimes. Do not copy MLX accuracy numbers into
this card: the quantization formats and inference implementations differ.
The MLX comparison measured 87/100 for both BF16 and MLX 4-bit on a fixed
100-example GSM8K test sample (2026-09-21, M1 Pro). This is evidence for the
MLX artifacts only. **No GSM8K accuracy result is claimed for this GGUF/Ollama
distribution.** See the source repository for raw results and methodology.

## Limitations

- This GGUF card documents text-only Ollama use; image understanding was not
  validated in Ollama.
- Quantization can change accuracy, calibration, tool use, and refusal
  behaviour. Validate representative workloads before production use.
- The model may produce incorrect, biased, unsafe, or culturally insensitive
  outputs. Follow the upstream Mistral model card and add application-level
  safeguards.

## License

The upstream checkpoint is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
Preserve the upstream attribution and model card when redistributing this
derivative.
