"""Paired zero-shot GSM8K accuracy evaluation for unquantized and 4-bit MLX."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import re
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_URL = (
    f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}"
    "/grade_school_math/data/test.jsonl"
)
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
INSTRUCTION = (
    "Solve the problem. You may explain your reasoning, but end with a "
    "separate line containing exactly #### followed by the numeric answer.\n\n"
)
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def numeric_answer(text: str) -> str | None:
    """Score only a final answer, never a number embedded in reasoning."""
    final = text.strip().rsplit("####", 1)[-1].strip()
    if not NUMBER.fullmatch(final):
        return None
    return str(Decimal(final.replace(",", "")).normalize())


def load_examples(cache: Path, samples: int, seed: int) -> list[dict]:
    if not cache.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded GSM8K checksum mismatch")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    data = cache.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Cached GSM8K checksum mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    if not 1 <= samples <= len(rows):
        raise ValueError(f"samples must be between 1 and {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), samples))
    return [
        {
            "id": f"gsm8k/test/{i}",
            "question": rows[i]["question"],
            "expected": numeric_answer(rows[i]["answer"]),
        }
        for i in indices
    ]


def wilson_interval(correct: int, total: int) -> list[float]:
    p, z = correct / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_summary(baseline: list[dict], quantized: list[dict], seed: int) -> dict:
    if not baseline or [r["id"] for r in baseline] != [r["id"] for r in quantized]:
        raise ValueError("Paired results must contain the same nonempty ordered IDs")
    differences = [
        int(q["correct"]) - int(b["correct"]) for b, q in zip(baseline, quantized)
    ]
    rng = random.Random(seed)
    boot = sorted(
        100 * sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(2000)
    )
    return {
        "delta_accuracy_pp": 100 * sum(differences) / len(differences),
        "paired_bootstrap_95_ci_pp": [boot[49], boot[1949]],
        "bootstrap_resamples": 2000,
        "regression_ids": [b["id"] for b, d in zip(baseline, differences) if d == -1],
        "improvement_ids": [b["id"] for b, d in zip(baseline, differences) if d == 1],
    }


def evaluate(model_path: str, examples: list[dict], max_tokens: int, seed: int) -> dict:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    model, tokenizer = load(model_path, tokenizer_config={"fix_mistral_regex": True})
    rows = []
    try:
        for index, example in enumerate(examples, 1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION + example["question"]}],
                add_generation_prompt=True,
            )
            answer = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0),
                verbose=False,
            )
            prediction = numeric_answer(answer)
            rows.append(
                {
                    **example,
                    "answer": answer,
                    "prediction": prediction,
                    "correct": prediction is not None
                    and prediction == example["expected"],
                    "format_valid": prediction is not None,
                }
            )
            if index == 1 or index % 10 == 0 or index == len(examples):
                print(
                    f"  {index}/{len(examples)} | correct: "
                    f"{sum(row['correct'] for row in rows)}",
                    flush=True,
                )
    finally:
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    correct = sum(row["correct"] for row in rows)
    return {
        "model": model_path,
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson_95_ci": wilson_interval(correct, len(rows)),
        "invalid_answers": sum(not row["format_valid"] for row in rows),
        "examples": rows,
    }


def checkpoint_metadata(path: str, quantized: bool) -> dict:
    """Require local configs so dtype and quantization labels are evidence-backed."""
    root = Path(path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    quantization = (
        config.get("quantization")
        or text_config.get("quantization")
        or config.get("quantization_config")
        or text_config.get("quantization_config")
    )
    if quantized != bool(quantization):
        raise ValueError(
            f"{path}: expected {'quantized' if quantized else 'unquantized'} weights"
        )
    if quantized and quantization.get("bits") != 4:
        raise ValueError(f"{path}: this comparison expects 4-bit quantization")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"{path}: no local Safetensors weights found")
    dtypes = set()
    for weight in weights:
        with weight.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size <= 100_000_000:
                raise ValueError(f"{weight}: invalid Safetensors header size")
            header = json.loads(stream.read(size))
        dtypes.update(
            tensor["dtype"] for name, tensor in header.items() if name != "__metadata__"
        )
    return {
        "path": str(root.resolve()),
        "config": config,
        "weight_dtypes": sorted(dtypes),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "weights": [{"name": p.name, "bytes": p.stat().st_size} for p in weights],
    }


def validate_pair(checkpoints: dict) -> None:
    baseline, quantized = checkpoints.values()
    filenames = ["tokenizer.json", "tokenizer_config.json"]
    if any(
        (Path(item["path"]) / "chat_template.jinja").exists()
        for item in checkpoints.values()
    ):
        filenames.append("chat_template.jinja")
    for filename in filenames:
        original = (Path(baseline["path"]) / filename).read_bytes()
        derivative = (Path(quantized["path"]) / filename).read_bytes()
        if original != derivative:
            raise ValueError(
                f"Checkpoint {filename} differs; use the same tokenizer and chat template"
            )
        for checkpoint in checkpoints.values():
            checkpoint.setdefault("tokenizer_sha256", {})[filename] = hashlib.sha256(
                original
            ).hexdigest()
    for key in ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]:
        configs = [
            item["config"].get("text_config", item["config"])
            for item in checkpoints.values()
        ]
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Checkpoint architecture differs: {key}")


def write_report(payload: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    results = payload["results"]
    labels = list(results)
    values = [100 * results[label]["accuracy"] for label in labels]
    errors = [
        [
            max(0.0, value - 100 * results[label]["wilson_95_ci"][0])
            for label, value in zip(labels, values)
        ],
        [
            max(0.0, 100 * results[label]["wilson_95_ci"][1] - value)
            for label, value in zip(labels, values)
        ],
    ]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(labels, values, yerr=errors, capsize=6, color=["#64748b", "#2563eb"])
    ax.bar_label(
        bars,
        labels=[
            f"{results[label]['correct']}/{results[label]['total']}" for label in labels
        ],
        padding=5,
    )
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Numeric exact-match accuracy (%)")
    ax.set_xlabel("MLX checkpoint (same GSM8K test examples)")
    delta = payload["comparison"]["delta_accuracy_pp"]
    ax.set_title(
        f"GSM8K zero-shot, n={payload['samples']}; delta = {delta:+.2f} pp\n"
        "Error bars: 95% Wilson intervals"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=160)
    plt.close(fig)
    lines = [
        "# GSM8K accuracy comparison",
        "",
        "| Variant | Correct / total | Accuracy | 95% Wilson CI | Invalid answers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, result in results.items():
        lo, hi = result["wilson_95_ci"]
        lines.append(
            f"| {label} | {result['correct']}/{result['total']} | "
            f"{result['accuracy']:.2%} | {lo:.2%}–{hi:.2%} | {result['invalid_answers']} |"
        )
    comparison = payload["comparison"]
    lo, hi = comparison["paired_bootstrap_95_ci_pp"]
    lines += [
        "",
        (
            f"Quantized minus baseline: **{delta:+.2f} percentage points**; "
            f"paired bootstrap 95% interval: [{lo:+.2f}, {hi:+.2f}] pp."
        ),
        (
            f"Regressions: {len(comparison['regression_ids'])}; "
            f"improvements: {len(comparison['improvement_ids'])}."
        ),
        "",
        "![Accuracy comparison](accuracy_comparison.png)",
        "",
        (
            "Dataset revision, sample IDs, settings, checkpoint configs, versions, and raw "
            "answers are in [accuracy_results.json](accuracy_results.json)."
        ),
        "",
        (
            "This is a small, zero-shot math evaluation, not a general quality or vision "
            "benchmark. Missing, malformed, or incorrect final answers score zero. Test-set contamination "
            "in pretraining is unknown. A tied score (including a degenerate paired "
            "bootstrap interval) does not prove parity. Do not compare this protocol "
            "directly to leaderboard scores or transfer MLX scores to GGUF/Ollama."
        ),
    ]
    (output_dir / "accuracy_results.md").write_text("\n".join(lines) + "\n")


def run_accuracy(
    model_path: str,
    baseline_path: str,
    output_dir: Path,
    *,
    samples: int = 100,
    max_tokens: int = 512,
    seed: int = 42,
) -> dict:
    if max_tokens < 1 or samples < 1:
        raise ValueError("samples and max_tokens must be positive")
    if Path(model_path).resolve() == Path(baseline_path).resolve():
        raise ValueError("Baseline and quantized checkpoints must differ")
    checkpoints = {
        "MLX unquantized": checkpoint_metadata(baseline_path, False),
        "MLX 4-bit": checkpoint_metadata(model_path, True),
    }
    validate_pair(checkpoints)
    examples = load_examples(output_dir / "gsm8k-test.jsonl", samples, seed)
    results = {}
    for label, checkpoint in checkpoints.items():
        print(f"[{label}] GSM8K accuracy", flush=True)
        results[label] = evaluate(checkpoint["path"], examples, max_tokens, seed)
        (output_dir / "accuracy_partial.json").write_text(
            json.dumps(
                {
                    "status": "incomplete",
                    "samples": samples,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ["mlx", "mlx-lm", "transformers"]},
        "dataset": {
            "url": DATA_URL,
            "revision": DATA_REVISION,
            "sha256": DATA_SHA256,
            "split": "test",
            "selection": "sorted random.sample indices",
        },
        "samples": samples,
        "seed": seed,
        "max_tokens": max_tokens,
        "sampler": "greedy, temperature=0",
        "tokenizer_config": {"fix_mistral_regex": True},
        "instruction": INSTRUCTION,
        "metric": "numeric exact match of final #### answer or numeric-only response",
        "checkpoints": checkpoints,
        "results": results,
        "comparison": paired_summary(
            results["MLX unquantized"]["examples"],
            results["MLX 4-bit"]["examples"],
            seed,
        ),
    }
    (output_dir / "accuracy_results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_report(payload, output_dir)
    (output_dir / "accuracy_partial.json").unlink(missing_ok=True)
    print(
        f"Accuracy delta: {payload['comparison']['delta_accuracy_pp']:+.2f} pp",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local 4-bit MLX checkpoint")
    parser.add_argument(
        "--baseline-model", required=True, help="Local unquantized source checkpoint"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_artifacts/accuracy")
    )
    args = parser.parse_args()
    run_accuracy(
        args.model,
        args.baseline_model,
        args.output_dir,
        samples=args.samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
