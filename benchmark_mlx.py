"""Repeatable baseline-vs-quantized MLX benchmark with a chart artifact."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected: tuple[str, ...]


CASES = (
    Case(
        "arithmetic",
        "What is 17 multiplied by 6? Answer with the number and a short explanation.",
        ("102",),
    ),
    Case(
        "science",
        "Why does ice float on liquid water? Answer in one sentence.",
        ("less dense",),
    ),
    Case(
        "italian",
        "Rispondi in italiano: qual è la capitale d'Italia?",
        ("roma",),
    ),
    Case(
        "json_instruction",
        'Return exactly this JSON object and nothing else: {"status":"ok"}',
        ("status", "ok"),
    ),
)


def benchmark_model(model_path: str, max_tokens: int) -> dict:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, tokenizer_config={"fix_mistral_regex": True})
    rows = []
    for case in CASES:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": case.prompt}],
            add_generation_prompt=True,
        )
        started = time.perf_counter()
        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        normalized = answer.lower()
        ok = all(expected in normalized for expected in case.expected)
        output_tokens = len(tokenizer.encode(answer))
        rows.append(
            {
                "case": case.name,
                "seconds": elapsed,
                "output_tokens": output_tokens,
                "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
                "passed": ok,
                "answer": answer.strip(),
            }
        )
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return {
        "model": str(model_path),
        "cases": rows,
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "mean_seconds": statistics.mean(row["seconds"] for row in rows),
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "mean_output_tokens_per_second": statistics.mean(
            row["tokens_per_second"] for row in rows
        ),
    }


def write_chart(results: dict, output_path: Path, accuracy: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    labels = list(results)
    checkpoints = accuracy.get("checkpoints", {}) if accuracy else {}

    def display_label(label: str) -> str:
        if label == "MLX unquantized" and checkpoints.get(label, {}).get(
            "weight_dtypes"
        ) == ["BF16"]:
            return "MLX BF16"
        return label

    display_labels = [display_label(label) for label in labels]
    colors = ["#64748b", "#2563eb"]
    mean_latency = [results[label]["mean_seconds"] for label in labels]
    generation_speed = [
        results[label]["mean_output_tokens_per_second"] for label in labels
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(display_labels, mean_latency, color=colors)
    axes[0].set_title("Mean end-to-end latency")
    axes[0].set_ylabel("Seconds (lower is better)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(display_labels, generation_speed, color=colors)
    axes[1].set_title("Generated output speed")
    axes[1].set_ylabel("Tokens/s (higher is better)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].set_title("GSM8K numeric exact-match accuracy")
    if accuracy is None:
        axes[2].text(
            0.5,
            0.5,
            "Not measured",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    else:
        quality = accuracy["results"]
        values = [100 * row["accuracy"] for row in quality.values()]
        errors = [
            [
                max(0.0, value - 100 * row["wilson_95_ci"][0])
                for value, row in zip(values, quality.values())
            ],
            [
                max(0.0, 100 * row["wilson_95_ci"][1] - value)
                for value, row in zip(values, quality.values())
            ],
        ]
        bars = axes[2].bar(
            [display_label(label) for label in quality],
            values,
            color=colors,
            yerr=errors,
            capsize=5,
        )
        axes[2].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=6)
        axes[2].set_ylim(0, 110)
        axes[2].set_yticks(range(0, 101, 20))
        axes[2].set_ylabel("Accuracy (%) · 95% Wilson intervals")
        axes[2].set_xlabel(
            f"n={accuracy['samples']}; quantized − baseline: "
            f"{accuracy['comparison']['delta_accuracy_pp']:+.1f} pp"
        )
        axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Ministral 3 MLX baseline vs 4-bit quantized")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(
    model_path: str,
    baseline_path: str | None,
    max_tokens: int,
    output_dir: Path,
    *,
    smoke_only: bool = False,
    accuracy_samples: int = 100,
    accuracy_max_tokens: int = 512,
    seed: int = 42,
) -> int:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    paths = {"MLX 4-bit": model_path}
    if baseline_path:
        paths = {"MLX unquantized": baseline_path, "MLX 4-bit": model_path}
    results = {}
    print(f"platform: {platform.platform()}")
    print(f"cases: {len(CASES)}, max_tokens: {max_tokens}, sampler: default greedy")
    for label, path in paths.items():
        print(f"\n[{label}] {path}")
        result = benchmark_model(path, max_tokens)
        results[label] = result
        for row in result["cases"]:
            status = "PASS" if row["passed"] else "CHECK"
            print(f"[{status}] {row['case']}: {row['seconds']:.2f}s | {row['answer']}")
        print(
            f"summary: {result['pass_rate']:.0%} passed | "
            f"{result['mean_output_tokens_per_second']:.1f} output tok/s"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "platform": platform.platform(),
        "max_tokens": max_tokens,
        "sampler": "default greedy",
        "tokenizer_config": {"fix_mistral_regex": True},
        "results": results,
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_chart(results, output_dir / "benchmark_comparison.png")
    if baseline_path and not smoke_only:
        from benchmark_accuracy import run_accuracy

        accuracy = run_accuracy(
            model_path,
            baseline_path,
            output_dir / "accuracy",
            samples=accuracy_samples,
            max_tokens=accuracy_max_tokens,
            seed=seed,
        )
        write_chart(results, output_dir / "benchmark_comparison.png", accuracy)
    else:
        print(
            "Accuracy comparison not run: "
            + (
                "--smoke-only selected."
                if smoke_only
                else "--baseline-model is required."
            )
        )
    passed = all(result["pass_rate"] == 1.0 for result in results.values())
    print(f"\nartifacts: {output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".", help="Local MLX checkpoint directory")
    parser.add_argument("--baseline-model", help="Optional unquantized MLX checkpoint")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument(
        "--smoke-only", action="store_true", help="Skip GSM8K accuracy evaluation"
    )
    parser.add_argument("--accuracy-samples", type=int, default=100)
    parser.add_argument("--accuracy-max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.model,
            args.baseline_model,
            args.max_tokens,
            args.output_dir,
            smoke_only=args.smoke_only,
            accuracy_samples=args.accuracy_samples,
            accuracy_max_tokens=args.accuracy_max_tokens,
            seed=args.seed,
        )
    )
