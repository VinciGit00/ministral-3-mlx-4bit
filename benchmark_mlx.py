"""Small, repeatable functional benchmark for the converted MLX checkpoint."""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass

from mlx_lm import generate, load


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


def run(model_path: str, max_tokens: int) -> int:
    model, tokenizer = load(model_path)
    passed = 0

    print(f"platform: {platform.platform()}")
    print(f"model: {model_path}")
    print(f"cases: {len(CASES)}, max_tokens: {max_tokens}, sampler: default greedy")

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
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        normalized = answer.lower()
        ok = all(expected in normalized for expected in case.expected)
        passed += int(ok)
        status = "PASS" if ok else "CHECK"
        print(f"[{status}] {case.name}: {elapsed:.2f}s | {answer.strip()}")

    print(f"summary: {passed}/{len(CASES)} keyword checks passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".", help="Local MLX checkpoint directory")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    raise SystemExit(run(args.model, args.max_tokens))
