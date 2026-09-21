"""CPU-only validation of scoring, pairing, dataset integrity, and report output."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_accuracy as accuracy
import benchmark_mlx


class AccuracyTests(unittest.TestCase):
    def test_numeric_answers(self):
        for text in ["42", "  +0042\n", "Reasoning: 2 + 40 = 42\n#### 42.0"]:
            self.assertEqual(accuracy.numeric_answer(text), "42")
        self.assertEqual(accuracy.numeric_answer("#### -1,234.50"), "-1234.5")
        self.assertEqual(accuracy.numeric_answer(" +0000 "), "0")

    def test_rejects_ambiguous_or_incomplete_answers(self):
        for text in [
            "",
            "The answer is 42",
            "#### 42 or 43",
            "#### 42 dollars",
            "#### 1,23",
            "42\n43",
            "#### NaN",
            "#### 42\nmore reasoning",
            "42.0.1",
            "####",
        ]:
            self.assertIsNone(accuracy.numeric_answer(text), text)
        self.assertNotEqual(
            accuracy.numeric_answer("142"), accuracy.numeric_answer("42")
        )
        self.assertNotEqual(
            accuracy.numeric_answer("123.45"), accuracy.numeric_answer("123")
        )

    def test_paired_regressions_and_gains(self):
        baseline = [
            {"id": str(i), "correct": v}
            for i, v in enumerate([True, True, False, True])
        ]
        quantized = [
            {"id": str(i), "correct": v}
            for i, v in enumerate([False, False, True, True])
        ]
        result = accuracy.paired_summary(baseline, quantized, 42)
        self.assertEqual(result["delta_accuracy_pp"], -25)
        self.assertEqual(result["regression_ids"], ["0", "1"])
        self.assertEqual(result["improvement_ids"], ["2"])
        self.assertEqual(result, accuracy.paired_summary(baseline, quantized, 42))
        with self.assertRaises(ValueError):
            accuracy.paired_summary(baseline, quantized[::-1], 42)
        with self.assertRaises(ValueError):
            accuracy.paired_summary([], [], 42)

    def test_wilson_extremes_retain_uncertainty(self):
        self.assertGreater(accuracy.wilson_interval(0, 100)[1], 0)
        self.assertLess(accuracy.wilson_interval(100, 100)[0], 1)

    def test_dataset_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "test.jsonl"
            cache.write_text('{"question": "modified"}\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                accuracy.load_examples(cache, 1, 42)

    def test_same_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            accuracy.run_accuracy(".", ".", Path("unused"))

    def test_checkpoint_precision_and_tokenizer_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = {}
            for name, quantized in [("baseline", False), ("quantized", True)]:
                root = Path(directory) / name
                root.mkdir()
                config = {"model_type": "test", "hidden_size": 8}
                if quantized:
                    config["quantization"] = {"bits": 4, "group_size": 64}
                (root / "config.json").write_text(json.dumps(config))
                for filename in ["tokenizer.json", "tokenizer_config.json"]:
                    (root / filename).write_text("{}")
                header = json.dumps(
                    {"weight": {"dtype": "U32" if quantized else "BF16"}}
                ).encode()
                (root / "model.safetensors").write_bytes(
                    len(header).to_bytes(8, "little") + header
                )
                checkpoints[name] = accuracy.checkpoint_metadata(str(root), quantized)
            accuracy.validate_pair(checkpoints)
            self.assertEqual(checkpoints["baseline"]["weight_dtypes"], ["BF16"])
            with self.assertRaisesRegex(ValueError, "expected quantized"):
                accuracy.checkpoint_metadata(checkpoints["baseline"]["path"], True)
            (
                Path(checkpoints["quantized"]["path"]) / "tokenizer_config.json"
            ).write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "tokenizer_config.json differs"):
                accuracy.validate_pair(checkpoints)

    def test_pipeline_and_chart_without_model_inference(self):
        examples = [{"id": "a", "question": "example", "expected": "1"}]

        def fake_evaluate(path, examples, max_tokens, seed):
            correct = path == "baseline"
            return {
                "correct": int(correct),
                "total": 1,
                "accuracy": float(correct),
                "wilson_95_ci": accuracy.wilson_interval(int(correct), 1),
                "invalid_answers": 0,
                "examples": [{**examples[0], "correct": correct}],
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(
                    accuracy,
                    "checkpoint_metadata",
                    side_effect=lambda path, q: {"path": path},
                ),
                patch.object(accuracy, "validate_pair"),
                patch.object(accuracy, "load_examples", return_value=examples),
                patch.object(accuracy, "evaluate", side_effect=fake_evaluate),
                patch.object(accuracy, "version", return_value="test"),
            ):
                accuracy.run_accuracy("quantized", "baseline", output, samples=1)
            payload = json.loads((output / "accuracy_results.json").read_text())
            self.assertEqual(payload["comparison"]["delta_accuracy_pp"], -100)
            self.assertGreater(
                (output / "accuracy_comparison.png").stat().st_size, 1000
            )
            self.assertIn("-100.00", (output / "accuracy_results.md").read_text())

    def test_combined_benchmark_runs_accuracy_by_default(self):
        smoke = {"cases": [], "pass_rate": 1.0, "mean_output_tokens_per_second": 1.0}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(benchmark_mlx, "benchmark_model", return_value=smoke),
            patch.object(benchmark_mlx, "write_chart"),
            patch.object(accuracy, "run_accuracy") as evaluate,
        ):
            output = Path(directory)
            self.assertEqual(benchmark_mlx.run("quantized", "baseline", 96, output), 0)
            evaluate.assert_called_once_with(
                "quantized",
                "baseline",
                output / "accuracy",
                samples=100,
                max_tokens=512,
                seed=42,
            )
            evaluate.reset_mock()
            benchmark_mlx.run("quantized", "baseline", 96, output, smoke_only=True)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
