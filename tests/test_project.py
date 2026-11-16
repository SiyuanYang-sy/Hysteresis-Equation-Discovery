"""Regression checks for the repository layout and shared experiment utilities."""

from contextlib import redirect_stdout
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hysteresis_discovery.cli import main, parse_args
from hysteresis_discovery.registry import EXPERIMENTS
from hysteresis_discovery.runtime import RunConfig, experiment_session
from hysteresis_discovery.symbolic import (
    evaluate_expression,
    has_independent_constant,
    select_expression,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class RepositoryTests(unittest.TestCase):
    def test_dataset_checksums(self):
        """Verify each dataset against its published checksum."""
        manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 5)
        self.assertEqual(len({entry["path"] for entry in manifest}), 5)
        for entry in manifest:
            with self.subTest(path=entry["path"]):
                file = ROOT / "data" / entry["path"]
                self.assertEqual(file.stat().st_size, entry["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(file.read_bytes()).hexdigest(), entry["sha256"]
                )

    def test_experiment_imports_do_not_run_training(self):
        previous_stdout = sys.stdout
        previous_cwd = Path.cwd()
        for name in EXPERIMENTS:
            for stage in ("step12", "step3"):
                module = importlib.import_module(
                    f"hysteresis_discovery.experiments.{name}.{stage}"
                )
                self.assertTrue(callable(module.run))
        self.assertIs(sys.stdout, previous_stdout)
        self.assertEqual(Path.cwd(), previous_cwd)


class CommandLineTests(unittest.TestCase):
    def test_cli_overrides_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "runtime.json"
            config.write_text(
                json.dumps(
                    {"epochs": 3, "niterations": 2, "show": False, "device": "cpu"}
                ),
                encoding="utf-8",
            )
            _, args = parse_args(["bouc_wen", "--config", str(config), "--epochs", "7"])
        self.assertEqual(args.epochs, 7)
        self.assertEqual(args.niterations, 2)
        self.assertFalse(args.show)
        self.assertEqual(args.device, "cpu")

    def test_invalid_training_budget_is_rejected(self):
        with self.assertRaises(SystemExit) as error, redirect_stdout(io.StringIO()):
            parse_args(["bouc_wen", "--epochs", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_unknown_config_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.json"
            config.write_text('{"epoch": 10}', encoding="utf-8")
            with self.assertRaises(SystemExit) as error:
                parse_args(["bouc_wen", "--config", str(config)])
            self.assertEqual(error.exception.code, 2)

    def test_checkpoint_is_required_before_step3(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as error:
                main(["bouc_wen", "--stage", "step3", "--output-root", directory])
            self.assertEqual(error.exception.code, 1)

    def test_checkout_entry_point_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_experiment.py"),
                    "--check-data",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("[OK]"), len(EXPERIMENTS))

    def test_dry_run_leaves_output_directory_uncreated(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "unused"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "mdof_2dof",
                            "--stage",
                            "all",
                            "--dry-run",
                            "--output-root",
                            str(destination),
                        ]
                    ),
                    0,
                )
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["stages"], ["step12", "step3"])
            self.assertFalse(destination.exists())


class SymbolicTests(unittest.TestCase):
    def test_numeric_evaluation_matches_direct_substitution(self):
        velocity, z = sympy.symbols("velocity z")
        expression = (
            4 * velocity
            - 5 * sympy.Abs(velocity) * sympy.Abs(z) ** sympy.Rational(1, 2) * z
        )
        v = np.array([-0.5, 0.0, 0.5])
        hysteresis = np.array([-0.3, 0.0, 0.2])
        expected = [
            float(expression.subs({velocity: a, z: b})) for a, b in zip(v, hysteresis)
        ]
        actual = evaluate_expression(expression, ["velocity", "z"], v, hysteresis)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.dtype, np.float64)

    def test_constant_expression_is_broadcast(self):
        actual = evaluate_expression(sympy.Integer(3), ["x"], np.arange(4))
        np.testing.assert_array_equal(actual, [3.0] * 4)

    def test_mdof_expression_uses_velocity_columns(self):
        v1, v2, z = sympy.symbols("v1 v2 z")
        expression = 50 * (v2 - v1) - 2 * sympy.Abs(v2 - v1) * z
        velocity1 = np.array([1.0, -2.0])
        velocity2 = np.array([3.0, 1.0])
        hysteresis = np.array([0.1, 0.2])
        expected = (
            50 * (velocity2 - velocity1)
            - 2 * np.abs(velocity2 - velocity1) * hysteresis
        )
        actual = evaluate_expression(
            expression, ["v1", "v2", "z"], velocity1, velocity2, hysteresis
        )
        np.testing.assert_allclose(actual, expected)

    def test_non_finite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_expression(sympy.oo, ["x"], np.ones(3))

    def test_original_candidate_filter_is_preserved(self):
        x = sympy.Symbol("x")
        self.assertFalse(has_independent_constant(x + 2, ["x"]))
        self.assertTrue(has_independent_constant(x + sympy.pi, ["x"]))
        model = SimpleNamespace(
            equations_=pd.DataFrame(
                {
                    "score": [3.0, 2.0, 1.0],
                    "sympy_format": [x + sympy.pi, x * 2, x],
                }
            )
        )
        expression, score = select_expression(model, ["x"])
        self.assertEqual(expression, x * 2)
        self.assertEqual(score, 2.0)

    def test_empty_candidate_set_has_clear_error(self):
        model = SimpleNamespace(
            equations_=pd.DataFrame(columns=["score", "sympy_format"])
        )
        with self.assertRaisesRegex(RuntimeError, "No symbolic expression"):
            select_expression(model, ["x"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class NumericalTests(unittest.TestCase):
    def test_exact_rmse_match_has_finite_zero_gradient(self):
        import torch
        from hysteresis_discovery.numerics import rmse

        prediction = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        loss = rmse(prediction, prediction.detach().clone())
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_rmse_matches_original_for_nonzero_residuals(self):
        import torch
        from hysteresis_discovery.numerics import rmse

        prediction = torch.tensor(
            [1.0, 3.0, -2.0], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([2.0, 0.0, -1.0], dtype=torch.float64)
        reference = torch.sqrt(torch.mean((prediction - target) ** 2))
        actual = rmse(prediction, target)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            torch.autograd.grad(actual, prediction)[0],
            torch.autograd.grad(reference, prediction)[0],
        )

    def test_interpolation_and_endpoint_clamping(self):
        import torch
        from hysteresis_discovery.numerics import lerp_series

        signal = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)
        values = torch.stack(
            [
                lerp_series(signal, torch.tensor(t), 0.0, 1.0)
                for t in (-1.0, 0.5, 1.5, 5.0)
            ]
        )
        torch.testing.assert_close(
            values, torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float64)
        )
        values.sum().backward()
        torch.testing.assert_close(
            signal.grad, torch.tensor([1.5, 1.0, 1.5], dtype=torch.float64)
        )

    def test_central_and_endpoint_differences(self):
        import torch
        from hysteresis_discovery.numerics import diff_series

        series = torch.tensor(
            [0.0, 1.0, 4.0, 9.0], dtype=torch.float64, requires_grad=True
        )
        result = diff_series(series, 1.0)
        torch.testing.assert_close(
            result, torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=torch.float64)
        )
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(series.grad).all())

    def test_shared_acceleration_network_matches_original_architecture(self):
        import torch
        from torch import nn
        from hysteresis_discovery.networks import XDotDotNet

        network = XDotDotNet(hidden_dim=8).double()
        reference = nn.Sequential(
            nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
        ).double()
        reference.load_state_dict(network.net.state_dict())
        inputs = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        torch.testing.assert_close(
            network(*inputs.unbind(dim=1)), reference(inputs).squeeze(-1)
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class RuntimeTests(unittest.TestCase):
    def test_runtime_restores_state_after_failure(self):
        import torch
        import matplotlib.pyplot as plt

        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        previous_dtype, previous_rc = torch.get_default_dtype(), plt.rcParams.copy()
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                output_root=Path(directory), device="cpu", seed=42, dpi=80
            )
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with experiment_session(config, "bouc_wen", "step12"):
                    self.assertEqual(torch.get_default_dtype(), torch.float64)
                    raise RuntimeError("intentional failure")
            metadata = json.loads(
                (Path(directory) / "bouc_wen/run_step12.json").read_text()
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("intentional failure", metadata["error"])
        self.assertIs(sys.stdout, previous_stdout)
        self.assertIs(sys.stderr, previous_stderr)
        self.assertEqual(torch.get_default_dtype(), previous_dtype)
        self.assertEqual(plt.rcParams["text.usetex"], previous_rc["text.usetex"])
        self.assertEqual(Path.cwd(), previous_cwd)

    def test_seeded_sessions_are_repeatable_and_restore_rng(self):
        import torch

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        samples = []
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(output_root=Path(directory), device="cpu", seed=42)
            for _ in range(2):
                with experiment_session(config, "bouc_wen", "step12") as session:
                    samples.append(
                        (random.random(), np.random.random(), torch.rand(1).item())
                    )
                    self.assertFalse(
                        session.checkpoint(0, 5, include_first=True, interval=0)
                    )
                    self.assertTrue(session.checkpoint(4, 5))
                    self.assertFalse(session.checkpoint(3, 5))
            metadata = json.loads(
                (Path(directory) / "bouc_wen/run_step12.json").read_text()
            )
            self.assertEqual(metadata["status"], "completed")
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)


if __name__ == "__main__":
    unittest.main()
