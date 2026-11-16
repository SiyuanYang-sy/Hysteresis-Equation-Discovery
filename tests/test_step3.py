"""Check Step 3 checkpoint compatibility without starting a Julia search."""

from contextlib import redirect_stdout
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hysteresis_discovery.registry import EXPERIMENTS
from hysteresis_discovery.runtime import RunConfig


class FixedRegressor:
    """Provide known expressions so the test exercises only data plumbing."""

    def __init__(self, **kwargs):
        self.options = kwargs

    def fit(self, features, target, variable_names):
        self.expression = sympy.Symbol(variable_names[0]) + sympy.Symbol(
            variable_names[-1]
        )
        self.equations_ = pd.DataFrame(
            {"score": [1.0], "sympy_format": [self.expression]}
        )
        return self


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class Step3IntegrationTests(unittest.TestCase):
    def test_all_cases_consume_checkpoints_and_export_numeric_results(self):
        t = np.arange(1000) * 0.01
        values = {
            "t": t,
            "u": np.sin(t),
            "x0": np.zeros(3),
            "m": 1.0,
            "c": 0.8,
            "k": 0.5,
            "alpha": 1.0,
        }
        for name in (
            "x_true",
            "x_pred",
            "x_dot_true",
            "x_dot_pred",
            "x_dotdot_true",
            "x_dotdot_pred",
            "z_true",
            "z_learned",
            "z_phy",
            "z_dot_true",
            "z_dot_learned",
        ):
            values[name] = 0.1 * np.sin(t) + 0.2
        for name in ("x1_true", "x1_pred", "x2_true", "x2_pred"):
            values[name] = 100 + np.sin(t)
        for name in (
            "v1_true",
            "v1_pred",
            "v2_true",
            "v2_pred",
            "z_pred",
            "z_dot_pred",
        ):
            values[name] = 0.1 + 0.01 * np.cos(t)
        for name in (
            "m1",
            "m2",
            "c11",
            "c12",
            "c21",
            "c22",
            "k11",
            "k12",
            "k21",
            "k22",
            "h1",
            "h2",
        ):
            values[name] = 1.0
        fake_pysr = ModuleType("pysr")
        fake_pysr.PySRRegressor = FixedRegressor
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                output_root=Path(directory), device="cpu", dpi=40, seed=42
            )
            for case in EXPERIMENTS:
                with self.subTest(case=case):
                    destination = Path(directory) / case
                    destination.mkdir()
                    np.savez(destination / "data_Final.npz", **values)
                    module = importlib.import_module(
                        f"hysteresis_discovery.experiments.{case}.step3"
                    )
                    with (
                        patch.dict(sys.modules, {"pysr": fake_pysr}),
                        redirect_stdout(io.StringIO()),
                    ):
                        module.run(config)
                    with np.load(
                        destination / "data_step3.npz", allow_pickle=False
                    ) as result:
                        for name in result.files:
                            self.assertTrue(np.isfinite(result[name]).all(), name)
                        if case == "mdof_2dof":
                            expected = values["v1_pred"][:-1] + values["z_pred"][:-1]
                            np.testing.assert_allclose(result["z_dot_pred"], expected)
                    equations = json.loads((destination / "equations.json").read_text())
                    self.assertIn("z_dot", equations)
                    if case == "nonlinear_fractional":
                        self.assertIn("x_dotdot", equations)
                    metadata = json.loads((destination / "run_step3.json").read_text())
                    self.assertEqual(metadata["status"], "completed")


if __name__ == "__main__":
    unittest.main()
