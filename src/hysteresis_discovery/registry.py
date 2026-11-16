"""Experiment names, source modules, and required input files."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Experiment:
    """Describe one independently runnable experimental case."""

    title: str
    section: str
    files: tuple[str, ...] = ()


EXPERIMENTS = {
    "bouc_wen": Experiment(
        "Bouc-Wen hysteretic system benchmark",
        "4.1",
        (
            "uval_sinesweep.mat",
            "yval_sinesweep.mat",
            "uval_multisine.mat",
            "yval_multisine.mat",
        ),
    ),
    "nonlinear_fractional": Experiment(
        "High-order nonlinear and fractional terms", "4.2"
    ),
    "mdof_2dof": Experiment(
        "Two-degree-of-freedom system", "4.4", ("mdof_2dof_truth_data.npz",)
    ),
}
