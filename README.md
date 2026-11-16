# Hysteresis Equation Discovery

Research code for **A unified framework for equation discovery and dynamic
prediction of hysteretic systems**, by Siyuan Yang, Wei Liu, and Zhilu Lai.

*Mechanical Systems and Signal Processing*, 258 (2026), 114725.
[Paper](https://doi.org/10.1016/j.ymssp.2026.114725) ·
[Reproduction guide](docs/reproduction.md)

The framework combines solver-based learning of unobserved hysteretic variables
with symbolic regression to identify explicit governing equations. The experiments
cover a Bouc-Wen benchmark, nonlinear and fractional dynamics, and a coupled
two-degree-of-freedom system.

## Repository structure

```text
.
├── .github/workflows/               # Automated checks
├── configs/                        # Default experiment runtime settings
├── data/                           # Input datasets, grouped by experiment
│   ├── bouc_wen/
│   ├── mdof_2dof/
│   └── manifest.json               # File sizes and SHA-256 checksums
├── docs/reproduction.md             # Experiment settings and usage
├── scripts/run_experiment.py        # Entry point for a source checkout
├── src/hysteresis_discovery/
│   ├── cli.py                      # Shared command-line interface
│   ├── runtime.py                  # Paths, logs, plotting, seeds, and metadata
│   ├── numerics.py                 # Interpolation, finite differences, and RMSE
│   ├── io.py                       # Numeric checkpoint loading and validation
│   ├── networks.py                 # Shared acceleration network
│   ├── symbolic.py                 # Expression selection and evaluation
│   └── experiments/                # Case-specific step12.py and step3.py modules
├── tests/                          # Data integrity and regression checks
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

`outputs/` is created automatically and excluded from Git. Experiment 2 generates
its synthetic signals in code and needs no raw-data directory.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Choose the appropriate PyTorch build for your CPU or CUDA environment using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).
Symbolic regression uses [PySR](https://ai.damtp.cam.ac.uk/pysr/), whose Julia
backend may require downloads and initial compilation on first use. Step 1-2
training does not import PySR or start Julia.

The dependency ranges describe compatible package families; they are not a
fully pinned lock file. Each run records its actual package versions.

## Quick start

Run these commands from the repository root:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --check-data
python scripts/run_experiment.py bouc_wen --stage step12
python scripts/run_experiment.py bouc_wen --stage step3
```

Each experiment has two stages:

- `step12`: initialize structural parameters and learn hysteretic trajectories.
- `step3`: discover symbolic equations from `data_Final.npz` produced by `step12`.
- `all`: run `step12` followed by `step3` for the selected experiment.

The default stage is `step12`. The default budgets are **10,000 training epochs**
and **2,000 PySR iterations per fit**.

After an editable installation, `hysteresis-discovery` and
`python -m hysteresis_discovery` expose the same interface. The checkout script
resolves data and output paths from its own location, so it can also be launched
from another working directory.

## Experiments

| Paper | Experiment argument | Description | Input |
| --- | --- | --- | --- |
| Section 4.1 | `bouc_wen` | Bouc-Wen benchmark | Four MATLAB files |
| Section 4.2 | `nonlinear_fractional` | Cubic stiffness and fractional hysteresis | Generated in code |
| Section 4.4 | `mdof_2dof` | Coupled two-degree-of-freedom system | NPZ dataset |

For example:

```bash
python scripts/run_experiment.py nonlinear_fractional --stage all --seed 42
python scripts/run_experiment.py bouc_wen --stage step12 --device cpu
python scripts/run_experiment.py mdof_2dof --stage step3
```

## Configuration and outputs

Runtime settings can be supplied through JSON or command-line flags; flags take
precedence. Scientific parameters remain in each experiment module.

```bash
python scripts/run_experiment.py nonlinear_fractional --config configs/default.json --epochs 500 --seed 42
python scripts/run_experiment.py bouc_wen --plot-interval 0 --dpi 150
python scripts/run_experiment.py mdof_2dof --stage all --dry-run
```

Use `--help` for all options. Plots run without a GUI and use Matplotlib's built-in
math rendering by default. Enable interactive windows with `--show`; enable
external LaTeX rendering with `--usetex` if a working TeX installation is available.

Results are written to `outputs/<experiment>/`:

- `data_Final.npz`: learned trajectories and parameters consumed by Step 3.
- `log_step12.txt` and `log_step3.txt`: console output for each stage.
- `run_step12.json` and `run_step3.json`: settings, package versions, timing, and status.
- PNG figures and intermediate NPZ snapshots at the configured checkpoint intervals.
- `symbolic/fit_*/`: PySR search artifacts, created during Step 3.

Each Step 3 run exports numeric predictions in `data_step3.npz` and discovered
expressions in `equations.json`.
Repeated runs with the same output root overwrite files with the same names;
use `--output-root outputs/<run-name>` to keep runs separate.

## Development checks

```bash
python -m pip install -r requirements.txt
python -m black --check src scripts tests
python -m unittest discover -s tests -v
```

The tests cover shared numerical operations, expression evaluation, CLI behavior,
runtime cleanup, stage interfaces, and dataset checksums. Step 3
interface tests use known expressions in place of a live Julia search.
CI runs these checks without starting training or Julia searches.

## Citation

```bibtex
@article{yang2026hysteretic,
  title   = {A unified framework for equation discovery and dynamic prediction of hysteretic systems},
  author  = {Yang, Siyuan and Liu, Wei and Lai, Zhilu},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {258},
  pages   = {114725},
  year    = {2026},
  doi     = {10.1016/j.ymssp.2026.114725}
}
```

## License

No software license is specified for this repository. The publication and
third-party datasets retain their respective rights.
