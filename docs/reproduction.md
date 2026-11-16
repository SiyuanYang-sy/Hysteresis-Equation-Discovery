# Reproduction guide

## Execution order

Each case has two stages. `step12` initializes parameters and learns hysteretic
variables. Its `data_Final.npz` is the input for
`step3`, which performs symbolic regression. Keep the same `--output-root` for
both stages.

```bash
python scripts/run_experiment.py bouc_wen --stage step12 --seed 42
python scripts/run_experiment.py bouc_wen --stage step3 --seed 42
```

Use `--stage all` to run both stages sequentially. Replace `bouc_wen` with any
experiment listed by `--list`. A missing Step 2 checkpoint is reported before
PySR or Julia is imported.

## Scientific settings

Case-specific equations, loss definitions, and training settings are defined in
`src/hysteresis_discovery/experiments/`:

| Case | Sampling and training selection | Initialization and learning |
| --- | --- | --- |
| `bouc_wen` | 750 Hz; generate 1,000 samples and select indices 500:800 | Least-squares linear parameters, then learned hysteresis and acceleration network |
| `nonlinear_fractional` | 100 Hz over 10 s | Cubic restoring term, fractional exponent 1.5, learned acceleration network |
| `mdof_2dof` | Training split, 3,000 samples | Perturbed-parameter initialization and coupled equations |

Training defaults to 10,000 epochs. Each symbolic fit defaults to 2,000 PySR
iterations, with case-specific operators, constraints, and repeated fits.
The nonlinear/fractional case learns both the motion and hysteresis equations.
The other Step 3 scripts focus on the hysteresis
equation; the structural parameters are also recorded by Step 2.

The expression filter permits standalone numeric literals.

The entry points cover equation discovery for the three cases. Paper figures,
appendix ablations, and held-out prediction studies are not exposed as separate
commands.

## Runtime options

| Option | Default | Purpose |
| --- | --- | --- |
| `--epochs` | 10000 | Step 2 optimizer updates |
| `--niterations` | 2000 | Iterations for each symbolic fit |
| `--device` | `auto` | Automatically select CUDA or use `cpu` / `cuda` explicitly |
| `--seed` | Unset | Seed Python, NumPy, PyTorch, and PySR |
| `--plot-interval` | Per-case interval | Override intermediate plot/NPZ frequency; `0` disables them |
| `--dpi` | 600 | Resolution of saved figures |
| `--show` | Disabled | Display interactive figures |
| `--usetex` | Disabled | Use external LaTeX instead of built-in math rendering |
| `--data-root` | Repository `data/` | Root containing the case-specific input folders |
| `--output-root` | Repository `outputs/` | Root containing the case-specific result folders |
| `--config` | None | Read runtime defaults from JSON; flags override JSON |
| `--dry-run` | Disabled | Print resolved settings without imports or execution |

Paths supplied as arguments or in JSON are resolved relative to the calling
working directory. The built-in default paths resolve from the source checkout.
Use an editable installation for this repository layout. If distributing only
a built wheel, supply data and output roots explicitly; the wheel does not bundle
the datasets or paper.

A supplied seed configures PySR for deterministic serial search. With no seed,
its normal search settings apply. A seed helps comparisons within the same
software and hardware environment; it does not guarantee identical GPU results
or identical search results across package versions. The benchmark and nonlinear
scripts use float64; the MDOF script uses float32 training tensors.

## Repository checks

```bash
python scripts/run_experiment.py --check-data
python -m unittest discover -s tests -v
```

The committed interface tests use a controlled regressor with known expressions
to check checkpoint compatibility and exported predictions for all three cases.
These tests verify data flow and numeric evaluation; they do not measure live
PySR search behavior.

## Outputs

`run_step12.json` and `run_step3.json` record actual dependencies, resolved
device, runtime arguments, UTC start time, elapsed time, and completion/failure
status. Detailed console output is written to the matching stage log.

`equations.json` records the discovered expressions. PySR search artifacts live
under `symbolic/fit_01/`, `symbolic/fit_02/`, and so on. These files and the stage
logs provide the symbolic search record. Numerical predictions are stored as
numeric arrays so they can be reopened with `allow_pickle=False`.

For a full study, use separate output roots for configurations or random seeds,
record package versions with `python -m pip freeze`, and compare prediction
errors only after training and symbolic search have converged.
