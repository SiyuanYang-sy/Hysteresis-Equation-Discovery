# Input datasets

The five data files cover two dataset groups. `manifest.json` records their paths,
file sizes, and SHA-256 checksums. The regression tests verify these checksums.

| Directory | Files | Use |
| --- | --- | --- |
| `bouc_wen/` | Four `.mat` files | Sine-sweep and multisine excitation/response records |
| `mdof_2dof/` | One `.npz` file | Two-degree-of-freedom truth data and train/test splits |

`nonlinear_fractional` generates its input and reference response in code.

## Bouc-Wen benchmark

Each MATLAB file contains a variable with the same name as the file stem:

- `uval_sinesweep.mat`, `yval_sinesweep.mat`: 153,000 samples each.
- `uval_multisine.mat`, `yval_multisine.mat`: 8,192 samples each.

The benchmark training script uses the sine-sweep input at 750 Hz and generates
reference states with its specified Bouc-Wen solver. The dataset also includes
response measurements and multisine records.

## Two-degree-of-freedom data

`mdof_2dof_truth_data.npz` contains system matrices `M`, `C`, `K`, `H`, `B`,
scalar parameters and timing information, and the time series `t`, `u`, `x1`,
`x2`, `v1`, `v2`, `a1`, `a2`, `z`, and `zdot`. Train/test arrays use `_train`
and `_test` suffixes. There are 5,001 total samples, split into 3,000 training
and 2,001 test samples. All entries are numeric, so loading requires no pickle.

## Provenance

Refer to the [paper](https://doi.org/10.1016/j.ymssp.2026.114725) for the
experimental setup and dataset references. Generated predictions and learned
trajectories are written to `outputs/`.
