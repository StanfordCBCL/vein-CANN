# Sensitivity analysis

This folder contains the first model-selection stage for vein-CANN. The script
uses variance-based Sobol indices to rank the candidate strain-energy terms
learned across a cohort. It does **not** select the final constitutive equation
on its own. Its ranked output is passed to the decorrelation and family-collapse
stages in the neighboring folders.

## Method

For every candidate term, the script evaluates its strain-energy basis function
on two biaxial stretch grids:

- a physiological grid, which represents the deformation range used for model
  selection; and
- an extended grid, which checks whether the ranking is sensitive to the chosen
  deformation window.

The resulting basis matrix is denoted by `Psi`. For each Sobol coefficient
sample `w`, the scalar response is

```text
W = Psi @ w
Y = sqrt(mean(W**2))
```

SALib then estimates the first-order index `S1` and total-order index `ST` for
each term. The headline ranking is based on `ST`; a smaller rank is better. Four
coefficient regimes are reported:

| Regime | Basis and coefficient treatment | Purpose |
| --- | --- | --- |
| `normalized_cohort` | Unit-norm basis columns; bounds are ± the across-specimen coefficient standard deviation | Default cohort-aware ranking |
| `coeff_max` | Raw basis columns; bounds are ± the largest observed coefficient magnitude | Retains fitted coefficient scale |
| `normalized` | Unit-norm basis columns; common bounds of `[-1, 1]` | Compares terms on equal footing |
| `raw_uniform` | Raw basis columns; common bounds of `[-1, 1]` | Measures intrinsic basis-shape influence |

The script also performs leave-one-specimen-out ranking for the selected
cohort-aware regime. The reduced term set contains terms that occur in the
top `K` on both stretch grids. Decorrelation must still be performed because
high Sobol importance does not imply that two terms are distinguishable.

## Files

- `sobol_sensitivity_analysis.py`: self-contained sensitivity-analysis script.
- `requirements.txt`: lightweight environment requirements for this stage.

The candidate catalogue mirrors the default single-network library in
`2. CANN training/neural_net.py`: seven shapes for `I1`, `I2`, `I4f`, `I4s`,
`I4el`, and `I4smc`; five shapes for `I8fs`; and the enabled Green-strain
polynomials.

## Required input

Supply one or more cohort roots. Each immediate child directory is treated as
one specimen, and the script searches recursively below it for
`active_terms.csv`, which is written by the CANN training stage.

```text
training_outputs/
├── specimen_01/
│   └── .../active_terms.csv
├── specimen_02/
│   └── .../active_terms.csv
└── specimen_03/
    └── .../active_terms.csv
```

Each `active_terms.csv` must contain these columns:

| Column | Meaning |
| --- | --- |
| `group` | `invariant` or `strain` |
| `inv` | Invariant name, or `strain` for Green-strain terms |
| `term` | Candidate shape name |
| `w_sub` | Weight inside the candidate subnet |
| `w_ann` | Weight in the shared strain-energy layer |

The observed coefficient used for cohort-derived bounds is
`w_sub * w_ann`. Missing catalogue terms are assigned zero for that specimen.
Use training runs with the same candidate library and strain-basis settings in
one cohort. In particular, `--poly_n_terms` and
`--poly-include-linear`/`--no-poly-include-linear` must match training.

## Environment

Python 3.10 or later is recommended. From this folder, create an isolated
environment and install the small dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This separate environment avoids coupling the analysis to the older
TensorFlow pins that may be required to reproduce a historical training run.

## Run

Run from the repository root. `--cohort NAME=PATH` is repeatable; the name is
used as the corresponding output subfolder.

```bash
python "3. Model Selection/1. Sensitivity analysis/sobol_sensitivity_analysis.py" \
  --cohort ALL_VEINS="path/to/training_outputs" \
  --out_root "3. Model Selection/1. Sensitivity analysis/results" \
  --sobol_N 4096 \
  --primary_regime normalized_cohort \
  --loo \
  --top_k 15 \
  --poly_n_terms 7 \
  --poly-include-linear
```

For separate anatomical cohorts, repeat the option:

```bash
python "3. Model Selection/1. Sensitivity analysis/sobol_sensitivity_analysis.py" \
  --cohort NATIVE_IVC="path/to/ivc_training_outputs" \
  --cohort JUGULAR="path/to/jugular_training_outputs" \
  --cohort SAPHENOUS="path/to/saphenous_training_outputs" \
  --out_root "3. Model Selection/1. Sensitivity analysis/results"
```

The legacy `--native_root`, `--tevg_root`, and `--saphenous_root` options remain
available for compatibility, but `--cohort` is clearer for new analyses.

Use a power-of-two base sample size for `--sobol_N`. With second-order indices
disabled, each Sobol run evaluates `N * (D + 2)` coefficient vectors, where
`D` is the number of candidate terms. Runtime therefore grows with the number
of terms, grids, regimes, cohorts, and leave-one-out reruns.

## Main outputs

For a cohort named `ALL_VEINS`, results are written under
`<out_root>/ALL_VEINS/`:

| Output | Contents |
| --- | --- |
| `ranked_terms_sobol.csv` | Unified ranking across both grids and all regimes; primary input to decorrelation |
| `sobol_indices_<grid>_<regime>.csv` | `S1`, `ST`, confidence intervals, and ranks for one analysis |
| `loo_rank_stability_<grid>.csv` | Leave-one-specimen-out stability results |
| `psi_matrix_physio.npz` | Physiological basis matrix and term metadata |
| `psi_matrix_extended.npz` | Extended-grid basis matrix and term metadata |
| `reduced_term_set.txt` | Top-`K` intersections across grids and regimes |
| `summary.txt` | Human-readable run summary and leading terms |

Keep the two `psi_matrix_*.npz` files with `ranked_terms_sobol.csv`; the next
stage uses them to identify highly correlated response columns.

## Optional recruitment-aware physiological matrix

By default, the basis uses raw invariants with generic fiber directions. This
is appropriate for intrinsic library sensitivity, but symmetric helical fibers
can make corresponding `I4f` and `I4s` columns identical. If a validated
recruitment-aware matrix has already been generated, pass it with
`--psi_matrix_override path/to/psi_matrix_physio_recruited.npz`.

The NPZ file must contain `psi` with shape `(points, terms)` and `term_keys` in
the same `group | inv | term` format used by the script. The override affects
only the physiological grid; the extended grid remains an unrecruited
extrapolation. Record how the override was generated because it changes the
scientific interpretation of the analysis.

## Reproducibility and interpretation

- Record the CANN training commit, specimen list, active-term threshold,
  candidate-library settings, command, seed, and output folder.
- Do not interpret `S1` or `ST` as a fitted material coefficient. They measure
  variance in the scalar response under the specified coefficient distribution.
- Compare the four regimes and the leave-one-out results. A term that is highly
  ranked in only one regime is not a robust population-level choice.
- Sensitivity ranking is a screening step. Continue with decorrelation, family
  collapse, and independent equation refitting before choosing a final model.
