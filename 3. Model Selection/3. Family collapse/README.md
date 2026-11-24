# Family collapse

This folder contains the third vein-CANN model-selection stage. It reduces the
decorrelated candidate set to a limited number of response shapes per
constitutive family.

The input is `ranked_terms_sobol_pruned.csv` from the
[decorrelation stage](<../2.%20Decorrelation/README.md>). The output is a compact
candidate equation that can be passed to independent coefficient and nonlinear
parameter refitting in `4. Evaluation`.

## Why this stage is separate

Pairwise decorrelation removes terms whose evaluated basis columns are nearly
identical over the selected stretch grids. It does not prevent a model from
retaining several distinguishable shapes for the same scalar mechanism, such
as linear, quadratic, and exponential functions of `I1 - 3`.

Family collapse adds an interpretability constraint: alternative shapes of the
same family compete for a limited number of slots. This is a structural model
selection rule, not another correlation test.

The clean core implementation is deliberately **rank-based**. It does not fit
an aggregate response, alter CANN coefficients, or use animal-specific elbow
data. Those are separate response-approximation experiments and should not be
mixed silently into the baseline selection pipeline.

## Family definitions

Each invariant is its own family:

```text
I1, I2, I4f, I4s, I4el, I4smc, I8fs
```

Green-strain terms are grouped by directional structure:

| Family | Terms |
| --- | --- |
| `strain_Etheta_pure` | `E_θ`, `E_θ²`, `E_θ³`, `E_θ⁴` |
| `strain_Ez_pure` | `E_z`, `E_z²`, `E_z³`, `E_z⁴` |
| `strain_mixed` | `E_θ·E_z`, `E_θ²·E_z`, `E_θ·E_z²`, `E_θ²·E_z²` |

Unrecognized terms cause an error by default. This prevents a spelling change
or a new library term from bypassing the collapse unintentionally.
`--allow_unclassified` permits explicit pass-through when needed.

## Selection rule

Within each family:

1. Protected terms are retained first and bypass the family cap.
2. Remaining terms are ordered by `--rank_col`; smaller values are better.
3. With the default `--prefer_simple_shapes`, all candidates within
   `--tie_tolerance` of the best remaining rank form a tie band.
4. The simplest shapes in that band fill the available family slots.
5. If slots remain, the next best-ranked terms fill them.

The default caps are one shape per invariant family and one shape per strain
family. Protected terms can make a family exceed its cap; this is reported in
the audit log.

The invariant simplicity order is:

```text
Linear < Quadratic < Cubic < xlog linear < xlog quadratic
       < Exp linear < Exp quadratic
```

For strain monomials, total polynomial degree is the complexity measure. The
complexity rule is only a tie-breaker within the configured rank tolerance; it
does not replace the Sobol ranking globally.

## Files

- `collapse_term_families.py`: self-contained structural family-collapse
  implementation.
- `requirements.txt`: lightweight dependency set.

## Environment

Python 3.10 or later is recommended. The sensitivity or decorrelation virtual
environment can be reused. To create a separate environment from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Run from the repository root:

```bash
python "3. Model Selection/3. Family collapse/collapse_term_families.py" \
  --input "3. Model Selection/2. Decorrelation/results/ALL_VEINS/ranked_terms_sobol_pruned.csv" \
  --out_dir "3. Model Selection/3. Family collapse/results/ALL_VEINS" \
  --rank_col mean_ST_rank_primary \
  --max_shapes_per_invariant 1 \
  --max_shapes_per_strain_family 1 \
  --prefer_simple_shapes \
  --tie_tolerance 5
```

Use `--no-prefer_simple_shapes` to retain the best-ranked terms without the
simplicity tie-breaker. Set a family cap greater than one only when multiple
shapes per scalar family are scientifically intended.

## Optional family mergers

Two mergers are available but disabled by default because they require
additional constitutive assumptions.

### Helical families

```bash
--merge_helical_i4
```

This treats `I4f` and `I4s` as one logical `I4h` family during the cap. It is
appropriate only when the two diagonal fiber families are intended to be
exchangeable—for example, symmetric `+/-` helix angles with matching
dispersion and recruitment assumptions. The selected output row retains its
original `I4f` or `I4s` identity; `collapse_family=I4h` records the logical
merger.

### Direction-matched strain and invariant families

```bash
--merge_direction_strain_pairs
```

This folds pure circumferential strain terms into `I4smc` and pure axial strain
terms into `I4el`. On a raw aligned-fiber grid without dispersion or
recruitment,

```text
I4smc - 1 = lambda_theta^2 - 1 = 2 E_theta
I4el  - 1 = lambda_z^2     - 1 = 2 E_z
```

The equivalence does not generally hold after family-specific dispersion,
orientation, or recruitment transformations. Enable this merger only when its
assumptions match the model being selected.

Both mergers can be enabled together, but each must be reported as part of the
model-selection configuration.

## Protected terms

The decorrelation output carries its `protected` column into this stage. A
protected term is always retained even if another term from the same family is
better ranked or simpler. Protection therefore overrides the family cap and
should be used only for a documented mechanistic requirement.

## Outputs

| Output | Contents |
| --- | --- |
| `ranked_terms_family_collapsed.csv` | Final compact term set with all original Sobol/decorrelation metadata |
| `family_collapse_log.csv` | One decision record for every decorrelated input term |
| `family_collapse_summary.txt` | Configuration, family representation, retained terms, and removals |

The final CSV adds:

- `collapse_family`: logical family used for the cap;
- `family_rank_used`: value from the selected ranking column; and
- `shape_complexity`: tie-breaker complexity score.

The log identifies the selected representative or representatives for each
family and records whether a term was retained, protected, or removed.

## Robustness and interpretation

- Compare results with and without the simplicity preference.
- Repeat the collapse over plausible tie tolerances, such as `0`, `2`, and `5`.
- Report any family cap above one and every protected term.
- Do not interpret the retained Sobol rank as a fitted coefficient.
- Do not treat rank-based collapse as proof that the selected shape reproduces
  the sum of all removed family responses. That claim requires a separate
  response-level approximation analysis.
- Refit the final equation to the experimental data in `4. Evaluation`; the
  coefficients from the original overcomplete CANN do not automatically carry
  over to the collapsed equation.
