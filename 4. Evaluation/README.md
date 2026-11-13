# Evaluation: parameter estimation for candidate SEFs

`pure_sedf_nnls_refit_per_animal_fung.py` fits a selected strain-energy function (SEF) to each specimen's biaxial stress data. 

## Method

For each specimen and candidate equation, Latin hypercube sampling explores nonlinear recruitment, fiber-angle, dispersion, and exponential-curvature parameters. Each nonlinear candidate defines a stress-basis matrix; weighted nonnegative least squares (NNLS) determines its linear coefficients. The best candidate can then initialize bounded L-BFGS-B refinement, with NNLS solved again at each nonlinear evaluation. This uses NumPy/SciPy and does not retrain a neural network.

## Required inputs

- `--animal_root`: a saved CANN results folder or a root containing specimen results. Keep `metrics.txt`, `learned_recruitment.csv`, `final_equation.txt`, and `trained_weights.txt` together, plus `learned_curve_offsets.csv` when available.
- `--ranked_terms`: a candidate-equation CSV with `group`, `inv`, and `term` columns. `--ks` controls the number of terms selected; ranking uses `rank_robust`, then `robust_score`, or otherwise row order. To evaluate one complete equation, supply only its terms and set `--ks` to its term count.
- The original experimental workbook, located through `sample_file` in `metrics.txt`. Excel loading calculates mid-wall stretch. See the [sample data README](<../1.%20Sample%20data%20and%20pre-processing/README.md>) for workbook structure.

**Excel setup:** the copied script requires `input_mode: excel` in the `HYPERPARAMETERS` section of each specimen's `metrics.txt`, along with a valid `sample_file` path. The current Excel-only trainer does not write `input_mode`; add this metadata entry before evaluation. Use an absolute workbook path or run from the repository root when the recorded path is relative. Without the entry, the upstream script defaults to its legacy data loader.

Use Python 3.10 or later with NumPy, pandas, SciPy, and openpyxl; matplotlib is needed for plots. The training environment file has older version pins and has not been validated for this evaluation script.

## Example

Run from the repository root. Replace `path/to/...` with existing inputs and adjust `--ks 8` to the candidate's term count; candidate equations and fitted training results are not bundled with this script.

```bash
python "4. Evaluation/pure_sedf_nnls_refit_per_animal_fung.py" \
  --animal_root "path/to/training_results" \
  --ranked_terms "path/to/candidate_equation.csv" \
  --out_root "4. Evaluation/refit_results/candidate_01" \
  --ks 8 \
  --lhs_samples 4096 --lhs_seed 12345 \
  --lhs_recruitment_frac 0.20 --lhs_shape_frac 0.50 \
  --lhs_tau_frac 0.50 --lhs_kappa_delta 0.05 --lhs_angle_deg 15 \
  --exp_coeff_lo 1e-4 --exp_coeff_hi 5.0 \
  --exp_coeff_invariants I4el I4smc \
  --lhs_objective weighted_RMSE \
  --refine_top_k 1 --refine_method L-BFGS-B --refine_maxiter 100 \
  --make_plots
```

`--lhs_samples 4096` evaluates the trained baseline plus 4095 LHS vectors. `I4el` and `I4smc` identify the axial and circumferential curvature contributions. Recruitment, shape, and dispersion bounds include the script's admissibility limits; angles use the trained value ±15° without additional angle-specific clipping. Upper recruitment stretches are adjusted after sampling to remain above lower recruitment stretches. These are the bounds qualifications discussed for this script.

## Outputs

| Output                                                | Contents                                                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Per-specimen `k<N>` folders                         | Fitted coefficients, fit metrics, stress predictions, and equation reports for each term count. |
| `lhs_candidate_metrics.csv`, `polish_records.csv` | Sampling results and local-refinement records when generated.                                   |
| `selected_nonlinear_parameters.json`                | Selected nonlinear parameters, terms, coefficients, and stress scales.                          |
| `pure_nnls_summary_all_animals.csv`                 | Combined fit statistics across specimens and term counts.                                       |
| Fit figures                                           | Stress-stretch plots when `--make_plots` is enabled.                                          |

Use a separate `--out_root` for each candidate equation. The script retains the upstream stress/derivative conventions; copying it does not validate those conventions or reproduce a published fit.
