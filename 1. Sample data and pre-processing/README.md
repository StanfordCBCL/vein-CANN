# Sample data and pre-processing

This folder contains the jugular vein 106 example, its raw measurements, processed training inputs, and the copied conversion scripts. The format descriptions and protocol mappings are adapted from the original VeinCANN `Input_data/README.md` and checked against the scripts in this folder.

## What is included

```text
1. Sample data and pre-processing/
├── README.md
├── sample data/
│   ├── JV_VG_106_19Jun20_Fl.mat
│   ├── JV_VG_106_19Jun20_Pd.mat
│   ├── JV_VG_106_19Jun20_unloading_data.xlsx
│   ├── 106_fl10.csv
│   ├── 106_fl60.csv
│   ├── 106_fl100.csv
│   ├── 106_fl140.csv
│   ├── 106_pd95.csv
│   ├── 106_pd100.csv
│   └── 106_pd105.csv
└── Data_processing_scripts/
    ├── split_mat_to_excel.py
    ├── combined_mat_to_excel.py
    ├── input_data_processing_ivc.py
    └── mat_to_csv/
        ├── combined_mat_to_curve_csv.m
        ├── compute_fl_stress_stretch.m
        └── compute_pd_stress_stretch.m
```

`FL` denotes force-length tests and `PD` pressure-diameter tests. The MAT files contain separate FL and PD measurements for the same specimen. The Excel workbook and seven curve-level CSV files are already processed inputs accepted by `neural_net.py`; raw MAT files cannot be passed directly to training.

## Use the processed data directly

No conversion is required to train with the included workbook or CSVs. Run commands from the repository root, with the training dependencies installed. For a short CSV trial:

```bash
python "2. CANN training/neural_net.py" \
  --input_mode clean_csv \
  --clean_data_root "1. Sample data and pre-processing/sample data" \
  --path_root "4. Evaluation/Fitted_results" \
  --filename jugular_106_trial \
  --report_label jugular_106 \
  --objective stress \
  --epochs 100 --n_restarts 1
```

The loader selects PD/FL CSV filenames and ignores the MAT and Excel files in the same folder. This trial checks the training workflow; use the full settings in the [CANN training README](../2.%20CANN%20training/README.md) for the documented jugular 106 fit. That README also lists dependencies, hyperparameters, and outputs.

## Regenerate Excel from the raw MAT pair

Use `Data_processing_scripts/split_mat_to_excel.py` for the included pair. This converter needs Python, SciPy, pandas, and an Excel writer such as openpyxl. In your chosen Python environment:

```bash
python -m pip install scipy pandas openpyxl
```

1. Open `Data_processing_scripts/split_mat_to_excel.py` and change the three values under `EDIT THESE PATHS` to the following. These relative paths assume you run from the repository root:

   ```python
   FL_PATH = "1. Sample data and pre-processing/sample data/JV_VG_106_19Jun20_Fl.mat"
   PD_PATH = "1. Sample data and pre-processing/sample data/JV_VG_106_19Jun20_Pd.mat"
   OUTPUT_DIR = "1. Sample data and pre-processing/reprocessed"
   ```

2. Run the converter from the repository root:

   ```bash
   python "1. Sample data and pre-processing/Data_processing_scripts/split_mat_to_excel.py"
   ```

3. Read the printed output path. The script creates `OUTPUT_DIR` and writes `<sample_name>_unloading_data.xlsx`; it derives `sample_name` from the FL file's `name` field when present, otherwise from its filename with the trailing `_Fl` removed. The separate `reprocessed` folder keeps the included workbook intact.

4. Train with the resulting workbook using `--input_mode excel --sample_file "<printed output path>"`. Provide `--path_root` and `--filename` as in the example above.

The copied script still contains its original hard-coded paths until you change them. It has no command-line path arguments; passing filenames after the script name does not configure the conversion.

### What the Excel converter does

It reads only `passive.unloading`, extracting `biaxial_Fl` from the FL MAT file and `biaxial_Pd` from the PD MAT file. Each raw test array must have five columns in this order:

| Raw column | Excel column | Expected units |
| --- | --- | --- |
| 1 | `Pressure_mmHg` | mmHg |
| 2 | `Outer_Diameter_microm` | micrometres |
| 3 | `Force_g` | grams |
| 4 | `Axial_Stretch` | Dimensionless |
| 5 | `Loaded_Length_mm` | mm |

It copies unloaded geometry `H`, `L`, `OD`, and `PHI` from each file onto the corresponding rows and adds `test_case`. It does not perform unit conversion or smooth the measurements. For Excel training, `H` and `OD` must be in micrometres, `L` in mm, and `PHI` in degrees. Check the raw units before applying this converter to modified data.

| MATLAB test field | FL label | PD label |
| --- | --- | --- |
| `test1` | `p10` | `invivo-5%` |
| `test2` | `p60` | `invivo` |
| `test3` | `p100` | `invivo+5%` |
| `test4` | `p140` | — |

The result is one workbook with `FL` and `PD` sheets. Missing test fields are reported and skipped. Stress/stretch calculations happen later in `neural_net.py`: it converts pressure and force units, reconstructs loaded inner geometry under constant wall volume, adds the pressure contribution to vessel force, and computes stresses and stretches.

## CSV processing and the MATLAB scripts

The seven included CSVs are ready for `--input_mode clean_csv`. Each file represents one protocol and contains these columns:

| Columns | Meaning / units |
| --- | --- |
| `lambda_theta`, `lambda_z` | Circumferential and axial stretches, dimensionless |
| `sigma_theta_kPa`, `sigma_z_kPa` | Circumferential and axial stresses, kPa |
| `pressure_kPa` | Pressure, kPa |
| `od_mm`, `length_mm` | Loaded outer diameter and length, mm |
| `F_transducer_mN`, `F_vessel_mN` | Measured and pressure-corrected vessel force, mN |

The copied `mat_to_csv/combined_mat_to_curve_csv.m` is a MATLAB batch exporter for **combined** MAT structures containing geometry plus both `biaxial_Pd` and `biaxial_Fl` in one measurement block. It does not directly accept the included split MAT pair. There is no split-pair-to-CSV driver among the copied scripts; use the supplied CSVs or the split-pair-to-Excel workflow above for this sample.

For a compatible combined input, the batch script requires editing its `specimens` list and output location, keeping `analysis = '2D'`, and making both helper functions available on the MATLAB path. It uses PD tags `[-5, 0, 5]` and FL tags `[10, 60, 100, 140]`, then exports one CSV per protocol and combined PD/FL plots. Its default output folder is `out` under MATLAB's working directory.

The helpers assume raw `H`, `OD`, and loaded diameter are in **mm**, while the split Excel converter labels the corresponding diameter/thickness values as **micrometres** without rescaling them. The MATLAB helpers also use outer-radius circumferential stretch, whereas Excel training computes mid-wall circumferential stretch. Treat these as separate processing routes; they are not guaranteed to regenerate numerically identical stress/stretch curves.

The MATLAB exporter writes the nine columns above but does not add `unloaded_od_mm`, `unloaded_h_mm`, or `unloaded_length_mm`. Those additional unloaded-geometry columns are needed for `--objective obj6`; use `--objective stress` with the included CSVs.

## Role of each copied script

| Script | Purpose and use here |
| --- | --- |
| `split_mat_to_excel.py` | Convert the jugular 106 split MAT pair to one Excel workbook; configure paths before running. |
| `combined_mat_to_excel.py` | Legacy combined-MAT converter with hard-coded input/output paths; not the route for the included split pair. |
| `input_data_processing_ivc.py` | Combined-MAT converter with command-line paths and explicit mm-to-micrometre conversion; not the route for the included split pair. |
| `mat_to_csv/combined_mat_to_curve_csv.m` | Batch export compatible combined MAT data to protocol CSVs and plots; requires path/specimen configuration. |
| `mat_to_csv/compute_fl_stress_stretch.m` | MATLAB helper for FL geometry, stress, and stretch. |
| `mat_to_csv/compute_pd_stress_stretch.m` | MATLAB helper for PD geometry, stress, and stretch. |

## Check the processed inputs

- Confirm the workbook has `FL` and `PD` sheets and the columns/units described above. Geometry should be positive, with unloaded `OD > 2*H`.
- Confirm protocol labels match the Excel training selection. In CSV mode, all discovered protocol curves are used for training.
- Check stresses and stretches for non-finite or complex values. The CSV loader drops invalid required entries and reports the number dropped; the Excel converter does not clean them.
- Inspect stress-stretch curves and the number of retained points. The current MATLAB PD helper filters points with reconstructed inner-diameter-squared below `1` (in its mm-based units); account for that filter when comparing processing routes.
- Keep regenerated CSVs in a separate folder and point training to that folder. CSV discovery is recursive, so keeping duplicate curves beneath the selected data root would include both copies.
