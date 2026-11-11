# Sample data and pre-processing

This folder contains the jugular vein 106 example, its raw measurements, processed training inputs, and the copied conversion scripts. The format descriptions and protocol mappings are adapted from the original VeinCANN `Input_data/README.md` and checked against the scripts in this folder.

## What is included

The `sample data` folder contains experimental data for jugular vein 106 and a processed Excel workbook:

| Format | Contents | Use |
| --- | --- | --- |
| MATLAB (`.mat`) | Experimental measurements and specimen geometry in separate force-length (FL) and pressure-diameter (PD) files. | Convert the pair to Excel before training. |
| Excel (`.xlsx`) | One workbook with `FL` and `PD` sheets containing measurements and specimen geometry. | Load directly into `neural_net.py`, which calculates stresses and mid-wall stretches before fitting. |

The raw files are `JV_VG_106_19Jun20_Fl.mat` and `JV_VG_106_19Jun20_Pd.mat`; the included processed workbook is `JV_VG_106_19Jun20_unloading_data.xlsx`. The `Data_processing_scripts` folder contains the Python MAT-to-Excel converters.

The included workbook is ready for training. See the [CANN training README](../2.%20CANN%20training/README.md) for training commands, dependencies, hyperparameters, and outputs.

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

4. Open the resulting workbook and confirm that it contains `FL` and `PD` sheets with the columns and protocol labels listed below.

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

The result is one workbook with `FL` and `PD` sheets. Missing test fields are reported and skipped. The workbook stores measurements and geometry; stresses and mid-wall stretches are calculated later by the training script.

## Role of each pre-processing script

| Script | Purpose and use here |
| --- | --- |
| `split_mat_to_excel.py` | Convert the jugular 106 split MAT pair to one Excel workbook; configure paths before running. |
| `combined_mat_to_excel.py` | Legacy combined-MAT-to-Excel converter with hard-coded input/output paths; not the route for the included split pair. |
| `input_data_processing_ivc.py` | Combined-MAT-to-Excel converter with command-line paths and explicit mm-to-micrometre conversion; not the route for the included split pair. |

## Check the processed workbook

- Confirm the workbook has `FL` and `PD` sheets and the columns/units described above. Geometry should be positive, with unloaded `OD > 2*H`.
- Confirm each measurement row has the appropriate `test_case` label from the table above.
- Check measurement columns for missing or non-finite values. The Excel converter organizes measurements without cleaning them.
