# Sample data and pre-processing

## What is included

The `sample data` folder contains inflation-extension biaxial testing data [1] for jugular vein 106 [2] and a processed Excel workbook. The experimental protocol included 3 force-length tests and 4 pressure-diameter tests as described in section 2.1 of [3].

| Format            | Contents                                                                                                        | Use                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| MATLAB (`.mat`) | Experimental measurements and specimen geometry in separate force-length (FL) and pressure-diameter (PD) files. | Convert the pair to Excel before training.                                                             |
| Excel (`.xlsx`) | One workbook with `FL` and `PD` sheets containing measurements and specimen geometry.                       | Load directly into `neural_net.py`, which calculates stresses and mid-wall stretches before fitting. |

The raw files are `JV_VG_106_19Jun20_Fl.mat` and `JV_VG_106_19Jun20_Pd.mat`; the included processed workbook is `JV_VG_106_19Jun20_unloading_data.xlsx`. The `Data_processing_scripts` folder contains the Python MAT-to-Excel converters.

The included processed excel workbook is ready for training. See the [CANN training README](<../2.%20CANN%20training/README.md>) for training commands, dependencies, hyperparameters, and outputs.

## Role of each pre-processing script

- `split_mat_to_excel.py` combines FL and PD measurements stored in two separate MAT files into one Excel workbook with `FL` and `PD` sheets. **We use this script for the jugular vein dataset because its FL and PD measurements are split across two files.**
- `combined_mat_to_excel.py` converts a single MAT file containing both FL and PD measurements into one Excel workbook; input and output paths are set inside the script.
- `input_data_processing_ivc.py` converts combined MAT files with diameter and thickness measurements in millimetres into Excel workbooks, converting those values to micrometres as required by the training input format; input files and the output folder are supplied through command-line arguments.

## Regenerate Excel from the raw MAT pair

If you want to convert the .mat files to excel that can be used for training, use `Data_processing_scripts/split_mat_to_excel.py` for the included pair of data. In your chosen Python environment:

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

### What the Excel converter does

It reads only `passive.unloading`, extracting `biaxial_Fl` from the FL MAT file and `biaxial_Pd` from the PD MAT file. Each raw test array must have five columns in this order:

| Raw column | Excel column              | Expected units |
| ---------- | ------------------------- | -------------- |
| 1          | `Pressure_mmHg`         | mmHg           |
| 2          | `Outer_Diameter_microm` | micrometres    |
| 3          | `Force_g`               | grams          |
| 4          | `Axial_Stretch`         | Dimensionless  |
| 5          | `Loaded_Length_mm`      | mm             |

It copies unloaded geometry `H`, `L`, `OD`, and `PHI` from each file onto the corresponding rows and adds `test_case`. It does not perform unit conversion or smooth the measurements. For Excel training, `H` and `OD` must be in micrometres, `L` in mm, and `PHI` in degrees. Check the raw units before applying this converter to modified data.

| MATLAB test field | FL label | PD label      |
| ----------------- | -------- | ------------- |
| `test1`         | `p10`  | `invivo-5%` |
| `test2`         | `p60`  | `invivo`    |
| `test3`         | `p100` | `invivo+5%` |
| `test4`         | `p140` | —            |

The result is one workbook with `FL` and `PD` sheets. Missing test fields are reported and skipped. The workbook stores measurements and geometry; stresses and mid-wall stretches are calculated later by the training script.

## Check the processed workbook

- Confirm the workbook has `FL` and `PD` sheets and the columns/units described above. Geometry should be positive, with unloaded `OD > 2*H`.
- Confirm each measurement row has the appropriate `test_case` label from the table above.
- Check measurement columns for missing or non-finite values. The Excel converter organizes measurements without cleaning them.

## References

1. Ferruzzi, J., Bersi, M.R., Humphrey, J.D.: Biomechanical phenotyping of central arteries in health and disease: Advantages of and methods for murine models. *Annals of Biomedical Engineering* **41**(7), 1311–1330 (2013). [DOI](https://doi.org/10.1007/s10439-013-0799-1).
2. Ramachandra, A.B., Wang, H., Wnorowski, A., Schwarz, E.L., Pickering, J., Heiler, J.C., Lucian, H.J., Hironaka, C.E., Tran, N.A., Liu, Y., Khan, M.O., Obafemi, O., Tada, Y., Kahn, A.M., Sayed, N., Wu, J.C., Humphrey, J.D., Boyd, J.H., Marsden, A.L.: Biodegradable external wrapping promotes favorable adaptation in an ovine vein graft model. *Acta Biomaterialia* **151**, 414–425 (2022). [DOI](https://doi.org/10.1016/j.actbio.2022.08.029).
3. Bazzi, M., Adil, D., Mays, G., Kuhl, E., Breuer, C.K., Humphrey, J.D., Marsden, A.L.: Data-Driven Discovery of a Hyperelastic Strain-Energy Function for Veins Using Constitutive Artificial Neural Networks. *arXiv preprint* arXiv:2610.15229 (2026). [arXiv](https://arxiv.org/abs/2610.15229).
