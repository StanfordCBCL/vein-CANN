# Convert IVC combined .mat files to CANN-ready Excel workbooks.
# Each output workbook contains FL and PD sheets.
#
# Supported .mat structures:
#   - top-level "passive" struct
#   - top-level "S" wrapper containing "passive"

import argparse
import os

import pandas as pd
import scipy.io as sio


COLUMNS = [
    "Pressure_mmHg",
    "Outer_Diameter_microm",
    "Force_g",
    "Axial_Stretch",
    "Loaded_Length_mm",
]

FL_MAPPING = {
    "test1": "p10",
    "test2": "p60",
    "test3": "p100",
    "test4": "p140",
}

PD_MAPPING = {
    "test1": "invivo-5%",
    "test2": "invivo",
    "test3": "invivo+5%",
}


def _passive_struct(mat_contents):
    if "S" in mat_contents and hasattr(mat_contents["S"], "passive"):
        return mat_contents["S"].passive
    return mat_contents["passive"]


def _metadata_from_unloading(unloading):
    # H and OD are stored in mm in these .mat files; CANN Excel input expects microns.
    return {
        "H": unloading.H * 1000,
        "L": unloading.L,
        "OD": unloading.OD * 1000,
        "PHI": unloading.PHI,
    }


def _build_dataframe(struct, mapping, metadata):
    dfs = []
    for test_field, label in mapping.items():
        test_array = getattr(struct, test_field)
        df = pd.DataFrame(test_array, columns=COLUMNS)
        df["Outer_Diameter_microm"] = df["Outer_Diameter_microm"] * 1000
        for key, value in metadata.items():
            df[key] = value
        df["test_case"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def convert_mat_to_excel(mat_file_path, output_dir):
    mat_contents = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    passive = _passive_struct(mat_contents)
    unloading = passive.unloading
    metadata = _metadata_from_unloading(unloading)

    fl_df = _build_dataframe(unloading.biaxial_Fl, FL_MAPPING, metadata)
    pd_df = _build_dataframe(unloading.biaxial_Pd, PD_MAPPING, metadata)

    stem = os.path.splitext(os.path.basename(mat_file_path))[0]
    output_path = os.path.join(output_dir, f"unloading_data_{stem}.xlsx")
    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        fl_df.to_excel(writer, sheet_name="FL", index=False)
        pd_df.to_excel(writer, sheet_name="PD", index=False)

    print(f"Excel file written successfully: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one or more IVC combined .mat files to CANN-ready Excel."
    )
    parser.add_argument("mat_files", nargs="+", help="Input IVC .mat files.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder. Defaults to each input .mat file's folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for mat_file_path in args.mat_files:
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(mat_file_path))
        convert_mat_to_excel(mat_file_path, output_dir)


if __name__ == "__main__":
    main()
