# This script converts a pair of .mat files (Fl and Pd tests, from the same
# sample) into a single Excel file with two sheets ("FL" and "PD"), in the
# format expected by the CANN processing code.
#
# Only the UNLOADING data (passive.unloading) is used, matching the original
# single-file workflow.
#
# Set FL_PATH, PD_PATH, and OUTPUT_DIR below, then just run:
#   python mat_to_excel.py

import os
import re

import pandas as pd
import scipy.io as sio

# ---------------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------------
FL_PATH = "/Users/divya/CANN/VEIN/JV_dataset/JV_VG_Stanford_143_Fl.mat"
PD_PATH = "/Users/divya/CANN/VEIN/JV_dataset/JV_VG_Stanford_143_Pd.mat"
OUTPUT_DIR = "/Users/divya/CANN/VEIN/JV_dataset"
# ---------------------------------------------------------------------------

# Column names for each test (same order as in the raw arrays)
COLUMNS = [
    "Pressure_mmHg",
    "Outer_Diameter_microm",
    "Force_g",
    "Axial_Stretch",
    "Loaded_Length_mm",
]

# Mappings from MATLAB test field name -> descriptive label
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


def get_sample_name(mat_contents, fallback_path):
    """Derive a clean sample name, stripping trailing _Fl/_Pd if present."""
    name = str(mat_contents.get("name", "")).strip()
    if not name:
        name = os.path.splitext(os.path.basename(fallback_path))[0]
    name = re.sub(r"_(Fl|Pd)$", "", name, flags=re.IGNORECASE)
    return name


def build_dataframe(biaxial_struct, mapping, metadata):
    """Build a long-format dataframe from a biaxial_Fl/biaxial_Pd struct."""
    dfs = []
    for test_field, label in mapping.items():
        if not hasattr(biaxial_struct, test_field):
            # Not every sample necessarily has every test (e.g. missing test4)
            print(f"  Warning: {test_field} ({label}) not found, skipping.")
            continue
        test_array = getattr(biaxial_struct, test_field)
        df = pd.DataFrame(test_array, columns=COLUMNS)
        for k, v in metadata.items():
            df[k] = v
        df["test_case"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_unloading(mat_path, biaxial_field):
    """Load the passive.unloading struct and pull out metadata + biaxial data."""
    mat_contents = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    passive = mat_contents["passive"]
    unloading = passive.unloading

    metadata = {
        "H": unloading.H,
        "L": unloading.L,
        "OD": unloading.OD,
        "PHI": unloading.PHI,
    }

    biaxial_struct = getattr(unloading, biaxial_field)
    return mat_contents, metadata, biaxial_struct


def convert_pair(fl_path, pd_path, output_dir=None):
    print(f"Reading FL file: {fl_path}")
    fl_contents, fl_metadata, fl_biaxial = load_unloading(fl_path, "biaxial_Fl")

    print(f"Reading PD file: {pd_path}")
    pd_contents, pd_metadata, pd_biaxial = load_unloading(pd_path, "biaxial_Pd")

    sample_name = get_sample_name(fl_contents, fl_path)

    fl_df = build_dataframe(fl_biaxial, FL_MAPPING, fl_metadata)
    pd_df = build_dataframe(pd_biaxial, PD_MAPPING, pd_metadata)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(fl_path))
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{sample_name}_unloading_data.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        fl_df.to_excel(writer, sheet_name="FL", index=False)
        pd_df.to_excel(writer, sheet_name="PD", index=False)

    print(f"Excel file written successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    convert_pair(FL_PATH, PD_PATH, OUTPUT_DIR)