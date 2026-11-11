# This code converts mat file with cleaned data to excel files that in a format that can be processed by the CANN code.
# The excel file contains 2 sheets - FL and PD.
import scipy.io as sio
import pandas as pd

# Load the .mat file
mat_file_path = "/Users/divya/CANN/VEIN/Shortlisted_grafts_wk28/486_3075_EJ_graft_gaussian_window_9_truncate.mat"
mat_contents = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)

# Extract unloading struct
passive = mat_contents["passive"]
unloading = passive.unloading

# Metadata
metadata = {
    "H": unloading.H,
    "L": unloading.L,
    "OD": unloading.OD,
    "PHI": unloading.PHI,
}

# Column names for each test
columns = [
    "Pressure_mmHg", 
    "Outer_Diameter_microm", 
    "Force_g", 
    "Axial_Stretch", 
    "Loaded_Length_mm"
]

# Mappings for test labels
fl_mapping = {
    "test1": "p10",
    "test2": "p60",
    "test3": "p100",
    "test4": "p140",
}
pd_mapping = {
    "test1": "invivo-5%",
    "test2": "invivo",
    "test3": "invivo+5%",
}

def build_dataframe(struct, mapping, metadata):
    dfs = []
    for test_field, label in mapping.items():
        # Extract the numeric array
        test_array = getattr(struct, test_field)
        df = pd.DataFrame(test_array, columns=columns)
        # Add metadata + label
        for k, v in metadata.items():
            df[k] = v
        df["test_case"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Build both dataframes
fl_df = build_dataframe(unloading.biaxial_Fl, fl_mapping, metadata)
pd_df = build_dataframe(unloading.biaxial_Pd, pd_mapping, metadata)

# Write to Excel
with pd.ExcelWriter("Shortlisted_grafts_wk28/unloading_data_bethany_graft.xlsx") as writer:
    fl_df.to_excel(writer, sheet_name="FL", index=False)
    pd_df.to_excel(writer, sheet_name="PD", index=False)

print("Excel file 'unloading_data.xlsx' written successfully!")
