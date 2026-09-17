import os
import pandas as pd
import numpy as np


# ==========================================================
# PATHS
# ==========================================================

# Main LearnLift project folder
project_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift"

# Original dataset path
input_file = os.path.join(
    project_dir,
    "Dataset",
    "learnlift_updated_dataset_raw.csv"
)

# Output folder
output_dir = os.path.join(
    project_dir,
    "Dataset"
)

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Output file
output_file = os.path.join(
    output_dir,
    "clean_target_encode.csv"
)


# ==========================================================
# Load LearnLift Dataset
# Original dataset is NOT modified
# ==========================================================

df = pd.read_csv(input_file)

# Create a copy
data = df.copy()


print("Dataset Loaded Successfully")
print("Original Dataset Shape:", data.shape)

print("-" * 50)


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(
    include=["object", "string"]
).columns:

    data[col] = data[col].str.strip()


print("Leading and trailing spaces removed.")


# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("\nMissing Values Before Cleaning:")

print(data.isnull().sum())

print("-" * 50)


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

duplicate_count = data.duplicated().sum()

data = data.drop_duplicates()

print(
    "\nDuplicate Records Removed:",
    duplicate_count
)


# ==========================================================
# 4. Identify Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\nNumerical Columns:")
print(num_cols)


print("\nCategorical Columns:")
print(cat_cols)


# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================

for col in num_cols:

    mean_value = data[col].mean()

    data[col] = data[col].fillna(
        mean_value
    )


print("\nMissing numerical values filled using mean.")


# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

for col in cat_cols:

    mode_values = data[col].mode()

    if len(mode_values) > 0:

        mode_value = mode_values[0]

        data[col] = data[col].fillna(
            mode_value
        )


print("Missing categorical values filled using mode.")


# ==========================================================
# 7. Select Target Column
# ==========================================================

target_column = "dropout"


print("\nTarget Column:")
print(target_column)


# Check whether target column exists
if target_column not in data.columns:

    raise ValueError(
        f"Target column '{target_column}' "
        "was not found in the dataset."
    )


# ==========================================================
# 8. Apply Target Encoding Using Pandas
# ==========================================================

target_encoded_df = pd.DataFrame()


for col in cat_cols:

    # Do not encode target column itself
    if col != target_column:

        mean_encoding = (
            data.groupby(col)[target_column]
            .mean()
        )

        target_encoded_df[
            "Target_" + col
        ] = data[col].map(
            mean_encoding
        )


# ==========================================================
# 9. Merge Numerical Columns
#    and Target Encoded Columns
# ==========================================================

final_output = pd.concat(
    [
        data[num_cols].reset_index(drop=True),
        target_encoded_df.reset_index(drop=True)
    ],
    axis=1
)


# ==========================================================
# 10. Check Missing Values After Processing
# ==========================================================

print("\nMissing Values After Cleaning:")

print(
    final_output.isnull().sum()
)


total_missing_after = (
    final_output.isnull().sum().sum()
)


print(
    "\nTotal Missing Values:",
    total_missing_after
)


# ==========================================================
# 11. Display Final Dataset Information
# ==========================================================

print("\nFinal Dataset Shape:")

print(
    final_output.shape
)


print("\nFirst 5 Rows After Target Encoding:")

print(
    final_output.head()
)


# ==========================================================
# 12. Save Final Result
# ==========================================================

final_output.to_csv(
    output_file,
    index=False
)


# ==========================================================
# Final Message
# ==========================================================

print("\n" + "=" * 50)

print(
    "LearnLift Target Encoding Completed Successfully"
)

print(
    "Original dataset is NOT modified"
)

print(
    "Target Variable:",
    target_column
)

print(
    "Output file:"
)

print(
    output_file
)

print("=" * 50)