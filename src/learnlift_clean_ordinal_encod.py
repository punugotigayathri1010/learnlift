import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# PATHS
# ==========================================================

# Main LearnLift project folder
project_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift"

# Dataset path
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
    "clean_ordinal_encode.csv"
)


# ==========================================================
# Load LearnLift Dataset
# Original dataset will NOT be modified
# ==========================================================

df = pd.read_csv(input_file)

# Create a copy for processing
data = df.copy()

print("Dataset Loaded Successfully.")
print("Original Dataset Shape:", data.shape)

print("-" * 60)


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

print("-" * 60)


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before_duplicates = data.shape[0]

data = data.drop_duplicates()

after_duplicates = data.shape[0]

duplicate_removed = (
    before_duplicates - after_duplicates
)

print(
    "\nDuplicate Records Removed:",
    duplicate_removed
)


# ==========================================================
# 4. Separate Numerical and Categorical Columns
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

if len(num_cols) > 0:

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    data[num_cols] = num_imputer.fit_transform(
        data[num_cols]
    )


print("\nMissing numerical values filled using mean.")


# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

if len(cat_cols) > 0:

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


print("Missing categorical values filled using mode.")


# ==========================================================
# 7. Apply Ordinal Encoding
# ==========================================================

if len(cat_cols) > 0:

    ordinal_encoder = OrdinalEncoder()

    encoded_values = ordinal_encoder.fit_transform(
        data[cat_cols]
    )


    encoded_df = pd.DataFrame(
        encoded_values,
        columns=[
            "Ordinal_" + col
            for col in cat_cols
        ]
    )


    # Keep numerical columns and encoded columns
    final_output = pd.concat(
        [
            data[num_cols].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

else:

    final_output = data.copy()


# ==========================================================
# 8. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")

print(
    final_output.isnull().sum()
)


total_missing = (
    final_output.isnull().sum().sum()
)

print(
    "\nTotal Missing Values:",
    total_missing
)


# ==========================================================
# 9. Display Final Dataset Information
# ==========================================================

print("\nFinal Dataset Shape:")

print(
    final_output.shape
)


print("\nFirst 5 Rows After Ordinal Encoding:")

print(
    final_output.head()
)


# ==========================================================
# 10. Save Final Result
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
    "Original LearnLift dataset is NOT modified."
)

print(
    "Ordinal Encoding completed successfully."
)

print(
    "Target Variable: dropout"
)

print(
    "Output file:"
)

print(
    output_file
)

print("=" * 50)