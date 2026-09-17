import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# LEARNLIFT - FINAL PREPROCESSING
# Duplicate Removal, Missing Value Handling,
# Text Cleaning, Label Encoding and Standard Scaling
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

project_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift"

input_file = os.path.join(
    project_dir,
    "Dataset",
    "learnlift_updated_dataset_raw.csv"
)

output_file = os.path.join(
    project_dir,
    "Dataset",
    "final_preprocess.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(input_file)

processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)


# ============================================================
# REMOVE DUPLICATE RECORDS
# ============================================================

duplicates_before = processed_df.shape[0]

processed_df.drop_duplicates(
    inplace=True
)

duplicates_removed = (
    duplicates_before - processed_df.shape[0]
)

print(
    "Duplicate Records Removed:",
    duplicates_removed
)


# ============================================================
# IDENTIFY NUMERICAL COLUMNS
# ============================================================

numeric_cols = processed_df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# TARGET COLUMN
# ============================================================

target_column = "dropout"


# ============================================================
# NUMERICAL MISSING VALUE HANDLING
# ============================================================

for col in numeric_cols:

    # Do not impute target column
    # because dropout is the prediction target

    if col != target_column:

        processed_df[col] = (
            processed_df[col].fillna(
                processed_df[col].median()
            )
        )


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_cols = processed_df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


# ============================================================
# CATEGORICAL MISSING VALUE HANDLING
# ============================================================

for col in categorical_cols:

    mode_values = processed_df[col].mode()

    if len(mode_values) > 0:

        processed_df[col] = (
            processed_df[col].fillna(
                mode_values.iloc[0]
            )
        )


# ============================================================
# CLEAN TEXT DATA
# ============================================================

for col in categorical_cols:

    # Remove leading and trailing spaces

    processed_df[col] = (
        processed_df[col]
        .astype(str)
        .str.strip()
    )

    # Convert text to lowercase

    processed_df[col] = (
        processed_df[col]
        .str.lower()
    )


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = (
        encoder.fit_transform(
            processed_df[col]
        )
    )

    label_encoders[col] = encoder


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()


# Do not scale dropout target

features_to_scale = [
    col
    for col in numeric_cols
    if col != target_column
]


if len(features_to_scale) > 0:

    processed_df[features_to_scale] = (
        scaler.fit_transform(
            processed_df[features_to_scale]
        )
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing Values After Preprocessing:")

print(
    processed_df.isnull().sum()
)


# ============================================================
# SAVE PREPROCESSED DATASET
# ============================================================

processed_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nOriginal Dataset Shape :",
    df.shape
)

print(
    "Processed Dataset Shape:",
    processed_df.shape
)

print(
    "Target Column          :",
    target_column
)

print(
    "Features Scaled        :",
    features_to_scale
)

print(
    "\nSaved File:"
)

print(output_file)

print("=" * 60)