# ---------------------------------------------------------
# LearnLift - Numeric Column Pre-processing Techniques
# Mean, Median, Mode
# Feature Scaling, Standardization, and Normalization
# Save all results in ONE CSV file
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)

import matplotlib.pyplot as plt


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

# Create output folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Output file
output_file = os.path.join(
    output_dir,
    "clean_minmax_stand_normal.csv"
)


# ==========================================================
# Step 1: Load Dataset
# ==========================================================

df = pd.read_csv(input_file)

print("Original LearnLift Dataset")
print("-" * 40)

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print("-" * 40)
print(df.dtypes)

print("\nMissing Values:")
print("-" * 40)
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ==========================================================
# Step 2: Remove Duplicate Records
# ==========================================================

df = df.drop_duplicates()

print(
    "\nDuplicate Records After Removal:",
    df.duplicated().sum()
)


# ==========================================================
# Step 3: Handle Missing Values
# ==========================================================

# Numerical Columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# Fill numerical missing values with mean

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].mean()
    )


# Categorical Columns

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# Fill categorical missing values with mode

for column in categorical_columns:

    mode_values = df[column].mode()

    if len(mode_values) > 0:

        df[column] = df[column].fillna(
            mode_values[0]
        )


# ==========================================================
# Step 4: Remove Leading and Trailing Spaces
# ==========================================================

for column in categorical_columns:

    df[column] = df[column].str.strip()


# ==========================================================
# Step 5: Select Numeric Columns
# ==========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


print("\nNumeric Columns:")
print(numeric_columns)


# ==========================================================
# Step 6: Exclude Target and ID Columns
# ==========================================================

target_column = "dropout"

id_columns = [
    "id_student"
]


scaling_columns = [
    col
    for col in numeric_columns
    if col != target_column
    and col not in id_columns
]


print("\nNumeric Columns Selected for Scaling:")
print(scaling_columns)


print("\nTarget Column:")
print(target_column)


# ==========================================================
# Step 7: Standardization (Z-score)
# Mean = 0
# Standard Deviation = 1
# ==========================================================

standard_scaler = StandardScaler()


standardized = standard_scaler.fit_transform(
    df[scaling_columns]
)


for i, col in enumerate(scaling_columns):

    df[col + "_Standardized"] = standardized[:, i]


# ==========================================================
# Step 8: Feature Scaling (Min-Max Scaling)
# Values between 0 and 1
# ==========================================================

minmax_scaler = MinMaxScaler()


scaled = minmax_scaler.fit_transform(
    df[scaling_columns]
)


for i, col in enumerate(scaling_columns):

    df[col + "_Scaled"] = scaled[:, i]


# ==========================================================
# Step 9: Normalization (L2 Normalization)
# Each row becomes a unit vector
# ==========================================================

normalizer = Normalizer(
    norm="l2"
)


normalized = normalizer.fit_transform(
    df[scaling_columns]
)


for i, col in enumerate(scaling_columns):

    df[col + "_Normalized"] = normalized[:, i]


# ==========================================================
# Step 10: Display Results After Pre-processing
# ==========================================================

print("\nDisplay Results After Preprocessing")
print("-" * 50)

print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Information:")
df.info()


print("\nColumns in Dataset:")
print(df.columns.tolist())


print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())


print(
    "\nTotal Missing Values After Preprocessing:",
    df.isnull().sum().sum()
)


print(
    "\nDuplicate Records After Preprocessing:",
    df.duplicated().sum()
)


# ==========================================================
# Step 11: Save Preprocessed Dataset
# ==========================================================

df.to_csv(
    output_file,
    index=False
)


print(
    "\nPreprocessed LearnLift dataset saved successfully."
)

print("\nOutput File:")
print(output_file)


# ==========================================================
# Step 12: Display Histograms
# ==========================================================

pf = pd.read_csv(
    output_file
)


print("\nDisplaying Histograms...")


pf.hist(
    figsize=(16, 12),
    bins=10,
    edgecolor="black"
)


plt.suptitle(
    "Histogram of Preprocessed LearnLift Dataset"
)


plt.tight_layout()

plt.show()


# ==========================================================
# Final Message
# ==========================================================

print("\n" + "=" * 50)

print(
    "LearnLift Numerical Preprocessing Completed"
)

print("=" * 50)

print(
    "Original Dataset was NOT overwritten."
)

print("\nTechniques Applied:")

print("1. Mean Imputation")

print("2. Mode Imputation")

print("3. Standardization")

print("4. Min-Max Scaling")

print("5. L2 Normalization")

print("\nTarget Variable:")
print("dropout")

print("\nOutput:")
print(output_file)

print("=" * 50)