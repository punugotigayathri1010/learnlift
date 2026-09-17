
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Project Paths
# ==========================================================

project_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift"

input_file = os.path.join(
    project_dir,
    "Dataset",
    "learnlift_updated_dataset_raw.csv"
)

output_dir = os.path.join(
    project_dir,
    "Dataset"
)

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "clean_label_encode.csv"
)


# ==========================================================
# Load the LearnLift Dataset
# ==========================================================

df = pd.read_csv(input_file)

data = df.copy()


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before = data.shape[0]

data = data.drop_duplicates()

after = data.shape[0]

print("\nDuplicate Records Removed:", before - after)


# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = data.select_dtypes(
    exclude=np.number
).columns.tolist()


print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================

if len(num_cols) > 0:

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    data[num_cols] = num_imputer.fit_transform(
        data[num_cols]
    )


# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================

if len(cat_cols) > 0:

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


# ==========================================================
# 6. Label Encoding
# ==========================================================

label_encoders = {}

for col in cat_cols:

    encoder = LabelEncoder()

    data[col] = encoder.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = encoder


# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 8. Display Encoded Dataset
# ==========================================================

print("\nFirst 5 Rows After Label Encoding:")
print(data.head())


# ==========================================================
# 9. Display Encoding Information
# ==========================================================

print("\n======================================")
print("LABEL ENCODING INFORMATION")
print("======================================")

for col, encoder in label_encoders.items():

    print(f"\nColumn: {col}")

    encoding_dict = dict(
        zip(
            encoder.classes_,
            encoder.transform(encoder.classes_)
        )
    )

    print(encoding_dict)


# ==========================================================
# 10. Save Result
# ==========================================================

data.to_csv(
    output_file,
    index=False
)


# ==========================================================
# Final Message
# ==========================================================

print("\n======================================")
print("Original dataset is NOT modified.")
print("LearnLift Label Encoding completed successfully.")
print("Target Variable: dropout")
print("Output file:")
print(output_file)
print("======================================")