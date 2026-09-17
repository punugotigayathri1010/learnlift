import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Load LearnLift Dataset
# Original dataset will NOT be modified
# ==========================================================

df = pd.read_csv(
    r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv"
)


# Create a copy for processing
data = df.copy()


print("Original LearnLift Dataset")
print("------------------------")
print(data.head())


print("Dataset Shape:", df.shape)

print("\nData Types:")
print("------------------------")
print(data.dtypes)


print("\nDuplicate Records:", df.duplicated().sum())


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(
    include=["object", "string"]
).columns:

    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before_duplicates = data.shape[0]

data = data.drop_duplicates()

after_duplicates = data.shape[0]

print(
    "\nDuplicate Records Removed:",
    before_duplicates - after_duplicates
)


# ==========================================================
# Separate Numerical and Categorical Columns
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
# 6. One-Hot Encoding
# ==========================================================

if len(cat_cols) > 0:

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )


    encoded_values = encoder.fit_transform(
        data[cat_cols]
    )


    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(
            cat_cols
        )
    )


    # Reset index for merging

    encoded_df.reset_index(
        drop=True,
        inplace=True
    )


    # Keep numerical columns

    numeric_df = data[num_cols].reset_index(
        drop=True
    )


    # Merge numerical + encoded columns

    final_output = pd.concat(
        [
            numeric_df,
            encoded_df
        ],
        axis=1
    )

else:

    final_output = data.copy()


# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")

print(
    final_output.isnull().sum()
)


print(
    "\nTotal Missing Values:",
    final_output.isnull().sum().sum()
)


# ==========================================================
# 8. Check Final Dataset Information
# ==========================================================

print("\nFinal Dataset Shape:")

print(
    final_output.shape
)


print("\nFinal Dataset Data Types:")

print(
    final_output.dtypes
)


# ==========================================================
# 9. Save Final Result
# ==========================================================

output_file = (
    r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\clean_one_hot_encoding.csv"
)


final_output.to_csv(
    output_file,
    index=False
)


# ==========================================================
# Final Message
# ==========================================================

print("\n======================================")
print("Original LearnLift dataset is NOT modified.")
print("Cleaning and One-Hot Encoding completed.")
print("Target Variable: dropout")
print("Output file:")
print("clean_one_hot_encoding.csv")
print("======================================")