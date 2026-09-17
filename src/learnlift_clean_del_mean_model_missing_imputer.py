import os
import pandas as pd
import numpy as np


# ============================================================
# LEARNLIFT - MISSING VALUE HANDLING TECHNIQUES
# Deletion, Mean, Median, Model-Based Imputation,
# Missing Indicator Features
# ============================================================


# ============================================================
# PROJECT PATHS
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
    "clean_del_mean_model.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(input_file)

print("=" * 70)
print("LEARNLIFT DATASET")
print("=" * 70)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values in Original Dataset:")
print(df.isnull().sum())


# Keep original dataset unchanged
df_original = df.copy()


# ============================================================
# 1. DELETION
# ============================================================

df_deletion = df_original.dropna().copy()

print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)

print("Original rows:", len(df_original))
print("Rows after deletion:", len(df_deletion))
print(
    "Rows deleted:",
    len(df_original) - len(df_deletion)
)


# Missing-value indicator for each row
deletion_indicator = (
    df_original.isnull().any(axis=1)
).astype(int)


# ============================================================
# FIND NUMERICAL COLUMNS
# ============================================================

numeric_columns = df_original.select_dtypes(
    include="number"
).columns.tolist()

print("\nNumerical Columns:")
print(numeric_columns)


# ============================================================
# 2. MEAN IMPUTATION
# ============================================================

df_mean = df_original.copy()

for column in numeric_columns:

    if df_mean[column].isnull().any():

        mean_value = df_mean[column].mean()

        df_mean[column] = df_mean[column].fillna(
            mean_value
        )

        print(
            "Mean used for",
            column,
            "=",
            mean_value
        )


print("\n" + "=" * 70)
print("2. MEAN IMPUTATION")
print("=" * 70)

print(df_mean.head())


# ============================================================
# 3. MEDIAN IMPUTATION
# ============================================================

df_median = df_original.copy()

for column in numeric_columns:

    if df_median[column].isnull().any():

        median_value = df_median[column].median()

        df_median[column] = df_median[column].fillna(
            median_value
        )

        print(
            "Median used for",
            column,
            "=",
            median_value
        )


print("\n" + "=" * 70)
print("3. MEDIAN IMPUTATION")
print("=" * 70)

print(df_median.head())


# ============================================================
# 4. MODEL-BASED IMPUTATION
# ============================================================

df_model = df_original.copy()

print("\n" + "=" * 70)
print("4. MODEL-BASED IMPUTATION")
print("=" * 70)


if len(numeric_columns) >= 2:

    for target_column in numeric_columns:

        # Skip target column
        if target_column == "dropout":
            continue

        # Continue if target column has no missing values
        if not df_model[target_column].isnull().any():
            continue


        # ----------------------------------------------------
        # Select another numerical column as predictor
        # ----------------------------------------------------

        predictor_column = None

        for column in numeric_columns:

            if column != target_column:

                if (
                    df_model[column].notnull().sum() >= 2
                ):

                    predictor_column = column
                    break


        # If predictor is not available
        if predictor_column is None:
            continue


        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        training_data = df_model[
            df_model[target_column].notnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        # Need at least two records
        if len(training_data) < 2:
            continue


        x = training_data[predictor_column]
        y = training_data[target_column]


        # ----------------------------------------------------
        # Calculate linear regression manually
        # y = slope*x + intercept
        # ----------------------------------------------------

        x_mean = x.mean()
        y_mean = y.mean()


        numerator = (
            (x - x_mean) *
            (y - y_mean)
        ).sum()


        denominator = (
            (x - x_mean) ** 2
        ).sum()


        if denominator == 0:
            continue


        slope = numerator / denominator

        intercept = y_mean - (
            slope * x_mean
        )


        # ----------------------------------------------------
        # Find rows where target is missing
        # ----------------------------------------------------

        missing_rows = df_model[
            df_model[target_column].isnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        # ----------------------------------------------------
        # Predict missing values
        # ----------------------------------------------------

        if len(missing_rows) > 0:

            predicted_values = (
                slope *
                missing_rows[predictor_column]
                +
                intercept
            )


            df_model.loc[
                missing_rows.index,
                target_column
            ] = predicted_values


            print(
                "\nTarget column:",
                target_column
            )

            print(
                "Predictor column:",
                predictor_column
            )

            print(
                "Slope:",
                slope
            )

            print(
                "Intercept:",
                intercept
            )

            print(
                "Number of values predicted:",
                len(missing_rows)
            )


# ============================================================
# MEDIAN FALLBACK FOR REMAINING NUMERICAL MISSING VALUES
# ============================================================

for column in numeric_columns:

    # Do not modify dropout target through imputation
    if column == "dropout":
        continue

    if df_model[column].isnull().any():

        median_value = df_model[column].median()

        df_model[column] = df_model[column].fillna(
            median_value
        )


print("\nModel-Based Imputation Result:")
print(df_model.head())


# ============================================================
# HANDLE CATEGORICAL COLUMNS FOR MODEL RESULT
# ============================================================

categorical_columns = df_original.select_dtypes(
    exclude="number"
).columns.tolist()


# Fill categorical missing values using mode
# This is a supporting step for categorical data.

for column in categorical_columns:

    if df_model[column].isnull().any():

        mode_values = df_model[column].mode()

        if len(mode_values) > 0:

            df_model[column] = df_model[column].fillna(
                mode_values.iloc[0]
            )


# ============================================================
# 5. MISSING INDICATOR FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING INDICATOR FEATURES")
print("=" * 70)


df_indicator = df_original.copy()


# Create an indicator for every column

for column in df_original.columns:

    indicator_column = (
        column + "_Missing"
    )

    df_indicator[indicator_column] = (
        df_original[column].isnull()
    ).astype(int)


print("\nMissing Indicator Result:")
print(df_indicator.head())


# ============================================================
# CREATE ONE FINAL OUTPUT DATAFRAME
# ============================================================

final_result = df_original.copy()


# ------------------------------------------------------------
# A. Deletion indicator
# ------------------------------------------------------------

final_result[
    "Deletion_Row_Removed"
] = deletion_indicator


# ------------------------------------------------------------
# B. Mean-imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Mean_Imputed"
    ] = df_mean[column]


# ------------------------------------------------------------
# C. Median-imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Median_Imputed"
    ] = df_median[column]


# ------------------------------------------------------------
# D. Model-based imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Model_Imputed"
    ] = df_model[column]


# ------------------------------------------------------------
# E. Missing Indicator features
# ------------------------------------------------------------

missing_indicator_data = {}

for column in df_original.columns:

    missing_indicator_data[
        column + "_Missing"
    ] = (
        df_original[column].isnull()
    ).astype(int).values


missing_indicator_df = pd.DataFrame(
    missing_indicator_data,
    index=df_original.index
)


# Add missing indicators to final result

final_result = pd.concat(
    [
        final_result,
        missing_indicator_df
    ],
    axis=1
)


# ============================================================
# SAVE ALL RESULTS INTO ONE CSV FILE
# ============================================================

final_result.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(final_result.head())

print("\nFinal Dataset Shape:")
print(final_result.shape)


# ============================================================
# CHECK REMAINING MISSING VALUES
# ============================================================

print("\nMissing Values in Mean-Imputed Dataset:")
print(df_mean.isnull().sum())


print("\nMissing Values in Median-Imputed Dataset:")
print(df_median.isnull().sum())


print("\nMissing Values in Model-Imputed Dataset:")
print(df_model.isnull().sum())


# ============================================================
# VERIFY ORIGINAL DATASET IS NOT MODIFIED
# ============================================================

df_check = pd.read_csv(input_file)


if df_original.equals(df_check):

    print(
        "\nOriginal LearnLift dataset was NOT modified."
    )

else:

    print(
        "\nWARNING: Original LearnLift dataset was modified!"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("LEARNLIFT MISSING-VALUE PROCESSING COMPLETED")
print("=" * 70)

print("\nInput file:")
print(input_file)

print("\nOutput file:")
print(output_file)

print(
    "\nAll missing-value handling techniques "
    "are stored in ONE CSV file."
)

print("=" * 70)