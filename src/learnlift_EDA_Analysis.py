# ============================================================
# LEARNLIFT - EDA ANALYSIS
# Student Dropout Prediction
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = (r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv")

OUTPUT_FOLDER = (r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\outputs\EDA_Analysis_outputs")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("LEARNLIFT - EXPLORATORY DATA ANALYSIS")
print("Student Dropout Prediction")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nFirst Five Records")
print("-" * 70)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")
print("-" * 70)
print(df.columns.tolist())

print("\nData Types")
print("-" * 70)
print(df.dtypes)

print("\nDataset Information")
print("-" * 70)
df.info()

print("\nMissing Values")
print("-" * 70)
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ============================================================
# 4. TARGET VARIABLE
# ============================================================

# LearnLift target variable
target = "dropout"

if target in df.columns:

    print("\nTarget Variable:", target)

    print("\nTarget Value Counts")
    print("-" * 70)
    print(df[target].value_counts())

    print("\nTarget Value Percentages")
    print("-" * 70)

    target_percentage = (
        df[target]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(target_percentage)

else:
    print("\nERROR: Target column 'dropout' not found.")

# ============================================================
# 5. SELECT NUMERICAL COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

# Remove target from feature list
numeric_features = [
    col for col in numeric_cols
    if col != target
]

print("\nNumerical Columns")
print("-" * 70)
print(numeric_cols)

print("\nNumerical Features")
print("-" * 70)
print(numeric_features)

# ============================================================
# 6. SELECT CATEGORICAL COLUMNS
# ============================================================

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nCategorical Columns")
print("-" * 70)
print(categorical_cols)

# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

summary = df.describe(include="all")

print("\nStatistical Summary")
print("-" * 70)
print(summary)

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Statistical_Summary.csv"
    )
)

# ============================================================
# 8. UNIVARIATE ANALYSIS - HISTOGRAMS
# ============================================================

print("\nGenerating Histograms...")

for col in numeric_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=col,
        bins=20,
        kde=True
    )

    plt.title(f"Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Histogram_{col}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 9. BOX PLOTS
# ============================================================

print("\nGenerating Boxplots...")

for col in numeric_features:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        y=df[col]
    )

    plt.title(f"Box Plot - {col}")
    plt.ylabel(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Boxplot_{col}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 10. OUTLIER DETECTION USING IQR
# ============================================================

print("\nDetecting Outliers...")

outlier_results = []

for col in numeric_features:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_percentage = (
        outlier_count / len(df)
    ) * 100

    outlier_results.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outlier_count,
        "Outlier_Percentage": round(
            outlier_percentage, 2
        )
    })

outlier_summary = pd.DataFrame(
    outlier_results
)

print("\nOutlier Summary")
print("-" * 70)
print(outlier_summary)

outlier_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Outlier_Summary.csv"
    ),
    index=False
)

# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

print("\nGenerating Correlation Matrix...")

numeric_df = df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

corr.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Matrix.csv"
    )
)

plt.figure(
    figsize=(
        max(10, len(numeric_df.columns) * 0.7),
        max(8, len(numeric_df.columns) * 0.6)
    )
)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Correlation Matrix - LearnLift"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Correlation_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 12. COUNT PLOTS FOR CATEGORICAL COLUMNS
# ============================================================

print("\nGenerating Categorical Count Plots...")

for col in categorical_cols:

    # Skip columns with too many unique values
    if df[col].nunique() > 30:

        print(
            f"Skipping {col} - "
            f"{df[col].nunique()} unique values"
        )

        continue

    plt.figure(figsize=(8, 5))

    order = df[col].value_counts().index

    sns.countplot(
        data=df,
        x=col,
        order=order
    )

    plt.title(
        f"Count Plot - {col}"
    )

    plt.xlabel(col)
    plt.ylabel("Count")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Countplot_{col}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 13. DROPOUT TARGET DISTRIBUTION
# ============================================================

if target in df.columns:

    print("\nGenerating Dropout Target Distribution...")

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=target,
        order=df[target].value_counts().index
    )

    plt.title(
        "Dropout Target Variable Distribution"
    )

    plt.xlabel(
        "Dropout (0 = No, 1 = Yes)"
    )

    plt.ylabel(
        "Number of Students"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Dropout_Target_Distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 14. DROPOUT PIE CHART
# ============================================================

if target in df.columns:

    plt.figure(figsize=(7, 7))

    target_counts = df[target].value_counts()

    plt.pie(
        target_counts.values,
        labels=[
            "Not Dropout (0)",
            "Dropout (1)"
        ],
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Student Dropout Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Dropout_Distribution_Pie.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 15. NUMERICAL FEATURES VS DROPOUT
# ============================================================

print(
    "\nGenerating Numerical Features vs Dropout..."
)

for col in numeric_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=target,
        y=col
    )

    plt.title(
        f"{col} vs Dropout"
    )

    plt.xlabel(
        "Dropout (0 = No, 1 = Yes)"
    )

    plt.ylabel(col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{col}_vs_Dropout.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 16. CATEGORICAL FEATURES VS DROPOUT
# ============================================================

print(
    "\nGenerating Categorical Features vs Dropout..."
)

for col in categorical_cols:

    # Skip target if present
    if col == target:
        continue

    # Skip high-cardinality columns
    if df[col].nunique() > 20:
        continue

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=col,
        hue=target
    )

    plt.title(
        f"{col} vs Dropout"
    )

    plt.xlabel(col)
    plt.ylabel("Number of Students")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{col}_vs_Dropout_Countplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 17. SCATTER PLOT
# ============================================================

if len(numeric_features) >= 2:

    x_col = numeric_features[0]
    y_col = numeric_features[1]

    print(
        f"\nGenerating Scatter Plot: "
        f"{x_col} vs {y_col}"
    )

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=target,
        s=60
    )

    plt.title(
        f"Scatter Plot: {x_col} vs {y_col}"
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Scatterplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 18. PAIR PLOT
# ============================================================

# Pairplot can be very large.
# Therefore, use a maximum of 5 numerical features.

if len(numeric_features) >= 2:

    pair_cols = numeric_features[:5]

    print(
        "\nGenerating Pair Plot using:"
    )

    print(pair_cols)

    pair_data = df[
        pair_cols + [target]
    ].copy()

    pair = sns.pairplot(
        pair_data,
        hue=target
    )

    pair.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Pairplot.png"
        ),
        dpi=300
    )

    plt.close("all")

# ============================================================
# 19. MISSING VALUE HEATMAP
# ============================================================

print(
    "\nGenerating Missing Values Heatmap..."
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title(
    "Missing Values Heatmap - LearnLift"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Missing_Values_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 20. MISSING VALUE SUMMARY
# ============================================================

missing_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage":
        df.isnull().mean().values * 100
})

missing_summary = missing_summary.sort_values(
    by="Missing_Count",
    ascending=False
)

missing_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Missing_Value_Summary.csv"
    ),
    index=False
)

# ============================================================
# 21. DUPLICATE SUMMARY
# ============================================================

duplicate_count = df.duplicated().sum()

duplicate_summary = pd.DataFrame({
    "Total_Rows": [len(df)],
    "Duplicate_Rows": [duplicate_count]
})

duplicate_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Duplicate_Summary.csv"
    ),
    index=False
)

# ============================================================
# 22. TARGET-WISE NUMERICAL SUMMARY
# ============================================================

if target in df.columns:

    print(
        "\nGenerating Target-wise Numerical Summary..."
    )

    target_summary = df.groupby(
        target

    )[numeric_features].mean()

    print(
        "\nMean Numerical Features by Dropout:"
    )

    print(target_summary)

    target_summary.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Numerical_Features_by_Dropout.csv"
        )
    )

# ============================================================
# 23. FINAL EDA OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nDataset Shape:",
    df.shape
)

print(
    "Target Variable:",
    target
)

print(
    "Numerical Features:",
    len(numeric_features)
)

print(
    "Categorical Features:",
    len(categorical_cols)
)

print(
    "Total Missing Values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate Rows:",
    df.duplicated().sum()
)

print(
    "\nAll EDA outputs are saved in:"
)

print(OUTPUT_FOLDER)

print("=" * 70)