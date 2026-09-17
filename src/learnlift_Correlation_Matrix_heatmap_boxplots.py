import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# PATHS
# =============================================================

# Main LearnLift project folder
project_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift"

# Dataset path
dataset_path = os.path.join(
    project_dir,
    "Dataset",
    "learnlift_updated_dataset_raw.csv"
)

# Output folder
output_dir = os.path.join(
    project_dir,
    "outputs",
    "Boxplots_correlation"
)

# Create output folder if it does not exist
os.makedirs(output_dir, exist_ok=True)


# =============================================================
# 1. LOAD LEARNLIFT DATASET
# =============================================================

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully.")
print("Dataset Shape:", df.shape)

print("-" * 60)


# =============================================================
# 2. COMPUTE CORRELATION MATRIX
# =============================================================

# Select only numerical columns
numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Numerical Columns:")
print(numerical_cols)

print("-" * 60)


# Calculate correlation matrix
corr_matrix = df[numerical_cols].corr()


# Print correlation matrix
print("\n--- Correlation Matrix ---")
print(corr_matrix)

print("-" * 60)


# =============================================================
# 3. GENERATE CORRELATION HEATMAP
# =============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()


# Save heatmap
heatmap_path = os.path.join(
    output_dir,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Exported heatmap to:")
print(heatmap_path)

print("-" * 60)


# =============================================================
# 4. PRODUCE BOXPLOTS
# Numerical Features vs Dropout
# =============================================================

target_col = "dropout"


if target_col in df.columns:

    print("\n--- Generating Boxplots ---")

    for col in numerical_cols:

        # Do not create dropout vs dropout boxplot
        if col == target_col:
            continue

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            hue=target_col,
            legend=False
        )

        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel(
            "Dropout (0 = No, 1 = Yes)"
        )

        plt.ylabel(col)

        plt.tight_layout()


        # File name
        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )

        # Complete path
        boxplot_path = os.path.join(
            output_dir,
            boxplot_filename
        )


        # Save boxplot
        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Exported boxplot: {boxplot_filename}"
        )

else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset."
    )

    print("Skipping boxplots.")


# =============================================================
# 5. DROPOUT TARGET DISTRIBUTION
# =============================================================

if target_col in df.columns:

    print("\n--- Dropout Target Distribution ---")

    print(
        df[target_col].value_counts()
    )


    print("\n--- Dropout Target Percentage ---")

    print(
        df[target_col]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


# =============================================================
# 6. COMPLETION MESSAGE
# =============================================================

print("\n" + "=" * 60)

print(
    "All LearnLift Correlation and Boxplot "
    "tasks completed successfully!"
)

print("=" * 60)

print("\nOutput folder:")
print(output_dir)