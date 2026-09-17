import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =============================================================
# Load LearnLift Dataset
# =============================================================

file_path = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv"

df = pd.read_csv(file_path)


# =============================================================
# 1. View first five rows
# =============================================================

print("--- First 5 Rows ---")
print(df.head())

print("-" * 40)


# =============================================================
# Print specific columns
# =============================================================

print("---- Print First 6 Columns ----")

subset = df.iloc[:, 0:6]

print(subset)

print("-" * 40)


# =============================================================
# 2. Identify missing values per column
# =============================================================

missing_counts = df.isnull().sum()

print("----- Missing Values Per Column -----")
print(missing_counts)

print("-" * 40)


# Total missing values in the complete dataset

total_missing = df.isnull().sum().sum()

print("Total Missing Values in Dataset:", total_missing)

print("-" * 40)


# =============================================================
# 3. Detect duplicate rows
# =============================================================

duplicate_rows = df[df.duplicated()]

print(
    f"Total Duplicate Rows Detected: {len(duplicate_rows)}"
)

print(duplicate_rows)

print("-" * 40)


# =============================================================
# 4. Produce a Missingness Heatmap
# =============================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap - LearnLift")
plt.xlabel("Features")
plt.ylabel("Students")

plt.tight_layout()

plt.show()


# =============================================================
# 5. Save Missingness Heatmap
# =============================================================

# Create output directory

output_dir = r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\outputs\EDA_Analysis_outputs"

os.makedirs(output_dir, exist_ok=True)


# Complete output file path

output_path = os.path.join(
    output_dir,
    "Missing_Values_Heatmap.png"
)


# Create the heatmap again for saving

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap - LearnLift")
plt.xlabel("Features")
plt.ylabel("Students")

plt.tight_layout()


# Save image

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nMissing Values Heatmap exported to:")
print(output_path)

print("-" * 40)


# =============================================================
# 6. Dataset Summary
# =============================================================

print("\n" + "=" * 50)
print("LEARNLIFT DATASET SUMMARY")
print("=" * 50)

print("Dataset Shape:", df.shape)

print("Number of Rows:", df.shape[0])

print("Number of Columns:", df.shape[1])

print(
    "Target Variable:",
    "dropout"
    if "dropout" in df.columns
    else "Not Found"
)

print(
    "Total Missing Values:",
    total_missing
)

print(
    "Total Duplicate Rows:",
    len(duplicate_rows)
)

print("=" * 50)