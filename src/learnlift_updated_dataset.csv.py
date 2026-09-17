import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("1. Load the Dataset")
print("-----------------------------------")

file_path =r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv"

try:

    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)

    print("-----------------------------------")
    print("Dataset Loaded Successfully")
    print("-----------------------------------")


    # ========================================================
    # 2. DATASET SIZE
    # ========================================================

    print("-----------------------------------")
    print("\n2. Number of Rows and Columns:")
    print("-----------------------------------")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Shape:", df.shape)


    # ========================================================
    # 3. COLUMN NAMES
    # ========================================================

    print("-----------------------------------")
    print("\n3. Column Names:")
    print("-----------------------------------")

    print(df.columns.tolist())


    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


    # ========================================================
    # 4. FIRST 10 RECORDS
    # ========================================================

    print("-----------------------------------")
    print("\n4. First 10 Records:")
    print("-----------------------------------")

    print(df.head(10))


    # ========================================================
    # 5. LAST 10 RECORDS
    # ========================================================

    print("-----------------------------------")
    print("\n5. Last 10 Records:")
    print("-----------------------------------")

    print(df.tail(10))


    # ========================================================
    # 6. DATA TYPES
    # ========================================================

    print("-----------------------------------")
    print("\n6. Data Types of Columns:")
    print("-----------------------------------")

    print(df.dtypes)


    # ========================================================
    # 7. COLUMN NAME + DATA TYPE
    # ========================================================

    print("-----------------------------------")
    print("\n7. Column Names with Data Types:")
    print("-----------------------------------")

    print("\nColumn Name\t\t\tData Type")
    print("-" * 50)

    for column in df.columns:
        print(f"{column:<30} {df[column].dtype}")


    # ========================================================
    # 8. DATASET INFORMATION
    # ========================================================

    print("-----------------------------------")
    print("\n8. Dataset Summary and Information:")
    print("-----------------------------------")

    df.info()


    # ========================================================
    # 9. NUMERICAL COLUMNS
    # ========================================================

    print("-----------------------------------")
    print("\n9. Numerical Columns:")
    print("-----------------------------------")

    numeric_df = df.select_dtypes(
        include=['int64', 'float64']
    )

    print(numeric_df.columns.tolist())


    # ========================================================
    # 10. MISSING VALUES IN NUMERICAL COLUMNS
    # ========================================================

    print("-----------------------------------")
    print("\n10. Missing Values in Numerical Columns:")
    print("-----------------------------------")

    print(numeric_df.isnull().sum())

    print(
        "\nTotal Missing Numerical Values:",
        numeric_df.isnull().sum().sum()
    )


    # ========================================================
    # 11. CATEGORICAL COLUMNS
    # ========================================================

    print("-----------------------------------")
    print("\n11. Categorical Columns:")
    print("-----------------------------------")

    categorical_df = df.select_dtypes(
        include=['object']
    )

    print(categorical_df.columns.tolist())


    # ========================================================
    # 12. MISSING VALUES IN CATEGORICAL COLUMNS
    # ========================================================

    print("-----------------------------------")
    print("\n12. Missing Values in Categorical Columns:")
    print("-----------------------------------")

    print(categorical_df.isnull().sum())

    print(
        "\nTotal Missing Categorical Values:",
        categorical_df.isnull().sum().sum()
    )


    # ========================================================
    # 13. MISSING VALUES IN EACH COLUMN
    # ========================================================

    print("-----------------------------------")
    print("\n13. Missing Values in Each Column:")
    print("-----------------------------------")

    print(df.isnull().sum())


    # ========================================================
    # 14. TOTAL MISSING VALUES
    # ========================================================

    print("-----------------------------------")
    print("\n14. Total Missing Values:")
    print("-----------------------------------")

    total_missing = df.isnull().sum().sum()

    print(total_missing)


    # ========================================================
    # 15. DUPLICATE RECORDS
    # ========================================================

    print("-----------------------------------")
    print("\n15. Number of Duplicate Records:")
    print("-----------------------------------")

    duplicate_count = df.duplicated().sum()

    print(duplicate_count)


    # ========================================================
    # 16. STATISTICAL SUMMARY
    # ========================================================

    print("-----------------------------------")
    print("\n16. Statistical Overview:")
    print("-----------------------------------")

    print(df.describe())


    # ========================================================
    # 17. FINAL RESULT DISTRIBUTION
    # ========================================================

    print("-----------------------------------")
    print("\n17. Final Result Distribution:")
    print("-----------------------------------")

    print(df['final_result'].value_counts())


    # ========================================================
    # 18. DROPOUT DISTRIBUTION
    # ========================================================

    print("-----------------------------------")
    print("\n18. Dropout Distribution:")
    print("-----------------------------------")

    print(df['dropout'].value_counts())


    # ========================================================
    # 19. FINAL SCORE STATISTICS
    # ========================================================

    print("-----------------------------------")
    print("\n19. Final Score Statistics:")
    print("-----------------------------------")

    print(df['final_score'].describe())


    # ========================================================
    # 20. AVERAGE ASSESSMENT SCORE STATISTICS
    # ========================================================

    print("-----------------------------------")
    print("\n20. Average Assessment Score Statistics:")
    print("-----------------------------------")

    print(df['avg_assessment_score'].describe())


    # ========================================================
    # 21. HISTOGRAM - FINAL SCORE
    # ========================================================

    print("-----------------------------------")
    print("\n21. Display Histogram of Final Score:")
    print("-----------------------------------")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df['final_score'].dropna(),
        bins=10,
        edgecolor='black'
    )

    plt.title("Distribution of Final Score")
    plt.xlabel("Final Score")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()


# ============================================================
# ERROR HANDLING
# ============================================================

except FileNotFoundError:

    print("-----------------------------------")
    print("ERROR: Dataset file not found.")
    print("-----------------------------------")
    print(f"Please check the file path:")
    print(file_path)


except Exception as e:

    print("-----------------------------------")
    print("An error occurred:")
    print("-----------------------------------")
    print(e)