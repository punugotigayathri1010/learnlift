import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


class LearnLiftPipeline:

    def __init__(self, dataset_path, output_dir):

        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df = pd.read_csv(self.dataset_path)

    # ==========================================================
    # BASIC DATASET INFORMATION
    # ==========================================================

    def basic_info(self):

        df = self.df

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "numeric_columns": int(
                len(df.select_dtypes(include=np.number).columns)
            ),
            "categorical_columns": int(
                len(df.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns)
            )
        }

    # ==========================================================
    # STATISTICAL SUMMARY
    # ==========================================================

    def statistical_summary(self):

        summary = (
            self.df
            .describe(include="all")
            .fillna("")
            .round(3)
        )

        return summary.to_html(
            classes="data-table",
            border=0
        )

    # ==========================================================
    # MISSING VALUES
    # ==========================================================

    def missing_values(self):

        result = (
            self.df
            .isnull()
            .sum()
            .reset_index()
        )

        result.columns = [
            "Feature",
            "Missing Values"
        ]

        result["Percentage"] = (
            result["Missing Values"]
            / len(self.df)
            * 100
        ).round(2)

        result = result[
            result["Missing Values"] > 0
        ]

        return result.to_dict(
            orient="records"
        )

    # ==========================================================
    # TARGET ANALYSIS
    # ==========================================================

    def target_analysis(self):

        if "dropout" not in self.df.columns:

            return {
                "available": False
            }

        counts = (
            self.df["dropout"]
            .value_counts()
            .sort_index()
        )

        return {
            "available": True,
            "labels": [
                "Retained",
                "Dropout"
            ],
            "values": [
                int(counts.get(0, 0)),
                int(counts.get(1, 0))
            ]
        }

    # ==========================================================
    # MODULE RISK
    # ==========================================================

    def module_risk(self):

        if not {
            "code_module",
            "dropout"
        }.issubset(self.df.columns):

            return {
                "labels": [],
                "values": []
            }

        result = (
            self.df
            .groupby("code_module")["dropout"]
            .mean()
            .mul(100)
            .round(2)
        )

        return {
            "labels": result.index.tolist(),
            "values": result.tolist()
        }

    # ==========================================================
    # AGE RISK
    # ==========================================================

    def age_risk(self):

        if not {
            "age_band",
            "dropout"
        }.issubset(self.df.columns):

            return {
                "labels": [],
                "values": []
            }

        result = (
            self.df
            .groupby("age_band")["dropout"]
            .mean()
            .mul(100)
        )

        preferred_order = [
            "0-35",
            "35-55",
            "55<="
        ]

        result = result.reindex(
            [
                x for x in preferred_order
                if x in result.index
            ]
        )

        return {
            "labels": result.index.tolist(),
            "values": result.round(2).tolist()
        }

    # ==========================================================
    # EDUCATION RISK
    # ==========================================================

    def education_risk(self):

        if not {
            "highest_education",
            "dropout"
        }.issubset(self.df.columns):

            return {
                "labels": [],
                "values": []
            }

        result = (
            self.df
            .groupby("highest_education")["dropout"]
            .mean()
            .mul(100)
            .sort_values()
        )

        return {
            "labels": result.index.tolist(),
            "values": result.round(2).tolist()
        }

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    def final_result(self):

        if "final_result" not in self.df.columns:

            return {
                "labels": [],
                "values": []
            }

        result = (
            self.df["final_result"]
            .value_counts()
        )

        return {
            "labels": result.index.tolist(),
            "values": result.astype(int).tolist()
        }

    # ==========================================================
    # CORRELATION
    # ==========================================================

    def correlation(self):

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if numeric.empty:

            return {
                "columns": [],
                "values": []
            }

        corr = numeric.corr().round(2)

        return {
            "columns": corr.columns.tolist(),
            "values": corr.values.tolist()
        }

    # ==========================================================
    # OUTLIERS
    # ==========================================================

    def outliers(self):

        numeric = self.df.select_dtypes(
            include=np.number
        )

        result = []

        for column in numeric.columns:

            if column == "dropout":
                continue

            q1 = numeric[column].quantile(0.25)
            q3 = numeric[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = (
                (
                    (numeric[column] < lower)
                    |
                    (numeric[column] > upper)
                )
                .sum()
            )

            result.append({
                "feature": column,
                "count": int(count),
                "percentage": round(
                    count / len(self.df) * 100,
                    2
                )
            })

        return result

    # ==========================================================
    # GENERATE EDA IMAGES
    # ==========================================================

    def generate_eda_images(self):

        sns.set_theme(
            style="whitegrid"
        )

        numeric = self.df.select_dtypes(
            include=np.number
        )

        # --------------------------------------------
        # Correlation heatmap
        # --------------------------------------------

        if not numeric.empty:

            plt.figure(
                figsize=(12, 8)
            )

            sns.heatmap(
                numeric.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm"
            )

            plt.title(
                "LearnLift Correlation Matrix"
            )

            plt.tight_layout()

            plt.savefig(
                self.output_dir /
                "correlation_heatmap.png",
                dpi=150
            )

            plt.close()

        # --------------------------------------------
        # Histograms
        # --------------------------------------------

        for column in numeric.columns:

            if column == "dropout":
                continue

            plt.figure(
                figsize=(8, 5)
            )

            sns.histplot(
                self.df[column],
                kde=True
            )

            plt.title(
                f"Distribution - {column}"
            )

            plt.tight_layout()

            safe_name = (
                column
                .replace("/", "_")
                .replace(" ", "_")
            )

            plt.savefig(
                self.output_dir /
                f"hist_{safe_name}.png",
                dpi=150
            )

            plt.close()

        # --------------------------------------------
        # Dropout distribution
        # --------------------------------------------

        if "dropout" in self.df.columns:

            plt.figure(
                figsize=(7, 5)
            )

            sns.countplot(
                data=self.df,
                x="dropout"
            )

            plt.title(
                "Student Dropout Distribution"
            )

            plt.tight_layout()

            plt.savefig(
                self.output_dir /
                "dropout_distribution.png",
                dpi=150
            )

            plt.close()

    # ==========================================================
    # COMPLETE ANALYSIS
    # ==========================================================

    def run(self):

        self.generate_eda_images()

        return {
            "info": self.basic_info(),
            "missing": self.missing_values(),
            "target": self.target_analysis(),
            "module": self.module_risk(),
            "age": self.age_risk(),
            "education": self.education_risk(),
            "result": self.final_result(),
            "correlation": self.correlation(),
            "outliers": self.outliers(),
            "summary": self.statistical_summary()
        }