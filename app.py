from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import time

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# PATHS
# ==========================================================

BASE = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE / "uploads"
DATASET_FOLDER = BASE / "Dataset"
OUTPUT_FOLDER = DATASET_FOLDER

UPLOAD_FOLDER.mkdir(exist_ok=True)
DATASET_FOLDER.mkdir(exist_ok=True)


# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ==========================================================
# ACTIVE DATASET
#
# IMPORTANT:
# None means no dataset has been uploaded yet.
# Therefore the UI will initially show blank values.
# ==========================================================

active_dataset = None
df = None
model = None


# ==========================================================
# MODEL FEATURES
# ==========================================================

FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "avg_assessment_score",
    "assessment_count",
    "total_vle_clicks",
    "date_registration",
    "date_unregistration",
    "module_presentation_length"
]

TARGET = "dropout"


CAT_COLS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]


# ==========================================================
# LOAD ACTIVE DATASET
# ==========================================================

def load_active_dataset(path):

    global active_dataset
    global df
    global model

    active_dataset = Path(path)

    df = pd.read_csv(active_dataset)

    model = None

    train_model()

    print()
    print("=" * 60)
    print("ACTIVE LEARNLIFT DATASET")
    print("=" * 60)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Missing:", int(df.isnull().sum().sum()))
    print("Duplicates:", int(df.duplicated().sum()))
    print("=" * 60)


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model():

    global model

    if df is None:
        return

    if TARGET not in df.columns:
        return

    available_features = [
        c for c in FEATURES
        if c in df.columns
    ]

    if not available_features:
        return

    categorical = [
        c for c in available_features
        if c in CAT_COLS
    ]

    numerical = [
        c for c in available_features
        if c not in categorical
    ]

    transformers = []

    if categorical:

        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical
            )
        )

    if numerical:

        transformers.append(
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )
                ]),
                numerical
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        (
            "prep",
            preprocessor
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=180,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    try:

        model.fit(
            df[available_features],
            df[TARGET]
        )

        print("Random Forest model trained.")

    except Exception as e:

        model = None

        print(
            "Model training skipped:",
            e
        )


# ==========================================================
# BASIC VALUE CLEANING
# ==========================================================

def clean_value(value):

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (np.integer,)
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):
        return round(
            float(value),
            3
        )

    return value


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        uploaded=df is not None
    )


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        uploaded=df is not None
    )


# ==========================================================
# EDA PAGE
# ==========================================================

@app.route("/eda")
def eda():

    return render_template(
        "eda.html",
        uploaded=df is not None
    )


# ==========================================================
# PREPROCESSING PAGE
# ==========================================================

@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html",
        uploaded=df is not None
    )


# ==========================================================
# DATASET STUDIO
# ==========================================================

@app.route(
    "/dataset-studio",
    methods=["GET", "POST"]
)
def dataset_studio():

    message = None
    error = None

    if request.method == "POST":

        file = request.files.get(
            "dataset"
        )

        if not file:

            error = "Please choose a CSV file."

        elif not file.filename.lower().endswith(
            ".csv"
        ):

            error = "Only CSV files are supported."

        else:

            # --------------------------------------------------
            # DO NOT overwrite the original dataset.
            # --------------------------------------------------

            filename = (
                f"uploaded_{int(time.time())}.csv"
            )

            upload_path = (
                UPLOAD_FOLDER /
                filename
            )

            file.save(upload_path)

            try:

                # Check whether CSV is readable
                test_df = pd.read_csv(
                    upload_path
                )

                if test_df.empty:

                    raise ValueError(
                        "The uploaded CSV is empty."
                    )

                load_active_dataset(
                    upload_path
                )

                message = (
                    f"'{file.filename}' uploaded successfully "
                    f"and is now the active dataset."
                )

            except Exception as e:

                error = (
                    f"Could not process the file: {e}"
                )


    rows = None
    cols = None
    nulls = None
    dup = None
    summary = None

    if df is not None:

        rows, cols = df.shape

        nulls = int(
            df.isnull()
            .sum()
            .sum()
        )

        dup = int(
            df.duplicated()
            .sum()
        )

        summary = (
            df.describe(
                include="all"
            )
            .fillna("")
            .round(2)
            .to_html(
                classes="summary-table"
            )
        )

    return render_template(
        "dataset_studio.html",
        uploaded=df is not None,
        rows=rows,
        cols=cols,
        nulls=nulls,
        dup=dup,
        summary=summary,
        message=message,
        error=error
    )


# ==========================================================
# ABOUT
# ==========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================================
# API: OVERVIEW
# ==========================================================

@app.get("/api/overview")
def overview():

    if df is None:

        return jsonify({
            "uploaded": False
        })

    total = len(df)

    if "dropout" in df.columns:

        dropouts = int(
            pd.to_numeric(
                df["dropout"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        retained = total - dropouts

        risk_rate = (
            round(
                dropouts / total * 100,
                1
            )
            if total
            else 0
        )

    else:

        dropouts = None
        retained = None
        risk_rate = None


    return jsonify({

        "uploaded": True,

        "total": total,

        "dropouts": dropouts,

        "retained": retained,

        "risk_rate": risk_rate,

        "modules": (
            int(df["code_module"].nunique())
            if "code_module" in df.columns
            else None
        ),

        "avg_clicks": (
            round(
                pd.to_numeric(
                    df["total_vle_clicks"],
                    errors="coerce"
                ).mean()
            )
            if "total_vle_clicks" in df.columns
            else None
        ),

        "avg_score": (
            round(
                pd.to_numeric(
                    df["avg_assessment_score"],
                    errors="coerce"
                ).mean(),
                1
            )
            if "avg_assessment_score" in df.columns
            else None
        ),

        "columns": int(df.shape[1]),

        "missing": int(
            df.isnull().sum().sum()
        ),

        "duplicates": int(
            df.duplicated().sum()
        ),

        "numeric_columns": int(
            len(
                df.select_dtypes(
                    include=np.number
                ).columns
            )
        ),

        "categorical_columns": int(
            len(
                df.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns
            )
        )
    })


# ==========================================================
# API: DASHBOARD CHARTS
# ==========================================================

@app.get("/api/charts")
def charts():

    if df is None:

        return jsonify({
            "uploaded": False
        })

    response = {
        "uploaded": True
    }


    # ------------------------------------------------------
    # MODULE
    # ------------------------------------------------------

    if {
        "code_module",
        "dropout"
    }.issubset(df.columns):

        module = (
            df.groupby(
                "code_module"
            )["dropout"]
            .agg(["sum", "count"])
            .reset_index()
        )

        module["rate"] = (
            module["sum"]
            / module["count"]
            * 100
        )

        response["module"] = {
            "labels":
                module["code_module"].tolist(),

            "values":
                [
                    round(x, 1)
                    for x in module["rate"]
                ]
        }

    else:

        response["module"] = {
            "labels": [],
            "values": []
        }


    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------

    if {
        "highest_education",
        "dropout"
    }.issubset(df.columns):

        education = (
            df.groupby(
                "highest_education"
            )["dropout"]
            .mean()
            .mul(100)
            .sort_values()
        )

        response["education"] = {

            "labels":
                education.index.tolist(),

            "values":
                [
                    round(x, 1)
                    for x in education.values
                ]
        }

    else:

        response["education"] = {
            "labels": [],
            "values": []
        }


    # ------------------------------------------------------
    # AGE
    # ------------------------------------------------------

    if {
        "age_band",
        "dropout"
    }.issubset(df.columns):

        age = (
            df.groupby(
                "age_band"
            )["dropout"]
            .mean()
            .mul(100)
        )

        order = [
            "0-35",
            "35-55",
            "55<="
        ]

        age = age.reindex(
            [
                x
                for x in order
                if x in age.index
            ]
        )

        response["age"] = {

            "labels":
                age.index.tolist(),

            "values":
                [
                    round(x, 1)
                    for x in age.values
                ]
        }

    else:

        response["age"] = {
            "labels": [],
            "values": []
        }


    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    if "final_result" in df.columns:

        result = (
            df["final_result"]
            .value_counts()
        )

        response["result"] = {

            "labels":
                result.index.tolist(),

            "values":
                result.astype(
                    int
                ).tolist()
        }

    else:

        response["result"] = {
            "labels": [],
            "values": []
        }


    return jsonify(response)


# ==========================================================
# API: EDA
# ==========================================================

@app.get("/api/eda")
def eda_api():

    if df is None:

        return jsonify({
            "uploaded": False
        })


    # ------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "feature",
        "count"
    ]

    missing["percentage"] = (
        missing["count"]
        / len(df)
        * 100
    ).round(2)

    missing = missing[
        missing["count"] > 0
    ]

    missing_data = (
        missing
        .sort_values(
            "count",
            ascending=False
        )
        .to_dict(
            orient="records"
        )
    )


    # ------------------------------------------------------
    # OUTLIERS
    # ------------------------------------------------------

    outliers = []

    numeric = df.select_dtypes(
        include=np.number
    )

    for column in numeric.columns:

        if column == "dropout":
            continue

        series = numeric[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (series < lower)
                |
                (series > upper)
            ).sum()
        )

        outliers.append({

            "feature": column,

            "count": count,

            "percentage": round(
                count / len(series) * 100,
                2
            )
        })


    # ------------------------------------------------------
    # DATA TYPES
    # ------------------------------------------------------

    dtype_data = []

    for column in df.columns:

        dtype_data.append({

            "feature": column,

            "dtype": str(
                df[column].dtype
            ),

            "unique": int(
                df[column].nunique(
                    dropna=True
                )
            ),

            "missing": int(
                df[column].isnull().sum()
            )
        })


    # ------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------

    summary = (
        df.describe(
            include="all"
        )
        .fillna("")
        .round(3)
    )

    summary_html = summary.to_html(
        classes="summary-table",
        border=0
    )


    return jsonify({

        "uploaded": True,

        "missing": missing_data,

        "outliers": outliers,

        "dtypes": dtype_data,

        "summary": summary_html
    })


# ==========================================================
# API: PREPROCESSING INFORMATION
# ==========================================================

@app.get("/api/preprocessing/info")
def preprocessing_info():

    if df is None:

        return jsonify({
            "uploaded": False
        })


    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )


    missing = []

    for column in df.columns:

        count = int(
            df[column].isnull().sum()
        )

        if count > 0:

            missing.append({

                "feature": column,

                "missing": count,

                "percentage": round(
                    count / len(df) * 100,
                    2
                )
            })


    return jsonify({

        "uploaded": True,

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "numeric": numeric_columns,

        "categorical": categorical_columns,

        "missing": missing
    })


# ==========================================================
# API: RUN EXISTING PREPROCESSING SCRIPT
# ==========================================================

@app.post("/api/preprocessing/run")
def run_preprocessing():

    if df is None:

        return jsonify({
            "success": False,
            "error":
                "Please upload a dataset first."
        }), 400


    data = request.get_json(
        force=True
    )

    technique = data.get(
        "technique"
    )


    scripts = {

        "missing":
            "learnlift_clean_del_mean_model_missing_imputer.py",

        "label":
            "learnlift_clean_label_encode.py",

        "onehot":
            "learnlift_clean_one_hot_encod.py",

        "ordinal":
            "learnlift_clean_ordinal_encod.py",

        "target":
            "learnlift_clean_target_encode.py",

        "scaling":
            "learnlift_clean_minmax_stand_normal.py"
    }


    if technique not in scripts:

        return jsonify({

            "success": False,

            "error":
                "Unknown preprocessing technique."
        }), 400


    script_path = (
        BASE
        / "src"
        / scripts[technique]
    )


    if not script_path.exists():

        return jsonify({

            "success": False,

            "error":
                f"Script not found: {script_path.name}"
        }), 404


    # ------------------------------------------------------
    # Environment variable tells the existing script
    # which dataset to process.
    # ------------------------------------------------------

    env = os.environ.copy()

    env[
        "LEARNLIFT_INPUT_FILE"
    ] = str(active_dataset)

    env[
        "LEARNLIFT_PROJECT_DIR"
    ] = str(BASE)


    try:

        result = subprocess.run(

            [
                sys.executable,
                str(script_path)
            ],

            cwd=str(BASE),

            env=env,

            capture_output=True,

            text=True,

            timeout=180
        )


        if result.returncode != 0:

            return jsonify({

                "success": False,

                "error":
                    result.stderr[-4000:]

            }), 500


        return jsonify({

            "success": True,

            "technique": technique,

            "message":
                "Preprocessing completed successfully.",

            "output":
                result.stdout[-5000:]
        })


    except subprocess.TimeoutExpired:

        return jsonify({

            "success": False,

            "error":
                "Preprocessing took too long."
        }), 500


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================================
# API: STUDENTS
# ==========================================================

@app.get("/api/students")
def students():

    if df is None:

        return jsonify([])


    q = (
        request.args
        .get("q", "")
        .strip()
        .lower()
    )


    try:

        limit = min(
            int(
                request.args.get(
                    "limit",
                    100
                )
            ),
            250
        )

    except:

        limit = 100


    columns = [

        "id_student",
        "code_module",
        "gender",
        "age_band",
        "highest_education",
        "avg_assessment_score",
        "total_vle_clicks",
        "final_result",
        "dropout"
    ]


    available = [
        c
        for c in columns
        if c in df.columns
    ]


    view = df[
        available
    ].copy()


    if q:

        mask = pd.Series(
            False,
            index=view.index
        )

        for column in available:

            mask |= (
                view[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    q,
                    na=False
                )
            )

        view = view[mask]


    view = view.head(
        limit
    )


    rows = [

        {
            key: clean_value(value)
            for key, value in row.items()
        }

        for row in
        view.to_dict(
            orient="records"
        )
    ]


    return jsonify(rows)


# ==========================================================
# PREDICTION
# ==========================================================

@app.post("/api/predict")
def predict():

    if df is None:

        return jsonify({

            "error":
                "Upload a dataset first."
        }), 400


    if model is None:

        return jsonify({

            "error":
                "The dataset cannot train the prediction model. "
                "Make sure it contains the dropout column "
                "and required learner features."
        }), 400


    data = request.get_json(
        force=True
    )


    available_features = [
        c
        for c in FEATURES
        if c in df.columns
    ]


    row = {}


    for col in available_features:

        value = data.get(col)

        if col not in CAT_COLS:

            if value in ("", None):

                row[col] = np.nan

            else:

                try:

                    row[col] = float(
                        value
                    )

                except:

                    row[col] = np.nan

        else:

            row[col] = (
                value
                if value not in ("", None)
                else np.nan
            )


    sample = pd.DataFrame(
        [row],
        columns=available_features
    )


    probability = float(
        model
        .predict_proba(sample)[0][1]
    )


    prediction = int(
        probability >= 0.5
    )


    if probability >= 0.70:

        level = "High"

        message = (
            "The learner shows several signals "
            "associated with withdrawal risk."
        )

    elif probability >= 0.40:

        level = "Moderate"

        message = (
            "The learner may benefit from "
            "closer academic engagement."
        )

    else:

        level = "Low"

        message = (
            "The learner currently shows a "
            "lower predicted withdrawal risk."
        )


    return jsonify({

        "prediction": prediction,

        "probability": round(
            probability * 100,
            1
        ),

        "level": level,

        "message": message
    })


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )