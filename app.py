from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename

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


# ============================================================
# APPLICATION
# ============================================================

BASE = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE / "uploads"
DATASET_FOLDER = BASE / "Dataset"
OUTPUT_FOLDER = DATASET_FOLDER
SRC_FOLDER = BASE / "src"

UPLOAD_FOLDER.mkdir(exist_ok=True)
DATASET_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)

# Maximum upload size = 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# GLOBAL DATA
# ============================================================

active_dataset = None
df = None
model = None


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

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


# ============================================================
# PREPROCESSING SCRIPTS
# ============================================================

PREPROCESSING = {

    "missing": {
        "label": "Missing value handling",
        "description":
            "Deletion, mean/median imputation and model-based imputation.",
        "script":
            "learnlift_clean_del_mean_model_missing_imputer.py",
        "output":
            "clean_del_mean_model.csv"
    },

    "label": {
        "label": "Label encoding",
        "description":
            "Converts categorical values into integer labels.",
        "script":
            "learnlift_clean_label_encode.py",
        "output":
            "clean_label_encode.csv"
    },

    "onehot": {
        "label": "One-hot encoding",
        "description":
            "Creates binary indicator columns for categorical features.",
        "script":
            "learnlift_clean_one_hot_encod.py",
        "output":
            "clean_one_hot_encoding.csv"
    },

    "ordinal": {
        "label": "Ordinal encoding",
        "description":
            "Maps ordered categories to numeric values.",
        "script":
            "learnlift_clean_ordinal_encod.py",
        "output":
            "clean_ordinal_encode.csv"
    },

    "target": {
        "label": "Target encoding",
        "description":
            "Encodes categories using the target mean.",
        "script":
            "learnlift_clean_target_encode.py",
        "output":
            "clean_target_encode.csv"
    },

    "scaling": {
        "label": "Scaling & normalization",
        "description":
            "Applies standardization, min-max scaling and normalization.",
        "script":
            "learnlift_clean_minmax_stand_normal.py",
        "output":
            "clean_minmax_stand_normal.csv"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    name = secure_filename(name)

    if not name:
        return "processed_dataset.csv"

    return name


def clean_value(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return round(float(value), 3)

    return value


def current_df():

    global df

    if active_dataset is not None:

        path = Path(active_dataset)

        if path.exists():

            try:
                df = pd.read_csv(path)

            except Exception as exc:

                print("Could not reload active dataset:", exc)

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global model

    model = None

    if df is None:
        return

    if TARGET not in df.columns:
        print("Model training skipped: dropout column not found.")
        return

    available = [
        c for c in FEATURES
        if c in df.columns
    ]

    if not available:
        print("Model training skipped: no usable features.")
        return

    target = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid = target.notna()

    if valid.sum() == 0:
        print("Model training skipped: no valid target values.")
        return

    target = target.loc[valid].astype(int)

    if target.nunique() < 2:
        print(
            "Model training skipped: target has fewer than 2 classes."
        )
        return

    categorical = [
        c for c in available
        if c in CAT_COLS
    ]

    numerical = [
        c for c in available
        if c not in categorical
    ]

    transformers = []

    # --------------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------------

    if categorical:

        try:

            onehot = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        except TypeError:

            onehot = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        transformers.append(
            (
                "cat",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "onehot",
                            onehot
                        )
                    ]
                ),

                categorical
            )
        )

    # --------------------------------------------------------
    # NUMERICAL PIPELINE
    # --------------------------------------------------------

    if numerical:

        transformers.append(
            (
                "num",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        )
                    ]
                ),

                numerical
            )
        )

    if not transformers:
        return

    # --------------------------------------------------------
    # COMPLETE MODEL
    # --------------------------------------------------------

    model = Pipeline(
        steps=[

            (
                "prep",

                ColumnTransformer(
                    transformers=transformers
                )
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
        ]
    )

    try:

        model.fit(
            df.loc[valid, available],
            target
        )

        print("Random Forest model trained successfully.")

    except Exception as exc:

        model = None

        print(
            "Model training skipped:",
            exc
        )


# ============================================================
# LOAD ACTIVE DATASET
# ============================================================

def load_active_dataset(path):

    global active_dataset
    global df
    global model

    active_dataset = Path(path)

    df = pd.read_csv(active_dataset)

    model = None

    print(
        "Dataset loaded:",
        active_dataset
    )

    print(
        "Shape:",
        df.shape
    )

    train_model()


# ============================================================
# RESTORE LAST UPLOADED DATASET
# ============================================================

def initialize_active_dataset():

    global active_dataset
    global df

    uploaded_files = list(
        UPLOAD_FOLDER.glob("uploaded_*.csv")
    )

    if not uploaded_files:
        return

    latest = max(
        uploaded_files,
        key=lambda p: p.stat().st_mtime
    )

    try:

        load_active_dataset(latest)

        print(
            "Restored last uploaded dataset:",
            latest.name
        )

    except Exception as exc:

        print(
            "Could not restore previous dataset:",
            exc
        )


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template(
        "overview.html",
        uploaded=df is not None
    )


@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        uploaded=df is not None
    )


@app.route("/eda")
def eda():

    return render_template(
        "eda.html",
        uploaded=df is not None
    )


@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html",
        uploaded=df is not None
    )


@app.route("/explore")
def explore():

    return render_template(
        "explore.html",
        uploaded=df is not None
    )


@app.route("/predict")
def predict_page():

    return render_template(
        "predict.html",
        uploaded=df is not None
    )


@app.route("/about")
def about():

    return render_template(
        "about.html",
        uploaded=df is not None
    )


# ============================================================
# DATASET STUDIO / CSV UPLOAD
# ============================================================

@app.route(
    "/dataset-studio",
    methods=["GET", "POST"]
)
def dataset_studio():

    # --------------------------------------------------------
    # POST = UPLOAD
    # --------------------------------------------------------

    if request.method == "POST":

        print("\n========== DATASET UPLOAD ==========")

        print(
            "request.files:",
            request.files
        )

        # ----------------------------------------------------
        # ACCEPT BOTH FIELD NAMES
        #
        # New overview.html uses:
        #       file
        #
        # Older Dataset Studio uses:
        #       dataset
        #
        # This prevents the current error.
        # ----------------------------------------------------

        file = request.files.get("file")

        if file is None:

            file = request.files.get("dataset")

        # ----------------------------------------------------
        # NO FILE
        # ----------------------------------------------------

        if file is None:

            print("ERROR: No file field received.")

            return jsonify(
                {
                    "success": False,
                    "message": "No file selected."
                }
            ), 400

        # ----------------------------------------------------
        # EMPTY FILE NAME
        # ----------------------------------------------------

        if not file.filename:

            print("ERROR: Empty filename.")

            return jsonify(
                {
                    "success": False,
                    "message": "No file selected."
                }
            ), 400

        # ----------------------------------------------------
        # CSV CHECK
        # ----------------------------------------------------

        if not file.filename.lower().endswith(".csv"):

            return jsonify(
                {
                    "success": False,
                    "message":
                        "Only CSV files are supported."
                }
            ), 400

        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        original_name = safe_filename(
            file.filename
        )

        timestamp = int(
            time.time() * 1000
        )

        filename = (
            f"uploaded_{timestamp}_{original_name}"
        )

        upload_path = (
            UPLOAD_FOLDER / filename
        )

        try:

            # Save file
            file.save(upload_path)

            print(
                "Saved:",
                upload_path
            )

            # ------------------------------------------------
            # READ CSV
            # ------------------------------------------------

            test = pd.read_csv(
                upload_path
            )

            # ------------------------------------------------
            # EMPTY DATASET CHECK
            # ------------------------------------------------

            if (
                test.empty
                or len(test.columns) == 0
            ):

                upload_path.unlink(
                    missing_ok=True
                )

                return jsonify(
                    {
                        "success": False,
                        "message":
                            "The uploaded CSV is empty."
                    }
                ), 400

            # ------------------------------------------------
            # LOAD ACTIVE DATASET
            # ------------------------------------------------

            load_active_dataset(
                upload_path
            )

            print(
                "Dataset successfully activated."
            )

            print(
                "Rows:",
                len(df)
            )

            print(
                "Columns:",
                len(df.columns)
            )

            # ------------------------------------------------
            # RETURN JSON
            # ------------------------------------------------

            return jsonify(
                {
                    "success": True,

                    "message":
                        f"{original_name} is now the active dataset.",

                    "filename":
                        original_name,

                    "rows":
                        int(df.shape[0]),

                    "columns":
                        int(df.shape[1])
                }
            )

        except Exception as exc:

            print(
                "UPLOAD ERROR:",
                exc
            )

            return jsonify(
                {
                    "success": False,
                    "message":
                        f"Could not process the file: {exc}"
                }
            ), 500

    # ========================================================
    # GET
    # ========================================================

    data = current_df()

    rows = (
        int(data.shape[0])
        if data is not None
        else None
    )

    cols = (
        int(data.shape[1])
        if data is not None
        else None
    )

    nulls = (
        int(data.isnull().sum().sum())
        if data is not None
        else None
    )

    dup = (
        int(data.duplicated().sum())
        if data is not None
        else None
    )

    active_name = (
        Path(active_dataset).name
        if active_dataset is not None
        else None
    )

    return render_template(
        "dataset_studio.html",

        uploaded=data is not None,

        rows=rows,
        cols=cols,
        nulls=nulls,
        dup=dup,

        active_name=active_name
    )


# ============================================================
# OVERVIEW API
# ============================================================

@app.get("/api/overview")
def overview_api():

    data = current_df()

    if data is None:

        return jsonify(
            {
                "uploaded": False
            }
        )

    total = len(data)

    if TARGET in data.columns:

        dropout_values = pd.to_numeric(
            data[TARGET],
            errors="coerce"
        )

        dropouts = int(
            dropout_values
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
            else None
        )

    else:

        dropouts = None
        retained = None
        risk_rate = None

    avg_clicks = None

    if "total_vle_clicks" in data.columns:

        avg_clicks = round(
            pd.to_numeric(
                data["total_vle_clicks"],
                errors="coerce"
            ).mean(),
            1
        )

    avg_score = None

    if "avg_assessment_score" in data.columns:

        avg_score = round(
            pd.to_numeric(
                data["avg_assessment_score"],
                errors="coerce"
            ).mean(),
            1
        )

    return jsonify(
        {
            "uploaded": True,

            "filename":
                Path(active_dataset).name
                if active_dataset
                else "dataset.csv",

            "total":
                total,

            "dropouts":
                dropouts,

            "retained":
                retained,

            "risk_rate":
                risk_rate,

            "modules":
                int(
                    data["code_module"].nunique()
                )
                if "code_module" in data.columns
                else None,

            "avg_clicks":
                avg_clicks,

            "avg_score":
                avg_score,

            "rows":
                int(data.shape[0]),

            "columns":
                int(data.shape[1]),

            "missing":
                int(
                    data.isnull()
                    .sum()
                    .sum()
                ),

            "duplicates":
                int(
                    data.duplicated()
                    .sum()
                ),

            "numeric_columns":
                int(
                    len(
                        data.select_dtypes(
                            include=np.number
                        ).columns
                    )
                ),

            "categorical_columns":
                int(
                    len(
                        data.select_dtypes(
                            include=[
                                "object",
                                "category",
                                "bool"
                            ]
                        ).columns
                    )
                ),

            "target":
                TARGET
                if TARGET in data.columns
                else None
        }
    )


# ============================================================
# CHART API
# ============================================================

@app.get("/api/charts")
def charts():

    data = current_df()

    if data is None:

        return jsonify(
            {
                "uploaded": False
            }
        )

    response = {
        "uploaded": True
    }

    # --------------------------------------------------------
    # MODULE
    # --------------------------------------------------------

    if {
        "code_module",
        TARGET
    }.issubset(data.columns):

        g = data.groupby(
            "code_module"
        )[TARGET].agg(
            ["sum", "count"]
        )

        values = (
            g["sum"]
            / g["count"]
            * 100
        ).round(1)

        response["module"] = {
            "labels":
                g.index.astype(str).tolist(),

            "values":
                values.tolist()
        }

    else:

        response["module"] = {
            "labels": [],
            "values": []
        }

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if {
        "highest_education",
        TARGET
    }.issubset(data.columns):

        g = (
            data.groupby(
                "highest_education"
            )[TARGET]
            .mean()
            .mul(100)
            .sort_values()
        )

        response["education"] = {
            "labels":
                g.index.astype(str).tolist(),

            "values":
                g.round(1).tolist()
        }

    else:

        response["education"] = {
            "labels": [],
            "values": []
        }

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if {
        "age_band",
        TARGET
    }.issubset(data.columns):

        g = (
            data.groupby(
                "age_band"
            )[TARGET]
            .mean()
            .mul(100)
        )

        order = [
            "0-35",
            "35-55",
            "55<="
        ]

        existing = [
            x for x in order
            if x in g.index
        ]

        if existing:

            g = g.reindex(existing)

        response["age"] = {
            "labels":
                g.index.astype(str).tolist(),

            "values":
                g.round(1).tolist()
        }

    else:

        response["age"] = {
            "labels": [],
            "values": []
        }

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if "final_result" in data.columns:

        g = data[
            "final_result"
        ].value_counts(
            dropna=False
        )

        response["result"] = {
            "labels":
                g.index.astype(str).tolist(),

            "values":
                g.astype(int).tolist()
        }

    else:

        response["result"] = {
            "labels": [],
            "values": []
        }

    return jsonify(response)


# ============================================================
# EDA API
# ============================================================

@app.get("/api/eda")
def eda_api():

    data = current_df()

    if data is None:

        return jsonify(
            {
                "uploaded": False
            }
        )

    numeric_cols = (
        data
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_cols = (
        data
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing = []

    for column in data.columns:

        count = int(
            data[column].isnull().sum()
        )

        missing.append(
            {
                "feature":
                    column,

                "count":
                    count,

                "percentage":
                    round(
                        count / len(data) * 100,
                        2
                    )
                    if len(data)
                    else 0
            }
        )

    missing = sorted(
        [
            x for x in missing
            if x["count"] > 0
        ],
        key=lambda x: x["count"],
        reverse=True
    )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    outliers = []

    for column in numeric_cols:

        if column == TARGET:
            continue

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

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

        outliers.append(
            {
                "feature":
                    column,

                "count":
                    count,

                "percentage":
                    round(
                        count / len(series) * 100,
                        2
                    ),

                "lower":
                    round(
                        float(lower),
                        3
                    ),

                "upper":
                    round(
                        float(upper),
                        3
                    )
            }
        )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    dtype_data = []

    for column in data.columns:

        dtype_data.append(
            {
                "feature":
                    column,

                "dtype":
                    str(data[column].dtype),

                "unique":
                    int(
                        data[column]
                        .nunique(
                            dropna=True
                        )
                    ),

                "missing":
                    int(
                        data[column]
                        .isnull()
                        .sum()
                    )
            }
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = []

    for column in numeric_cols:

        series = pd.to_numeric(
            data[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        statistics.append(
            {
                "feature":
                    column,

                "mean":
                    round(
                        float(series.mean()),
                        3
                    ),

                "median":
                    round(
                        float(series.median()),
                        3
                    ),

                "std":
                    round(
                        float(series.std()),
                        3
                    ),

                "min":
                    round(
                        float(series.min()),
                        3
                    ),

                "max":
                    round(
                        float(series.max()),
                        3
                    )
            }
        )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    correlation = {}

    if len(numeric_cols) >= 2:

        matrix = (
            data[numeric_cols]
            .corr(
                numeric_only=True
            )
            .round(3)
        )

        correlation = {
            str(i):
                {
                    str(j):
                        clean_value(value)

                    for j, value
                    in row.items()
                }

            for i, row
            in matrix.to_dict().items()
        }

    return jsonify(
        {
            "uploaded": True,

            "rows":
                int(len(data)),

            "columns":
                int(data.shape[1]),

            "numeric":
                numeric_cols,

            "categorical":
                categorical_cols,

            "missing":
                missing,

            "outliers":
                outliers,

            "dtypes":
                dtype_data,

            "statistics":
                statistics,

            "correlation":
                correlation,

            "target":
                TARGET
                if TARGET in data.columns
                else None
        }
    )


# ============================================================
# PREPROCESSING INFORMATION API
# ============================================================

@app.get("/api/preprocessing/info")
def preprocessing_info():

    data = current_df()

    if data is None:

        return jsonify(
            {
                "uploaded": False
            }
        )

    missing = []

    for column in data.columns:

        count = int(
            data[column].isnull().sum()
        )

        if count:

            missing.append(
                {
                    "feature":
                        column,

                    "missing":
                        count,

                    "percentage":
                        round(
                            count / len(data) * 100,
                            2
                        )
                }
            )

    return jsonify(
        {
            "uploaded": True,

            "filename":
                Path(active_dataset).name
                if active_dataset
                else "dataset.csv",

            "rows":
                int(data.shape[0]),

            "columns":
                int(data.shape[1]),

            "numeric":
                data
                .select_dtypes(
                    include=np.number
                )
                .columns
                .tolist(),

            "categorical":
                data
                .select_dtypes(
                    include=[
                        "object",
                        "category",
                        "bool"
                    ]
                )
                .columns
                .tolist(),

            "missing":
                missing,

            "duplicates":
                int(
                    data.duplicated()
                    .sum()
                )
        }
    )


# ============================================================
# RUN PREPROCESSING
# ============================================================

@app.post("/api/preprocessing/run")
def run_preprocessing():

    data = current_df()

    if (
        data is None
        or active_dataset is None
    ):

        return jsonify(
            {
                "success": False,
                "error":
                    "Please upload a dataset first."
            }
        ), 400

    payload = (
        request
        .get_json(
            silent=True
        )
        or {}
    )

    technique = payload.get(
        "technique"
    )

    if technique not in PREPROCESSING:

        return jsonify(
            {
                "success": False,
                "error":
                    "Unknown preprocessing technique."
            }
        ), 400

    info = PREPROCESSING[
        technique
    ]

    script_path = (
        SRC_FOLDER
        /
        info["script"]
    )

    if not script_path.exists():

        return jsonify(
            {
                "success": False,
                "error":
                    f"Script not found: "
                    f"{info['script']}"
            }
        ), 404

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

            timeout=240
        )

    except subprocess.TimeoutExpired:

        return jsonify(
            {
                "success": False,
                "error":
                    "Preprocessing took too long."
            }
        ), 500

    if result.returncode != 0:

        return jsonify(
            {
                "success": False,

                "error":
                    (
                        result.stderr[-5000:]
                        or
                        result.stdout[-5000:]
                    )
            }
        ), 500

    output_path = (
        DATASET_FOLDER
        /
        info["output"]
    )

    # --------------------------------------------------------
    # OUTPUT PREVIEW
    # --------------------------------------------------------

    preview = []
    shape = None

    if output_path.exists():

        out_df = pd.read_csv(
            output_path
        )

        shape = [
            int(out_df.shape[0]),
            int(out_df.shape[1])
        ]

        preview = [
            {
                key:
                    clean_value(value)

                for key, value
                in row.items()
            }

            for row
            in out_df
            .head(8)
            .to_dict(
                orient="records"
            )
        ]

    return jsonify(
        {
            "success": True,

            "technique":
                technique,

            "label":
                info["label"],

            "message":
                "Preprocessing completed successfully.",

            "output":
                result.stdout[-6000:],

            "filename":
                output_path.name
                if output_path.exists()
                else info["output"],

            "shape":
                shape,

            "preview":
                preview,

            "download":
                (
                    "/api/preprocessing/download/"
                    +
                    safe_filename(
                        output_path.name
                    )
                )
                if output_path.exists()
                else None
        }
    )


# ============================================================
# PREPROCESSING DOWNLOAD
# ============================================================

@app.get(
    "/api/preprocessing/download/<filename>"
)
def download_preprocessed(filename):

    filename = safe_filename(
        filename
    )

    return send_from_directory(
        DATASET_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# LEARNER EXPLORER
#
# IMPORTANT:
# MAXIMUM 100 RECORDS
# ============================================================

@app.get("/api/students")
def students():

    data = current_df()

    if data is None:

        return jsonify([])

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    query = (
        request
        .args
        .get("q", "")
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # ONLY THESE COLUMNS ARE SHOWN
    # --------------------------------------------------------

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
        column
        for column in columns
        if column in data.columns
    ]

    view = data[
        available
    ].copy()

    # --------------------------------------------------------
    # SEARCH ACROSS AVAILABLE COLUMNS
    # --------------------------------------------------------

    if query:

        mask = pd.Series(
            False,
            index=view.index
        )

        for column in available:

            mask |= (
                view[column]
                .astype(str)
                .str
                .lower()
                .str
                .contains(
                    query,
                    na=False
                )
            )

        view = view[
            mask
        ]

    # --------------------------------------------------------
    # HARD LIMIT = 100
    #
    # Even if the user sends ?limit=5000,
    # only 100 records are returned.
    # --------------------------------------------------------

    view = view.head(100)

    records = [
        {
            key:
                clean_value(value)

            for key, value
            in row.items()
        }

        for row
        in view.to_dict(
            orient="records"
        )
    ]

    return jsonify(
        records
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
def predict():

    if (
        df is None
        or model is None
    ):

        return jsonify(
            {
                "error":
                    "Upload a suitable dataset first. "
                    "The dataset must contain a usable "
                    "dropout target."
            }
        ), 400

    payload = (
        request
        .get_json(
            silent=True
        )
        or {}
    )

    available = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    row = {}

    for column in available:

        value = payload.get(
            column
        )

        if column not in CAT_COLS:

            try:

                row[column] = (
                    np.nan
                    if value in (None, "")
                    else float(value)
                )

            except Exception:

                row[column] = np.nan

        else:

            row[column] = (
                np.nan
                if value in (None, "")
                else value
            )

    sample = pd.DataFrame(
        [row],
        columns=available
    )

    try:

        probability = float(
            model
            .predict_proba(
                sample
            )[0][1]
        )

    except Exception as exc:

        return jsonify(
            {
                "error":
                    f"Prediction failed: {exc}"
            }
        ), 500

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.70:

        level = "High"

    elif probability >= 0.40:

        level = "Moderate"

    else:

        level = "Low"

    message = {

        "High":
            "The learner shows several signals associated with withdrawal risk.",

        "Moderate":
            "The learner may benefit from closer academic engagement.",

        "Low":
            "The learner currently shows a lower predicted withdrawal risk."

    }[level]

    return jsonify(
        {
            "prediction":
                int(
                    probability >= 0.5
                ),

            "probability":
                round(
                    probability * 100,
                    1
                ),

            "level":
                level,

            "message":
                message
        }
    )


# ============================================================
# STARTUP
# ============================================================

initialize_active_dataset()


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )