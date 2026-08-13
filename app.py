from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv")

app = Flask(__name__)
df = pd.read_csv(DATA_PATH)

# Features chosen for early student-risk prediction.
# final_result, final_score and dropout are excluded because they are outcomes.
FEATURES = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability",
    "avg_assessment_score", "assessment_count", "total_vle_clicks",
    "date_registration", "date_unregistration",
    "module_presentation_length"
]
TARGET = "dropout"

CAT_COLS = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability"
]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_COLS),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), NUM_COLS)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(df[FEATURES], df[TARGET])


def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 3)
    return v


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/explore")
def explore():
    return render_template("explore.html")


@app.route("/dataset-studio", methods=["GET", "POST"])
def dataset_studio():
    data_path = Path(
        r"C:\Users\punug\OneDrive\Desktop\fourth-sem\machine_learning\LearnLift\Dataset\learnlift_updated_dataset_raw.csv")
    message = None

    if request.method == "POST":
        file = request.files.get("dataset")
        if file and file.filename.lower().endswith(".csv"):
            file.save(data_path)
            message = f"'{file.filename}' uploaded and now in use."
        else:
            message = "Please upload a valid .csv file."

    d = pd.read_csv(data_path)
    rows, cols = d.shape
    nulls = int(d.isnull().sum().sum())
    dup = int(d.duplicated().sum())
    summary = d.describe(include='all').fillna("").round(2).to_html(classes="summary-table")

    return render_template(
        "dataset_studio.html",
        rows=rows, cols=cols, nulls=nulls, dup=dup,
        summary=summary, message=message
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.get("/api/overview")
def overview():
    total = len(df)
    dropouts = int(df["dropout"].sum())
    retained = total - dropouts
    return jsonify({
        "total": total,
        "dropouts": dropouts,
        "retained": retained,
        "risk_rate": round(dropouts / total * 100, 1),
        "modules": int(df["code_module"].nunique()),
        "avg_clicks": round(df["total_vle_clicks"].mean()),
        "avg_score": round(df["avg_assessment_score"].mean(), 1)
    })


@app.get("/api/charts")
def charts():
    module = df.groupby("code_module")["dropout"].agg(["sum", "count"]).reset_index()
    module["rate"] = module["sum"] / module["count"] * 100

    education = df.groupby("highest_education")["dropout"].mean().mul(100).sort_values()
    age = df.groupby("age_band")["dropout"].mean().mul(100).reindex(["0-35", "35-55", "55<="])
    result = df["final_result"].value_counts()

    return jsonify({
        "module": {
            "labels": module["code_module"].tolist(),
            "values": [round(x, 1) for x in module["rate"]]
        },
        "education": {
            "labels": education.index.tolist(),
            "values": [round(x, 1) for x in education.values]
        },
        "age": {
            "labels": age.index.tolist(),
            "values": [round(x, 1) for x in age.values]
        },
        "result": {
            "labels": result.index.tolist(),
            "values": result.astype(int).tolist()
        }
    })


@app.get("/api/students")
def students():
    q = request.args.get("q", "").strip().lower()
    limit = min(int(request.args.get("limit", 100)), 250)

    cols = [
        "id_student", "code_module", "gender", "age_band",
        "highest_education", "avg_assessment_score",
        "total_vle_clicks", "final_result", "dropout"
    ]
    view = df[cols].copy()

    if q:
        mask = (
            view["id_student"].astype(str).str.contains(q, na=False)
            | view["code_module"].str.lower().str.contains(q, na=False)
            | view["final_result"].str.lower().str.contains(q, na=False)
        )
        view = view[mask]

    view = view.head(limit)
    rows = [{k: clean_value(v) for k, v in row.items()}
            for row in view.to_dict(orient="records")]
    return jsonify(rows)


@app.post("/api/predict")
def predict():
    data = request.get_json(force=True)

    row = {}
    for col in FEATURES:
        value = data.get(col)
        if col in NUM_COLS:
            if value in ("", None):
                row[col] = np.nan
            else:
                row[col] = float(value)
        else:
            row[col] = value if value not in ("", None) else np.nan

    sample = pd.DataFrame([row], columns=FEATURES)
    probability = float(model.predict_proba(sample)[0][1])
    prediction = int(probability >= 0.5)

    if probability >= 0.70:
        level = "High"
        message = "The learner shows several signals associated with withdrawal risk."
    elif probability >= 0.40:
        level = "Moderate"
        message = "The learner may benefit from closer academic engagement."
    else:
        level = "Low"
        message = "The learner currently shows a lower predicted withdrawal risk."

    return jsonify({
        "prediction": prediction,
        "probability": round(probability * 100, 1),
        "level": level,
        "message": message
    })


if __name__ == "__main__":
    app.run(debug=True)