# import pickle
import joblib
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

import utils


project_dir = Path(__file__).resolve().parents[1]
app = Flask(__name__)


# with open(project_dir / "models" / "best_model.pickle", "rb") as f:
#     model = pickle.load(f)

with open(project_dir / "models" / "best_model.pickle", "rb") as f:
    model = joblib.load(f)


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     form_data = request.form.to_dict()
#     preprocessed_data = utils.preprocess(form_data)
#     probability = model.predict_proba(pd.DataFrame(preprocessed_data, index=[0]))
#     return render_template("result.html", probability=f"{probability[0, 1] * 100:.2f}")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    preprocessed_data = utils.preprocess(form_data)
    # Ensure DataFrame columns match model's expected input features
    input_df = pd.DataFrame([preprocessed_data])
    # Get expected feature names from the pipeline
    expected_features = model.named_steps['columntransformer'].feature_names_in_
    # Reorder columns to match training
    input_df = input_df.reindex(columns=expected_features)
    probability = model.predict_proba(input_df)
    return render_template("result.html", probability=f"{probability[0, 1] * 100:.2f}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
