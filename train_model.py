import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline
import joblib

# Paths
DATA_PATH = "data/processed/train.csv"
MODEL_PATH = "models/best_model.pickle"

# Load data
df = pd.read_csv(DATA_PATH)

# Auto-detect target column (prefers 'fraud_reported', then 'claim_status', then 'fraud')
target_candidates = [col for col in df.columns if col.lower() in ["fraud_reported", "claim_status", "fraud"]]
if not target_candidates:
    raise ValueError("No target column found. Please ensure your data contains 'fraud_reported', 'claim_status', or 'fraud'.")
target_col = target_candidates[0]

# Remove ID columns and target from features
drop_cols = [col for col in ["claim_number", target_col] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df[target_col]

# Detect categorical features
categorical_features = X.select_dtypes(include="object").columns.tolist()

# Preprocessing
column_transformer = make_column_transformer(
    (OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    remainder="passthrough"
)
scaler = MinMaxScaler()

# Model
xgb_clf = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Pipeline
pipeline = make_pipeline(column_transformer, scaler, xgb_clf)

# Train
pipeline.fit(X, y)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Model trained and saved to {MODEL_PATH}")