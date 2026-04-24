from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
import joblib
import os
import json
from typing import List
from datetime import datetime

app = FastAPI(title="Fraud Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "model", "scaler.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "model", "shap_explainer.pkl")
STATS_PATH     = os.path.join(BASE_DIR, "model", "stats.json")

# ─── Globals ─────────────────────────────────────────────────────────────────
pipeline  = None
scaler    = None
explainer = None

@app.on_event("startup")
def load_model():
    global pipeline, scaler, explainer
    if os.path.exists(MODEL_PATH):
        pipeline  = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        explainer = joblib.load(EXPLAINER_PATH)
        print("✅ Model loaded")
    else:
        print("⚠️  Model not found. Run train.py first.")

# ─── Schemas ─────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    @validator("Amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    shap_values: List[float]
    top_features: List[dict]
    timestamp: str

# ─── Helpers ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def get_risk_level(prob: float) -> str:
    if prob > 0.7:   return "HIGH"
    elif prob > 0.3: return "MEDIUM"
    else:            return "LOW"

def get_top_features(shap_vals, n=5):
    pairs = sorted(
        zip(FEATURE_NAMES, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:n]
    return [{"feature": k, "shap_value": round(float(v), 6)} for k, v in pairs]

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([[
        tx.Time, tx.Amount,
        *[getattr(tx, f"V{i}") for i in range(1, 29)]
    ]], columns=FEATURE_NAMES)

    prob = float(pipeline.predict_proba(row)[0][1])
    pred = int(prob > 0.5)

    # SHAP uses scaler separately
    row_scaled = scaler.transform(row)
    sv = explainer.shap_values(row_scaled)
    if isinstance(sv, list):
        sv = sv[1]
    shap_vals = sv[0].tolist()

    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 6),
        risk_level=get_risk_level(prob),
        shap_values=shap_vals,
        top_features=get_top_features(shap_vals),
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/stats")
def stats():
    if not os.path.exists(STATS_PATH):
        raise HTTPException(status_code=404, detail="Stats not available. Run train.py first.")
    with open(STATS_PATH) as f:
        return json.load(f)