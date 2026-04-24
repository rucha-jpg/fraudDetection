"""
train.py  –  Run once to train the model and save artifacts.
Usage:  python train.py
"""

import os, json
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = "/Users/rucha/fraudDetection/data/creditcard.csv"
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("📦 Loading data...")
df = pd.read_csv(DATA_PATH)

FEATURE_NAMES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
X = df[FEATURE_NAMES].values
y = df["Class"].values

print(f"   Total rows : {len(df):,}")
print(f"   Fraud rows : {y.sum():,}  ({100*y.mean():.3f}%)")

# ─── Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── SMOTE to handle class imbalance ────────────────────────────────────────
print("⚖️  Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"   After SMOTE: {y_res.sum():,} fraud / {(y_res==0).sum():,} normal")

# ─── Scale ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_res_sc  = scaler.fit_transform(X_res)
X_test_sc = scaler.transform(X_test)

# ─── XGBoost ─────────────────────────────────────────────────────────────────
print("🚀 Training XGBoost...")
clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
clf.fit(
    X_res_sc, y_res,
    eval_set=[(X_test_sc, y_test)],
    verbose=50,
)

# ─── Evaluation ──────────────────────────────────────────────────────────────
print("\n📊 Evaluation on held-out test set:")
y_prob = clf.predict_proba(X_test_sc)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

roc   = roc_auc_score(y_test, y_prob)
pr    = average_precision_score(y_test, y_prob)
cm    = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, output_dict=True)

print(f"   ROC-AUC : {roc:.4f}")
print(f"   PR-AUC  : {pr:.4f}")
print(classification_report(y_test, y_pred))

# ─── SHAP Explainer ──────────────────────────────────────────────────────────
print("🔍 Building SHAP explainer (this takes ~1 min)...")
explainer = shap.TreeExplainer(clf)

# ─── Save artifacts ──────────────────────────────────────────────────────────
print("💾 Saving artifacts...")

# We save scaler + clf together so main.py only loads one pipeline object
class WrappedModel:
    """Wraps scaler + clf so API calls .predict_proba() directly."""
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf    = clf
    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

wrapped = WrappedModel(scaler, clf)
joblib.dump(wrapped,   f"{MODEL_DIR}/fraud_model.pkl")
joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")

# SHAP explainer expects raw (unscaled) — wrap it too
class WrappedExplainer:
    def __init__(self, explainer, scaler):
        self.explainer = explainer
        self.scaler    = scaler
    def shap_values(self, X):
        return self.explainer.shap_values(self.scaler.transform(X))

wrapped_exp = WrappedExplainer(explainer, scaler)
joblib.dump(wrapped_exp, f"{MODEL_DIR}/shap_explainer.pkl")

# Save stats for /stats endpoint
stats = {
    "roc_auc":  round(roc, 4),
    "pr_auc":   round(pr, 4),
    "confusion_matrix": cm,
    "classification_report": report,
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "fraud_rate": round(float(y.mean()), 6),
}
with open(f"{MODEL_DIR}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ All artifacts saved to ./model/")
print(f"   ROC-AUC={roc:.4f}  |  PR-AUC={pr:.4f}")