import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)

PHASE = "Intraoperative"
MODEL_DIR = "./Model_File"
EXTERNAL_XLSX = "./dataset"
EXTERNAL_CENTER = "external_center"
TARGET = "target"

model_path = os.path.join(MODEL_DIR, f"LGB_{PHASE}.pkl")
le_path = os.path.join(MODEL_DIR, f"LGB_label_encoders_{PHASE}.pkl")
feat_path = os.path.join(MODEL_DIR, f"LGB_feature_list_{PHASE}.pkl")

lgbm_model = joblib.load(model_path)
label_encoders = joblib.load(le_path)
feature_list = joblib.load(feat_path)

catvars = list(label_encoders.keys())
print(f"Loaded model: {model_path}")
print(f"Loaded encoders for: {catvars}")
print(f"Loaded feature list ({len(feature_list)} columns)")

df_ext = pd.read_excel(EXTERNAL_XLSX)
df_ext = df_ext[df_ext["Center"] == EXTERNAL_CENTER].copy()

def safe_le_transform(series: pd.Series, le):
    values = series.astype(str).fillna("NA")
    known = set(le.classes_)
    fallback = le.classes_[0]
    values = values.apply(lambda x: x if x in known else fallback)
    return le.transform(values)

for col in catvars:
    if col not in df_ext.columns:
        raise ValueError(f"missing feature columns: {col}")
    df_ext[col] = safe_le_transform(df_ext[col], label_encoders[col])

missing_cols = [c for c in feature_list if c not in df_ext.columns]
if missing_cols:
    raise ValueError(f"External data missing the below features：{missing_cols}")

X_external = df_ext[feature_list].copy()
y_external = df_ext[TARGET].values

num_cols = [c for c in X_external.columns if c not in catvars]
for c in num_cols:
    if X_external[c].isna().any():
        X_external[c] = X_external[c].astype(float).fillna(X_external[c].median())

y_pred_external = lgbm_model.predict(X_external)
y_prob_external = lgbm_model.predict_proba(X_external)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_external, y_pred_external).ravel()

accuracy_external = accuracy_score(y_external, y_pred_external)
auc_external = roc_auc_score(y_external, y_prob_external)
recall_external = recall_score(y_external, y_pred_external)
specificity_external = tn / (tn + fp) if (tn + fp) else 0
balanced_acc_external = balanced_accuracy_score(y_external, y_pred_external)
precision_external = precision_score(y_external, y_pred_external) if (tp + fp) else 0
npv_external = tn / (tn + fn) if (tn + fn) else 0
f1_external = f1_score(y_external, y_pred_external) if (precision_external + recall_external) else 0

print("\n===== External Validation (Intraoperative - LightGBM) =====")
print(f"Center: {EXTERNAL_CENTER}, n={len(y_external)}")
print(f"Accuracy: {accuracy_external:.4f}")
print(f"AUC: {auc_external:.4f}")
print(f"Recall (Sensitivity): {recall_external:.4f}")
print(f"Specificity: {specificity_external:.4f}")
print(f"Balanced Accuracy: {balanced_acc_external:.4f}")
print(f"PPV (Precision): {precision_external:.4f}")
print(f"NPV: {npv_external:.4f}")
print(f"F1 Score: {f1_external:.4f}")

try:
    explainer = shap.Explainer(lambda x: lgbm_model.predict_proba(x)[:, 1], X_external)
    shap_values = explainer(X_external)

    shap.summary_plot(shap_values, X_external, show=False)
    plt.title(f"SHAP Summary (External - {PHASE}, GBM)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_external, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Bar, External - {PHASE}, GBM)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[SHAP warning] unable to create SHAP figure：{e}")
