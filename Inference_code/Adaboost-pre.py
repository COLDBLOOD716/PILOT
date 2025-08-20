import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)

PHASE = "Preoperative"
MODEL_DIR = "./Model_File"
EXTERNAL_XLSX = "./dataset"
EXTERNAL_CENTER = "external_center"
TARGET = "target"

model_path = os.path.join(MODEL_DIR, f"Adaboost_{PHASE}.pkl")
le_path = os.path.join(MODEL_DIR, f"Adaboost_label_encoders_{PHASE}.pkl")
feat_path = os.path.join(MODEL_DIR, f"Adaboost_feature_list_{PHASE}.pkl")

model = joblib.load(model_path)
label_encoders = joblib.load(le_path)
feature_list = joblib.load(feat_path)

catvars = list(label_encoders.keys())

print(f"Loaded model: {model_path}")
print(f"Loaded encoders: {le_path} -> {catvars}")
print(f"Loaded features: {feat_path} -> {len(feature_list)} columns")

df_ext = pd.read_excel(EXTERNAL_XLSX)
df_ext = df_ext[df_ext["Center"] == EXTERNAL_CENTER].copy()

def safe_le_transform(series: pd.Series, le):
    values = series.astype(str).fillna("NA")
    known = set(le.classes_)
    if "NA" in known:
        values = values.apply(lambda x: x if x in known else "NA")
    else:
        fallback = le.classes_[0]
        values = values.apply(lambda x: x if x in known else fallback)
    return le.transform(values)

for col in catvars:
    if col not in df_ext.columns:
        raise ValueError(f"[External data missing columns] missing features: {col}")
    df_ext[col] = safe_le_transform(df_ext[col], label_encoders[col])

missing_cols = [c for c in feature_list if c not in df_ext.columns]
if missing_cols:
    raise ValueError(f"[External data missing columns] External data missing the below features：{missing_cols}")

X_external = df_ext[feature_list].copy()

num_cols = [c for c in X_external.columns if c not in catvars]
for c in num_cols:
    if X_external[c].isna().any():
        X_external[c] = X_external[c].astype(float)
        X_external[c] = X_external[c].fillna(X_external[c].median())

if TARGET not in df_ext.columns:
    raise ValueError(f"[External data missing columns] unable to locate the feature column: {TARGET}")
y_external = df_ext[TARGET].values

y_pred_ext = model.predict(X_external)
y_prob_ext = model.predict_proba(X_external)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_external, y_pred_ext).ravel()
acc = accuracy_score(y_external, y_pred_ext)
auc = roc_auc_score(y_external, y_prob_ext)
recall = recall_score(y_external, y_pred_ext)
specificity = tn / (tn + fp) if (tn + fp) else 0
bal_acc = balanced_accuracy_score(y_external, y_pred_ext)
precision = precision_score(y_external, y_pred_ext) if (tp + fp) else 0
npv = tn / (tn + fn) if (tn + fn) else 0
f1 = f1_score(y_external, y_pred_ext) if (precision + recall) else 0

print("\n===== External Validation (Preoperative) =====")
print(f"Center: {EXTERNAL_CENTER}, n={len(y_external)}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"PPV (Precision): {precision:.4f}")
print(f"NPV: {npv:.4f}")
print(f"F1 Score: {f1:.4f}")

try:
    explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], X_external)
    shap_values = explainer(X_external)

    shap.summary_plot(shap_values, X_external, show=False)
    plt.title("SHAP Summary (External - Preoperative)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_external, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar, External - Preoperative)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[SHAP warning] unable to create SHAP figure：{e}")
