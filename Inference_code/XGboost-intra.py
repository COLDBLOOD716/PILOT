import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)

PHASE = "Intraoperative"
TARGET = "target"
MODEL_CENTER = "model_center"
EXTERNAL_CENTER = "external_center"
EXCEL_FILE = "./dataset"

df = pd.read_excel(EXCEL_FILE)
df_model = df[df["Center"] == MODEL_CENTER].copy()

catvars = ["Sex", "Smoke", "Alcohol", "MASLD", "Diabete", "Antiviral",
           "HCV", "CSPH", "Ascites", "Cirrhosis", "Diagnosis"]
convars = ["Age", "BMI", "AFP", "ICG", "TP0", "ALB0", "PALB0", "PLT0",
           "INR0", "ALT0", "AST0", "GGT0", "ALP0", "CR0", "TBI0", "Phos0",
           "Block time", "Blood loss", "TP1", "ALB1", "PALB1", "PLT1", "INR1", "PT1",
           "AST1", "GGT1", "ALP1", "CR1", "TBI1", "Liver GATA3", "Liver RAMP2", "LRGR"]

label_encoders = {}
def safe_le_transform(series: pd.Series, le):
    values = series.astype(str).fillna("NA")
    known = set(le.classes_)
    fallback = le.classes_[0]
    values = values.apply(lambda x: x if x in known else fallback)
    return le.transform(values)

for col in catvars:
    le = LabelEncoder()
    df_model[col] = df_model[col].astype(str).fillna("NA")
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model[catvars + convars].copy()
y = df_model[TARGET].values

best_params = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.2,
    'max_depth': 7,
    'n_estimators': 50,
    'reg_alpha': 1,
    'reg_lambda': 2,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'scale_pos_weight': len(y[y == 0]) / len(y[y == 1])
}

xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X, y)

y_pred = xgb_model.predict(X)
y_prob = xgb_model.predict_proba(X)[:, 1]

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
recall = recall_score(y, y_pred)
specificity = tn / (tn + fp) if (tn + fp) else 0
balanced_acc = balanced_accuracy_score(y, y_pred)
precision = precision_score(y, y_pred) if (tp + fp) else 0
npv = tn / (tn + fn) if (tn + fn) else 0
f1 = f1_score(y, y_pred) if (precision + recall) else 0

print(f"\n===== Modeling Group (XGBoost - {PHASE}) =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"PPV (Precision): {precision:.4f}")
print(f"NPV: {npv:.4f}")
print(f"F1 Score: {f1:.4f}")

df_ext = df[df["Center"] == EXTERNAL_CENTER].copy()

for col in catvars:
    if col not in df_ext.columns:
        raise ValueError(f"External data missing the below features: {col}")
    df_ext[col] = safe_le_transform(df_ext[col], label_encoders[col])

X_external = df_ext[catvars + convars].copy()
y_external = df_ext[TARGET].values

y_pred_external = xgb_model.predict(X_external)
y_prob_external = xgb_model.predict_proba(X_external)[:, 1]

tn_ext, fp_ext, fn_ext, tp_ext = confusion_matrix(y_external, y_pred_external).ravel()
accuracy_external = accuracy_score(y_external, y_pred_external)
auc_external = roc_auc_score(y_external, y_prob_external)
recall_external = recall_score(y_external, y_pred_external)
specificity_external = tn_ext / (tn_ext + fp_ext) if (tn_ext + fp_ext) else 0
balanced_acc_external = balanced_accuracy_score(y_external, y_pred_external)
precision_external = precision_score(y_external, y_pred_external) if (tp_ext + fp_ext) else 0
npv_external = tn_ext / (tn_ext + fn_ext) if (tn_ext + fn_ext) else 0
f1_external = f1_score(y_external, y_pred_external) if (precision_external + recall_external) else 0

print(f"\n===== External Validation (XGBoost - {PHASE}) =====")
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
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_external)

    shap.summary_plot(shap_values, X_external, show=False)
    plt.title(f"SHAP Summary (External - {PHASE}, XGBoost)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_external, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Bar, External - {PHASE}, XGBoost)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[SHAP error] Unable to create SHAP figure：{e}")
