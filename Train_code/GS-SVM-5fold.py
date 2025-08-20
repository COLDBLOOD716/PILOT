import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import bootstrap
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings("ignore")

INPUT_XLSX = "./dataset"
CENTER_FILTER = "internal_center"
TARGET = "target"
seeds = [42, 7, 2023]
outer_folds = 5
inner_folds = 3
n_bootstrap = 5000
OUTPUT_DIR = "./Model_File"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(INPUT_XLSX)
df = df[df["Center"] == CENTER_FILTER].reset_index(drop=True)

variable_sets = {
    "Preoperative": {
        "catvars": ["Sex", "Smoke", "Alcohol", "MASLD", "Diabete", "Antiviral",
                    "HCV", "CSPH", "Ascites", "Cirrhosis", "Diagnosis"],
        "convars": ["Age", "BMI", "AFP", "ICG", "TP0", "ALB0", "PALB0", "PLT0",
                    "INR0", "ALT0", "AST0", "GGT0", "ALP0", "CR0", "TBI0", "Phos0"]
    },
    "Intraoperative": {
        "catvars": ["Sex", "Smoke", "Alcohol", "MASLD", "Diabete", "Antiviral",
                    "HCV", "CSPH", "Ascites", "Cirrhosis", "Diagnosis"],
        "convars": ["Age", "BMI", "AFP", "ICG", "TP0", "ALB0", "PALB0", "PLT0", "INR0",
                    "ALT0", "AST0", "GGT0", "ALP0", "CR0", "TBI0", "Phos0", "Block time",
                    "Blood loss", "TP1", "ALB1", "PALB1", "PLT1", "INR1", "PT1", "AST1",
                    "GGT1", "ALP1", "CR1", "TBI1", "Liver GATA3", "Liver RAMP2", "LRGR", "Phos1"]
    },
    "Postoperative": {
        "catvars": ["Sex", "Smoke", "Alcohol", "MASLD", "Diabete", "Antiviral",
                    "HCV", "CSPH", "Ascites", "Cirrhosis", "Diagnosis"],
        "convars": ["Age", "BMI", "AFP", "ICG", "TP0", "ALB0", "PALB0", "PLT0",
                    "INR0", "ALT0", "AST0", "GGT0", "ALP0", "CR0", "TBI0",
                    "Phos0", "Block time", "Blood loss", "TP1", "ALB1", "PALB1",
                    "PLT1", "INR1", "PT1", "AST1", "GGT1", "ALP1", "CR1", "TBI1",
                    "Phos1", "Liver GATA3", "Liver RAMP2", "LRGR", "Serum VEGFA-post",
                    "SPVI-post", "TP3", "ALB3", "PALB3", "PLT3", "INR3", "AST3",
                    "ALP3", "CR3", "TBI3", "Phos3"]
    }
}

param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__gamma": ["scale", "auto"]
}

def compute_ci(arr, confidence=0.95):
    arr = np.array(arr)
    if len(arr) <= 1:
        return "NA"
    boot = bootstrap((arr,), np.mean, confidence_level=confidence,
                     n_resamples=n_bootstrap, method="percentile")
    return f"{boot.confidence_interval.low:.4f} - {boot.confidence_interval.high:.4f}"

def complexity_svm(params):
    C = params.get("svc__C", 1)
    gamma = params.get("svc__gamma", "scale")
    if gamma in ["scale", "auto"]:
        gamma_val = 1.0
    else:
        gamma_val = float(gamma)
    return C * (1.0 / gamma_val if gamma_val > 0 else 1.0)

results_dict = {}

for var_set_name, var_set in variable_sets.items():
    print(f"\n===== 阶段: {var_set_name} =====")

    df_local = df.copy()

    label_encoders = {}
    for col in var_set["catvars"]:
        le = LabelEncoder()
        vals = df_local[col].astype(str).fillna("NA")
        le.fit(vals)
        df_local[col] = le.transform(vals)
        label_encoders[col] = le

    le_filename = os.path.join(OUTPUT_DIR, f"SVM_label_encoders_{var_set_name}.pkl")
    joblib.dump(label_encoders, le_filename)
    print(f"Saved LabelEncoders: {le_filename}")

    feature_list = var_set["catvars"] + var_set["convars"]
    feat_filename = os.path.join(OUTPUT_DIR, f"SVM_feature_list_{var_set_name}.pkl")
    joblib.dump(feature_list, feat_filename)
    print(f"Saved feature list: {feat_filename}")

    X = df_local[feature_list].reset_index(drop=True)
    y = df_local[TARGET].reset_index(drop=True)

    all_seed_results = []
    all_models = []

    for seed in seeds:
        np.random.seed(seed)
        outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        pbar = tqdm(total=outer_folds, desc=f"{var_set_name} CV (seed={seed})", ncols=100)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(probability=True, random_state=seed))
            ])

            gs = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=inner_folds, n_jobs=-1)
            gs.fit(X_train, y_train)

            cvres = gs.cv_results_
            means = cvres["mean_test_score"]
            stds = cvres["std_test_score"]
            params_list = cvres["params"]

            best_idx = np.argmax(means)
            best_mean = means[best_idx]
            threshold = best_mean - stds[best_idx]

            candidate_idxs = [i for i, m in enumerate(means) if m >= threshold]
            candidate_idxs_sorted = sorted(candidate_idxs, key=lambda i: complexity_svm(params_list[i]))
            chosen_idx = candidate_idxs_sorted[0]
            chosen_params = params_list[chosen_idx]

            best_model = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(probability=True, random_state=seed, **{k.replace("svc__", ""): v for k, v in chosen_params.items()}))
            ])
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics = {
                "Seed": seed,
                "Fold": fold_idx,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Recall": recall_score(y_test, y_pred),
                "Specificity": tn / (tn + fp) if (tn + fp) else 0,
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "NPV": tn / (tn + fn) if (tn + fn) else 0,
                "F1": f1_score(y_test, y_pred),
                "Chosen Params": chosen_params
            }
            all_seed_results.append(metrics)

            try:
                full_auc = roc_auc_score(y, best_model.predict_proba(X)[:, 1])
            except Exception:
                full_auc = -np.inf
            all_models.append((best_model, chosen_params, seed, full_auc))

            pbar.set_postfix({"AUC": f"{metrics['AUC']:.4f}"})
            pbar.update(1)

        pbar.close()

    df_results = pd.DataFrame(all_seed_results)
    summary = df_results.drop(columns=["Seed", "Fold", "Chosen Params"]).agg(["mean", "std", "median"]).T
    for col in summary.index:
        ci = compute_ci(df_results[col].values)
        summary.loc[col, "95% CI"] = ci

    results_dict[var_set_name] = (df_results, summary)

    all_models_sorted = sorted(all_models, key=lambda x: (-x[3], complexity_svm(x[1])))
    final_model = all_models_sorted[0][0]
    model_filename = os.path.join(OUTPUT_DIR, f"SVM_{var_set_name}.pkl")
    joblib.dump(final_model, model_filename)
    print(f"Saved best model: {model_filename}")

output_path = os.path.join(OUTPUT_DIR, "5-fold-CV-SVM.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for var_set_name, (df_results, summary_df) in results_dict.items():
        df_results.to_excel(writer, sheet_name=f"{var_set_name} Results", index=False)
        summary_df.to_excel(writer, sheet_name=f"{var_set_name} Summary")

print(f"\n All train results saved: {output_path}")
print(f"Best model saved: {OUTPUT_DIR}")
