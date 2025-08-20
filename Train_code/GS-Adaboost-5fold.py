import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.stats import bootstrap
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

INPUT_XLSX = "./dataset"
CENTER_FILTER = "internal_center"
TARGET = "target"
seeds = [42, 7, 2023]
outer_folds = 5
inner_folds = 3
n_bootstrap = 10000
OUTPUT_DIR = "./Model_File"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(INPUT_XLSX)
df = df[df["Center"] == CENTER_FILTER]

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
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 3, 5],
    "estimator__min_samples_split": [2, 5, 10],
    "estimator__min_samples_leaf": [1, 2, 5]
}

def compute_ci(arr, confidence=0.95):
    arr = np.array(arr)
    if len(arr) <= 1:
        return "NA"
    boot = bootstrap((arr,), np.mean, confidence_level=confidence,
                     n_resamples=n_bootstrap, method='percentile')
    return f"{boot.confidence_interval.low:.4f} - {boot.confidence_interval.high:.4f}"

def complexity_ada(params):
    ne = params.get("n_estimators", 100)
    md = params.get("estimator__max_depth", 1)
    return ne * md

for var_set_name, var_set in variable_sets.items():
    print(f"\n===== stage: {var_set_name} =====")
    X = df[var_set["catvars"] + var_set["convars"]].copy()
    y = df[TARGET].reset_index(drop=True)

    label_encoders = {}
    for col in var_set["catvars"]:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna("NA")
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    feature_list = var_set["catvars"] + var_set["convars"]
    joblib.dump(feature_list, os.path.join(OUTPUT_DIR, f"feature_list_{var_set_name}.pkl"))

    results_across_seeds = []
    total_iters = len(seeds) * outer_folds
    pbar = tqdm(total=total_iters, desc=f"{var_set_name} Total progress", ncols=100)

    best_model = None
    best_auc = -np.inf
    best_complexity = np.inf

    for seed in seeds:
        outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test  = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test  = y.iloc[test_idx].reset_index(drop=True)

            base = DecisionTreeClassifier(random_state=seed)
            model = AdaBoostClassifier(estimator=base, random_state=seed)

            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            gs = GridSearchCV(model, param_grid, scoring="roc_auc",
                              cv=inner_cv, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            cvres = gs.cv_results_
            means = cvres["mean_test_score"]
            stds = cvres["std_test_score"]
            params_list = cvres["params"]

            best_idx = np.argmax(means)
            best_mean = means[best_idx]
            best_std = stds[best_idx]
            threshold = best_mean - best_std
            candidate_idxs = [i for i, m in enumerate(means) if m >= threshold]
            candidate_idxs_sorted = sorted(candidate_idxs,
                                           key=lambda i: complexity_ada(params_list[i]))
            chosen_idx = candidate_idxs_sorted[0]
            chosen_params = params_list[chosen_idx]

            model.set_params(**chosen_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Recall": recall_score(y_test, y_pred),
                "Specificity": tn / (tn + fp) if (tn + fp) else 0,
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "NPV": tn / (tn + fn) if (tn + fn) else 0,
                "F1": f1_score(y_test, y_pred),
                "Fold": fold_idx,
                "Chosen Params": chosen_params
            }
            fold_metrics.append(metrics)

            auc = metrics["AUC"]
            complexity = complexity_ada(chosen_params)
            if (auc > best_auc) or (auc == best_auc and complexity < best_complexity):
                best_auc = auc
                best_complexity = complexity
                best_model = model

            print(f"[Seed {seed}] Fold {fold_idx} done -> "
                  f"AUC: {metrics['AUC']:.4f}, "
                  f"Acc: {metrics['Accuracy']:.4f}, "
                  f"F1: {metrics['F1']:.4f}, "
                  f"Recall: {metrics['Recall']:.4f}")

            pbar.set_postfix({"AUC": f"{metrics['AUC']:.4f}"})
            pbar.update(1)

        df_folds = pd.DataFrame(fold_metrics)
        summary = df_folds.drop(columns=["Fold", "Chosen Params"]).agg(["mean", "std", "median"]).T
        for col in summary.index:
            ci = compute_ci(df_folds[col].values)
            summary.loc[col, "95% CI"] = ci

        results_across_seeds.append((seed, df_folds, summary))

    pbar.close()

    results_filename = os.path.join(OUTPUT_DIR, f"nested_adaboost_results_{var_set_name}.pkl")
    joblib.dump(results_across_seeds, results_filename)
    print(f"All train results saved：{results_filename}")

    le_filename = os.path.join(OUTPUT_DIR, f"label_encoders_{var_set_name}.pkl")
    joblib.dump(label_encoders, le_filename)
    print(f"LabelEncoder saved：{le_filename}")

    model_filename = os.path.join(OUTPUT_DIR, f"best_model_{var_set_name}.pkl")
    joblib.dump(best_model, model_filename)
    print(f"Best model saved：{model_filename} (AUC={best_auc:.4f}, complexity={best_complexity})")
