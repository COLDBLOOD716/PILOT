import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
    balanced_accuracy_score, precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import bootstrap
from tqdm import tqdm
import joblib
import os
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

label_encoders = {}
for col in set(sum([v["catvars"] for v in variable_sets.values()], [])):
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

param_grid = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.01, 0.1, 1, 10],
    "solver": ["saga", "liblinear"],
    "max_iter": [100, 500, 1000]
}

def compute_ci(arr, confidence=0.95):
    arr = np.array(arr)
    if len(arr) <= 1: return "NA"
    boot = bootstrap((arr,), np.mean, confidence_level=confidence,
                     n_resamples=n_bootstrap, method="percentile")
    return f"{boot.confidence_interval.low:.4f} - {boot.confidence_interval.high:.4f}"

results_dict = {}

for var_set_name, var_set in variable_sets.items():
    print(f"\n===== stage: {var_set_name} =====")
    X = df[var_set["catvars"] + var_set["convars"]]
    y = df[TARGET].reset_index(drop=True)

    all_seed_results = []
    all_models = []

    for seed in seeds:
        np.random.seed(seed)
        outer_kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        pbar = tqdm(total=outer_folds, desc=f"{var_set_name} CV (seed={seed})", ncols=100)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = LogisticRegression(random_state=seed, max_iter=1000)
            gs = GridSearchCV(model, param_grid, scoring="roc_auc", cv=inner_folds, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

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
                "Chosen Params": best_model.get_params()
            }
            all_seed_results.append(metrics)
            all_models.append((best_model, roc_auc_score(y, best_model.predict_proba(X)[:, 1]), seed))
            pbar.set_postfix({"AUC": f"{metrics['AUC']:.4f}"})
            pbar.update(1)
        pbar.close()

    df_results = pd.DataFrame(all_seed_results)
    summary = df_results.drop(columns=["Seed", "Fold", "Chosen Params"]).agg(["mean", "std", "median"]).T
    for col in summary.index:
        ci = compute_ci(df_results[col].values)
        summary.loc[col, "95% CI"] = ci
    results_dict[var_set_name] = (df_results, summary)

    all_models_sorted = sorted(all_models, key=lambda x: (-x[1], x[2]))
    final_model = all_models_sorted[0][0]

    joblib.dump(final_model, os.path.join(OUTPUT_DIR, f"ENlog_{var_set_name}.pkl"))
    joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, f"ENlog_label_encoders_{var_set_name}.pkl"))
    joblib.dump(var_set["catvars"] + var_set["convars"],
                os.path.join(OUTPUT_DIR, f"ENlog_feature_list_{var_set_name}.pkl"))

output_path = os.path.join(OUTPUT_DIR, "5-fold-CV-ENlog.xlsx")
with pd.ExcelWriter(output_path) as writer:
    for var_set_name, (df_results, summary_df) in results_dict.items():
        df_results.to_excel(writer, sheet_name=f"{var_set_name} Results", index=False)
        summary_df.to_excel(writer, sheet_name=f"{var_set_name} Summary")

print(f"\n All train results saved: {output_path}")
print(f"Best model saved: {OUTPUT_DIR}")
