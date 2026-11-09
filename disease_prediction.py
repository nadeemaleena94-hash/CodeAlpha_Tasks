"""
CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data
Dataset: Breast Cancer Wisconsin (UCI, bundled with scikit-learn -> no download needed)
Models: Logistic Regression, SVM, Random Forest, XGBoost (if installed)

Install:  pip install scikit-learn pandas matplotlib seaborn xgboost
Run:      python disease_prediction.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed - skipping XGBoost (pip install xgboost)")

# 1. Load data -------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
print("Shape:", X.shape, "| Class balance:\n", y.value_counts(normalize=True).round(3))
print("Missing values:", int(X.isna().sum().sum()))

# 2. Train / test split ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Models (scaling inside pipeline avoids data leakage) --------------------
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "SVM": Pipeline([("sc", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                      eval_metric="logloss", random_state=42)

# 4. Train + evaluate --------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, fitted = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (5-fold)": cv_auc,
    })
results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n=== Model comparison (test set) ===\n", results)

best = results["ROC-AUC"].idxmax()
print(f"\nBest model by ROC-AUC: {best}")
print(classification_report(y_test, fitted[best].predict(X_test),
                            target_names=data.target_names))

# 5. Hyper-parameter tuning on Random Forest -------------------------------
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    {"n_estimators": [200, 400], "max_depth": [None, 5, 10],
                     "min_samples_leaf": [1, 2]},
                    cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("Tuned RF best params:", grid.best_params_,
      "| test ROC-AUC:", round(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]), 4))

# 6. Plots -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_estimator(fitted[best], X_test, y_test,
                                      display_labels=data.target_names, ax=ax[0])
ax[0].set_title(f"Confusion Matrix - {best}")
for name, m in fitted.items():
    RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax[1])
ax[1].set_title("ROC Curves")
plt.tight_layout()
plt.savefig("evaluation.png", dpi=150)

rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10).sort_values()
imp.plot.barh(figsize=(7, 5), title="Top 10 Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved evaluation.png and feature_importance.png")