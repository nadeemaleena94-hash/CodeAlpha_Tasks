"""
CodeAlpha ML Internship - Task 1: Credit Scoring Model
Predict creditworthiness (1 = good, 0 = default risk) from financial history.

Usage:
  python credit_scoring.py                      # runs on a synthetic dataset
  python credit_scoring.py --csv my_data.csv --target default
      (use a real dataset, e.g. "Give Me Some Credit" / German Credit from Kaggle/UCI;
       the target column should be binary 0/1 where 1 = BAD credit / default)

Install: pip install scikit-learn pandas numpy matplotlib
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, RocCurveDisplay, ConfusionMatrixDisplay)


def make_synthetic(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.5, n)
    age = rng.integers(21, 70, n)
    debt = income * rng.uniform(0.0, 1.2, n)
    credit_limit = income * rng.uniform(0.1, 0.8, n)
    credit_used = credit_limit * rng.uniform(0, 1.1, n)
    late_30 = rng.poisson(0.4, n)
    late_90 = rng.poisson(0.1, n)
    open_lines = rng.integers(0, 15, n)
    history_years = np.clip(age - 18 - rng.integers(0, 10, n), 0, None)
    employment = rng.choice(["salaried", "self-employed", "unemployed"], n, p=[.65, .27, .08])
    home = rng.choice(["rent", "own", "mortgage"], n, p=[.4, .25, .35])
    z = (-3.2 + 1.8 * (debt / income) + 1.5 * (credit_used / credit_limit)
         + 0.6 * late_30 + 1.2 * late_90 - 0.02 * history_years
         + 1.0 * (employment == "unemployed") - 0.3 * (home == "own") + rng.normal(0, .6, n))
    default = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(dict(income=income, age=age, total_debt=debt, credit_limit=credit_limit,
                           credit_used=credit_used, late_30=late_30, late_90=late_90,
                           open_credit_lines=open_lines, credit_history_years=history_years,
                           employment=employment, home_ownership=home, default=default))
    # inject some missing values to show imputation
    df.loc[rng.choice(n, 150, replace=False), "income"] = np.nan
    return df


def engineer(df):
    """Feature engineering from financial history."""
    df = df.copy()
    if {"total_debt", "income"} <= set(df.columns):
        df["debt_to_income"] = df["total_debt"] / (df["income"] + 1)
    if {"credit_used", "credit_limit"} <= set(df.columns):
        df["credit_utilization"] = df["credit_used"] / (df["credit_limit"] + 1)
    if {"late_30", "late_90"} <= set(df.columns):
        df["total_late_payments"] = df["late_30"] + df["late_90"]
        df["severe_delinquency"] = (df["late_90"] > 0).astype(int)
    return df


def main(csv, target):
    df = pd.read_csv(csv) if csv else make_synthetic()
    df = engineer(df)
    print("Data shape:", df.shape, "| Default rate:", round(df[target].mean(), 3))

    X, y = df.drop(columns=[target]), df[target]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
    }
    fitted, rows = {}, []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
        fitted[name] = pipe
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append(dict(Model=name, Precision=precision_score(y_te, pred),
                         Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred),
                         ROC_AUC=roc_auc_score(y_te, proba)))
    res = pd.DataFrame(rows).set_index("Model").round(4)
    print("\n=== Model comparison ===\n", res)

    best = res["ROC_AUC"].idxmax()
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, fitted[best].predict(X_te),
                                target_names=["Good credit", "Default"]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in fitted.items():
        RocCurveDisplay.from_estimator(p, X_te, y_te, name=name, ax=ax[0])
    ax[0].set_title("ROC Curves")
    ConfusionMatrixDisplay.from_estimator(fitted[best], X_te, y_te, ax=ax[1])
    ax[1].set_title(f"Confusion Matrix - {best}")
    plt.tight_layout()
    plt.savefig("credit_evaluation.png", dpi=150)

    rf = fitted["Random Forest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(10).sort_values()
    imp.plot.barh(figsize=(7, 5), title="Top 10 Features (Random Forest)")
    plt.tight_layout()
    plt.savefig("credit_feature_importance.png", dpi=150)
    print("Saved credit_evaluation.png and credit_feature_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default="default")
    a = ap.parse_args()
    main(a.csv, a.target)