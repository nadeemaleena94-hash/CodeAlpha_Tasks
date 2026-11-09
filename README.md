# CodeAlpha Machine Learning Internship - Tasks

Python projects completed for the CodeAlpha Machine Learning internship.

| Task | Folder | Summary |
|------|--------|---------|
| 1 - Credit Scoring | `task1_credit_scoring` | Predicts creditworthiness with Logistic Regression, Decision Tree and Random Forest. Includes feature engineering (debt-to-income, credit utilization, late payments) and evaluation with Precision, Recall, F1 and ROC-AUC. Runs on a synthetic dataset by default; use `--csv` for a real one. |
| 3 - Handwritten Character Recognition | `task3_handwritten_character_recognition` | CNN trained on MNIST digits (optional EMNIST letters with `--emnist`). |
| 4 - Disease Prediction | `task4_disease_prediction` | Breast cancer prediction (UCI dataset) with Logistic Regression, SVM, Random Forest and XGBoost, plus cross-validation and tuning. |

## Setup
```
python -m venv ml_env
ml_env\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run
```
python task4_disease_prediction/disease_prediction.py
python task1_credit_scoring/credit_scoring.py
python task3_handwritten_character_recognition/handwritten_cnn.py
```
Each script prints metrics in the terminal and saves charts as `.png` files in the folder you run it from. Task 3 downloads MNIST (about 11 MB) automatically on the first run.

## Notes
- Task 1 uses generated data unless you pass a real CSV: `python credit_scoring.py --csv data.csv --target default`.
- Models and datasets are not stored in this repo.
