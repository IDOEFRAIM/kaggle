import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

import joblib


DATA_PATH = os.path.join(os.path.dirname(__file__), "PCOS_infertility.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip column names and standardize
    df = df.rename(columns=lambda c: c.strip())
    # rename PCOS target to `target`
    if 'PCOS (Y/N)' in df.columns:
        df = df.rename(columns={'PCOS (Y/N)': 'target'})
    # drop Sl. No and Patient File No. if present
    for col in ['Sl. No', 'Patient File No.', 'Sl. No.', 'Patient File No']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def preprocess(df: pd.DataFrame):
    df = clean_columns(df)
    # Convert columns to numeric when possible
    for col in df.columns:
        if col != 'target':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # target may be 0/1 as numeric or strings
    df['target'] = pd.to_numeric(df['target'], errors='coerce').astype('Int64')

    # Basic EDA printouts
    print("Dataset shape:", df.shape)
    print(df.describe(include='all'))

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)

    # Imputation and scaling pipeline for numeric features
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    return X, y, preprocessor



def evaluate_model(model, split_data):
    X_train, X_test, y_train, y_test = split_data

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = None

    results = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0)
    }
    if probs is not None:
        results['roc_auc'] = roc_auc_score(y_test, probs)
    else:
        results['roc_auc'] = None
    return results

# ...existing code...

def run_models(X, y, preprocessor, random_state=42):
    X_proc = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=random_state)
    split_data = (X_train, X_test, y_train, y_test)

    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=random_state),
        'rf': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'gb': GradientBoostingClassifier(random_state=random_state),
        'svc': SVC(probability=True, random_state=random_state)
    }
    if XGB_AVAILABLE:
        # `use_label_encoder` is deprecated/ignored in newer xgboost
        models['xgb'] = xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=random_state
        )

    results = {}
    for name, clf in models.items():
        print(f"Training and evaluating {name}")
        res = evaluate_model(clf, split_data)
        results[name] = res
        print(name, res)

    # pick best by roc_auc; if missing, fallback to f1
    best = max(
        results.items(),
        key=lambda kv: (
            -1 if kv[1].get('roc_auc') is None else kv[1]['roc_auc'],
            kv[1]['f1']
        )
    )[0]

    return results, best

# ...existing code...

def main():
    df = load_data()
    X, y, preprocessor = preprocess(df)
    results, best = run_models(X, y, preprocessor)
    # Save results summary
    pd.DataFrame(results).T.to_csv('model_comparison_summary.csv')
    print('Wrote model_comparison_summary.csv',bests)


if __name__ == '__main__':
    main()
