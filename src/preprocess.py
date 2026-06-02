"""Reproducible preprocessing pipeline for Pyrenex Crédit scoring.

Build a scikit-learn Pipeline + ColumnTransformer that handles:
- Numeric features: imputation + scaling
- Categorical features: imputation + one-hot encoding
- Target: binary mapping

This module is imported by `train.py` and `evaluate.py`.

---

Objectif: construire un préprocesseur réutilisable (scikit-learn ColumnTransformer)
pour numeric/categorical et charger le dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# liste des colonnes numériques
NUMERIC_FEATURES: list[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "revol_util",
]
# liste des colonnes catégorielles
CATEGORICAL_FEATURES: list[str] = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length",
]
# nom de la colonne cible ("loan_status").
TARGET_COLUMN: str = "loan_status"
# mapping de la cible en binaire (0/1).
TARGET_MAPPING: dict[str, int] = {"Fully Paid": 0, "Charged Off": 1}


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load Lending Club CSV and split features/target.

    Args:
        path: CSV path.

    Returns:
        (X, y) where X is the feature DataFrame and y the target Series.

    ---

    Lit le CSV (pd.read_csv(path)),
    mappe la colonne cible via TARGET_MAPPING,
    retourne X (DataFrame sans la colonne cible) et y (Series binaire).
    """
    df = pd.read_csv(path)
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer applying numeric + categorical pipelines."""
    # Vérifie que NUMERIC_FEATURES et CATEGORICAL_FEATURES ne sont pas vides (lève ValueError sinon).
    if not NUMERIC_FEATURES or not CATEGORICAL_FEATURES:
        raise ValueError(
            "NUMERIC_FEATURES and CATEGORICAL_FEATURES are empty in "
            "src/preprocess.py. Fill them from your EDA before training. "
            "See ressources/01_Pandas_Sklearn_split_essentiel.md."
        )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # Retourne un ColumnTransformer qui applique ces pipelines et droppe le reste.
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
