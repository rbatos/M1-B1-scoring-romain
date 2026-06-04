"""Train Pyrenex Crédit risk model — M1-B1.

Usage:
    python src/train.py --config default
    python src/train.py --config balanced

Each run writes:
    models/pyrenex_risk_v2_<config>.joblib   (full Pipeline)
    models/pyrenex_risk_v2_<config>.json     (metadata, no holdout metric yet)

Once you have chosen which configuration to retain, promote it to the
canonical name expected by `evaluate.py` and `contract_test.py`:

    cp models/pyrenex_risk_v2_<chosen>.joblib models/pyrenex_risk_v2.joblib
    cp models/pyrenex_risk_v2_<chosen>.json   models/pyrenex_risk_v2.json

Then `python src/evaluate.py --update-meta` fills in `metrics_holdout`.

---

But: train.py entraîne un modèle de classification (RandomForest) pour le risque de crédit et écrit le pipeline entraîné + métadonnées. Fichier: train.py:1-200.

Entrées / Sorties :
Entrée par défaut: data/lending_club_train.csv (option --data).
Sorties: models/pyrenex_risk_v2_<config>.joblib (pipeline joblib) et models/pyrenex_risk_v2_<config>.json (métadonnées).

Configs: défini dans le dict CONFIGS (default, balanced, ...), choisi via --config.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    load_dataset,
)


def build_configs() -> dict[str, dict]:
    # Paramètres RF et leur impact :
    # - n_estimators : plus d'arbres diminue la variance, mais augmente le temps.
    # - max_depth : profondeur maximale des arbres; plus bas = plus de régularisation.
    # - min_samples_leaf : nombre minimal d'échantillons par feuille; plus haut = moins de bruit.
    # - class_weight : permet de corriger le déséquilibre entre Fully Paid / Charged Off.
    # - max_features : nombre de variables candidates à chaque split ; sqrt/log2 favorisent la diversité.

    configs: dict[str, dict] = {
        "default": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced_recall": {
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced_precise": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced_deep": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced_shallow": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "balanced_wide": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    return configs


CONFIGS = build_configs()


def train(config_name: str, data_path: Path, output_dir: Path) -> dict:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(CONFIGS)}")
    params = CONFIGS[config_name]

    start_time = time.perf_counter()

    # Charge les données via load_dataset(data_path) (importé depuis preprocess.py)
    X, y = load_dataset(data_path)
    # Sépare en train 80% / test 20% (test_size=0.2, stratify=y => conserve la même proportion de classes dans train et test, random_state=42 => fige le tirage aléatoire → reproductibilité).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Construit un Pipeline avec: 
    # étape preprocess: build_preprocessor() (feature engineering / encodage dans preprocess.py),
    # étape classifier: RandomForestClassifier(**params).
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )
    # Entraîne le pipeline sur l'ensemble d'entraînement.
    pipeline.fit(X_train, y_train)

    # Calcule probabilités et métriques sur le test interne.
    # Attention `model.predict()` applique implicitement un seuil de 0.5 sur la probabilité.
    # **Ce seuil n'est pas optimal** en déséquilibre. Pour Pyrenex, baisser le seuil à 0.3 peut faire chuter la précision mais explose le recall sur les défauts — souvent ce que le métier veut.
    # TODO reste à faire de jour avec le seuil : (model.predict_proba(X_test)[:, 1] > 0.3).astype(int). Recompute la matrice de confusion. Qu'est-ce qui bouge ?
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_default": f1_score(y_test, y_pred, pos_label=1),
        "precision_default": precision_score(y_test, y_pred, pos_label=1),
        "recall_default": recall_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm.tolist(),
    }

    # Sauvegarde le modèle (joblib.dump) et écrit un fichier JSON meta contenant: versions, horodatage, sha256 du dataset, hyperparamètres, métriques internes, colonnes de features et mapping de la cible.
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"pyrenex_risk_v2_{config_name}.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    metrics_test_internal = {
        k: round(v, 4)
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }
    metrics_test_internal["confusion_matrix"] = metrics["confusion_matrix"]

    meta = {
        "model_name": "pyrenex_risk_v2",
        "model_version": "v2.0.0",
        "config_name": config_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "hyperparameters": params,
        "metrics_test_internal": metrics_test_internal,
        "feature_columns": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "target": {"column": TARGET_COLUMN, "mapping": TARGET_MAPPING},
    }
    end_time = time.perf_counter()
    duration_seconds = round(end_time - start_time, 4)
    metrics["duration_seconds"] = duration_seconds

    meta["metrics_test_internal"]["duration_seconds"] = duration_seconds

    meta_path = output_dir / f"pyrenex_risk_v2_{config_name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"model_path": model_path, "meta_path": meta_path, "metrics": metrics}


def print_summary_table(results: list[tuple[str, dict]]) -> None:
    headers = (
        "Config",
        "Accuracy",
        "F1 macro",
        "F1 défaut",
        "Precision défaut",
        "Recall défaut",
        "ROC-AUC",
        "Duration(s)",
    )
    row_format = "{:<22} {:>8} {:>8} {:>9} {:>15} {:>13} {:>8} {:>12}"
    print("\nSummary of all configs:")
    print(row_format.format(*headers))
    print("-" * 104)
    for config_name, metrics in results:
        print(
            row_format.format(
                config_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['f1_macro']:.4f}",
                f"{metrics['f1_default']:.4f}",
                f"{metrics['precision_default']:.4f}",
                f"{metrics['recall_default']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics.get('duration_seconds', 0.0):.4f}",
            )
        )


def main() -> None:
    # parse les arguments et appelle train(config, data, output)
    parser = argparse.ArgumentParser(description="Train Pyrenex risk model")
    parser.add_argument("--config", default="default", choices=["all"] + list(CONFIGS))
    parser.add_argument("--data", default="data/lending_club_train.csv", type=Path)
    parser.add_argument("--output", default="models/", type=Path)
    args = parser.parse_args()

    if args.config == "all":
        results = []
        for config_name in CONFIGS:
            print(f"\n=== Training config: {config_name} ===")
            result = train(config_name, args.data, args.output)
            results.append((config_name, result["metrics"]))
            print(f"Saved {result['model_path']}")
            print(f"Duration: {result['metrics']['duration_seconds']} sec")
        print_summary_table(results)
        return

    result = train(args.config, args.data, args.output)
    print(f"Model saved to {result['model_path']}")
    print(f"Metadata saved to {result['meta_path']}")
    print(f"Metrics (test internal): {result['metrics']}")
    print(
        "\nNext step: once you have chosen your retained config, promote it:\n"
        f"  cp {result['model_path']} {args.output}/pyrenex_risk_v2.joblib\n"
        f"  cp {result['meta_path']} {args.output}/pyrenex_risk_v2.json\n"
        "  python src/evaluate.py --update-meta"
    )


if __name__ == "__main__":
    main()