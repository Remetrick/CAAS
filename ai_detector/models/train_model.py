from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_detector.config import CONFIG
from ai_detector.core.feature_extraction import extract_features
from ai_detector.core.preprocessing import preprocess_text
from ai_detector.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


def build_manual_feature_frame(texts: list[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        processed = preprocess_text(text)
        features = extract_features(processed).values
        rows.append(features)
    return pd.DataFrame(rows)


def build_hybrid_matrix(texts: list[str], manual_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    emb = np.zeros((len(texts), 384), dtype=np.float32)
    emb_feature_names = [f"emb_{i}" for i in range(emb.shape[1])]

    if SentenceTransformer is not None:
        model = SentenceTransformer(CONFIG.sentence_model_name)
        emb = model.encode(texts, normalize_embeddings=True)

    hybrid = np.hstack([manual_df.values, emb])
    feature_names = list(manual_df.columns) + emb_feature_names
    return hybrid, feature_names


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("El CSV debe incluir columnas: text, label")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    manual_df = build_manual_feature_frame(texts)
    x_hybrid, feature_names_hybrid = build_hybrid_matrix(texts, manual_df)

    x_train, x_tmp, y_train, y_tmp = train_test_split(x_hybrid, labels, test_size=0.3, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    base_model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(x_train, y_train)

    val_probs = calibrated.predict_proba(x_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    results = {
        "validation": metrics_dict(y_val, val_preds, val_probs),
        "test": metrics_dict(y_test, test_preds, test_probs),
        "confusion_matrix_test": confusion_matrix(y_test, test_preds).tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": calibrated,
            "feature_names": feature_names_hybrid,
            "model_name": "hybrid_histgb_calibrated",
        },
        output_dir / "hybrid_calibrated_model.joblib",
    )

    # Modelo interpretable (solo features manuales)
    manual_x = manual_df.values
    x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(
        manual_x, labels, test_size=0.2, random_state=42, stratify=labels
    )

    simple_pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_depth=4, random_state=42)),
        ]
    )
    simple_pipe.fit(x_train_m, y_train_m)
    simple_probs = simple_pipe.predict_proba(x_test_m)[:, 1]
    simple_preds = (simple_probs >= 0.5).astype(int)

    results["manual_model_test"] = metrics_dict(y_test_m, simple_preds, simple_probs)

    joblib.dump(
        {
            "model": simple_pipe,
            "feature_names": list(manual_df.columns),
            "model_name": "manual_histgb",
        },
        output_dir / "manual_model.joblib",
    )

    (output_dir / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Entrenamiento completado. Métricas: %s", json.dumps(results, indent=2))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Entrenamiento de detector IA")
    parser.add_argument("--csv", type=Path, required=True, help="Ruta al dataset CSV con text,label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.model_dir,
        help="Carpeta de salida para modelos y métricas",
    )
    args = parser.parse_args()

    train(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
