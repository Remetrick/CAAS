from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ai_detector.config import CONFIG
from ai_detector.core.thresholds import RiskThresholds

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreResult:
    probability_ia: float
    score_0_100: int
    risk_level: str
    model_name: str


def _fallback_heuristic_probability(features: dict[str, float]) -> float:
    weighted_sum = (
        1.25 * features.get("structural_uniformity", 0)
        + 1.1 * features.get("paragraph_embedding_similarity", 0)
        + 0.85 * features.get("trigram_repetition", 0)
        + 0.75 * max(0, 0.55 - features.get("ttr", 0))
        + 0.55 * max(0, 0.5 - features.get("burstiness", 0))
        + 0.35 * features.get("generic_voice_indicator", 0) * 10
    )
    return float(1 / (1 + np.exp(-3 * (weighted_sum - 1.1))))


def score_features(features: dict[str, float], model_path: Path | None = None) -> ScoreResult:
    thresholds = RiskThresholds(*CONFIG.risk_thresholds)
    used_model = "heuristic_fallback"

    probability = None
    chosen_path = model_path or (CONFIG.model_dir / "hybrid_calibrated_model.joblib")

    if chosen_path.exists():
        try:
            bundle = joblib.load(chosen_path)
            model = bundle["model"]
            feature_names = bundle["feature_names"]
            x = np.array([[features.get(name, 0.0) for name in feature_names]], dtype=float)
            probability = float(model.predict_proba(x)[0, 1])
            used_model = bundle.get("model_name", "trained_model")
        except Exception as exc:  # pragma: no cover
            logger.warning("Model inference failed, using fallback: %s", exc)

    if probability is None:
        probability = _fallback_heuristic_probability(features)

    score = int(round(probability * 100))
    return ScoreResult(
        probability_ia=probability,
        score_0_100=score,
        risk_level=thresholds.get_risk_level(score),
        model_name=used_model,
    )
