from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ai_detector.core.explainability import generate_explanations
from ai_detector.core.feature_extraction import extract_features
from ai_detector.core.preprocessing import preprocess_text
from ai_detector.core.scoring import score_features


def analyze_text(text: str) -> dict[str, Any]:
    processed = preprocess_text(text)
    feature_set = extract_features(processed)
    score_result = score_features(feature_set.values)
    explanation = generate_explanations(feature_set.values)

    return {
        "language": processed.language,
        "word_count": len(processed.words),
        "sentence_count": len(processed.sentences),
        "paragraph_count": len(processed.paragraphs),
        "features": feature_set.values,
        "score": asdict(score_result),
        "explanation": asdict(explanation),
        "sentence_analysis": feature_set.sentence_analysis,
        "paragraph_analysis": feature_set.paragraph_analysis,
        "disclaimer": "Este resultado es orientativo y no constituye una prueba definitiva.",
    }
