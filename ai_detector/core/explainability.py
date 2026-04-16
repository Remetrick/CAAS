from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Explanation:
    headline: str
    details: list[str]


def generate_explanations(features: dict[str, float]) -> Explanation:
    details: list[str] = []

    if features.get("burstiness", 0) < 0.45:
        details.append("El texto presenta baja variación entre oraciones (burstiness baja).")
    if features.get("structural_uniformity", 0) > 0.3:
        details.append("Se detecta alta uniformidad estructural en inicios de oración.")
    if features.get("trigram_repetition", 0) > 0.12:
        details.append("Existe repetición de patrones de n-gramas por encima del rango esperado.")
    if features.get("ttr", 1) < 0.45:
        details.append("La diversidad léxica (TTR) es menor a la esperada para escritura espontánea.")
    if features.get("paragraph_embedding_similarity", 0) > 0.75:
        details.append("Los párrafos son semánticamente muy similares, lo que sugiere uniformidad excesiva.")
    if features.get("generic_voice_indicator", 0) > 0.04:
        details.append("Se observaron expresiones genéricas/recurrentes asociadas a redacción plantilla.")

    if not details:
        details.append(
            "No se observaron señales extremas; la evaluación depende de una combinación probabilística de rasgos."
        )

    headline = "Análisis explicable basado en señales lingüísticas y semánticas."
    return Explanation(headline=headline, details=details)
