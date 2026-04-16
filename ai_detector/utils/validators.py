from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    warnings: list[str]


def validate_input_text(text: str, min_words: int, max_chars: int) -> ValidationResult:
    warnings: list[str] = []
    cleaned = text.strip()

    if not cleaned:
        return ValidationResult(False, ["Debes ingresar texto para analizar."])
    if len(cleaned) > max_chars:
        return ValidationResult(
            False,
            [f"El texto supera el máximo permitido de {max_chars} caracteres."],
        )

    word_count = len(cleaned.split())
    if word_count < min_words:
        warnings.append(
            "Texto corto: el análisis es menos confiable y puede producir más error."
        )

    return ValidationResult(True, warnings)
