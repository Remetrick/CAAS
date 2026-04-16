from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    app_name: str = "AI Text Detector"
    debug: bool = False
    min_words_for_reliable_analysis: int = 80
    max_chars_input: int = 50000
    risk_thresholds: tuple[int, int] = (30, 65)
    default_language: str = "unknown"
    logical_connectors: tuple[str, ...] = (
        "por lo tanto",
        "además",
        "sin embargo",
        "en consecuencia",
        "por otra parte",
        "en resumen",
        "therefore",
        "however",
        "moreover",
        "in conclusion",
        "furthermore",
    )
    suspicious_phrases: tuple[str, ...] = (
        "en conclusión",
        "en resumen",
        "en definitiva",
        "it is important to note",
        "in conclusion",
        "overall",
    )
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir: Path = field(default_factory=lambda: Path("ai_detector/models/saved_model"))


CONFIG = AppConfig()
