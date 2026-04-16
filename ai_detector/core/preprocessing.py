from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect

from ai_detector.config import CONFIG
from ai_detector.utils.text_helpers import normalize_whitespace, split_paragraphs, split_sentences, tokenize_words

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0


@dataclass(slots=True)
class ProcessedText:
    original_text: str
    cleaned_text: str
    language: str
    words: list[str]
    sentences: list[str]
    paragraphs: list[str]


def clean_text(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_whitespace(text)


def detect_language(text: str) -> str:
    try:
        if len(text) < 20:
            return CONFIG.default_language
        return detect(text)
    except Exception as exc:  # pragma: no cover
        logger.warning("Language detection failed: %s", exc)
        return CONFIG.default_language


def preprocess_text(text: str) -> ProcessedText:
    cleaned = clean_text(text)
    paragraphs = split_paragraphs(text)
    sentences = split_sentences(cleaned)
    words = tokenize_words(cleaned)
    language = detect_language(cleaned)

    return ProcessedText(
        original_text=text,
        cleaned_text=cleaned,
        language=language,
        words=words,
        sentences=sentences,
        paragraphs=paragraphs,
    )
