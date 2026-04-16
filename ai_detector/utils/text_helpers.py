from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        return [text.strip()]
    return paragraphs


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_PATTERN.findall(text)]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def top_repetition_ratio(items: Iterable[tuple[str, ...] | str]) -> float:
    items = list(items)
    if not items:
        return 0.0
    counts = Counter(items)
    top = counts.most_common(1)[0][1]
    return safe_ratio(top, len(items))
