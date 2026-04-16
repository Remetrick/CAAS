from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ai_detector.config import CONFIG
from ai_detector.core.preprocessing import ProcessedText
from ai_detector.utils.text_helpers import ngrams, safe_ratio, top_repetition_ratio

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureSet:
    values: dict[str, float]
    sentence_analysis: list[dict[str, Any]]
    paragraph_analysis: list[dict[str, Any]]


try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


def _mean_sentence_length(sentences: list[str]) -> float:
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return mean(lengths) if lengths else 0.0


def _std_sentence_length(sentences: list[str]) -> float:
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return pstdev(lengths) if len(lengths) > 1 else 0.0


def _burstiness(sentences: list[str]) -> float:
    avg = _mean_sentence_length(sentences)
    std = _std_sentence_length(sentences)
    return safe_ratio(std, avg)


def _type_token_ratio(words: list[str]) -> float:
    return safe_ratio(len(set(words)), len(words))


def _unique_word_frequency(words: list[str]) -> float:
    counts = Counter(words)
    hapax = sum(1 for _, c in counts.items() if c == 1)
    return safe_ratio(hapax, len(words))


def _ngram_repetition(words: list[str], n: int = 3) -> float:
    grams = ngrams(words, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return safe_ratio(repeated, len(grams))


def _punctuation_density(text: str) -> float:
    punct = re.findall(r"[.,;:!?¿¡()\-\"]", text)
    return safe_ratio(len(punct), len(text))


def _punctuation_variability(text: str) -> float:
    punct = re.findall(r"[.,;:!?¿¡()\-\"]", text)
    if not punct:
        return 0.0
    counts = np.array(list(Counter(punct).values()), dtype=float)
    return float(np.std(counts) / (np.mean(counts) + 1e-9))


def _logical_connector_frequency(text: str) -> float:
    lower = text.lower()
    hits = sum(lower.count(c) for c in CONFIG.logical_connectors)
    return safe_ratio(hits, len(text.split()))


def _generic_voice_indicator(text: str) -> float:
    lower = text.lower()
    suspicious_hits = sum(lower.count(p) for p in CONFIG.suspicious_phrases)
    hedging = len(re.findall(r"\b(probablemente|posiblemente|generalmente|typically|generally)\b", lower))
    return safe_ratio(suspicious_hits + hedging, len(text.split()))


def _perplexity_proxy(words: list[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    probs = np.array([c / len(words) for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(2 ** entropy)


def _embedding_similarity(paragraphs: list[str]) -> float:
    if len(paragraphs) < 2:
        return 0.0

    vectors = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(CONFIG.sentence_model_name)
            vectors = model.encode(paragraphs, normalize_embeddings=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("SentenceTransformer unavailable, using TF-IDF fallback: %s", exc)

    if vectors is None:
        vectors = TfidfVectorizer(max_features=768).fit_transform(paragraphs).toarray()

    sim = cosine_similarity(vectors)
    upper = sim[np.triu_indices_from(sim, k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def _structural_uniformity(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    starts = [" ".join(s.lower().split()[:3]) for s in sentences if s.split()]
    return top_repetition_ratio(starts)


def extract_features(processed: ProcessedText) -> FeatureSet:
    text = processed.cleaned_text
    words = processed.words
    sentences = processed.sentences
    paragraphs = processed.paragraphs

    sentence_lengths = [len(s.split()) for s in sentences]
    paragraph_lengths = [len(p.split()) for p in paragraphs]

    sentence_analysis = [
        {
            "sentence": s,
            "word_count": len(s.split()),
            "punctuation_density": _punctuation_density(s),
            "connector_density": _logical_connector_frequency(s),
        }
        for s in sentences
    ]

    paragraph_analysis = [
        {
            "paragraph": p,
            "word_count": len(p.split()),
            "sentence_count": max(1, len(re.split(r"(?<=[.!?])\s+", p.strip()))),
        }
        for p in paragraphs
    ]

    features = {
        "avg_sentence_length": _mean_sentence_length(sentences),
        "std_sentence_length": _std_sentence_length(sentences),
        "burstiness": _burstiness(sentences),
        "ttr": _type_token_ratio(words),
        "unique_word_frequency": _unique_word_frequency(words),
        "trigram_repetition": _ngram_repetition(words, n=3),
        "fourgram_repetition": _ngram_repetition(words, n=4),
        "punctuation_density": _punctuation_density(text),
        "punctuation_variability": _punctuation_variability(text),
        "logical_connector_frequency": _logical_connector_frequency(text),
        "paragraph_embedding_similarity": _embedding_similarity(paragraphs),
        "structural_uniformity": _structural_uniformity(sentences),
        "generic_voice_indicator": _generic_voice_indicator(text),
        "perplexity_proxy": _perplexity_proxy(words),
        "sentence_length_cv": safe_ratio(np.std(sentence_lengths), np.mean(sentence_lengths) + 1e-9)
        if sentence_lengths
        else 0.0,
        "paragraph_length_cv": safe_ratio(np.std(paragraph_lengths), np.mean(paragraph_lengths) + 1e-9)
        if paragraph_lengths
        else 0.0,
    }

    return FeatureSet(features, sentence_analysis, paragraph_analysis)
