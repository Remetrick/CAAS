from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template, request

from ai_detector.config import CONFIG
from ai_detector.models.inference import analyze_text
from ai_detector.utils.validators import validate_input_text

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", config=CONFIG)


@bp.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    validation = validate_input_text(
        text,
        min_words=CONFIG.min_words_for_reliable_analysis,
        max_chars=CONFIG.max_chars_input,
    )
    if not validation.is_valid:
        return jsonify({"ok": False, "errors": validation.warnings}), 400

    try:
        result = analyze_text(text)
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis error")
        return jsonify({"ok": False, "errors": [f"No fue posible analizar el texto: {exc}"]}), 500

    return jsonify({"ok": True, "warnings": validation.warnings, "result": result})
