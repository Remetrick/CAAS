from __future__ import annotations

from flask import Flask

from ai_detector.app.routes import bp
from ai_detector.config import CONFIG
from ai_detector.utils.logging_config import configure_logging


def create_app() -> Flask:
    configure_logging()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=CONFIG.debug)
