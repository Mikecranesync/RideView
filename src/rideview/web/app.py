"""
Flask application factory for RideView web interface.

Provides a web-based interface for:
- Live video streaming with detection overlay
- Real-time status monitoring
- Configuration management
- Snapshot capture
"""

import logging
from pathlib import Path

from flask import Flask, render_template

from ..core.camera import Camera
from ..core.config import Config
from ..core.detector import TorqueStripeDetector

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> Flask:
    """
    Application factory for Flask app.

    Args:
        config: RideView configuration, or None to load defaults

    Returns:
        Configured Flask application
    """
    # Get template and static directories
    app_dir = Path(__file__).parent
    template_dir = app_dir / "templates"
    static_dir = app_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    # Load configuration
    if config is None:
        config = Config()

    # Store config in app
    app.config["RIDEVIEW_CONFIG"] = config

    # Initialize camera and detector
    camera = Camera(config["camera"])
    detector = TorqueStripeDetector(config.as_dict)

    app.config["camera"] = camera
    app.config["detector"] = detector
    app.config["STREAM_QUALITY"] = config.get("web.stream_quality", 85)
    app.config["MAX_STREAM_FPS"] = config.get("web.max_stream_fps", 15)

    # Register blueprints
    from .routes import api, stream

    app.register_blueprint(api.bp, url_prefix="/api")
    app.register_blueprint(stream.bp, url_prefix="/stream")

    # Main routes
    @app.route("/")
    def index() -> str:
        """Main dashboard page."""
        return render_template(
            "index.html",
            version=config.get("app.version", "0.1.0"),
        )

    @app.route("/health")
    def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "version": config.get("app.version", "0.1.0")}

    # Cleanup on shutdown
    @app.teardown_appcontext
    def cleanup(exception: Exception | None = None) -> None:
        """Clean up resources on app shutdown."""
        cam = app.config.get("camera")
        if cam and cam.is_open:
            cam.release()

    logger.info("Flask app created")
    return app
