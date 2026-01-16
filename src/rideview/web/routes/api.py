"""
REST API routes for RideView.

Provides JSON endpoints for:
- Current detection status
- Configuration management
- Snapshot capture
"""

import logging
from datetime import datetime
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, request

from .stream import get_latest_analysis

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


@bp.route("/status")
def status():
    """
    Get current detection status.

    Returns:
        JSON with current detection result and metrics
    """
    analysis = get_latest_analysis()

    if analysis is None:
        return jsonify({
            "status": "no_data",
            "message": "No detection data available yet",
        })

    return jsonify(analysis.to_dict())


@bp.route("/config", methods=["GET"])
def get_config():
    """
    Get current configuration.

    Returns:
        JSON with current configuration
    """
    config = current_app.config["RIDEVIEW_CONFIG"]

    # Return safe subset of configuration
    return jsonify({
        "camera": {
            "source": config.get("camera.source"),
            "width": config.get("camera.width"),
            "height": config.get("camera.height"),
            "fps": config.get("camera.fps"),
        },
        "colors": {
            "enabled_colors": config.get("colors.enabled_colors"),
        },
        "thresholds": config.get("thresholds", {}),
    })


@bp.route("/config", methods=["POST"])
def update_config():
    """
    Update configuration at runtime.

    Accepts JSON body with configuration updates.

    Returns:
        JSON with updated configuration
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    detector = current_app.config["detector"]

    # Handle color range updates
    if "colors" in data:
        color_ranges = data["colors"].get("custom_ranges", {})
        for name, range_data in color_ranges.items():
            if "lower" in range_data and "upper" in range_data:
                detector.update_color_range(
                    name,
                    tuple(range_data["lower"]),
                    tuple(range_data["upper"]),
                )
                logger.info(f"Updated color range: {name}")

    return jsonify({"status": "ok", "message": "Configuration updated"})


@bp.route("/colors")
def get_colors():
    """
    Get current color detection ranges.

    Returns:
        JSON with HSV color ranges
    """
    detector = current_app.config["detector"]
    return jsonify(detector.get_color_ranges())


@bp.route("/snapshot", methods=["POST"])
def capture_snapshot():
    """
    Capture and save a snapshot.

    Returns:
        JSON with snapshot file path
    """
    camera = current_app.config["camera"]
    detector = current_app.config["detector"]
    config = current_app.config["RIDEVIEW_CONFIG"]

    # Get current frame
    frame = camera.read()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Run detection
    analysis = detector.analyze(frame)

    # Save snapshot
    snap_dir = Path(config.get("snapshots.directory", "snapshots"))
    snap_dir.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{analysis.result.value.lower()}.jpg"
    filepath = snap_dir / filename

    # Use annotated frame if available
    save_frame = analysis.annotated_frame if analysis.annotated_frame is not None else frame
    cv2.imwrite(str(filepath), save_frame)

    logger.info(f"Snapshot saved: {filepath}")

    return jsonify({
        "status": "ok",
        "filepath": str(filepath),
        "result": analysis.to_dict(),
    })


@bp.route("/cameras")
def list_cameras():
    """
    List available camera indices.

    Returns:
        JSON with list of available camera indices
    """
    from ...core.camera import Camera

    available = Camera.list_available_cameras()
    return jsonify({"cameras": available})
