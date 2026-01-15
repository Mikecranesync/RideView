"""
Video streaming routes for RideView.

Provides MJPEG video streaming with detection overlay for
remote viewing on phones or other devices.
"""

import logging
import time
from typing import Generator

import cv2
from flask import Blueprint, Response, current_app

logger = logging.getLogger(__name__)

bp = Blueprint("stream", __name__)

# Store latest analysis for API access
_latest_analysis = None


def generate_frames_with_detection() -> Generator[bytes, None, None]:
    """
    Generator that yields MJPEG frames with detection overlay.

    Yields:
        MJPEG frame bytes
    """
    global _latest_analysis

    camera = current_app.config["camera"]
    detector = current_app.config["detector"]
    quality = current_app.config.get("STREAM_QUALITY", 85)
    max_fps = current_app.config.get("MAX_STREAM_FPS", 15)

    min_frame_time = 1.0 / max_fps

    # Open camera if not already open
    if not camera.is_open:
        if not camera.open():
            logger.error("Failed to open camera for streaming")
            return

    while True:
        start_time = time.time()

        frame = camera.read()
        if frame is None:
            continue

        # Run detection
        analysis = detector.analyze(frame)
        _latest_analysis = analysis

        # Use annotated frame with overlay
        output_frame = analysis.annotated_frame if analysis.annotated_frame is not None else frame

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", output_frame, encode_params)

        # Yield as multipart frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        # Rate limiting
        elapsed = time.time() - start_time
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)


def generate_frames_raw() -> Generator[bytes, None, None]:
    """
    Generator that yields raw MJPEG frames without detection.

    Lower latency, useful for calibration.

    Yields:
        MJPEG frame bytes
    """
    camera = current_app.config["camera"]
    quality = current_app.config.get("STREAM_QUALITY", 85)
    max_fps = current_app.config.get("MAX_STREAM_FPS", 30)

    min_frame_time = 1.0 / max_fps

    if not camera.is_open:
        if not camera.open():
            logger.error("Failed to open camera for streaming")
            return

    while True:
        start_time = time.time()

        frame = camera.read()
        if frame is None:
            continue

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", frame, encode_params)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        # Rate limiting
        elapsed = time.time() - start_time
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)


@bp.route("/video_feed")
def video_feed() -> Response:
    """Video streaming route with detection overlay."""
    return Response(
        generate_frames_with_detection(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/raw_feed")
def raw_feed() -> Response:
    """Raw video feed without detection (lower latency)."""
    return Response(
        generate_frames_raw(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def get_latest_analysis():
    """Get the latest analysis result."""
    return _latest_analysis
