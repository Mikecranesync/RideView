"""
RideView CLI entry point.

Main application for live torque stripe detection.

Usage:
    python -m rideview              # Live detection mode
    python -m rideview --web        # Start web server
    python -m rideview --help       # Show help
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .core.camera import Camera
from .core.config import Config
from .core.detector import TorqueStripeDetector
from .core.result import DetectionResult
from .utils.visualization import draw_detection_overlay, create_comparison_view


def setup_logging(config: Config) -> None:
    """Configure logging based on config."""
    log_config = config["logging"]
    level = getattr(logging, log_config.get("level", "INFO"))

    # Create logs directory
    log_file = log_config.get("file", "logs/rideview.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def save_snapshot(
    frame: np.ndarray,
    result: DetectionResult,
    config: Config,
    include_overlay: bool = True,
) -> str:
    """
    Save a snapshot to disk.

    Args:
        frame: Frame to save
        result: Detection result
        config: Configuration
        include_overlay: Whether frame already has overlay

    Returns:
        Path to saved file
    """
    snap_config = config["snapshots"]
    snap_dir = Path(snap_config.get("directory", "snapshots"))
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename_format = snap_config.get("filename_format", "%Y%m%d_%H%M%S_{result}.jpg")
    filename = datetime.now().strftime(filename_format).format(result=result.value.lower())

    filepath = snap_dir / filename
    cv2.imwrite(str(filepath), frame)

    return str(filepath)


def run_live_detection(config: Config) -> None:
    """
    Run live detection mode with OpenCV window.

    Keyboard controls:
        q - Quit
        s - Save snapshot
        c - Toggle comparison view
        space - Pause/Resume
        r - Reset ROI
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting live detection mode...")

    # Initialize camera
    camera = Camera(config["camera"])
    if not camera.open():
        logger.error("Failed to open camera. Check connection and try again.")
        sys.exit(1)

    # Initialize detector
    detector = TorqueStripeDetector(config.as_dict)

    # State variables
    paused = False
    show_comparison = False
    roi = None
    selecting_roi = False
    roi_start = None

    # FPS calculation
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    # Window name
    window_name = "RideView - Torque Stripe Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ROI selection callback
    def mouse_callback(event: int, x: int, y: int, flags: int, param: None) -> None:
        nonlocal roi, selecting_roi, roi_start

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting_roi = True
            roi_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
            selecting_roi = False
            if roi_start:
                x1, y1 = roi_start
                x2, y2 = x, y
                # Ensure proper order
                roi = (
                    min(x1, x2),
                    min(y1, y2),
                    abs(x2 - x1),
                    abs(y2 - y1),
                )
                if roi[2] < 20 or roi[3] < 20:
                    roi = None  # Too small, ignore
                logger.info(f"ROI set: {roi}")

    cv2.setMouseCallback(window_name, mouse_callback)

    logger.info("Controls: q=quit, s=snapshot, c=comparison, space=pause, r=reset ROI")
    logger.info("Click and drag to select Region of Interest (ROI)")

    last_frame = None
    last_analysis = None

    try:
        while True:
            # Read frame
            if not paused:
                frame = camera.read()
                if frame is None:
                    logger.warning("Failed to read frame")
                    continue

                # Run detection
                analysis = detector.analyze(frame, roi)
                last_frame = frame
                last_analysis = analysis

                # Update FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
            else:
                frame = last_frame
                analysis = last_analysis
                if frame is None:
                    continue

            # Prepare display frame
            if show_comparison and analysis and analysis.stripe_mask is not None:
                display = create_comparison_view(
                    frame,
                    analysis.stripe_mask,
                    analysis.annotated_frame if analysis.annotated_frame is not None else frame,
                    scale=0.6,
                )
            else:
                if analysis:
                    display = draw_detection_overlay(
                        frame, analysis, show_metrics=True, show_fps=current_fps
                    )
                else:
                    display = frame.copy()

            # Draw ROI selection rectangle
            if selecting_roi and roi_start:
                # Get current mouse position (approximate with last known)
                pass  # Would need to track mouse position

            # Draw paused indicator
            if paused:
                cv2.putText(
                    display,
                    "PAUSED",
                    (display.shape[1] // 2 - 60, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            # Show frame
            cv2.imshow(window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit requested")
                break
            elif key == ord("s"):
                if analysis and analysis.annotated_frame is not None:
                    filepath = save_snapshot(analysis.annotated_frame, analysis.result, config)
                    logger.info(f"Snapshot saved: {filepath}")
            elif key == ord("c"):
                show_comparison = not show_comparison
                logger.info(f"Comparison view: {'on' if show_comparison else 'off'}")
            elif key == ord(" "):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord("r"):
                roi = None
                logger.info("ROI reset")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("RideView stopped")


def run_web_server(config: Config) -> None:
    """Start the Flask web server."""
    logger = logging.getLogger(__name__)
    logger.info("Starting web server...")

    # Import Flask app
    from .web.app import create_app

    # Create app with config
    app = create_app(config)

    # Get web config
    web_config = config["web"]
    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("port", 5000)
    debug = config.get("app.debug", False)

    logger.info(f"Web server starting at http://{host}:{port}")

    # Run Flask development server
    app.run(host=host, port=port, debug=debug, threaded=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RideView - Torque Stripe Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m rideview              Run live detection
    python -m rideview --web        Start web server
    python -m rideview --camera 1   Use camera index 1

Keyboard controls (live mode):
    q       Quit
    s       Save snapshot
    c       Toggle comparison view
    space   Pause/Resume
    r       Reset ROI
        """,
    )

    parser.add_argument(
        "--web", action="store_true", help="Start web server instead of live detection"
    )
    parser.add_argument(
        "--camera", type=int, help="Camera index to use (overrides config)"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration directory"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )

    args = parser.parse_args()

    # Load configuration
    config_dir = Path(args.config) if args.config else None
    config = Config(config_dir)

    # Apply command-line overrides
    if args.camera is not None:
        os.environ["RIDEVIEW_CAMERA_SOURCE"] = str(args.camera)
        config.reload()

    if args.debug:
        os.environ["RIDEVIEW_ENV"] = "development"
        config.reload()

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("RideView starting...")
    logger.info(f"Environment: {config.env}")

    # Run appropriate mode
    if args.web:
        run_web_server(config)
    else:
        run_live_detection(config)


if __name__ == "__main__":
    main()
