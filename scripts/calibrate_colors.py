#!/usr/bin/env python3
"""
Color Calibration Tool for RideView.

Interactive tool to tune HSV color ranges for torque stripe detection.
Uses trackbars to adjust color ranges in real-time.

Usage:
    python scripts/calibrate_colors.py
    python scripts/calibrate_colors.py --camera 1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rideview.core.camera import Camera
from rideview.core.config import Config


def nothing(x):
    """Trackbar callback (required but unused)."""
    pass


def create_trackbars(window_name: str, initial_values: dict):
    """Create HSV trackbars in a window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Get initial values
    lower = initial_values.get("lower", [0, 100, 100])
    upper = initial_values.get("upper", [10, 255, 255])

    cv2.createTrackbar("H Low", window_name, lower[0], 179, nothing)
    cv2.createTrackbar("S Low", window_name, lower[1], 255, nothing)
    cv2.createTrackbar("V Low", window_name, lower[2], 255, nothing)
    cv2.createTrackbar("H High", window_name, upper[0], 179, nothing)
    cv2.createTrackbar("S High", window_name, upper[1], 255, nothing)
    cv2.createTrackbar("V High", window_name, upper[2], 255, nothing)


def get_trackbar_values(window_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Get current trackbar values."""
    h_low = cv2.getTrackbarPos("H Low", window_name)
    s_low = cv2.getTrackbarPos("S Low", window_name)
    v_low = cv2.getTrackbarPos("V Low", window_name)
    h_high = cv2.getTrackbarPos("H High", window_name)
    s_high = cv2.getTrackbarPos("S High", window_name)
    v_high = cv2.getTrackbarPos("V High", window_name)

    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    return lower, upper


def main():
    parser = argparse.ArgumentParser(
        description="RideView Color Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
    q       Quit and print current values
    s       Save current values to config
    r       Reset to default red
    y       Reset to default yellow
    o       Reset to default orange
    space   Pause/Resume
        """,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--config", type=str, help="Path to config directory")

    args = parser.parse_args()

    # Load config
    config_dir = Path(args.config) if args.config else None
    config = Config(config_dir)

    # Override camera if specified
    camera_config = config["camera"].copy()
    camera_config["source"] = args.camera

    # Initialize camera
    camera = Camera(camera_config)
    if not camera.open():
        print("Error: Failed to open camera")
        sys.exit(1)

    # Default color ranges
    defaults = {
        "red_low": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
        "red_high": {"lower": [170, 100, 100], "upper": [179, 255, 255]},
        "yellow": {"lower": [20, 100, 100], "upper": [35, 255, 255]},
        "orange": {"lower": [10, 100, 100], "upper": [20, 255, 255]},
    }

    # Create windows
    main_window = "RideView Color Calibration"
    mask_window = "Color Mask"

    # Start with red_low
    current_color = "red_low"
    create_trackbars(main_window, defaults[current_color])
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)

    paused = False
    last_frame = None

    print("=== RideView Color Calibration ===")
    print(f"Current color: {current_color}")
    print("Adjust trackbars to tune HSV range")
    print("Press 'q' to quit, 's' to save, 'r/y/o' to switch colors")

    try:
        while True:
            # Read frame
            if not paused:
                frame = camera.read()
                if frame is not None:
                    last_frame = frame
            else:
                frame = last_frame

            if frame is None:
                continue

            # Get current HSV range
            lower, upper = get_trackbar_values(main_window)

            # Convert to HSV and create mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Create overlay
            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)  # Green overlay
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Add text info
            info_text = f"Color: {current_color}"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            range_text = f"Lower: [{lower[0]}, {lower[1]}, {lower[2]}]  Upper: [{upper[0]}, {upper[1]}, {upper[2]}]"
            cv2.putText(result, range_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if paused:
                cv2.putText(result, "PAUSED", (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show windows
            cv2.imshow(main_window, result)
            cv2.imshow(mask_window, mask)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                # Print final values
                print(f"\n=== Final Values for '{current_color}' ===")
                print(f"lower: [{lower[0]}, {lower[1]}, {lower[2]}]")
                print(f"upper: [{upper[0]}, {upper[1]}, {upper[2]}]")
                print("\nYAML format:")
                print(f"  {current_color}:")
                print(f"    lower: [{lower[0]}, {lower[1]}, {lower[2]}]")
                print(f"    upper: [{upper[0]}, {upper[1]}, {upper[2]}]")
                break

            elif key == ord("s"):
                print(f"\nSaved values for '{current_color}':")
                print(f"  lower: [{lower[0]}, {lower[1]}, {lower[2]}]")
                print(f"  upper: [{upper[0]}, {upper[1]}, {upper[2]}]")
                defaults[current_color] = {
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                }

            elif key == ord("r"):
                current_color = "red_low"
                cv2.destroyWindow(main_window)
                create_trackbars(main_window, defaults[current_color])
                print(f"Switched to: {current_color}")

            elif key == ord("y"):
                current_color = "yellow"
                cv2.destroyWindow(main_window)
                create_trackbars(main_window, defaults[current_color])
                print(f"Switched to: {current_color}")

            elif key == ord("o"):
                current_color = "orange"
                cv2.destroyWindow(main_window)
                create_trackbars(main_window, defaults[current_color])
                print(f"Switched to: {current_color}")

            elif key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    # Print all saved values
    print("\n=== All Saved Values ===")
    for name, values in defaults.items():
        print(f"\n{name}:")
        print(f"  lower: {values['lower']}")
        print(f"  upper: {values['upper']}")


if __name__ == "__main__":
    main()
