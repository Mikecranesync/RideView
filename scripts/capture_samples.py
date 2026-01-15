#!/usr/bin/env python3
"""
Sample Image Capture Tool for RideView.

Capture and organize sample images for testing and validation.
Images are saved to the samples/ directory organized by category.

Usage:
    python scripts/capture_samples.py
    python scripts/capture_samples.py --camera 1 --output samples/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rideview.core.camera import Camera
from rideview.core.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="RideView Sample Image Capture Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
    i       Save as INTACT (pass)
    b       Save as BROKEN (fail)
    w       Save as WARNING
    space   Save with auto-name
    q       Quit
        """,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output", type=str, default="samples", help="Output directory")
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

    # Create output directories
    output_dir = Path(args.output)
    intact_dir = output_dir / "intact"
    broken_dir = output_dir / "broken"
    warning_dir = output_dir / "warning"

    for d in [intact_dir, broken_dir, warning_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Counters
    counts = {"intact": 0, "broken": 0, "warning": 0}

    # Count existing files
    counts["intact"] = len(list(intact_dir.glob("*.jpg")))
    counts["broken"] = len(list(broken_dir.glob("*.jpg")))
    counts["warning"] = len(list(warning_dir.glob("*.jpg")))

    print("=== RideView Sample Capture ===")
    print(f"Output directory: {output_dir}")
    print(f"Existing samples: intact={counts['intact']}, broken={counts['broken']}, warning={counts['warning']}")
    print("\nControls:")
    print("  i - Save as INTACT (pass)")
    print("  b - Save as BROKEN (fail)")
    print("  w - Save as WARNING")
    print("  space - Save with timestamp")
    print("  q - Quit")

    window_name = "RideView Sample Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            # Add info overlay
            display = frame.copy()
            info_text = f"Intact: {counts['intact']} | Broken: {counts['broken']} | Warning: {counts['warning']}"
            cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "Press i/b/w to save, q to quit", (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("i"):
                # Save as intact
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"intact_{timestamp}.jpg"
                filepath = intact_dir / filename
                cv2.imwrite(str(filepath), frame)
                counts["intact"] += 1
                print(f"Saved: {filepath}")

            elif key == ord("b"):
                # Save as broken
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"broken_{timestamp}.jpg"
                filepath = broken_dir / filename
                cv2.imwrite(str(filepath), frame)
                counts["broken"] += 1
                print(f"Saved: {filepath}")

            elif key == ord("w"):
                # Save as warning
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"warning_{timestamp}.jpg"
                filepath = warning_dir / filename
                cv2.imwrite(str(filepath), frame)
                counts["warning"] += 1
                print(f"Saved: {filepath}")

            elif key == ord(" "):
                # Save with auto-name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sample_{timestamp}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"Saved: {filepath}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n=== Final Counts ===")
    print(f"Intact: {counts['intact']}")
    print(f"Broken: {counts['broken']}")
    print(f"Warning: {counts['warning']}")


if __name__ == "__main__":
    main()
