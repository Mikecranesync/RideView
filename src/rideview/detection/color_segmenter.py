"""
HSV color segmentation for torque stripe detection.

Isolates torque stripe colors (red, yellow, orange) from the background
using HSV color space thresholding. HSV is preferred over RGB because
it separates color information (Hue) from intensity (Value), making
detection more robust under varying lighting conditions.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ColorRange:
    """
    Represents an HSV color range for segmentation.

    Attributes:
        name: Identifier for this color range
        lower: Lower HSV bounds [H, S, V]
        upper: Upper HSV bounds [H, S, V]
    """

    name: str
    lower: np.ndarray
    upper: np.ndarray

    def __init__(self, name: str, lower: tuple[int, int, int], upper: tuple[int, int, int]):
        self.name = name
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)


class ColorSegmenter:
    """
    Segments torque stripes by color using HSV color space.

    HSV is preferred over RGB because it separates color information (Hue)
    from intensity (Value), making detection more robust under varying lighting.

    OpenCV HSV ranges:
    - H (Hue): 0-179 (half of standard 0-360)
    - S (Saturation): 0-255
    - V (Value/Brightness): 0-255

    Usage:
        segmenter = ColorSegmenter(config['colors'])
        mask = segmenter.segment(frame)
    """

    # Default color ranges optimized for common torque stripe colors
    DEFAULT_COLORS: dict[str, ColorRange] = {
        "red_low": ColorRange("red_low", (0, 100, 100), (10, 255, 255)),
        "red_high": ColorRange("red_high", (170, 100, 100), (179, 255, 255)),
        "yellow": ColorRange("yellow", (20, 100, 100), (35, 255, 255)),
        "orange": ColorRange("orange", (10, 100, 100), (20, 255, 255)),
    }

    def __init__(self, config: dict[str, Any]):
        """
        Initialize with color configuration.

        Args:
            config: Dictionary with color ranges:
                {
                    'enabled_colors': ['red', 'yellow'],
                    'custom_ranges': {
                        'red_low': {'lower': [0, 100, 100], 'upper': [10, 255, 255]}
                    }
                }
        """
        self.enabled_colors = config.get("enabled_colors", ["red", "yellow", "orange"])
        self.color_ranges = self._build_color_ranges(config)

    def _build_color_ranges(self, config: dict) -> list[ColorRange]:
        """Build list of color ranges from config."""
        ranges: list[ColorRange] = []
        custom = config.get("custom_ranges", {})

        for color in self.enabled_colors:
            if color == "red":
                # Red requires two ranges (wraps around hue 0)
                for key in ["red_low", "red_high"]:
                    if key in custom:
                        ranges.append(
                            ColorRange(
                                key,
                                tuple(custom[key]["lower"]),  # type: ignore
                                tuple(custom[key]["upper"]),  # type: ignore
                            )
                        )
                    else:
                        ranges.append(self.DEFAULT_COLORS[key])
            else:
                key = color
                if key in custom:
                    ranges.append(
                        ColorRange(
                            key,
                            tuple(custom[key]["lower"]),  # type: ignore
                            tuple(custom[key]["upper"]),  # type: ignore
                        )
                    )
                elif key in self.DEFAULT_COLORS:
                    ranges.append(self.DEFAULT_COLORS[key])

        return ranges

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the frame to isolate torque stripe colors.

        Args:
            frame: BGR image

        Returns:
            Binary mask where white (255) = detected stripe pixels
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Initialize empty mask
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Apply each color range and combine with OR
        for color_range in self.color_ranges:
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def segment_with_debug(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Segment with individual masks for debugging/calibration.

        Args:
            frame: BGR image

        Returns:
            Dictionary with 'combined' and individual color masks
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks: dict[str, np.ndarray] = {}
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)

        for color_range in self.color_ranges:
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
            masks[color_range.name] = mask
            combined = cv2.bitwise_or(combined, mask)

        masks["combined"] = combined
        return masks

    def update_color_range(
        self, name: str, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> None:
        """
        Update a color range at runtime.

        Args:
            name: Color range name to update
            lower: New lower HSV bounds
            upper: New upper HSV bounds
        """
        for i, cr in enumerate(self.color_ranges):
            if cr.name == name:
                self.color_ranges[i] = ColorRange(name, lower, upper)
                return

        # If not found, add new range
        self.color_ranges.append(ColorRange(name, lower, upper))

    def get_color_ranges(self) -> dict[str, dict[str, list[int]]]:
        """Get current color ranges as dictionary."""
        return {
            cr.name: {"lower": cr.lower.tolist(), "upper": cr.upper.tolist()}
            for cr in self.color_ranges
        }
