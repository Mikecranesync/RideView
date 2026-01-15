"""
Image preprocessing for torque stripe detection.

Applies preprocessing steps to improve detection accuracy:
- Gaussian blur to reduce noise
- CLAHE contrast enhancement
- Optional histogram equalization
"""

from typing import Any

import cv2
import numpy as np


class Preprocessor:
    """
    Image preprocessor for torque stripe detection.

    Applies a series of image processing steps to prepare frames
    for color segmentation and line detection.

    Usage:
        preprocessor = Preprocessor(config['preprocessing'])
        processed = preprocessor.process(frame)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration with keys:
                - blur_kernel: [int, int] - Gaussian blur kernel size
                - histogram_eq: bool - Enable histogram equalization
                - clahe.enabled: bool - Enable CLAHE
                - clahe.clip_limit: float - CLAHE clip limit
                - clahe.tile_grid_size: [int, int] - CLAHE tile grid size
        """
        self.blur_kernel = tuple(config.get("blur_kernel", [5, 5]))
        self.histogram_eq = config.get("histogram_eq", False)

        clahe_config = config.get("clahe", {})
        self.clahe_enabled = clahe_config.get("enabled", True)
        self.clahe_clip_limit = clahe_config.get("clip_limit", 2.0)
        self.clahe_tile_grid_size = tuple(clahe_config.get("tile_grid_size", [8, 8]))

        # Create CLAHE object if enabled
        if self.clahe_enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size
            )
        else:
            self._clahe = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a frame.

        Args:
            frame: BGR image from camera

        Returns:
            Preprocessed BGR image
        """
        result = frame.copy()

        # Apply Gaussian blur to reduce noise
        if self.blur_kernel[0] > 0:
            result = cv2.GaussianBlur(result, self.blur_kernel, 0)

        # Apply CLAHE contrast enhancement
        if self.clahe_enabled and self._clahe is not None:
            result = self._apply_clahe(result)

        # Apply histogram equalization
        if self.histogram_eq:
            result = self._apply_histogram_eq(result)

        return result

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE is applied to the L channel in LAB color space
        to enhance contrast without affecting colors.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        l_enhanced = self._clahe.apply(l_channel)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_histogram_eq(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization.

        Applied to V channel in HSV color space to
        enhance brightness without affecting colors.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Split channels
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # Apply histogram equalization to V channel
        v_enhanced = cv2.equalizeHist(v_channel)

        # Merge channels back
        hsv_enhanced = cv2.merge([h_channel, s_channel, v_enhanced])

        # Convert back to BGR
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
