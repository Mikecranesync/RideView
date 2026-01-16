"""
Line continuity analyzer for torque stripe detection.

Analyzes a binary mask to determine stripe continuity by detecting
gaps and discontinuities. The core insight is that an intact torque
stripe appears as a single continuous contour, while a broken stripe
appears as multiple disconnected contours with gaps between them.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class LineMetrics:
    """
    Metrics describing line/stripe continuity.

    Attributes:
        coverage: Percentage of expected line covered (0.0 to 1.0)
        gap_count: Number of distinct gaps
        max_gap: Largest gap in pixels
        total_gap_length: Sum of all gaps in pixels
        contour_count: Number of separate contours (1 = continuous)
        bounding_box: (x, y, width, height) of stripe region, or None
        confidence: Detection confidence (0.0 to 1.0)
    """

    coverage: float
    gap_count: int
    max_gap: int
    total_gap_length: int
    contour_count: int
    bounding_box: tuple[int, int, int, int] | None
    confidence: float


class LineAnalyzer:
    """
    Analyzes a binary mask to determine stripe continuity.

    The core insight: an intact torque stripe appears as a single continuous
    contour. A broken stripe appears as multiple disconnected contours with
    gaps between them.

    Uses projection profile analysis to detect gaps along the stripe direction.

    Usage:
        analyzer = LineAnalyzer(config['line_analysis'])
        metrics = analyzer.analyze(mask)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize line analyzer.

        Args:
            config: Configuration dictionary with keys:
                - min_stripe_area: Minimum pixels to consider valid stripe
                - expected_width_range: [min, max] expected stripe width
                - scan_direction: 'horizontal' or 'vertical'
                - gap_threshold: Minimum gap size to count (pixels)
        """
        self.min_area = config.get("min_stripe_area", 500)
        self.width_range = config.get("expected_width_range", [10, 100])
        self.scan_direction = config.get("scan_direction", "horizontal")
        self.gap_threshold = config.get("gap_threshold", 5)

    def analyze(self, mask: np.ndarray) -> LineMetrics:
        """
        Analyze a binary mask for line continuity.

        Args:
            mask: Binary mask (255 = stripe, 0 = background)

        Returns:
            LineMetrics with continuity analysis
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return LineMetrics(
                coverage=0.0,
                gap_count=0,
                max_gap=0,
                total_gap_length=0,
                contour_count=0,
                bounding_box=None,
                confidence=0.0,
            )

        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        if not valid_contours:
            return LineMetrics(
                coverage=0.0,
                gap_count=0,
                max_gap=0,
                total_gap_length=0,
                contour_count=len(contours),
                bounding_box=None,
                confidence=0.0,
            )

        # Get overall bounding box
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        bounding_box = (x, y, w, h)

        # Analyze gaps using projection profile
        gaps = self._analyze_gaps(mask, bounding_box)

        # Calculate coverage
        if self.scan_direction == "horizontal":
            expected_length = w
        else:
            expected_length = h

        actual_length = expected_length - gaps["total_gap"]
        coverage = max(0.0, min(1.0, actual_length / expected_length)) if expected_length > 0 else 0.0

        # Calculate total stripe area
        total_area = sum(cv2.contourArea(c) for c in valid_contours)

        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            coverage, gaps["count"], len(valid_contours), total_area
        )

        return LineMetrics(
            coverage=coverage,
            gap_count=gaps["count"],
            max_gap=gaps["max"],
            total_gap_length=gaps["total_gap"],
            contour_count=len(valid_contours),
            bounding_box=bounding_box,
            confidence=confidence,
        )

    def _analyze_gaps(self, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, int]:
        """
        Analyze gaps using projection profile.

        Projects the mask onto a 1D line and finds discontinuities.

        Args:
            mask: Binary mask
            bbox: Bounding box (x, y, w, h)

        Returns:
            Dictionary with 'count', 'max', and 'total_gap'
        """
        x, y, w, h = bbox
        roi = mask[y : y + h, x : x + w]

        # Project along the scan direction
        if self.scan_direction == "horizontal":
            # Sum columns to get horizontal profile
            profile = np.sum(roi, axis=0)
        else:
            # Sum rows to get vertical profile
            profile = np.sum(roi, axis=1)

        # Find gaps (where profile drops to near-zero)
        threshold = profile.max() * 0.1 if profile.max() > 0 else 0
        is_gap = profile < threshold

        # Count gap segments
        gap_count = 0
        gap_lengths: list[int] = []
        in_gap = False
        current_gap_length = 0

        for g in is_gap:
            if g:
                if not in_gap:
                    in_gap = True
                    current_gap_length = 1
                else:
                    current_gap_length += 1
            else:
                if in_gap:
                    if current_gap_length >= self.gap_threshold:
                        gap_count += 1
                        gap_lengths.append(current_gap_length)
                    in_gap = False
                    current_gap_length = 0

        # Handle gap at end
        if in_gap and current_gap_length >= self.gap_threshold:
            gap_count += 1
            gap_lengths.append(current_gap_length)

        return {
            "count": gap_count,
            "max": max(gap_lengths) if gap_lengths else 0,
            "total_gap": sum(gap_lengths),
        }

    def _calculate_confidence(
        self, coverage: float, gap_count: int, contour_count: int, area: float
    ) -> float:
        """
        Calculate confidence score for the detection.

        Args:
            coverage: Line coverage percentage
            gap_count: Number of gaps detected
            contour_count: Number of contours
            area: Total stripe area in pixels

        Returns:
            Confidence score from 0.0 to 1.0
        """
        # High coverage = high confidence
        coverage_score = coverage

        # Single contour = high confidence (continuous stripe)
        contour_score = 1.0 if contour_count == 1 else max(0.3, 1.0 - (contour_count - 1) * 0.2)

        # Few gaps = high confidence
        gap_score = max(0.3, 1.0 - gap_count * 0.15)

        # Sufficient area = confident detection
        area_score = min(1.0, area / (self.min_area * 3))

        # Weighted combination
        confidence = (
            coverage_score * 0.4 + contour_score * 0.3 + gap_score * 0.2 + area_score * 0.1
        )

        return round(confidence, 3)
