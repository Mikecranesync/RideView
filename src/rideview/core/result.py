"""
Detection result data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DetectionResult(Enum):
    """Classification result for torque stripe detection."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NO_STRIPE = "NO_STRIPE"

    def __str__(self) -> str:
        return self.value


@dataclass
class StripeAnalysis:
    """
    Complete analysis result from torque stripe detection.

    Attributes:
        result: Classification result (PASS, WARNING, FAIL, NO_STRIPE)
        confidence: Detection confidence from 0.0 to 1.0
        coverage_percent: Percentage of expected stripe line covered (0.0 to 1.0)
        gap_count: Number of discontinuities detected in stripe
        max_gap_size: Size of largest gap in pixels
        stripe_mask: Binary mask where white (255) indicates detected stripe pixels
        annotated_frame: Original frame with detection overlay
        processing_time_ms: Time taken for detection in milliseconds
        reason: Human-readable explanation of classification
    """

    result: DetectionResult
    confidence: float
    coverage_percent: float
    gap_count: int
    max_gap_size: int
    stripe_mask: np.ndarray | None
    annotated_frame: np.ndarray | None
    processing_time_ms: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary (excludes numpy arrays)."""
        return {
            "result": self.result.value,
            "confidence": round(self.confidence, 3),
            "coverage_percent": round(self.coverage_percent * 100, 1),
            "gap_count": self.gap_count,
            "max_gap_size": self.max_gap_size,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "reason": self.reason,
        }

    @property
    def is_pass(self) -> bool:
        """Check if result is PASS."""
        return self.result == DetectionResult.PASS

    @property
    def is_fail(self) -> bool:
        """Check if result is FAIL."""
        return self.result == DetectionResult.FAIL

    @property
    def is_warning(self) -> bool:
        """Check if result is WARNING."""
        return self.result == DetectionResult.WARNING

    @property
    def has_stripe(self) -> bool:
        """Check if a stripe was detected."""
        return self.result != DetectionResult.NO_STRIPE
