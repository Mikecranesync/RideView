"""
Stripe validator for classifying detection results.

Takes line analysis metrics and applies configurable thresholds
to classify the stripe condition as PASS, WARNING, FAIL, or NO_STRIPE.
"""

from typing import Any

from ..core.result import DetectionResult
from .line_analyzer import LineMetrics


class StripeValidator:
    """
    Validates stripe integrity based on analysis metrics.

    Classification thresholds are configurable to accommodate different
    use cases and tolerance levels.

    Classification logic:
    - PASS: High coverage, no gaps
    - WARNING: Moderate coverage, minor gaps
    - FAIL: Low coverage or significant gaps
    - NO_STRIPE: No valid stripe detected

    Usage:
        validator = StripeValidator(config['thresholds'])
        result = validator.classify(metrics)
    """

    DEFAULT_THRESHOLDS = {
        "pass": {
            "min_coverage": 0.95,  # 95%+ coverage = PASS
            "max_gaps": 0,  # No gaps for PASS
            "max_gap_size": 0,  # No gap size allowed
        },
        "warning": {
            "min_coverage": 0.80,  # 80%+ coverage = WARNING
            "max_gaps": 2,  # Up to 2 small gaps
            "max_gap_size": 20,  # Gap up to 20 pixels
        },
        # Below WARNING thresholds = FAIL
        "min_confidence": 0.5,  # Minimum confidence to classify
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize validator with thresholds.

        Args:
            config: Threshold configuration dictionary, or None for defaults
        """
        self.thresholds = config or self.DEFAULT_THRESHOLDS

    def classify(self, metrics: LineMetrics) -> DetectionResult:
        """
        Classify stripe condition based on metrics.

        Args:
            metrics: LineMetrics from line analyzer

        Returns:
            DetectionResult enum value
        """
        # No stripe detected
        if metrics.bounding_box is None or metrics.coverage == 0:
            return DetectionResult.NO_STRIPE

        # Low confidence - unreliable detection
        if metrics.confidence < self.thresholds.get("min_confidence", 0.5):
            return DetectionResult.NO_STRIPE

        # Check PASS thresholds
        pass_thresh = self.thresholds.get("pass", {})
        if (
            metrics.coverage >= pass_thresh.get("min_coverage", 0.95)
            and metrics.gap_count <= pass_thresh.get("max_gaps", 0)
            and metrics.max_gap <= pass_thresh.get("max_gap_size", 0)
        ):
            return DetectionResult.PASS

        # Check WARNING thresholds
        warn_thresh = self.thresholds.get("warning", {})
        if (
            metrics.coverage >= warn_thresh.get("min_coverage", 0.80)
            and metrics.gap_count <= warn_thresh.get("max_gaps", 2)
            and metrics.max_gap <= warn_thresh.get("max_gap_size", 20)
        ):
            return DetectionResult.WARNING

        # Below WARNING = FAIL
        return DetectionResult.FAIL

    def get_classification_reason(self, metrics: LineMetrics, result: DetectionResult) -> str:
        """
        Generate human-readable reason for classification.

        Args:
            metrics: LineMetrics from analysis
            result: Classification result

        Returns:
            Human-readable explanation string
        """
        if result == DetectionResult.NO_STRIPE:
            if metrics.bounding_box is None:
                return "No valid stripe detected in frame"
            if metrics.confidence < self.thresholds.get("min_confidence", 0.5):
                return f"Low detection confidence ({metrics.confidence:.0%})"
            return "No valid stripe detected"

        reasons: list[str] = []

        if result == DetectionResult.PASS:
            reasons.append(f"Coverage: {metrics.coverage * 100:.1f}%")
            reasons.append("No discontinuities detected")
        elif result == DetectionResult.WARNING:
            reasons.append(f"Coverage: {metrics.coverage * 100:.1f}% (degraded)")
            if metrics.gap_count > 0:
                reasons.append(f"{metrics.gap_count} minor gap(s) detected")
        else:  # FAIL
            reasons.append(f"Coverage: {metrics.coverage * 100:.1f}% (below threshold)")
            if metrics.gap_count > 0:
                reasons.append(f"{metrics.gap_count} gap(s), max size: {metrics.max_gap}px")
            if metrics.contour_count > 1:
                reasons.append(f"Stripe fragmented into {metrics.contour_count} pieces")

        return "; ".join(reasons)

    # Runtime threshold update methods

    def set_pass_coverage(self, min_coverage: float) -> None:
        """Set minimum coverage for PASS classification (0.0-1.0)."""
        if "pass" not in self.thresholds:
            self.thresholds["pass"] = {}
        self.thresholds["pass"]["min_coverage"] = max(0.0, min(1.0, min_coverage))

    def set_warning_coverage(self, min_coverage: float) -> None:
        """Set minimum coverage for WARNING classification (0.0-1.0)."""
        if "warning" not in self.thresholds:
            self.thresholds["warning"] = {}
        self.thresholds["warning"]["min_coverage"] = max(0.0, min(1.0, min_coverage))

    def set_pass_max_gaps(self, max_gaps: int) -> None:
        """Set maximum gaps allowed for PASS classification."""
        if "pass" not in self.thresholds:
            self.thresholds["pass"] = {}
        self.thresholds["pass"]["max_gaps"] = max(0, max_gaps)

    def set_warning_max_gaps(self, max_gaps: int) -> None:
        """Set maximum gaps allowed for WARNING classification."""
        if "warning" not in self.thresholds:
            self.thresholds["warning"] = {}
        self.thresholds["warning"]["max_gaps"] = max(0, max_gaps)

    def set_min_confidence(self, min_confidence: float) -> None:
        """Set minimum confidence threshold (0.0-1.0)."""
        self.thresholds["min_confidence"] = max(0.0, min(1.0, min_confidence))

    def get_settings(self) -> dict:
        """Get current threshold settings."""
        pass_thresh = self.thresholds.get("pass", {})
        warn_thresh = self.thresholds.get("warning", {})
        return {
            "pass_coverage": pass_thresh.get("min_coverage", 0.95),
            "warning_coverage": warn_thresh.get("min_coverage", 0.80),
            "pass_max_gaps": pass_thresh.get("max_gaps", 0),
            "warning_max_gaps": warn_thresh.get("max_gaps", 2),
            "min_confidence": self.thresholds.get("min_confidence", 0.5),
        }
