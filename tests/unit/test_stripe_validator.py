"""
Unit tests for StripeValidator.
"""

import pytest

from rideview.detection.stripe_validator import StripeValidator
from rideview.detection.line_analyzer import LineMetrics
from rideview.core.result import DetectionResult


class TestStripeValidator:
    """Tests for stripe classification logic."""

    @pytest.fixture
    def validator(self, test_config):
        return StripeValidator(test_config["thresholds"])

    def test_pass_classification(self, validator):
        """High coverage, no gaps should PASS."""
        metrics = LineMetrics(
            coverage=0.98,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.95,
        )
        assert validator.classify(metrics) == DetectionResult.PASS

    def test_fail_classification(self, validator):
        """Low coverage, multiple gaps should FAIL."""
        metrics = LineMetrics(
            coverage=0.50,
            gap_count=5,
            max_gap=50,
            total_gap_length=150,
            contour_count=4,
            bounding_box=(10, 10, 100, 20),
            confidence=0.7,
        )
        assert validator.classify(metrics) == DetectionResult.FAIL

    def test_warning_classification(self, validator):
        """Moderate degradation should trigger WARNING."""
        metrics = LineMetrics(
            coverage=0.85,
            gap_count=1,
            max_gap=15,
            total_gap_length=15,
            contour_count=2,
            bounding_box=(10, 10, 100, 20),
            confidence=0.8,
        )
        assert validator.classify(metrics) == DetectionResult.WARNING

    def test_no_stripe_no_bounding_box(self, validator):
        """No bounding box should return NO_STRIPE."""
        metrics = LineMetrics(
            coverage=0.0,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=0,
            bounding_box=None,
            confidence=0.0,
        )
        assert validator.classify(metrics) == DetectionResult.NO_STRIPE

    def test_no_stripe_low_confidence(self, validator):
        """Low confidence should return NO_STRIPE."""
        metrics = LineMetrics(
            coverage=0.90,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.3,  # Below threshold
        )
        assert validator.classify(metrics) == DetectionResult.NO_STRIPE

    def test_no_stripe_zero_coverage(self, validator):
        """Zero coverage should return NO_STRIPE."""
        metrics = LineMetrics(
            coverage=0.0,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.8,
        )
        assert validator.classify(metrics) == DetectionResult.NO_STRIPE

    @pytest.mark.parametrize(
        "coverage,expected",
        [
            (0.99, DetectionResult.PASS),
            (0.95, DetectionResult.PASS),
            (0.94, DetectionResult.WARNING),
            (0.80, DetectionResult.WARNING),
            (0.79, DetectionResult.FAIL),
            (0.50, DetectionResult.FAIL),
        ],
    )
    def test_coverage_thresholds(self, validator, coverage, expected):
        """Test coverage threshold boundaries."""
        metrics = LineMetrics(
            coverage=coverage,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.9,
        )
        assert validator.classify(metrics) == expected

    def test_classification_reason_pass(self, validator):
        """Test reason generation for PASS."""
        metrics = LineMetrics(
            coverage=0.98,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.95,
        )
        reason = validator.get_classification_reason(metrics, DetectionResult.PASS)
        assert "Coverage" in reason
        assert "98.0%" in reason

    def test_classification_reason_fail(self, validator):
        """Test reason generation for FAIL."""
        metrics = LineMetrics(
            coverage=0.50,
            gap_count=3,
            max_gap=30,
            total_gap_length=60,
            contour_count=4,
            bounding_box=(10, 10, 100, 20),
            confidence=0.7,
        )
        reason = validator.get_classification_reason(metrics, DetectionResult.FAIL)
        assert "gap" in reason.lower()
        assert "fragmented" in reason.lower()

    def test_default_thresholds(self):
        """Test validator with default thresholds."""
        validator = StripeValidator()

        # Should use default thresholds
        metrics = LineMetrics(
            coverage=0.96,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.9,
        )
        assert validator.classify(metrics) == DetectionResult.PASS
