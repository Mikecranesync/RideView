"""
Unit tests for LineAnalyzer.
"""

import pytest
import numpy as np
import cv2

from rideview.detection.line_analyzer import LineAnalyzer, LineMetrics


class TestLineAnalyzer:
    """Tests for line continuity analysis."""

    @pytest.fixture
    def analyzer(self, test_config):
        return LineAnalyzer(test_config["line_analysis"])

    def test_intact_stripe_analysis(self, analyzer, sample_intact_mask):
        """Intact stripe should have high coverage, no gaps."""
        metrics = analyzer.analyze(sample_intact_mask)

        assert metrics.coverage > 0.9
        assert metrics.gap_count == 0
        assert metrics.contour_count == 1
        assert metrics.bounding_box is not None
        assert metrics.confidence > 0.8

    def test_broken_stripe_analysis(self, analyzer, sample_broken_mask):
        """Broken stripe should show gaps."""
        metrics = analyzer.analyze(sample_broken_mask)

        assert metrics.gap_count >= 1
        assert metrics.max_gap > 0
        assert metrics.contour_count >= 1  # May be multiple contours

    def test_empty_mask(self, analyzer):
        """Empty mask should return zero metrics."""
        empty_mask = np.zeros((200, 400), dtype=np.uint8)
        metrics = analyzer.analyze(empty_mask)

        assert metrics.coverage == 0.0
        assert metrics.gap_count == 0
        assert metrics.contour_count == 0
        assert metrics.bounding_box is None
        assert metrics.confidence == 0.0

    def test_small_noise_ignored(self, analyzer):
        """Small noise blobs should be filtered out."""
        mask = np.zeros((200, 400), dtype=np.uint8)
        # Add small noise blobs (below min_stripe_area)
        cv2.circle(mask, (50, 50), 3, 255, -1)
        cv2.circle(mask, (100, 100), 5, 255, -1)

        metrics = analyzer.analyze(mask)

        # Small blobs should be filtered
        assert metrics.contour_count == 0 or metrics.coverage == 0.0

    def test_vertical_scan_direction(self, test_config):
        """Test vertical scan direction."""
        config = test_config["line_analysis"].copy()
        config["scan_direction"] = "vertical"
        analyzer = LineAnalyzer(config)

        # Create vertical stripe
        mask = np.zeros((400, 200), dtype=np.uint8)
        cv2.line(mask, (100, 50), (100, 350), 255, 15)

        metrics = analyzer.analyze(mask)

        assert metrics.coverage > 0.9
        assert metrics.gap_count == 0

    def test_multiple_gaps(self, analyzer):
        """Test detection of multiple gaps."""
        mask = np.zeros((200, 400), dtype=np.uint8)
        # Create stripe with multiple gaps
        cv2.line(mask, (20, 100), (80, 100), 255, 15)
        cv2.line(mask, (120, 100), (180, 100), 255, 15)
        cv2.line(mask, (220, 100), (280, 100), 255, 15)
        cv2.line(mask, (320, 100), (380, 100), 255, 15)

        metrics = analyzer.analyze(mask)

        assert metrics.gap_count >= 2

    def test_confidence_calculation(self, analyzer, sample_intact_mask):
        """Test that confidence is properly calculated."""
        metrics = analyzer.analyze(sample_intact_mask)

        # Confidence should be between 0 and 1
        assert 0.0 <= metrics.confidence <= 1.0

        # Intact stripe should have high confidence
        assert metrics.confidence > 0.7


class TestLineMetrics:
    """Tests for LineMetrics dataclass."""

    def test_line_metrics_creation(self):
        """Test creating LineMetrics instance."""
        metrics = LineMetrics(
            coverage=0.95,
            gap_count=0,
            max_gap=0,
            total_gap_length=0,
            contour_count=1,
            bounding_box=(10, 10, 100, 20),
            confidence=0.9,
        )

        assert metrics.coverage == 0.95
        assert metrics.gap_count == 0
        assert metrics.bounding_box == (10, 10, 100, 20)
