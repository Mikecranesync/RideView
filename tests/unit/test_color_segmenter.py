"""
Unit tests for ColorSegmenter.
"""

import pytest
import numpy as np
import cv2

from rideview.detection.color_segmenter import ColorSegmenter, ColorRange


class TestColorRange:
    """Tests for ColorRange dataclass."""

    def test_color_range_creation(self):
        """Test creating ColorRange instance."""
        cr = ColorRange("red", (0, 100, 100), (10, 255, 255))

        assert cr.name == "red"
        np.testing.assert_array_equal(cr.lower, np.array([0, 100, 100]))
        np.testing.assert_array_equal(cr.upper, np.array([10, 255, 255]))


class TestColorSegmenter:
    """Tests for HSV color segmentation."""

    @pytest.fixture
    def segmenter(self, test_config):
        return ColorSegmenter(test_config["colors"])

    def test_red_segmentation(self, segmenter):
        """Test segmentation of red stripe."""
        # Create image with red stripe (in BGR)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[40:60, 50:150] = (0, 0, 255)  # Red stripe

        mask = segmenter.segment(img)

        # Should detect the red stripe
        assert np.sum(mask > 0) > 0

    def test_non_red_not_detected(self, segmenter):
        """Test that non-red colors are not detected."""
        # Create image with blue stripe (shouldn't be detected with red config)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[40:60, 50:150] = (255, 0, 0)  # Blue stripe

        mask = segmenter.segment(img)

        # Should not detect blue
        assert np.sum(mask > 0) == 0

    def test_empty_image(self, segmenter):
        """Test segmentation of empty image."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = segmenter.segment(img)

        assert np.sum(mask > 0) == 0

    def test_segment_with_debug(self, segmenter):
        """Test debug segmentation returns individual masks."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[40:60, 50:150] = (0, 0, 255)  # Red stripe

        masks = segmenter.segment_with_debug(img)

        assert "combined" in masks
        assert "red_low" in masks or "red_high" in masks

    def test_update_color_range(self, segmenter):
        """Test updating color range at runtime."""
        segmenter.update_color_range("custom", (100, 100, 100), (120, 255, 255))

        ranges = segmenter.get_color_ranges()
        assert "custom" in ranges

    def test_get_color_ranges(self, segmenter):
        """Test getting current color ranges."""
        ranges = segmenter.get_color_ranges()

        assert isinstance(ranges, dict)
        for name, values in ranges.items():
            assert "lower" in values
            assert "upper" in values
            assert len(values["lower"]) == 3
            assert len(values["upper"]) == 3


class TestMultiColorSegmenter:
    """Tests for multi-color detection."""

    @pytest.fixture
    def multi_segmenter(self):
        config = {
            "enabled_colors": ["red", "yellow", "orange"],
            "custom_ranges": {
                "red_low": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
                "red_high": {"lower": [170, 100, 100], "upper": [179, 255, 255]},
                "yellow": {"lower": [20, 100, 100], "upper": [35, 255, 255]},
                "orange": {"lower": [10, 100, 100], "upper": [20, 255, 255]},
            },
        }
        return ColorSegmenter(config)

    def test_multiple_colors_detected(self, multi_segmenter):
        """Test that multiple colors can be detected."""
        # Create image with red and yellow areas
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        img[40:60, 50:150] = (0, 0, 255)  # Red
        img[40:60, 250:350] = (0, 255, 255)  # Yellow

        mask = multi_segmenter.segment(img)

        # Should detect both colors
        # Check for pixels in both regions
        red_region = mask[40:60, 50:150]
        yellow_region = mask[40:60, 250:350]

        assert np.sum(red_region > 0) > 0
        assert np.sum(yellow_region > 0) > 0
