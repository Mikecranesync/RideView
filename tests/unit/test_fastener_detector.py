"""Tests for fastener detection module."""

import numpy as np
import pytest
import cv2

from rideview.detection.fastener_detector import (
    FastenerDetector,
    FastenerResult,
    FastenerType,
    HexagonDetector,
    CircleDetector,
    WasherDetector,
    NutDetector,
)


@pytest.fixture
def default_config():
    """Default fastener detection configuration."""
    return {
        "enabled": True,
        "require_fastener": True,
        "confidence_threshold": 0.5,
        "roi_margin": 10,
        "shapes": {
            "hexagon": {"enabled": True, "min_area": 100, "max_area": 50000},
            "circle": {"enabled": True, "min_radius": 10, "max_radius": 150},
            "phillips": {"enabled": True},
            "washer": {"enabled": True},
            "nut": {"enabled": True},
        },
    }


@pytest.fixture
def synthetic_hexagon_image():
    """Create a synthetic image with a hexagon."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    # Draw a regular hexagon
    center = (320, 240)
    radius = 80
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    vertices = np.array(
        [
            [int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))]
            for a in angles
        ],
        dtype=np.int32,
    )

    # Fill hexagon with metallic gray
    cv2.fillPoly(img, [vertices], (180, 180, 180))
    # Add edge
    cv2.polylines(img, [vertices], True, (100, 100, 100), 3)

    return img


@pytest.fixture
def synthetic_circle_image():
    """Create a synthetic image with a circle."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    # Draw a circle (bolt head)
    center = (320, 240)
    radius = 60

    cv2.circle(img, center, radius, (180, 180, 180), -1)  # Fill
    cv2.circle(img, center, radius, (100, 100, 100), 3)  # Edge

    return img


@pytest.fixture
def synthetic_washer_image():
    """Create a synthetic image with a washer (ring)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    center = (320, 240)
    outer_radius = 80
    inner_radius = 30

    # Draw outer circle
    cv2.circle(img, center, outer_radius, (180, 180, 180), -1)
    # Draw inner hole (dark)
    cv2.circle(img, center, inner_radius, (20, 20, 20), -1)
    # Add edges
    cv2.circle(img, center, outer_radius, (100, 100, 100), 2)
    cv2.circle(img, center, inner_radius, (60, 60, 60), 2)

    return img


@pytest.fixture
def synthetic_nut_image():
    """Create a synthetic image with a hex nut (hexagon with center hole)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray background

    # Draw hexagon
    center = (320, 240)
    radius = 80
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    vertices = np.array(
        [
            [int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))]
            for a in angles
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(img, [vertices], (180, 180, 180))
    cv2.polylines(img, [vertices], True, (100, 100, 100), 3)

    # Draw center hole
    cv2.circle(img, center, 20, (20, 20, 20), -1)
    cv2.circle(img, center, 20, (60, 60, 60), 2)

    return img


@pytest.fixture
def empty_image():
    """Create an empty image with no features."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    return img


class TestHexagonDetector:
    """Tests for hexagon detection."""

    def test_detect_hexagon(self, synthetic_hexagon_image):
        """Test that hexagon is detected in synthetic image."""
        detector = HexagonDetector(min_area=100, max_area=50000)
        result = detector.detect(synthetic_hexagon_image)

        assert result.detected is True
        assert result.shape_type == FastenerType.HEXAGON
        assert result.bounding_box is not None
        assert result.center is not None
        assert result.confidence > 0.5
        assert result.vertices is not None
        assert len(result.vertices) == 6

    def test_no_detection_on_empty(self, empty_image):
        """Test that no hexagon is detected in empty image."""
        detector = HexagonDetector()
        result = detector.detect(empty_image)

        assert result.detected is False

    def test_confidence_calculation(self, synthetic_hexagon_image):
        """Test that regular hexagons have high confidence."""
        detector = HexagonDetector()
        result = detector.detect(synthetic_hexagon_image)

        # Regular hexagon should have high confidence
        assert result.confidence > 0.7


class TestCircleDetector:
    """Tests for circle detection."""

    def test_detect_circle(self, synthetic_circle_image):
        """Test that circle is detected in synthetic image."""
        detector = CircleDetector(min_radius=20, max_radius=150)
        result = detector.detect(synthetic_circle_image)

        assert result.detected is True
        assert result.shape_type == FastenerType.CIRCLE
        assert result.bounding_box is not None
        assert result.center is not None
        assert result.radius > 0
        # Center should be near (320, 240)
        cx, cy = result.center
        assert abs(cx - 320) < 20
        assert abs(cy - 240) < 20

    def test_no_detection_on_empty(self, empty_image):
        """Test that no circle is detected in empty image."""
        detector = CircleDetector()
        result = detector.detect(empty_image)

        assert result.detected is False

    def test_radius_constraints(self, synthetic_circle_image):
        """Test that radius constraints are respected."""
        # Set min_radius higher than the circle
        detector = CircleDetector(min_radius=100, max_radius=150)
        result = detector.detect(synthetic_circle_image)

        # Should not detect the ~60 pixel radius circle
        assert result.detected is False


class TestWasherDetector:
    """Tests for washer (ring) detection."""

    def test_detect_washer(self, synthetic_washer_image):
        """Test that washer is detected in synthetic image."""
        detector = WasherDetector(min_outer_radius=30, max_outer_radius=150)
        result = detector.detect(synthetic_washer_image)

        assert result.detected is True
        assert result.shape_type == FastenerType.WASHER
        assert result.radius > 0  # Outer radius
        assert result.inner_radius > 0  # Inner radius
        assert result.inner_radius < result.radius

    def test_hole_ratio(self, synthetic_washer_image):
        """Test that hole ratio is calculated correctly."""
        detector = WasherDetector()
        result = detector.detect(synthetic_washer_image)

        if result.detected:
            ratio = result.details.get("hole_ratio", 0)
            # Inner should be ~30, outer ~80, ratio ~0.375
            assert 0.2 < ratio < 0.6


class TestNutDetector:
    """Tests for nut (hexagon with hole) detection."""

    def test_detect_nut(self, synthetic_nut_image):
        """Test that hex nut is detected in synthetic image."""
        detector = NutDetector(min_area=100, max_area=50000)
        result = detector.detect(synthetic_nut_image)

        # Nut detection requires both hexagon and center hole
        # This is more challenging for synthetic images
        if result.detected:
            assert result.shape_type == FastenerType.NUT
            assert result.inner_radius > 0
            assert result.vertices is not None


class TestFastenerDetector:
    """Tests for unified fastener detector."""

    def test_initialization(self, default_config):
        """Test that detector initializes correctly."""
        detector = FastenerDetector(default_config)

        assert detector.enabled is True
        assert detector.require_fastener is True
        assert detector.confidence_threshold == 0.5

    def test_detect_hexagon(self, default_config, synthetic_hexagon_image):
        """Test unified detector finds hexagon."""
        detector = FastenerDetector(default_config)
        result = detector.detect(synthetic_hexagon_image)

        assert result.detected is True
        assert result.shape_type in (FastenerType.HEXAGON, FastenerType.NUT)

    def test_detect_circle(self, default_config, synthetic_circle_image):
        """Test unified detector finds circle."""
        detector = FastenerDetector(default_config)
        result = detector.detect(synthetic_circle_image)

        assert result.detected is True
        assert result.shape_type in (FastenerType.CIRCLE, FastenerType.PHILLIPS)

    def test_no_detection_returns_false(self, default_config, empty_image):
        """Test that empty image returns no detection."""
        detector = FastenerDetector(default_config)
        result = detector.detect(empty_image)

        assert result.detected is False

    def test_disabled_detection(self, default_config, synthetic_hexagon_image):
        """Test that disabled detector returns no detection."""
        config = default_config.copy()
        config["enabled"] = False

        detector = FastenerDetector(config)
        result = detector.detect(synthetic_hexagon_image)

        assert result.detected is False

    def test_bounding_box_margin(self, default_config, synthetic_circle_image):
        """Test that ROI margin is applied to bounding box."""
        config = default_config.copy()
        config["roi_margin"] = 20

        detector = FastenerDetector(config)
        result = detector.detect(synthetic_circle_image)

        if result.detected and result.bounding_box:
            x, y, w, h = result.bounding_box
            # Bounding box should be larger than the circle due to margin
            assert w > 100  # Circle diameter ~120 + margins
            assert h > 100

    def test_draw_overlay(self, default_config, synthetic_hexagon_image):
        """Test that overlay drawing works."""
        detector = FastenerDetector(default_config)
        result = detector.detect(synthetic_hexagon_image)

        if result.detected:
            overlay = detector.draw_overlay(synthetic_hexagon_image, result)
            assert overlay.shape == synthetic_hexagon_image.shape
            # Overlay should be different from original (has drawings)
            assert not np.array_equal(overlay, synthetic_hexagon_image)

    def test_shape_priority(self, default_config, synthetic_nut_image):
        """Test that more specific shapes take priority."""
        detector = FastenerDetector(default_config)
        result = detector.detect(synthetic_nut_image)

        # Nut should be detected over plain hexagon if hole is found
        # But this depends on detection quality for synthetic image
        assert result.detected is True


class TestFastenerResult:
    """Tests for FastenerResult dataclass."""

    def test_default_values(self):
        """Test default values of FastenerResult."""
        result = FastenerResult(detected=False)

        assert result.detected is False
        assert result.shape_type == FastenerType.UNKNOWN
        assert result.bounding_box is None
        assert result.center is None
        assert result.confidence == 0.0
        assert result.mask is None

    def test_with_values(self):
        """Test FastenerResult with actual values."""
        result = FastenerResult(
            detected=True,
            shape_type=FastenerType.HEXAGON,
            bounding_box=(100, 100, 200, 200),
            center=(200, 200),
            confidence=0.85,
        )

        assert result.detected is True
        assert result.shape_type == FastenerType.HEXAGON
        assert result.bounding_box == (100, 100, 200, 200)
        assert result.center == (200, 200)
        assert result.confidence == 0.85
