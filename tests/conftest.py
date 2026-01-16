"""
Pytest fixtures for RideView tests.

Provides common test fixtures including:
- Synthetic test images
- Test configuration
- Sample stripe images
"""

import pytest
import numpy as np
import cv2


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "preprocessing": {
            "blur_kernel": [5, 5],
            "histogram_eq": False,
            "clahe": {
                "enabled": True,
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
            },
        },
        "colors": {
            "enabled_colors": ["red"],
            "custom_ranges": {
                "red_low": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
                "red_high": {"lower": [170, 100, 100], "upper": [179, 255, 255]},
            },
        },
        "morphology": {
            "close_kernel": [5, 5],
            "close_iterations": 2,
            "open_kernel": [3, 3],
            "open_iterations": 1,
        },
        "line_analysis": {
            "min_stripe_area": 100,
            "expected_width_range": [10, 100],
            "scan_direction": "horizontal",
            "gap_threshold": 3,
        },
        "thresholds": {
            "pass": {
                "min_coverage": 0.95,
                "max_gaps": 0,
                "max_gap_size": 0,
            },
            "warning": {
                "min_coverage": 0.80,
                "max_gaps": 2,
                "max_gap_size": 20,
            },
            "min_confidence": 0.5,
        },
        "visualization": {
            "pass_color": [0, 255, 0],
            "warning_color": [0, 165, 255],
            "fail_color": [0, 0, 255],
            "no_stripe_color": [128, 128, 128],
            "font_scale": 1.0,
            "font_thickness": 2,
            "show_mask_overlay": True,
            "mask_opacity": 0.3,
        },
    }


@pytest.fixture
def sample_intact_stripe():
    """Generate synthetic image of intact red torque stripe."""
    return generate_synthetic_stripe(intact=True, color=(0, 0, 255))


@pytest.fixture
def sample_broken_stripe():
    """Generate synthetic image of broken red torque stripe."""
    return generate_synthetic_stripe(intact=False, color=(0, 0, 255))


@pytest.fixture
def sample_warning_stripe():
    """Generate synthetic image of partially degraded stripe."""
    return generate_synthetic_stripe(intact=False, gap_size=15, color=(0, 0, 255))


@pytest.fixture
def sample_no_stripe():
    """Generate synthetic image with no stripe."""
    return np.zeros((200, 400, 3), dtype=np.uint8)


def generate_synthetic_stripe(
    intact: bool = True,
    color: tuple = (0, 0, 255),  # Red in BGR
    width: int = 400,
    height: int = 200,
    stripe_thickness: int = 15,
    gap_size: int = 50,
) -> np.ndarray:
    """
    Generate synthetic stripe image for testing.

    Args:
        intact: If True, generate continuous stripe. If False, add gap.
        color: BGR color tuple for stripe
        width: Image width
        height: Image height
        stripe_thickness: Thickness of stripe line
        gap_size: Size of gap if not intact

    Returns:
        BGR image as numpy array
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some background noise for realism
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    y_center = height // 2

    if intact:
        # Continuous line
        cv2.line(img, (50, y_center), (width - 50, y_center), color, stripe_thickness)
    else:
        # Broken line with gap
        mid_x = width // 2
        gap_start = mid_x - gap_size // 2
        gap_end = mid_x + gap_size // 2

        cv2.line(img, (50, y_center), (gap_start, y_center), color, stripe_thickness)
        cv2.line(img, (gap_end, y_center), (width - 50, y_center), color, stripe_thickness)

    return img


@pytest.fixture
def sample_intact_mask():
    """Generate binary mask for intact stripe."""
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.line(mask, (50, 100), (350, 100), 255, 15)
    return mask


@pytest.fixture
def sample_broken_mask():
    """Generate binary mask for broken stripe."""
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.line(mask, (50, 100), (150, 100), 255, 15)
    cv2.line(mask, (200, 100), (350, 100), 255, 15)
    return mask
