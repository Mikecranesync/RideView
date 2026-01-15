"""
Visualization utilities for RideView.

Helper functions for drawing detection overlays, creating comparison views,
and generating debug visualizations.
"""

import cv2
import numpy as np

from ..core.result import DetectionResult, StripeAnalysis


def draw_detection_overlay(
    frame: np.ndarray,
    analysis: StripeAnalysis,
    show_metrics: bool = True,
    show_fps: float | None = None,
) -> np.ndarray:
    """
    Draw detection overlay on frame.

    Args:
        frame: BGR image
        analysis: StripeAnalysis result
        show_metrics: Whether to show detailed metrics
        show_fps: Optional FPS to display

    Returns:
        Annotated BGR frame
    """
    # Use the annotated frame from analysis if available
    if analysis.annotated_frame is not None:
        annotated = analysis.annotated_frame.copy()
    else:
        annotated = frame.copy()

    height, width = annotated.shape[:2]

    # Draw metrics panel on right side
    if show_metrics:
        panel_width = 200
        panel_x = width - panel_width - 10

        # Background
        cv2.rectangle(
            annotated, (panel_x, 10), (width - 10, 160), (0, 0, 0), -1
        )
        cv2.rectangle(
            annotated, (panel_x, 10), (width - 10, 160), (100, 100, 100), 1
        )

        # Metrics text
        metrics = [
            f"Coverage: {analysis.coverage_percent * 100:.1f}%",
            f"Gaps: {analysis.gap_count}",
            f"Max Gap: {analysis.max_gap_size}px",
            f"Confidence: {analysis.confidence:.0%}",
            f"Time: {analysis.processing_time_ms:.1f}ms",
        ]

        y_offset = 35
        for metric in metrics:
            cv2.putText(
                annotated,
                metric,
                (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 25

    # Draw FPS counter
    if show_fps is not None:
        cv2.putText(
            annotated,
            f"{show_fps:.1f} FPS",
            (width - 100, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return annotated


def create_comparison_view(
    original: np.ndarray,
    mask: np.ndarray,
    annotated: np.ndarray,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Create side-by-side comparison view for debugging.

    Args:
        original: Original BGR frame
        mask: Binary detection mask
        annotated: Annotated frame with overlay
        scale: Scale factor for output

    Returns:
        Combined BGR image showing all three views
    """
    # Resize all frames
    h, w = original.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    orig_small = cv2.resize(original, (new_w, new_h))
    annotated_small = cv2.resize(annotated, (new_w, new_h))

    # Convert mask to BGR for display
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_small = cv2.resize(mask_bgr, (new_w, new_h))

    # Add labels
    cv2.putText(orig_small, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(mask_small, "Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_small, "Detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stack horizontally
    combined = np.hstack([orig_small, mask_small, annotated_small])

    return combined


def draw_color_calibration(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    selected_color: str | None = None,
) -> np.ndarray:
    """
    Draw color calibration view showing individual color masks.

    Args:
        frame: BGR image
        masks: Dictionary of color name to mask
        selected_color: Currently selected color to highlight

    Returns:
        Visualization frame
    """
    height, width = frame.shape[:2]
    result = frame.copy()

    # Calculate grid layout
    num_masks = len(masks)
    if num_masks == 0:
        return result

    cols = min(3, num_masks)
    rows = (num_masks + cols - 1) // cols

    thumb_w = width // (cols + 1)
    thumb_h = height // (rows + 1)

    # Draw each mask as thumbnail
    y_offset = 10
    col = 0

    for name, mask in masks.items():
        x_offset = width - (cols - col) * thumb_w - 10

        # Resize mask for thumbnail
        mask_thumb = cv2.resize(mask, (thumb_w - 10, thumb_h - 10))
        mask_bgr = cv2.cvtColor(mask_thumb, cv2.COLOR_GRAY2BGR)

        # Highlight selected color
        border_color = (0, 255, 0) if name == selected_color else (100, 100, 100)
        cv2.rectangle(
            result,
            (x_offset, y_offset),
            (x_offset + thumb_w - 10, y_offset + thumb_h - 10),
            border_color,
            2,
        )

        # Place thumbnail
        result[y_offset : y_offset + thumb_h - 10, x_offset : x_offset + thumb_w - 10] = mask_bgr

        # Label
        cv2.putText(
            result,
            name,
            (x_offset + 5, y_offset + thumb_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        col += 1
        if col >= cols:
            col = 0
            y_offset += thumb_h

    return result


def result_to_color(result: DetectionResult) -> tuple[int, int, int]:
    """
    Convert detection result to BGR color.

    Args:
        result: DetectionResult enum

    Returns:
        BGR color tuple
    """
    colors = {
        DetectionResult.PASS: (0, 255, 0),  # Green
        DetectionResult.WARNING: (0, 165, 255),  # Orange
        DetectionResult.FAIL: (0, 0, 255),  # Red
        DetectionResult.NO_STRIPE: (128, 128, 128),  # Gray
    }
    return colors.get(result, (128, 128, 128))
