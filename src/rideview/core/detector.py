"""
Main torque stripe detector orchestrator.

Coordinates the detection pipeline:
1. Preprocessing
2. Color segmentation
3. Morphological cleanup
4. Line continuity analysis
5. Classification
"""

import time
from typing import Any

import cv2
import numpy as np

from ..detection.color_segmenter import ColorSegmenter
from ..detection.line_analyzer import LineAnalyzer
from ..detection.preprocessor import Preprocessor
from ..detection.stripe_validator import StripeValidator
from .result import DetectionResult, StripeAnalysis


class TorqueStripeDetector:
    """
    Main detector class orchestrating the detection pipeline.

    Coordinates preprocessing, color segmentation, morphological operations,
    line analysis, and classification to detect broken torque stripes.

    Usage:
        detector = TorqueStripeDetector(config)
        result = detector.analyze(frame)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize detector with configuration.

        Args:
            config: Full configuration dictionary containing:
                - preprocessing: Preprocessor settings
                - colors: Color segmentation settings
                - morphology: Morphological operation settings
                - line_analysis: Line analyzer settings
                - thresholds: Classification thresholds
                - visualization: Overlay settings
        """
        self.config = config

        # Initialize pipeline components
        self.preprocessor = Preprocessor(config.get("preprocessing", {}))
        self.color_segmenter = ColorSegmenter(config.get("colors", {}))
        self.line_analyzer = LineAnalyzer(config.get("line_analysis", {}))
        self.validator = StripeValidator(config.get("thresholds", {}))

        # Morphology settings
        morph_config = config.get("morphology", {})
        self.close_kernel = np.ones(
            tuple(morph_config.get("close_kernel", [5, 5])), dtype=np.uint8
        )
        self.close_iterations = morph_config.get("close_iterations", 2)
        self.open_kernel = np.ones(
            tuple(morph_config.get("open_kernel", [3, 3])), dtype=np.uint8
        )
        self.open_iterations = morph_config.get("open_iterations", 1)

        # Visualization settings
        vis_config = config.get("visualization", {})
        self.pass_color = tuple(vis_config.get("pass_color", [0, 255, 0]))
        self.warning_color = tuple(vis_config.get("warning_color", [0, 165, 255]))
        self.fail_color = tuple(vis_config.get("fail_color", [0, 0, 255]))
        self.no_stripe_color = tuple(vis_config.get("no_stripe_color", [128, 128, 128]))
        self.font_scale = vis_config.get("font_scale", 1.0)
        self.font_thickness = vis_config.get("font_thickness", 2)
        self.show_mask_overlay = vis_config.get("show_mask_overlay", True)
        self.mask_opacity = vis_config.get("mask_opacity", 0.3)

    def analyze(
        self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None
    ) -> StripeAnalysis:
        """
        Analyze a single frame for torque stripe integrity.

        Args:
            frame: BGR image from camera
            roi: Optional (x, y, w, h) region of interest

        Returns:
            StripeAnalysis with detection results
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess
        processed = self.preprocessor.process(frame)

        # Step 2: Apply ROI if specified
        if roi:
            x, y, w, h = roi
            roi_frame = processed[y : y + h, x : x + w]
        else:
            roi_frame = processed
            x, y = 0, 0

        # Step 3: Color segmentation (HSV)
        color_mask = self.color_segmenter.segment(roi_frame)

        # Step 4: Morphological cleanup
        cleaned_mask = self._apply_morphology(color_mask)

        # Step 5: Line continuity analysis
        line_metrics = self.line_analyzer.analyze(cleaned_mask)

        # Step 6: Validate and classify
        result = self.validator.classify(line_metrics)
        reason = self.validator.get_classification_reason(line_metrics, result)

        # Step 7: Generate annotated frame
        # Create full-size mask for annotation
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if roi:
            full_mask[y : y + h, x : x + w] = cleaned_mask
        else:
            full_mask = cleaned_mask

        annotated = self._annotate_frame(frame, full_mask, result, reason, roi)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return StripeAnalysis(
            result=result,
            confidence=line_metrics.confidence,
            coverage_percent=line_metrics.coverage,
            gap_count=line_metrics.gap_count,
            max_gap_size=line_metrics.max_gap,
            stripe_mask=full_mask,
            annotated_frame=annotated,
            processing_time_ms=elapsed_ms,
            reason=reason,
        )

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean the mask.

        Closing fills small gaps (dilation then erosion)
        Opening removes noise (erosion then dilation)
        """
        # Closing fills small gaps
        closed = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=self.close_iterations
        )

        # Opening removes noise
        opened = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, self.open_kernel, iterations=self.open_iterations
        )

        return opened

    def _annotate_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        result: DetectionResult,
        reason: str,
        roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """
        Create annotated frame with detection overlay.

        Args:
            frame: Original BGR frame
            mask: Detection mask
            result: Classification result
            reason: Classification reason
            roi: Region of interest coordinates

        Returns:
            Annotated BGR frame
        """
        annotated = frame.copy()

        # Get color based on result
        color = self._get_result_color(result)

        # Draw mask overlay
        if self.show_mask_overlay and np.any(mask):
            # Create colored overlay
            overlay = np.zeros_like(annotated)
            overlay[mask > 0] = color

            # Blend with original
            cv2.addWeighted(overlay, self.mask_opacity, annotated, 1 - self.mask_opacity, 0, annotated)

            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

        # Draw ROI rectangle if specified
        if roi:
            x, y, w, h = roi
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Draw status text
        status_text = f"{result.value}"
        text_size = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 1.5, self.font_thickness
        )[0]

        # Background rectangle for text
        padding = 10
        cv2.rectangle(
            annotated,
            (10, 10),
            (10 + text_size[0] + padding * 2, 10 + text_size[1] + padding * 2),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            annotated,
            (10, 10),
            (10 + text_size[0] + padding * 2, 10 + text_size[1] + padding * 2),
            color,
            2,
        )

        # Status text
        cv2.putText(
            annotated,
            status_text,
            (10 + padding, 10 + text_size[1] + padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale * 1.5,
            color,
            self.font_thickness,
        )

        # Reason text (smaller, below status)
        if reason:
            # Truncate if too long
            max_len = 60
            display_reason = reason[:max_len] + "..." if len(reason) > max_len else reason
            cv2.putText(
                annotated,
                display_reason,
                (10, 10 + text_size[1] + padding * 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.6,
                (255, 255, 255),
                1,
            )

        return annotated

    def _get_result_color(self, result: DetectionResult) -> tuple[int, int, int]:
        """Get BGR color for detection result."""
        if result == DetectionResult.PASS:
            return self.pass_color  # type: ignore
        elif result == DetectionResult.WARNING:
            return self.warning_color  # type: ignore
        elif result == DetectionResult.FAIL:
            return self.fail_color  # type: ignore
        else:
            return self.no_stripe_color  # type: ignore

    def get_color_ranges(self) -> dict[str, dict[str, list[int]]]:
        """Get current color segmentation ranges."""
        return self.color_segmenter.get_color_ranges()

    def update_color_range(
        self, name: str, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> None:
        """Update a color segmentation range."""
        self.color_segmenter.update_color_range(name, lower, upper)
