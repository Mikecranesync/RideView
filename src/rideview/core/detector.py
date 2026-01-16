"""
Main torque stripe detector orchestrator.

Coordinates the detection pipeline:
1. Preprocessing
2. Fastener detection (v2.0) - creates ROI boundary
3. Color segmentation
4. Morphological cleanup
5. Line continuity analysis
6. Classification
"""

import time
from typing import Any

import cv2
import numpy as np

from ..detection.color_segmenter import ColorSegmenter
from ..detection.fastener_detector import FastenerDetector, FastenerResult
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
                - fastener_detection: Fastener detector settings (v2.0)
                - colors: Color segmentation settings
                - morphology: Morphological operation settings
                - line_analysis: Line analyzer settings
                - thresholds: Classification thresholds
                - visualization: Overlay settings
        """
        self.config = config

        # Initialize pipeline components
        self.preprocessor = Preprocessor(config.get("preprocessing", {}))
        self.fastener_detector = FastenerDetector(config.get("fastener_detection", {}))
        self.color_segmenter = ColorSegmenter(config.get("colors", {}))
        self.line_analyzer = LineAnalyzer(config.get("line_analysis", {}))
        self.validator = StripeValidator(config.get("thresholds", {}))

        # Fastener detection settings
        fastener_config = config.get("fastener_detection", {})
        self.fastener_enabled = fastener_config.get("enabled", True)
        self.require_fastener = fastener_config.get("require_fastener", True)

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
            roi: Optional (x, y, w, h) region of interest (overrides fastener detection)

        Returns:
            StripeAnalysis with detection results
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess
        processed = self.preprocessor.process(frame)

        # Step 2: Fastener detection (v2.0) - creates automatic ROI
        fastener_result: FastenerResult | None = None
        if self.fastener_enabled and roi is None:
            fastener_result = self.fastener_detector.detect(frame)

            if fastener_result.detected and fastener_result.bounding_box:
                # Use fastener bounding box as ROI
                roi = fastener_result.bounding_box
            elif self.require_fastener:
                # No fastener detected and fastener required - return NO_STRIPE
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                annotated = self._annotate_frame(
                    frame,
                    np.zeros(frame.shape[:2], dtype=np.uint8),
                    DetectionResult.NO_STRIPE,
                    "No fastener detected in frame",
                    roi=None,
                    fastener_result=fastener_result,
                )
                return StripeAnalysis(
                    result=DetectionResult.NO_STRIPE,
                    confidence=0.0,
                    coverage_percent=0.0,
                    gap_count=0,
                    max_gap_size=0,
                    stripe_mask=None,
                    annotated_frame=annotated,
                    processing_time_ms=elapsed_ms,
                    reason="No fastener detected in frame",
                )

        # Step 3: Apply ROI (from manual selection or fastener detection)
        if roi:
            x, y, w, h = roi
            # Validate ROI bounds
            x = max(0, min(x, processed.shape[1] - 1))
            y = max(0, min(y, processed.shape[0] - 1))
            w = max(1, min(w, processed.shape[1] - x))
            h = max(1, min(h, processed.shape[0] - y))
            roi_frame = processed[y : y + h, x : x + w]
        else:
            roi_frame = processed
            x, y = 0, 0

        # Validate ROI frame is not empty
        if roi_frame.size == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            annotated = self._annotate_frame(
                frame,
                np.zeros(frame.shape[:2], dtype=np.uint8),
                DetectionResult.NO_STRIPE,
                "Invalid ROI - empty region",
                roi=roi,
                fastener_result=fastener_result,
            )
            return StripeAnalysis(
                result=DetectionResult.NO_STRIPE,
                confidence=0.0,
                coverage_percent=0.0,
                gap_count=0,
                max_gap_size=0,
                stripe_mask=None,
                annotated_frame=annotated,
                processing_time_ms=elapsed_ms,
                reason="Invalid ROI - empty region",
            )

        # Step 4: Color segmentation (HSV)
        color_mask = self.color_segmenter.segment(roi_frame)

        # Step 5: Morphological cleanup
        cleaned_mask = self._apply_morphology(color_mask)

        # Step 6: Line continuity analysis
        line_metrics = self.line_analyzer.analyze(cleaned_mask)

        # Step 7: Validate and classify
        result = self.validator.classify(line_metrics)
        reason = self.validator.get_classification_reason(line_metrics, result)

        # Step 8: Generate annotated frame
        # Create full-size mask for annotation
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if roi:
            full_mask[y : y + h, x : x + w] = cleaned_mask
        else:
            full_mask = cleaned_mask

        annotated = self._annotate_frame(
            frame, full_mask, result, reason, roi, fastener_result
        )

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
        fastener_result: FastenerResult | None = None,
    ) -> np.ndarray:
        """
        Create annotated frame with detection overlay.

        Args:
            frame: Original BGR frame
            mask: Detection mask
            result: Classification result
            reason: Classification reason
            roi: Region of interest coordinates
            fastener_result: Optional fastener detection result for overlay

        Returns:
            Annotated BGR frame
        """
        annotated = frame.copy()

        # Draw fastener detection overlay first (if detected)
        if fastener_result and fastener_result.detected:
            annotated = self.fastener_detector.draw_overlay(annotated, fastener_result)

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

        # Draw ROI rectangle if specified (and not from fastener - already drawn)
        if roi and fastener_result is None:
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
