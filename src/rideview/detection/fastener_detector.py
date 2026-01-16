"""
Fastener Shape Detection Module for RideView.

Detects common fastener shapes (hex bolts, nuts, washers, screws) to create
bounding regions for stripe detection, reducing false positives from
environmental features.

Supported shapes:
- Hexagonal bolt heads and nuts
- Circular bolt heads
- Phillips head screws
- Washers (rings)
- Nuts with center holes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np


class FastenerType(Enum):
    """Types of fasteners that can be detected."""

    HEXAGON = "hexagon"
    CIRCLE = "circle"
    PHILLIPS = "phillips"
    WASHER = "washer"
    NUT = "nut"
    UNKNOWN = "unknown"


@dataclass
class FastenerResult:
    """Result from fastener shape detection."""

    detected: bool
    shape_type: FastenerType = FastenerType.UNKNOWN
    bounding_box: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    center: tuple[int, int] | None = None
    confidence: float = 0.0
    mask: np.ndarray | None = None
    vertices: np.ndarray | None = None  # For polygons
    radius: int = 0  # For circles
    inner_radius: int = 0  # For washers/nuts
    details: dict[str, Any] = field(default_factory=dict)


class HexagonDetector:
    """Detects hexagonal shapes using contour approximation."""

    def __init__(self, min_area: int = 500, max_area: int = 100000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(
        self, frame: np.ndarray, preprocessed: np.ndarray | None = None
    ) -> FastenerResult:
        """
        Detect hexagonal bolt head in frame.

        Args:
            frame: BGR image
            preprocessed: Optional preprocessed grayscale image

        Returns:
            FastenerResult with detection status and metrics
        """
        if preprocessed is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            preprocessed = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(preprocessed, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return FastenerResult(detected=False)

        best_hex = None
        best_confidence = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check for hexagon (6 vertices)
            if len(approx) != 6:
                continue

            # Check if convex
            if not cv2.isContourConvex(approx):
                continue

            confidence = self._calculate_confidence(approx)

            if confidence > best_confidence:
                best_confidence = confidence
                best_hex = (contour, approx, area)

        if best_hex is None:
            return FastenerResult(detected=False)

        contour, approx, area = best_hex

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return FastenerResult(detected=False)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        return FastenerResult(
            detected=True,
            shape_type=FastenerType.HEXAGON,
            bounding_box=(x, y, w, h),
            center=(cx, cy),
            confidence=best_confidence,
            mask=mask,
            vertices=approx.reshape(-1, 2),
            details={"area": area},
        )

    def _calculate_confidence(self, vertices: np.ndarray) -> float:
        """Calculate confidence based on hexagon regularity."""
        if len(vertices) != 6:
            return 0.0

        vertices = vertices.reshape(-1, 2).astype(float)

        # Calculate edge lengths
        edges = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge_len = np.linalg.norm(p2 - p1)
            edges.append(edge_len)

        edges = np.array(edges)
        edge_mean = edges.mean()

        if edge_mean == 0:
            return 0.0

        # Edge length consistency (lower variance = more regular)
        edge_variance = edges.std() / edge_mean
        edge_score = max(0.0, 1.0 - edge_variance)

        return float(edge_score)


class CircleDetector:
    """Detects circular shapes using Hough Circle Transform."""

    def __init__(self, min_radius: int = 20, max_radius: int = 200):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(
        self, frame: np.ndarray, preprocessed: np.ndarray | None = None
    ) -> FastenerResult:
        """
        Detect circular bolt head in frame.

        Args:
            frame: BGR image
            preprocessed: Optional preprocessed grayscale image

        Returns:
            FastenerResult with detection status
        """
        if preprocessed is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preprocessed = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Hough Circle Transform
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None or len(circles[0]) == 0:
            return FastenerResult(detected=False)

        # Select best circle (first one, usually highest accumulator)
        circles = np.uint16(np.around(circles[0]))
        best_circle = circles[0]
        # Convert to int to avoid uint16 overflow in arithmetic
        cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])

        # Calculate confidence based on edge strength
        confidence = self._calculate_confidence(preprocessed, (cx, cy, r))

        # Create bounding box
        x = max(0, cx - r)
        y = max(0, cy - r)
        w = min(frame.shape[1] - x, 2 * r)
        h = min(frame.shape[0] - y, 2 * r)

        # Skip if bounding box is invalid
        if w <= 0 or h <= 0:
            return FastenerResult(detected=False)

        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        return FastenerResult(
            detected=True,
            shape_type=FastenerType.CIRCLE,
            bounding_box=(x, y, w, h),
            center=(cx, cy),
            confidence=confidence,
            mask=mask,
            radius=r,
            details={"area": np.pi * r * r},
        )

    def _calculate_confidence(
        self, image: np.ndarray, circle: tuple[int, int, int]
    ) -> float:
        """Calculate confidence based on edge consistency."""
        cx, cy, r = circle

        # Sample edge points
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        edge_values = []

        for angle in angles:
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))

            px = np.clip(px, 0, image.shape[1] - 1)
            py = np.clip(py, 0, image.shape[0] - 1)

            edge_values.append(image[py, px])

        edge_values = np.array(edge_values)
        edge_mean = edge_values.mean()
        edge_std = edge_values.std()

        if edge_mean > 0:
            consistency = 1.0 - (edge_std / edge_mean)
            return float(max(0.0, min(1.0, consistency)))

        return 0.0


class PhillipsDetector:
    """Detects Phillips head screws (circle with cross pattern)."""

    def __init__(self, min_radius: int = 15, max_radius: int = 150):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circle_detector = CircleDetector(min_radius, max_radius)

    def detect(
        self, frame: np.ndarray, preprocessed: np.ndarray | None = None
    ) -> FastenerResult:
        """Detect Phillips head screw."""
        # First detect circular outline
        circle_result = self.circle_detector.detect(frame, preprocessed)

        if not circle_result.detected:
            return FastenerResult(detected=False)

        cx, cy = circle_result.center
        r = circle_result.radius

        # Detect cross pattern within circle
        has_cross, cross_confidence = self._detect_cross_pattern(frame, (cx, cy), r)

        if not has_cross:
            return FastenerResult(detected=False)

        # Combine confidences
        confidence = circle_result.confidence * 0.5 + cross_confidence * 0.5

        return FastenerResult(
            detected=True,
            shape_type=FastenerType.PHILLIPS,
            bounding_box=circle_result.bounding_box,
            center=circle_result.center,
            confidence=confidence,
            mask=circle_result.mask,
            radius=r,
            details={"cross_confidence": cross_confidence},
        )

    def _detect_cross_pattern(
        self, frame: np.ndarray, center: tuple[int, int], radius: int
    ) -> tuple[bool, float]:
        """Detect cross pattern in Phillips head."""
        cx, cy = center

        # Extract region around center
        margin = int(radius * 0.8)
        x_min = max(0, cx - margin)
        x_max = min(frame.shape[1], cx + margin)
        y_min = max(0, cy - margin)
        y_max = min(frame.shape[0], cy + margin)

        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=radius // 3,
            maxLineGap=10,
        )

        if lines is None or len(lines) < 2:
            return False, 0.0

        # Find perpendicular lines (cross pattern)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        # Look for ~90 degree difference
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(angles[i] - angles[j])
                diff = min(diff, 180 - diff)

                if 75 < diff < 105:
                    confidence = 1.0 - abs(diff - 90) / 15
                    return True, float(confidence)

        return False, 0.0


class WasherDetector:
    """Detects washers (ring shapes with inner and outer circles)."""

    def __init__(self, min_outer_radius: int = 30, max_outer_radius: int = 250):
        self.min_outer_radius = min_outer_radius
        self.max_outer_radius = max_outer_radius

    def detect(
        self, frame: np.ndarray, preprocessed: np.ndarray | None = None
    ) -> FastenerResult:
        """Detect washer (ring shape)."""
        if preprocessed is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preprocessed = cv2.GaussianBlur(gray, (7, 7), 1.5)

        # Detect outer circle
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=60,
            param1=30,
            param2=20,
            minRadius=self.min_outer_radius,
            maxRadius=self.max_outer_radius,
        )

        if circles is None or len(circles[0]) == 0:
            return FastenerResult(detected=False)

        circles = np.uint16(np.around(circles[0]))

        # Check each circle for inner hole (washer characteristic)
        for circle in circles:
            # Convert to int to avoid uint16 overflow in arithmetic
            cx, cy, outer_r = int(circle[0]), int(circle[1]), int(circle[2])

            # Look for inner circle
            inner_result = self._detect_inner_circle(preprocessed, (cx, cy), outer_r)

            if inner_result is not None:
                inner_r = inner_result

                # Create bounding box (ensure valid coordinates)
                x = max(0, cx - outer_r)
                y = max(0, cy - outer_r)
                w = min(frame.shape[1] - x, 2 * outer_r)
                h = min(frame.shape[0] - y, 2 * outer_r)

                # Skip if bounding box is invalid
                if w <= 0 or h <= 0:
                    continue

                # Create ring mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (cx, cy), outer_r, 255, -1)
                cv2.circle(mask, (cx, cy), inner_r, 0, -1)

                ratio = inner_r / max(outer_r, 1)
                confidence = 0.8 if 0.25 < ratio < 0.75 else 0.5

                return FastenerResult(
                    detected=True,
                    shape_type=FastenerType.WASHER,
                    bounding_box=(x, y, w, h),
                    center=(cx, cy),
                    confidence=confidence,
                    mask=mask,
                    radius=outer_r,
                    inner_radius=inner_r,
                    details={"hole_ratio": ratio},
                )

        return FastenerResult(detected=False)

    def _detect_inner_circle(
        self, image: np.ndarray, center: tuple[int, int], outer_r: int
    ) -> int | None:
        """Detect inner circle of washer."""
        cx, cy = center

        # Expected inner radius range
        inner_r_min = int(outer_r * 0.2)
        inner_r_max = int(outer_r * 0.7)

        # Extract center region
        margin = int(outer_r * 0.8)
        x_min = max(0, cx - margin)
        x_max = min(image.shape[1], cx + margin)
        y_min = max(0, cy - margin)
        y_max = min(image.shape[0], cy + margin)

        roi = image[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None

        # Look for inner circle
        inner_circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=20,
            param2=15,
            minRadius=inner_r_min,
            maxRadius=inner_r_max,
        )

        if inner_circles is not None and len(inner_circles[0]) > 0:
            return int(inner_circles[0][0][2])

        return None


class NutDetector:
    """Detects hexagonal nuts (hexagon with center hole)."""

    def __init__(self, min_area: int = 500, max_area: int = 100000):
        self.hex_detector = HexagonDetector(min_area, max_area)

    def detect(
        self, frame: np.ndarray, preprocessed: np.ndarray | None = None
    ) -> FastenerResult:
        """Detect hexagonal nut with center hole."""
        # First detect hexagon
        hex_result = self.hex_detector.detect(frame, preprocessed)

        if not hex_result.detected:
            return FastenerResult(detected=False)

        # Look for center hole
        cx, cy = hex_result.center

        if preprocessed is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed

        hole_result = self._detect_center_hole(gray, frame, (cx, cy))

        if hole_result is None:
            return FastenerResult(detected=False)

        inner_r = hole_result

        # Update mask to include hole
        mask = hex_result.mask.copy()
        cv2.circle(mask, (cx, cy), inner_r, 0, -1)

        confidence = hex_result.confidence * 0.8

        return FastenerResult(
            detected=True,
            shape_type=FastenerType.NUT,
            bounding_box=hex_result.bounding_box,
            center=hex_result.center,
            confidence=confidence,
            mask=mask,
            vertices=hex_result.vertices,
            inner_radius=inner_r,
            details={"hex_confidence": hex_result.confidence},
        )

    def _detect_center_hole(
        self, gray: np.ndarray, frame: np.ndarray, center: tuple[int, int]
    ) -> int | None:
        """Detect center hole in nut."""
        cx, cy = center

        # Search region
        search_radius = 50
        x_min = max(0, cx - search_radius)
        x_max = min(gray.shape[1], cx + search_radius)
        y_min = max(0, cy - search_radius)
        y_max = min(gray.shape[0], cy + search_radius)

        roi = gray[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None

        # Look for dark circular region (hole)
        blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=30,
            param2=15,
            minRadius=3,
            maxRadius=search_radius // 2,
        )

        if circles is None or len(circles[0]) == 0:
            return None

        # Verify it's a dark region (hole)
        circle = circles[0][0]
        hole_x, hole_y, hole_r = int(circle[0]), int(circle[1]), int(circle[2])

        # Check darkness at center
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(mask, (hole_x, hole_y), hole_r, 255, -1)

        mean_intensity = cv2.mean(roi, mask=mask)[0]

        # Hole should be relatively dark
        if mean_intensity < 120:
            return hole_r

        return None


class FastenerDetector:
    """
    Unified fastener detector that tries multiple detection methods.

    Integrates into RideView pipeline to provide bounding boxes for
    stripe detection, reducing false positives from environmental features.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize fastener detector with configuration.

        Args:
            config: Configuration dict with keys:
                - enabled: bool
                - require_fastener: bool
                - confidence_threshold: float
                - shapes: dict with shape-specific settings
                - roi_margin: int
        """
        self.enabled = config.get("enabled", True)
        self.require_fastener = config.get("require_fastener", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.roi_margin = config.get("roi_margin", 10)

        shapes_config = config.get("shapes", {})

        # Initialize individual detectors
        hex_config = shapes_config.get("hexagon", {})
        self.hex_detector = (
            HexagonDetector(
                min_area=hex_config.get("min_area", 500),
                max_area=hex_config.get("max_area", 100000),
            )
            if hex_config.get("enabled", True)
            else None
        )

        circle_config = shapes_config.get("circle", {})
        self.circle_detector = (
            CircleDetector(
                min_radius=circle_config.get("min_radius", 20),
                max_radius=circle_config.get("max_radius", 200),
            )
            if circle_config.get("enabled", True)
            else None
        )

        phillips_config = shapes_config.get("phillips", {})
        self.phillips_detector = (
            PhillipsDetector() if phillips_config.get("enabled", True) else None
        )

        washer_config = shapes_config.get("washer", {})
        self.washer_detector = (
            WasherDetector() if washer_config.get("enabled", True) else None
        )

        nut_config = shapes_config.get("nut", {})
        self.nut_detector = NutDetector() if nut_config.get("enabled", True) else None

    def detect(self, frame: np.ndarray) -> FastenerResult:
        """
        Detect fastener in frame using all enabled detectors.

        Tries detectors in order of specificity:
        1. Nut (most specific - hexagon with hole)
        2. Washer (ring shape)
        3. Phillips (circle with cross)
        4. Hexagon
        5. Circle (catch-all)

        Args:
            frame: BGR image

        Returns:
            FastenerResult with best detection
        """
        if not self.enabled:
            return FastenerResult(detected=False)

        # Preprocess once for all detectors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

        results = []

        # Try each detector (most specific first)
        if self.nut_detector:
            result = self.nut_detector.detect(frame, enhanced)
            if result.detected and result.confidence >= self.confidence_threshold:
                results.append(result)

        if self.washer_detector:
            result = self.washer_detector.detect(frame, blurred)
            if result.detected and result.confidence >= self.confidence_threshold:
                results.append(result)

        if self.phillips_detector:
            result = self.phillips_detector.detect(frame, blurred)
            if result.detected and result.confidence >= self.confidence_threshold:
                results.append(result)

        if self.hex_detector:
            result = self.hex_detector.detect(frame, enhanced)
            if result.detected and result.confidence >= self.confidence_threshold:
                results.append(result)

        if self.circle_detector:
            result = self.circle_detector.detect(frame, blurred)
            if result.detected and result.confidence >= self.confidence_threshold * 0.8:
                results.append(result)

        if not results:
            return FastenerResult(detected=False)

        # Return highest confidence result
        best_result = max(results, key=lambda r: r.confidence)

        # Expand bounding box by margin
        if best_result.bounding_box and self.roi_margin > 0:
            x, y, w, h = best_result.bounding_box
            x = max(0, x - self.roi_margin)
            y = max(0, y - self.roi_margin)
            w = min(frame.shape[1] - x, w + 2 * self.roi_margin)
            h = min(frame.shape[0] - y, h + 2 * self.roi_margin)
            best_result.bounding_box = (x, y, w, h)

        return best_result

    def draw_overlay(
        self, frame: np.ndarray, result: FastenerResult, color: tuple[int, int, int] | None = None
    ) -> np.ndarray:
        """
        Draw fastener detection overlay on frame.

        Args:
            frame: BGR image
            result: FastenerResult from detect()
            color: Optional BGR color override

        Returns:
            Frame with overlay drawn
        """
        if not result.detected:
            return frame

        output = frame.copy()

        # Default colors by type
        type_colors = {
            FastenerType.HEXAGON: (0, 255, 0),  # Green
            FastenerType.CIRCLE: (255, 165, 0),  # Orange
            FastenerType.PHILLIPS: (255, 0, 255),  # Magenta
            FastenerType.WASHER: (0, 255, 255),  # Yellow
            FastenerType.NUT: (0, 128, 255),  # Orange-red
            FastenerType.UNKNOWN: (128, 128, 128),  # Gray
        }

        draw_color = color or type_colors.get(result.shape_type, (0, 255, 0))

        # Draw bounding box
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)

        # Draw shape-specific overlay
        if result.shape_type == FastenerType.HEXAGON and result.vertices is not None:
            cv2.polylines(output, [result.vertices.astype(int)], True, draw_color, 2)

        elif result.shape_type in (FastenerType.CIRCLE, FastenerType.PHILLIPS):
            if result.center and result.radius:
                cv2.circle(output, result.center, result.radius, draw_color, 2)

        elif result.shape_type == FastenerType.WASHER:
            if result.center:
                cv2.circle(output, result.center, result.radius, draw_color, 2)
                cv2.circle(output, result.center, result.inner_radius, draw_color, 2)

        elif result.shape_type == FastenerType.NUT:
            if result.vertices is not None:
                cv2.polylines(output, [result.vertices.astype(int)], True, draw_color, 2)
            if result.center and result.inner_radius:
                cv2.circle(output, result.center, result.inner_radius, draw_color, 2)

        # Draw center point
        if result.center:
            cv2.circle(output, result.center, 4, (0, 0, 255), -1)

        # Draw label
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            label = f"{result.shape_type.value.upper()} ({result.confidence:.2f})"
            cv2.putText(
                output,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_color,
                2,
            )

        return output
