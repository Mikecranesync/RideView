"""
Status overlay widget for RideView.

Displays detection result, confidence, and metrics in a semi-transparent overlay.
"""

import logging

from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from ...core.result import DetectionResult, StripeAnalysis

logger = logging.getLogger(__name__)


# Result color mapping (R, G, B, A) - normalized 0-1
RESULT_COLORS = {
    DetectionResult.PASS: (0.0, 0.8, 0.0, 1.0),  # Green
    DetectionResult.WARNING: (1.0, 0.65, 0.0, 1.0),  # Orange
    DetectionResult.FAIL: (0.9, 0.0, 0.0, 1.0),  # Red
    DetectionResult.NO_STRIPE: (0.5, 0.5, 0.5, 1.0),  # Gray
}


class StatusOverlay(BoxLayout):
    """
    Overlay widget showing detection status and metrics.

    Layout:
    ┌─────────────────────────┐
    │  PASS / WARNING / FAIL  │
    │  Confidence: 92%        │
    │  Coverage: 95%          │
    │  Gaps: 0                │
    └─────────────────────────┘
    """

    def __init__(self, **kwargs):
        """Initialize the status overlay."""
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("size", (300, 150))
        kwargs.setdefault("padding", [15, 10, 15, 10])
        kwargs.setdefault("spacing", 5)
        super().__init__(**kwargs)

        # Current result for color
        self._current_result = DetectionResult.NO_STRIPE

        # Create labels
        self._result_label = Label(
            text="NO SIGNAL",
            font_size="28sp",
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=0.4,
        )
        self._result_label.bind(size=self._result_label.setter("text_size"))

        self._confidence_label = Label(
            text="Confidence: --",
            font_size="16sp",
            halign="left",
            valign="middle",
            size_hint_y=0.2,
        )
        self._confidence_label.bind(size=self._confidence_label.setter("text_size"))

        self._coverage_label = Label(
            text="Coverage: --",
            font_size="16sp",
            halign="left",
            valign="middle",
            size_hint_y=0.2,
        )
        self._coverage_label.bind(size=self._coverage_label.setter("text_size"))

        self._gaps_label = Label(
            text="Gaps: --",
            font_size="16sp",
            halign="left",
            valign="middle",
            size_hint_y=0.2,
        )
        self._gaps_label.bind(size=self._gaps_label.setter("text_size"))

        # Add labels
        self.add_widget(self._result_label)
        self.add_widget(self._confidence_label)
        self.add_widget(self._coverage_label)
        self.add_widget(self._gaps_label)

        # Draw background
        self._draw_background()
        self.bind(pos=self._update_background, size=self._update_background)

    def _draw_background(self):
        """Draw semi-transparent background with rounded corners."""
        with self.canvas.before:
            # Background color
            Color(0, 0, 0, 0.7)  # Semi-transparent black
            self._bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[10],
            )

            # Border color based on result
            color = RESULT_COLORS.get(self._current_result, (0.5, 0.5, 0.5, 1.0))
            Color(*color)
            self._border_rect = RoundedRectangle(
                pos=(self.pos[0] - 2, self.pos[1] - 2),
                size=(self.size[0] + 4, self.size[1] + 4),
                radius=[12],
            )

    def _update_background(self, *args):
        """Update background when position/size changes."""
        if hasattr(self, "_bg_rect"):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size
        if hasattr(self, "_border_rect"):
            self._border_rect.pos = (self.pos[0] - 2, self.pos[1] - 2)
            self._border_rect.size = (self.size[0] + 4, self.size[1] + 4)

    def update(self, analysis: StripeAnalysis) -> None:
        """
        Update the overlay with new detection results.

        Args:
            analysis: StripeAnalysis from detector.
        """
        self._current_result = analysis.result

        # Update result label
        self._result_label.text = analysis.result.value
        color = RESULT_COLORS.get(analysis.result, (1, 1, 1, 1))
        self._result_label.color = color

        # Update metrics
        self._confidence_label.text = f"Confidence: {analysis.confidence * 100:.0f}%"
        self._coverage_label.text = f"Coverage: {analysis.coverage_percent * 100:.0f}%"
        self._gaps_label.text = f"Gaps: {analysis.gap_count} (max: {analysis.max_gap_size}px)"

        # Redraw background with new color
        self.canvas.before.clear()
        self._draw_background()

    def set_no_signal(self):
        """Set overlay to show no camera signal state."""
        self._current_result = DetectionResult.NO_STRIPE
        self._result_label.text = "NO SIGNAL"
        self._result_label.color = (0.5, 0.5, 0.5, 1.0)
        self._confidence_label.text = "Confidence: --"
        self._coverage_label.text = "Coverage: --"
        self._gaps_label.text = "Gaps: --"

        self.canvas.before.clear()
        self._draw_background()
