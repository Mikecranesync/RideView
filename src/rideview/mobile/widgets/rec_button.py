"""
Recording button widget for RideView.

Animated red REC button with pulsing effect when recording.
"""

import logging
from enum import Enum
from typing import Callable

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.uix.button import Button

logger = logging.getLogger(__name__)


class RecButtonState(Enum):
    """Recording button states."""

    IDLE = "idle"
    RECORDING = "recording"


class RecButton(Button):
    """
    Recording toggle button with animated states.

    States:
    - IDLE: Gray circle with "REC" text
    - RECORDING: Red pulsing circle with "STOP" text
    """

    def __init__(
        self,
        on_toggle: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        """
        Initialize the REC button.

        Args:
            on_toggle: Callback when recording state changes. Called with True when
                       recording starts, False when it stops.
        """
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("size", (80, 80))
        kwargs.setdefault("text", "REC")
        kwargs.setdefault("font_size", "14sp")
        kwargs.setdefault("bold", True)

        super().__init__(**kwargs)

        self._state = RecButtonState.IDLE
        self._on_toggle = on_toggle
        self._pulse_animation: Animation | None = None

        # Colors
        self._idle_color = (0.4, 0.4, 0.4, 1.0)  # Gray
        self._recording_color = (0.9, 0.1, 0.1, 1.0)  # Red

        # Set up appearance
        self.background_color = (0, 0, 0, 0)  # Transparent background
        self.color = (1, 1, 1, 1)  # White text

        # Draw custom circle background
        self._draw_background()
        self.bind(pos=self._update_background, size=self._update_background)

        # Bind press event
        self.bind(on_press=self._on_press)

    def _draw_background(self):
        """Draw circular button background."""
        self.canvas.before.clear()
        with self.canvas.before:
            if self._state == RecButtonState.RECORDING:
                Color(*self._recording_color)
            else:
                Color(*self._idle_color)

            # Draw circle
            center_x = self.center_x
            center_y = self.center_y
            radius = min(self.width, self.height) / 2 - 4
            self._circle = Ellipse(
                pos=(center_x - radius, center_y - radius),
                size=(radius * 2, radius * 2),
            )

    def _update_background(self, *args):
        """Update background when position/size changes."""
        self._draw_background()

    def _on_press(self, instance):
        """Handle button press."""
        if self._state == RecButtonState.IDLE:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Transition to recording state."""
        if self._state == RecButtonState.RECORDING:
            return

        self._state = RecButtonState.RECORDING
        self.text = "STOP"
        self._draw_background()

        # Start pulse animation
        self._start_pulse()

        # Call callback
        if self._on_toggle:
            self._on_toggle(True)

        logger.info("RecButton: Recording started")

    def stop_recording(self):
        """Transition to idle state."""
        if self._state == RecButtonState.IDLE:
            return

        self._state = RecButtonState.IDLE
        self.text = "REC"
        self.opacity = 1.0  # Reset opacity

        # Stop pulse animation
        self._stop_pulse()
        self._draw_background()

        # Call callback
        if self._on_toggle:
            self._on_toggle(False)

        logger.info("RecButton: Recording stopped")

    def _start_pulse(self):
        """Start pulsing animation."""
        if self._pulse_animation:
            self._pulse_animation.cancel(self)

        # Create pulse animation
        self._pulse_animation = Animation(opacity=0.5, duration=0.5) + Animation(
            opacity=1.0, duration=0.5
        )
        self._pulse_animation.repeat = True
        self._pulse_animation.start(self)

    def _stop_pulse(self):
        """Stop pulsing animation."""
        if self._pulse_animation:
            self._pulse_animation.cancel(self)
            self._pulse_animation = None
        self.opacity = 1.0

    @property
    def is_recording(self) -> bool:
        """Check if button is in recording state."""
        return self._state == RecButtonState.RECORDING

    def set_recording(self, recording: bool):
        """
        Set recording state programmatically (without triggering callback).

        Args:
            recording: True to set recording state, False for idle.
        """
        if recording and self._state == RecButtonState.IDLE:
            self._state = RecButtonState.RECORDING
            self.text = "STOP"
            self._draw_background()
            self._start_pulse()
        elif not recording and self._state == RecButtonState.RECORDING:
            self._state = RecButtonState.IDLE
            self.text = "REC"
            self.opacity = 1.0
            self._stop_pulse()
            self._draw_background()
