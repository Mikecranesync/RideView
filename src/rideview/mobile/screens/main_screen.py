"""
Main screen for RideView mobile app.

Primary UI showing camera preview, detection overlay, and recording controls.
"""

import logging

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

import numpy as np

from ...core.config import Config
from ...core.result import StripeAnalysis
from ..camera_provider import CameraProvider
from ..detection_worker import DetectionWorker
from ..recording_manager import RecordingManager
from ..widgets.camera_preview import CameraPreview
from ..widgets.rec_button import RecButton
from ..widgets.status_overlay import StatusOverlay
from .settings_screen import SettingsScreen

logger = logging.getLogger(__name__)


class MainScreen(FloatLayout):
    """
    Main screen with camera preview and detection UI.

    Layout:
    ┌─────────────────────────────────────┐
    │  LIVE CAMERA FEED (full screen)     │
    │                                     │
    │  ┌──────────────────┐               │
    │  │ Status Overlay   │               │
    │  │ PASS / 92%       │               │
    │  └──────────────────┘               │
    │                                     │
    │  [Settings]  [Files]     [REC] ●    │
    └─────────────────────────────────────┘
    """

    def __init__(
        self,
        camera_provider: CameraProvider,
        detection_worker: DetectionWorker,
        recording_manager: RecordingManager,
        config: Config,
        **kwargs,
    ):
        """
        Initialize the main screen.

        Args:
            camera_provider: Camera provider instance.
            detection_worker: Detection worker instance.
            recording_manager: Recording manager instance.
            config: Application configuration.
        """
        super().__init__(**kwargs)

        self.camera_provider = camera_provider
        self.detection_worker = detection_worker
        self.recording_manager = recording_manager
        self.config = config

        # Create UI components
        self._create_ui()

    def _create_ui(self):
        """Create all UI components."""
        # Camera preview (full screen background)
        self.camera_preview = CameraPreview(
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            allow_stretch=True,
            keep_ratio=True,
        )
        self.add_widget(self.camera_preview)

        # Status overlay (top-left)
        self.status_overlay = StatusOverlay(
            pos_hint={"x": 0.02, "top": 0.98},
        )
        self.add_widget(self.status_overlay)

        # Bottom control bar
        self._create_control_bar()

        # Recording time label (shown when recording)
        self.recording_label = Label(
            text="",
            font_size="16sp",
            color=(1, 0.2, 0.2, 1),
            bold=True,
            size_hint=(None, None),
            size=(150, 30),
            pos_hint={"right": 0.98, "top": 0.98},
        )
        self.add_widget(self.recording_label)

    def _create_control_bar(self):
        """Create bottom control bar with buttons."""
        control_bar = BoxLayout(
            orientation="horizontal",
            size_hint=(1, None),
            height=100,
            pos_hint={"x": 0, "y": 0},
            padding=[20, 10, 20, 10],
            spacing=20,
        )

        # Settings button
        settings_btn = Button(
            text="Settings",
            size_hint=(None, 1),
            width=100,
            font_size="14sp",
        )
        settings_btn.bind(on_press=self._on_settings_press)
        control_bar.add_widget(settings_btn)

        # Files button
        files_btn = Button(
            text="Files",
            size_hint=(None, 1),
            width=100,
            font_size="14sp",
        )
        files_btn.bind(on_press=self._on_files_press)
        control_bar.add_widget(files_btn)

        # Spacer
        control_bar.add_widget(BoxLayout(size_hint=(1, 1)))

        # REC button
        self.rec_button = RecButton(
            on_toggle=self._on_recording_toggle,
        )
        control_bar.add_widget(self.rec_button)

        self.add_widget(control_bar)

    def update_preview(self, frame: np.ndarray) -> None:
        """
        Update camera preview with raw frame.

        Args:
            frame: BGR numpy array from camera.
        """
        # For now, we update with raw frame
        # The annotated frame comes through update_detection_result
        pass  # Preview is updated in update_detection_result with annotated frame

    def update_detection_result(self, analysis: StripeAnalysis) -> None:
        """
        Update UI with detection result.

        Args:
            analysis: StripeAnalysis from detector.
        """
        # Update status overlay
        self.status_overlay.update(analysis)

        # Update camera preview with annotated frame
        if analysis.annotated_frame is not None:
            self.camera_preview.update_frame(analysis.annotated_frame)

    def _on_recording_toggle(self, is_recording: bool) -> None:
        """
        Handle recording toggle from REC button.

        Args:
            is_recording: True if recording started, False if stopped.
        """
        if is_recording:
            # Start recording
            if self.recording_manager.start():
                self.recording_label.text = "REC 00:00"
                logger.info("Recording started from UI")
            else:
                # Failed to start, reset button
                self.rec_button.set_recording(False)
                logger.error("Failed to start recording")
        else:
            # Stop recording
            filepath = self.recording_manager.stop()
            self.recording_label.text = ""
            if filepath:
                logger.info(f"Recording saved: {filepath}")

    def _on_settings_press(self, instance):
        """Handle settings button press."""
        logger.info("Settings button pressed")
        # Create settings screen overlay
        self.settings_screen = SettingsScreen(
            config=self.config,
            camera_provider=self.camera_provider,
            on_close=self._on_settings_close,
            size_hint=(0.9, 0.9),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        # Add a semi-transparent background
        from kivy.graphics import Color, Rectangle
        with self.settings_screen.canvas.before:
            Color(0.1, 0.1, 0.1, 0.95)
            self.settings_screen._bg_rect = Rectangle(
                pos=self.settings_screen.pos,
                size=self.settings_screen.size,
            )
        self.settings_screen.bind(
            pos=self._update_settings_bg,
            size=self._update_settings_bg,
        )
        self.add_widget(self.settings_screen)

    def _update_settings_bg(self, instance, value):
        """Update settings background rectangle."""
        if hasattr(self, 'settings_screen') and hasattr(self.settings_screen, '_bg_rect'):
            self.settings_screen._bg_rect.pos = self.settings_screen.pos
            self.settings_screen._bg_rect.size = self.settings_screen.size

    def _on_settings_close(self):
        """Handle settings screen close."""
        if hasattr(self, 'settings_screen'):
            self.remove_widget(self.settings_screen)
            del self.settings_screen
            logger.info("Settings closed")

    def _on_files_press(self, instance):
        """Handle files button press."""
        logger.info("Files button pressed")
        # TODO: Show recordings list

    def update_recording_time(self, elapsed_seconds: float) -> None:
        """
        Update recording time display.

        Args:
            elapsed_seconds: Elapsed recording time in seconds.
        """
        if self.recording_manager.is_recording:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            self.recording_label.text = f"REC {minutes:02d}:{seconds:02d}"
