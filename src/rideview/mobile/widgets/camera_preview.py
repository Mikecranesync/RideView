"""
Camera preview widget for RideView.

Displays live camera feed as a Kivy Image widget with efficient texture updates.
"""

import logging

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

logger = logging.getLogger(__name__)


class CameraPreview(Image):
    """
    Kivy Image widget for displaying camera frames.

    Efficiently converts OpenCV BGR frames to Kivy textures.
    """

    def __init__(self, **kwargs):
        """Initialize the camera preview widget."""
        super().__init__(**kwargs)

        # Default placeholder texture
        self._create_placeholder()

        # Frame dimensions (updated on first frame)
        self._frame_width = 1280
        self._frame_height = 720

    def _create_placeholder(self):
        """Create a placeholder texture for when no frame is available."""
        # Create a dark gray placeholder
        placeholder = np.full((720, 1280, 3), 50, dtype=np.uint8)

        # Add "No Camera" text
        cv2.putText(
            placeholder,
            "Camera Preview",
            (440, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (128, 128, 128),
            2,
        )

        self._update_texture(placeholder)

    def update_frame(self, frame: np.ndarray) -> None:
        """
        Update the preview with a new frame.

        Args:
            frame: BGR numpy array from camera/detection.
        """
        if frame is None:
            return

        # Schedule texture update on main thread
        Clock.schedule_once(lambda dt: self._update_texture(frame), 0)

    def _update_texture(self, frame: np.ndarray) -> None:
        """
        Update the Kivy texture with a frame (must be called from main thread).

        Args:
            frame: BGR numpy array.
        """
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]

            # Convert BGR to RGB for Kivy
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip vertically (Kivy textures are bottom-up)
            frame_rgb = cv2.flip(frame_rgb, 0)

            # Create or update texture
            if (
                self.texture is None
                or self.texture.width != width
                or self.texture.height != height
            ):
                self.texture = Texture.create(size=(width, height), colorfmt="rgb")
                self._frame_width = width
                self._frame_height = height

            # Update texture data
            self.texture.blit_buffer(
                frame_rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte"
            )

            # Trigger redraw
            self.canvas.ask_update()

        except Exception as e:
            logger.error(f"Error updating camera preview texture: {e}")

    @property
    def frame_size(self) -> tuple[int, int]:
        """Get current frame dimensions as (width, height)."""
        return (self._frame_width, self._frame_height)
