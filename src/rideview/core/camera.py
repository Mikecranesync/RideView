"""
Camera abstraction layer for RideView.

Provides a consistent interface for capturing video from USB cameras
with support for different backends and configuration options.
"""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Camera abstraction for video capture.

    Supports:
    - USB webcams (ArduCam UVC, standard webcams)
    - Video files for testing
    - Configurable resolution, FPS, and backend

    Usage:
        camera = Camera(config['camera'])
        camera.open()
        frame = camera.read()
        camera.release()

    Or as context manager:
        with Camera(config['camera']) as camera:
            frame = camera.read()
    """

    # Backend mappings for OpenCV
    BACKENDS = {
        "CAP_MSMF": cv2.CAP_MSMF,
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_V4L2": cv2.CAP_V4L2,
        "CAP_ANY": cv2.CAP_ANY,
    }

    def __init__(self, config: dict[str, Any]):
        """
        Initialize camera with configuration.

        Args:
            config: Camera configuration dictionary with keys:
                - source: int (device index) or str (video file path)
                - backend: str (CAP_MSMF, CAP_DSHOW, etc.)
                - width: int
                - height: int
                - fps: int
                - buffer_size: int
                - auto_exposure: bool
        """
        self.source = config.get("source", 0)
        self.backend_name = config.get("backend", "CAP_MSMF")
        self.width = config.get("width", 1280)
        self.height = config.get("height", 720)
        self.fps = config.get("fps", 30)
        self.buffer_size = config.get("buffer_size", 1)
        self.auto_exposure = config.get("auto_exposure", True)

        self._cap: cv2.VideoCapture | None = None
        self._is_open = False

    @property
    def backend(self) -> int:
        """Get OpenCV backend constant."""
        return self.BACKENDS.get(self.backend_name, cv2.CAP_ANY)

    @property
    def is_open(self) -> bool:
        """Check if camera is open and ready."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def open(self) -> bool:
        """
        Open the camera for capture.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._is_open:
            return True

        try:
            # Determine if source is a file or device
            if isinstance(self.source, str):
                # Video file
                self._cap = cv2.VideoCapture(self.source)
            else:
                # Device index
                self._cap = cv2.VideoCapture(self.source, self.backend)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False

            # Configure camera
            self._configure()

            self._is_open = True
            logger.info(
                f"Camera opened: source={self.source}, "
                f"resolution={self.width}x{self.height}, fps={self.fps}"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def _configure(self) -> None:
        """Apply camera configuration settings."""
        if self._cap is None:
            return

        # Resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Frame rate
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Buffer size (lower = less latency)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Auto exposure (Windows-specific)
        if self.auto_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

        # Log actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        if actual_width != self.width or actual_height != self.height:
            logger.warning(
                f"Camera resolution mismatch: requested {self.width}x{self.height}, "
                f"got {actual_width}x{actual_height}"
            )

        logger.debug(f"Camera actual settings: {actual_width}x{actual_height} @ {actual_fps} FPS")

    def read(self) -> np.ndarray | None:
        """
        Read a frame from the camera.

        Returns:
            BGR image as numpy array, or None if read failed.
        """
        if not self.is_open:
            if not self.open():
                return None

        ret, frame = self._cap.read()  # type: ignore
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None

        return frame

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Camera released")

    def get_frame_size(self) -> tuple[int, int]:
        """
        Get actual frame size.

        Returns:
            Tuple of (width, height)
        """
        if not self.is_open:
            return (self.width, self.height)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore
        return (width, height)

    def __enter__(self) -> "Camera":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.release()

    @staticmethod
    def list_available_cameras(max_index: int = 5) -> list[int]:
        """
        List available camera indices.

        Args:
            max_index: Maximum camera index to check.

        Returns:
            List of available camera indices.
        """
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
