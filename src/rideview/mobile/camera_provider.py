"""
Cross-platform camera provider abstraction.

Provides a unified interface for camera capture across different platforms:
- Desktop (Windows, macOS, Linux): OpenCV VideoCapture
- Android: Native camera via pyjnius (future)
- iOS: Native camera via pyobjus (future)
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class CameraProvider(Protocol):
    """Protocol for cross-platform camera providers."""

    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        ...

    def read(self) -> np.ndarray | None:
        """Read a frame. Returns BGR numpy array or None on failure."""
        ...

    def release(self) -> None:
        """Release the camera resources."""
        ...

    def get_frame_size(self) -> tuple[int, int]:
        """Get frame dimensions as (width, height)."""
        ...

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open."""
        ...


class OpenCVCameraProvider:
    """
    Desktop camera provider using OpenCV VideoCapture.

    Works on Windows, macOS, and Linux with USB cameras.
    """

    def __init__(self, config: dict):
        """
        Initialize OpenCV camera provider.

        Args:
            config: Camera configuration dict with keys:
                - source: Camera index (int) or video file path (str)
                - width: Desired frame width
                - height: Desired frame height
                - fps: Desired frames per second
                - backend: OpenCV backend (CAP_MSMF, CAP_DSHOW, etc.)
        """
        self.source = config.get("source", 0)
        self.width = config.get("width", 1280)
        self.height = config.get("height", 720)
        self.fps = config.get("fps", 30)
        self.backend_name = config.get("backend", "CAP_ANY")

        self._cap: cv2.VideoCapture | None = None
        self._is_open = False

        # Map backend name to OpenCV constant
        self._backend_map = {
            "CAP_ANY": cv2.CAP_ANY,
            "CAP_MSMF": cv2.CAP_MSMF,
            "CAP_DSHOW": cv2.CAP_DSHOW,
            "CAP_V4L2": cv2.CAP_V4L2,
            "CAP_AVFOUNDATION": cv2.CAP_AVFOUNDATION,
        }

    def open(self) -> bool:
        """Open the camera with configured settings."""
        try:
            backend = self._backend_map.get(self.backend_name, cv2.CAP_ANY)

            if isinstance(self.source, str):
                # Video file
                self._cap = cv2.VideoCapture(self.source)
            else:
                # Camera index
                self._cap = cv2.VideoCapture(self.source, backend)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False

            # Set desired properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            # Verify actual settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera opened: source={self.source}, "
                f"resolution={actual_width}x{actual_height}, "
                f"fps={actual_fps}"
            )

            if actual_width != self.width or actual_height != self.height:
                logger.warning(
                    f"Camera resolution differs from requested: "
                    f"got {actual_width}x{actual_height}, "
                    f"requested {self.width}x{self.height}"
                )

            self._is_open = True
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read(self) -> np.ndarray | None:
        """Read a frame from the camera."""
        if not self._is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
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
        """Get current frame dimensions."""
        if self._cap is not None and self._is_open:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (self.width, self.height)

    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open and self._cap is not None and self._cap.isOpened()


class AndroidCameraProvider:
    """
    Android camera provider using native Camera2 API via pyjnius.

    Note: This is a stub implementation. Full implementation requires
    Android-specific setup with pyjnius and Camera2 API bindings.
    """

    def __init__(self, config: dict):
        """Initialize Android camera provider."""
        self.config = config
        self._is_open = False
        logger.warning("AndroidCameraProvider: Using stub implementation")

    def open(self) -> bool:
        """Open the Android camera."""
        # TODO: Implement with pyjnius + Camera2 API
        logger.warning("AndroidCameraProvider: Camera not implemented yet")
        return False

    def read(self) -> np.ndarray | None:
        """Read a frame from the Android camera."""
        return None

    def release(self) -> None:
        """Release Android camera resources."""
        self._is_open = False

    def get_frame_size(self) -> tuple[int, int]:
        """Get frame dimensions."""
        return (1080, 1920)  # Default Android portrait

    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open


def get_camera_provider(config: dict, platform_type: str = "desktop") -> CameraProvider:
    """
    Factory function to get the appropriate camera provider for the current platform.

    Args:
        config: Camera configuration dictionary.
        platform_type: Platform type ("desktop", "android", "ios").

    Returns:
        CameraProvider instance appropriate for the platform.
    """
    if platform_type == "android":
        # Try Android camera, fall back to OpenCV
        try:
            provider = AndroidCameraProvider(config)
            logger.info("Using AndroidCameraProvider")
            return provider
        except Exception as e:
            logger.warning(f"Android camera not available, falling back to OpenCV: {e}")

    # Default to OpenCV (works on all desktop platforms)
    logger.info("Using OpenCVCameraProvider")
    return OpenCVCameraProvider(config)


def list_available_cameras(max_cameras: int = 10) -> list[int]:
    """
    List available camera indices.

    Args:
        max_cameras: Maximum number of camera indices to check.

    Returns:
        List of available camera indices.
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available
