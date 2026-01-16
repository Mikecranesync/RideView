"""
RideView Kivy Application - Cross-platform torque stripe detection.

Main entry point for the Kivy-based mobile/desktop application.
"""

import logging
import os
import platform as sys_platform
from pathlib import Path

# Prevent Kivy from consuming command-line arguments
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.logger import Logger

from ..core.config import Config
from ..core.detector import TorqueStripeDetector
from .camera_provider import get_camera_provider
from .detection_worker import DetectionWorker
from .recording_manager import RecordingManager
from .screens.main_screen import MainScreen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RideViewApp(App):
    """
    Main RideView Kivy application.

    Coordinates:
    - Camera capture (via CameraProvider)
    - Detection pipeline (via DetectionWorker)
    - Video recording (via RecordingManager)
    - UI updates (via MainScreen)
    """

    def __init__(self, app_config: Config | None = None, **kwargs):
        """
        Initialize the RideView app.

        Args:
            app_config: Optional Config object. If not provided, loads from default location.
        """
        super().__init__(**kwargs)

        # Load configuration (use app_config to avoid conflict with Kivy's config)
        if app_config is None:
            app_config = Config()
        self.app_config = app_config

        # Detect platform
        self.platform_type = self._detect_platform()

        # Set up recording directory
        self.recording_dir = self._get_recording_directory()
        os.makedirs(self.recording_dir, exist_ok=True)

        # Components (initialized in build())
        self.camera_provider = None
        self.detector = None
        self.detection_worker = None
        self.recording_manager = None
        self.main_screen = None

        # State
        self.is_running = False

        Logger.info(f"RideView: Initialized on {sys_platform.system()} ({self.platform_type})")
        Logger.info(f"RideView: Recording directory: {self.recording_dir}")

    def _detect_platform(self) -> str:
        """Detect current platform type."""
        system = sys_platform.system()

        if system == "Linux":
            # Check if running on Android
            try:
                import android  # noqa: F401
                return "android"
            except ImportError:
                return "desktop"
        elif system == "Darwin":
            # Could be macOS or iOS
            # For now, assume desktop (iOS would need different detection)
            return "desktop"
        else:
            # Windows or other
            return "desktop"

    def _get_recording_directory(self) -> str:
        """Get platform-appropriate recording directory."""
        if self.platform_type == "android":
            return "/sdcard/RideView/recordings/"
        else:
            # Desktop: ~/RideView/recordings/
            return str(Path.home() / "RideView" / "recordings")

    def build(self):
        """Build the application UI."""
        # Set window properties based on platform
        if self.platform_type == "desktop":
            Window.size = (1280, 720)
            self.title = "RideView - Torque Stripe Detection"
        else:
            # Mobile: use full screen
            Window.size = (1080, 1920)

        # Initialize detection pipeline
        self.detector = TorqueStripeDetector(self.app_config.as_dict)
        Logger.info("RideView: Detection pipeline initialized")

        # Initialize camera provider
        camera_config = self.app_config.get("camera", {})
        self.camera_provider = get_camera_provider(camera_config, self.platform_type)
        Logger.info(f"RideView: Camera provider initialized ({type(self.camera_provider).__name__})")

        # Initialize detection worker
        self.detection_worker = DetectionWorker(
            detector=self.detector,
            on_result=self._on_detection_result,
        )

        # Initialize recording manager
        recording_config = self.app_config.get("recording", {})
        self.recording_manager = RecordingManager(
            output_dir=self.recording_dir,
            config=recording_config,
            platform=self.platform_type,
        )

        # Create main screen
        self.main_screen = MainScreen(
            camera_provider=self.camera_provider,
            detection_worker=self.detection_worker,
            recording_manager=self.recording_manager,
            config=self.app_config,
        )

        return self.main_screen

    def on_start(self):
        """Called when the application starts."""
        Logger.info("RideView: Application starting")

        # Open camera
        if self.camera_provider and self.camera_provider.open():
            Logger.info("RideView: Camera opened successfully")

            # Start detection worker
            self.detection_worker.start()

            # Start frame capture loop
            self.is_running = True
            Clock.schedule_interval(self._capture_frame, 1.0 / 30.0)  # 30 FPS capture
        else:
            Logger.error("RideView: Failed to open camera")

    def on_stop(self):
        """Called when the application stops."""
        Logger.info("RideView: Application stopping")

        self.is_running = False

        # Stop detection worker
        if self.detection_worker:
            self.detection_worker.stop()

        # Stop recording if active
        if self.recording_manager and self.recording_manager.is_recording:
            self.recording_manager.stop()

        # Release camera
        if self.camera_provider:
            self.camera_provider.release()

        Logger.info("RideView: Application stopped")

    def _capture_frame(self, dt):
        """Capture a frame from the camera and process it."""
        if not self.is_running or not self.camera_provider:
            return

        frame = self.camera_provider.read()
        if frame is not None:
            # Send frame to detection worker
            self.detection_worker.submit_frame(frame)

            # Update camera preview (raw frame)
            if self.main_screen:
                self.main_screen.update_preview(frame)

    def _on_detection_result(self, analysis):
        """Handle detection result from worker thread."""
        if not self.is_running:
            return

        # Schedule UI update on main thread
        Clock.schedule_once(lambda dt: self._update_ui(analysis), 0)

        # If recording, add frame to recording
        if self.recording_manager and self.recording_manager.is_recording:
            self.recording_manager.add_frame(
                frame=analysis.annotated_frame,
                result=analysis.result.value,
                confidence=analysis.confidence,
                coverage=analysis.coverage_percent,
                gap_count=analysis.gap_count,
            )

    def _update_ui(self, analysis):
        """Update UI with detection result (called on main thread)."""
        if self.main_screen:
            self.main_screen.update_detection_result(analysis)


def run_mobile_app(config: Config | None = None):
    """
    Run the RideView mobile/desktop Kivy application.

    Args:
        config: Optional Config object.
    """
    app = RideViewApp(app_config=config)
    app.run()
