"""
Video recording manager for RideView.

Handles MP4 video recording with JSON metadata sidecar files
for training data collection.
"""

import json
import logging
import os
import platform as sys_platform
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """Recording state machine states."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RECORDING = "recording"
    FINALIZING = "finalizing"


@dataclass
class FrameMetadata:
    """Metadata for a single recorded frame."""

    frame_number: int
    timestamp_ms: int
    result: str
    confidence: float
    coverage: float
    gap_count: int


@dataclass
class RecordingMetadata:
    """Full metadata for a recording session."""

    filename: str
    filepath: str
    timestamp: str
    duration_seconds: float = 0.0
    frame_count: int = 0
    platform: str = "desktop"
    device: str = ""
    resolution: str = "1280x720"
    fps: int = 15
    inference_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        summary = self._calculate_summary()
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "frame_count": self.frame_count,
            "platform": self.platform,
            "device": self.device,
            "resolution": self.resolution,
            "fps": self.fps,
            "inference_results": self.inference_results,
            "summary": summary,
        }

    def _calculate_summary(self) -> dict[str, Any]:
        """Calculate summary statistics from inference results."""
        if not self.inference_results:
            return {
                "total_frames": 0,
                "pass_count": 0,
                "warning_count": 0,
                "fail_count": 0,
                "no_stripe_count": 0,
                "avg_confidence": 0.0,
            }

        pass_count = sum(1 for r in self.inference_results if r.get("result") == "PASS")
        warning_count = sum(
            1 for r in self.inference_results if r.get("result") == "WARNING"
        )
        fail_count = sum(1 for r in self.inference_results if r.get("result") == "FAIL")
        no_stripe_count = sum(
            1 for r in self.inference_results if r.get("result") == "NO_STRIPE"
        )

        confidences = [
            r.get("confidence", 0) for r in self.inference_results if r.get("confidence")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_frames": len(self.inference_results),
            "pass_count": pass_count,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "no_stripe_count": no_stripe_count,
            "avg_confidence": round(avg_confidence, 3),
        }


class RecordingManager:
    """
    Manages video recording with metadata for training data collection.

    Features:
    - MP4 video recording using cv2.VideoWriter
    - JSON metadata sidecar with per-frame inference results
    - Thread-safe frame addition
    - Automatic file naming with timestamps
    """

    def __init__(
        self,
        output_dir: str,
        config: dict | None = None,
        platform: str = "desktop",
    ):
        """
        Initialize recording manager.

        Args:
            output_dir: Directory to save recordings.
            config: Recording configuration dict.
            platform: Platform type ("desktop", "android", "ios").
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.platform = platform

        # Recording settings
        self.fps = self.config.get("fps", 15)
        self.codec = self.config.get("codec", "mp4v")

        # Resolution based on platform
        if platform in ("android", "ios"):
            self.resolution = (1080, 1920)  # Portrait
        else:
            self.resolution = (1280, 720)  # Landscape

        # State
        self._state = RecordingState.IDLE
        self._lock = threading.Lock()

        # Current recording
        self._video_writer: cv2.VideoWriter | None = None
        self._current_path: Path | None = None
        self._metadata: RecordingMetadata | None = None
        self._start_time: datetime | None = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._state == RecordingState.RECORDING

    @property
    def state(self) -> RecordingState:
        """Get current recording state."""
        return self._state

    def start(self) -> bool:
        """
        Start a new recording.

        Returns:
            True if recording started successfully.
        """
        with self._lock:
            if self._state != RecordingState.IDLE:
                logger.warning(f"Cannot start recording in state: {self._state}")
                return False

            self._state = RecordingState.INITIALIZING

        try:
            # Generate filename with timestamp
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"RideView_{timestamp_str}.mp4"
            self._current_path = self.output_dir / filename

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._video_writer = cv2.VideoWriter(
                str(self._current_path),
                fourcc,
                self.fps,
                self.resolution,
            )

            if not self._video_writer.isOpened():
                logger.error("Failed to open video writer")
                self._state = RecordingState.IDLE
                return False

            # Initialize metadata
            self._metadata = RecordingMetadata(
                filename=filename,
                filepath=str(self._current_path),
                timestamp=timestamp.isoformat(),
                platform=self.platform,
                device=sys_platform.system(),
                resolution=f"{self.resolution[0]}x{self.resolution[1]}",
                fps=self.fps,
            )
            self._start_time = timestamp

            with self._lock:
                self._state = RecordingState.RECORDING

            logger.info(f"Recording started: {self._current_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._state = RecordingState.IDLE
            return False

    def stop(self) -> str | None:
        """
        Stop the current recording and save metadata.

        Returns:
            Path to the recorded video file, or None on failure.
        """
        with self._lock:
            if self._state != RecordingState.RECORDING:
                logger.warning(f"Cannot stop recording in state: {self._state}")
                return None

            self._state = RecordingState.FINALIZING

        try:
            # Release video writer
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

            # Calculate duration
            if self._start_time is not None and self._metadata is not None:
                duration = (datetime.now() - self._start_time).total_seconds()
                self._metadata.duration_seconds = duration

            # Save metadata JSON
            if self._current_path is not None and self._metadata is not None:
                json_path = self._current_path.with_suffix(".json")
                with open(json_path, "w") as f:
                    json.dump(self._metadata.to_dict(), f, indent=2)
                logger.info(f"Metadata saved: {json_path}")

            result_path = str(self._current_path) if self._current_path else None

            # Log summary
            if self._metadata:
                summary = self._metadata._calculate_summary()
                logger.info(
                    f"Recording stopped: {self._metadata.frame_count} frames, "
                    f"{self._metadata.duration_seconds:.1f}s, "
                    f"PASS:{summary['pass_count']} WARN:{summary['warning_count']} "
                    f"FAIL:{summary['fail_count']}"
                )

            # Reset state
            self._current_path = None
            self._metadata = None
            self._start_time = None

            with self._lock:
                self._state = RecordingState.IDLE

            return result_path

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self._state = RecordingState.IDLE
            return None

    def add_frame(
        self,
        frame: np.ndarray,
        result: str = "",
        confidence: float = 0.0,
        coverage: float = 0.0,
        gap_count: int = 0,
    ) -> bool:
        """
        Add a frame to the current recording.

        Args:
            frame: BGR numpy array (will be resized if needed).
            result: Detection result string (PASS/WARNING/FAIL/NO_STRIPE).
            confidence: Detection confidence (0.0-1.0).
            coverage: Stripe coverage percentage (0.0-1.0).
            gap_count: Number of gaps detected.

        Returns:
            True if frame was added successfully.
        """
        if self._state != RecordingState.RECORDING:
            return False

        if self._video_writer is None or self._metadata is None:
            return False

        try:
            # Resize frame if needed
            if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                frame = cv2.resize(frame, self.resolution)

            # Write frame
            self._video_writer.write(frame)

            # Add metadata
            frame_meta = {
                "frame_number": self._metadata.frame_count,
                "timestamp_ms": int(datetime.now().timestamp() * 1000),
                "result": result,
                "confidence": round(confidence, 3),
                "coverage": round(coverage, 3),
                "gap_count": gap_count,
            }
            self._metadata.inference_results.append(frame_meta)
            self._metadata.frame_count += 1

            return True

        except Exception as e:
            logger.error(f"Error adding frame: {e}")
            return False

    def get_recordings(self) -> list[dict[str, Any]]:
        """
        Get list of available recordings with metadata.

        Returns:
            List of recording info dicts sorted by date (newest first).
        """
        recordings = []

        for video_file in self.output_dir.glob("*.mp4"):
            json_file = video_file.with_suffix(".json")

            info = {
                "filename": video_file.name,
                "filepath": str(video_file),
                "size_mb": round(video_file.stat().st_size / 1024 / 1024, 2),
            }

            # Load metadata if available
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        metadata = json.load(f)
                    info.update(
                        {
                            "timestamp": metadata.get("timestamp", ""),
                            "duration_seconds": metadata.get("duration_seconds", 0),
                            "frame_count": metadata.get("frame_count", 0),
                            "summary": metadata.get("summary", {}),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {video_file}: {e}")

            recordings.append(info)

        # Sort by timestamp (newest first)
        recordings.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return recordings

    def delete_recording(self, filepath: str) -> bool:
        """
        Delete a recording and its metadata.

        Args:
            filepath: Path to the video file.

        Returns:
            True if deletion successful.
        """
        try:
            video_path = Path(filepath)
            json_path = video_path.with_suffix(".json")

            if video_path.exists():
                video_path.unlink()
            if json_path.exists():
                json_path.unlink()

            logger.info(f"Deleted recording: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete recording: {e}")
            return False
