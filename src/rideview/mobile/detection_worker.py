"""
Threaded detection worker for RideView.

Runs the TorqueStripeDetector.analyze() in a background thread
to avoid blocking the UI thread.
"""

import logging
import queue
import threading
from typing import Callable

import numpy as np

from ..core.detector import TorqueStripeDetector
from ..core.result import StripeAnalysis

logger = logging.getLogger(__name__)


class DetectionWorker:
    """
    Background worker for running detection on camera frames.

    Uses a producer-consumer pattern:
    - Main thread submits frames to input queue
    - Worker thread processes frames and calls result callback
    """

    def __init__(
        self,
        detector: TorqueStripeDetector,
        on_result: Callable[[StripeAnalysis], None],
        max_queue_size: int = 2,
    ):
        """
        Initialize the detection worker.

        Args:
            detector: TorqueStripeDetector instance to use for analysis.
            on_result: Callback function called with each StripeAnalysis result.
            max_queue_size: Maximum frames to queue (older frames dropped if full).
        """
        self.detector = detector
        self.on_result = on_result
        self.max_queue_size = max_queue_size

        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._roi: tuple[int, int, int, int] | None = None

        # Statistics
        self.frames_processed = 0
        self.frames_dropped = 0

    def start(self) -> None:
        """Start the detection worker thread."""
        if self._running:
            logger.warning("DetectionWorker already running")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="DetectionWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("DetectionWorker started")

    def stop(self) -> None:
        """Stop the detection worker thread."""
        if not self._running:
            return

        self._running = False

        # Send stop signal
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        logger.info(
            f"DetectionWorker stopped: "
            f"processed={self.frames_processed}, "
            f"dropped={self.frames_dropped}"
        )

    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit a frame for detection.

        If the queue is full, the oldest frame is dropped (non-blocking).

        Args:
            frame: BGR numpy array to analyze.

        Returns:
            True if frame was queued, False if dropped.
        """
        if not self._running:
            return False

        try:
            # Non-blocking put - drop frame if queue full
            self._frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Queue full, drop oldest frame and add new one
            try:
                self._frame_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame.copy())
                return True
            except queue.Full:
                return False

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """
        Set the region of interest for detection.

        Args:
            roi: (x, y, w, h) tuple or None to use full frame / fastener detection.
        """
        self._roi = roi
        logger.info(f"DetectionWorker ROI set to: {roi}")

    def _worker_loop(self) -> None:
        """Main worker loop - processes frames from queue."""
        logger.debug("DetectionWorker loop started")

        while self._running:
            try:
                # Wait for frame with timeout
                frame = self._frame_queue.get(timeout=0.1)

                # Check for stop signal
                if frame is None:
                    break

                # Run detection
                try:
                    analysis = self.detector.analyze(frame, roi=self._roi)
                    self.frames_processed += 1

                    # Call result callback
                    if self.on_result is not None:
                        self.on_result(analysis)

                except Exception as e:
                    logger.error(f"Detection error: {e}")

            except queue.Empty:
                # No frame available, continue waiting
                continue

        logger.debug("DetectionWorker loop exited")

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._frame_queue.qsize()
