"""
RideView - Torque Stripe Verification System

Real-time computer vision system for detecting broken or compromised
torque stripes on industrial bolts.
"""

__version__ = "0.1.0"
__author__ = "RideView Team"

from .core.detector import TorqueStripeDetector
from .core.result import DetectionResult, StripeAnalysis

__all__ = ["TorqueStripeDetector", "DetectionResult", "StripeAnalysis", "__version__"]
