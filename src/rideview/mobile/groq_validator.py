"""
Groq API validator for RideView Tier 2 validation.

Uses Groq's Vision API to validate ambiguous detections when
OpenCV confidence is between the configured thresholds.
"""

import base64
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroqValidationResult:
    """Result from Groq API validation."""

    result: str  # "PASS", "WARNING", "FAIL", "ERROR"
    confidence: float
    explanation: str
    error: str | None = None


class GroqValidator:
    """
    Tier 2 validator using Groq Vision API.

    Used when OpenCV detection confidence is ambiguous (typically 0.60-0.85).
    Sends the frame to Groq's vision model for additional validation.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize Groq validator.

        Args:
            config: LLM configuration dict with groq settings.
        """
        self.config = config or {}
        groq_config = self.config.get("groq", {})

        # Get API key from environment
        api_key_env = groq_config.get("api_key_env", "GROQ_API_KEY")
        self.api_key = os.environ.get(api_key_env)

        if not self.api_key:
            logger.warning(
                f"Groq API key not found in environment variable: {api_key_env}"
            )

        # Configuration
        self.model = groq_config.get("model", "llama-3.2-90b-vision-preview")
        self.timeout = groq_config.get("timeout", 10)
        self.enabled = groq_config.get("enabled", True) and self.api_key is not None

        # Confidence thresholds
        threshold_config = groq_config.get("confidence_threshold", {})
        self.min_confidence = threshold_config.get("min", 0.60)
        self.max_confidence = threshold_config.get("max", 0.85)

        # Groq client
        self._client = None
        if self.enabled:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("Groq validator initialized")
            except ImportError:
                logger.warning("groq package not installed. Run: pip install groq")
                self.enabled = False

    def should_validate(self, opencv_confidence: float) -> bool:
        """
        Check if Groq validation should be triggered.

        Args:
            opencv_confidence: Confidence from OpenCV detection (0.0-1.0).

        Returns:
            True if validation should be run.
        """
        if not self.enabled:
            return False

        return self.min_confidence <= opencv_confidence <= self.max_confidence

    def validate(self, frame: np.ndarray) -> GroqValidationResult:
        """
        Validate a frame using Groq Vision API.

        Args:
            frame: BGR numpy array of the frame to validate.

        Returns:
            GroqValidationResult with result, confidence, and explanation.
        """
        if not self.enabled or self._client is None:
            return GroqValidationResult(
                result="ERROR",
                confidence=0.0,
                explanation="",
                error="Groq validator not enabled or configured",
            )

        try:
            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Create prompt
            prompt = """Analyze this image of a fastener (bolt, nut, or washer) for torque stripe integrity.

A torque stripe is a paint line applied across the fastener head and surrounding surface to visually indicate if the fastener has loosened.

Evaluate:
1. Is there a visible torque stripe (colored paint line)?
2. Is the stripe continuous and unbroken?
3. Is the stripe aligned (not showing rotation)?

Respond with EXACTLY one of these classifications:
- PASS: Stripe is clearly visible, continuous, and properly aligned
- WARNING: Stripe is visible but shows minor degradation (small chip, slight fading)
- FAIL: Stripe is broken, missing, misaligned, or shows significant damage
- NO_STRIPE: No torque stripe is visible on this fastener

Format your response as:
CLASSIFICATION: [PASS/WARNING/FAIL/NO_STRIPE]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief reason for classification]"""

            # Make API call
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent classification
            )

            # Parse response
            response_text = response.choices[0].message.content
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return GroqValidationResult(
                result="ERROR",
                confidence=0.0,
                explanation="",
                error=str(e),
            )

    def validate_async(
        self,
        frame: np.ndarray,
        callback: Callable[[GroqValidationResult], None],
    ) -> threading.Thread:
        """
        Validate a frame asynchronously in a background thread.

        Args:
            frame: BGR numpy array of the frame to validate.
            callback: Function to call with the result.

        Returns:
            The thread running the validation.
        """
        def _validate():
            result = self.validate(frame)
            callback(result)

        thread = threading.Thread(target=_validate, daemon=True)
        thread.start()
        return thread

    def _parse_response(self, response_text: str) -> GroqValidationResult:
        """
        Parse the Groq API response into a structured result.

        Args:
            response_text: Raw text response from the API.

        Returns:
            Parsed GroqValidationResult.
        """
        result = "ERROR"
        confidence = 0.0
        explanation = response_text

        try:
            lines = response_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("CLASSIFICATION:"):
                    result_text = line.replace("CLASSIFICATION:", "").strip()
                    if result_text in ("PASS", "WARNING", "FAIL", "NO_STRIPE"):
                        result = result_text
                elif line.startswith("CONFIDENCE:"):
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    try:
                        confidence = float(conf_text)
                    except ValueError:
                        pass
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()

        except Exception as e:
            logger.warning(f"Failed to parse Groq response: {e}")
            explanation = response_text

        return GroqValidationResult(
            result=result,
            confidence=confidence,
            explanation=explanation,
        )

    @property
    def is_available(self) -> bool:
        """Check if Groq validation is available."""
        return self.enabled and self._client is not None
