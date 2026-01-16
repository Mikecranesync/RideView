"""
RideView Mobile - Cross-platform Kivy UI for torque stripe detection.

This module provides a Kivy-based user interface that works on:
- Desktop (Windows, macOS, Linux)
- Mobile (Android, iOS)

Features:
- Live camera preview with detection overlay
- Video recording with JSON metadata
- LLM-powered Tier 2 validation (Groq API / local Llama 2)
"""

from .app import RideViewApp

__all__ = ["RideViewApp"]
