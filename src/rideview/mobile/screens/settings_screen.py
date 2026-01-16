"""
Settings screen for RideView.

Configuration UI for camera, recording, color calibration, and LLM settings.
"""

import logging
from typing import Callable

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch

from ...core.config import Config

logger = logging.getLogger(__name__)


class SettingRow(BoxLayout):
    """A single setting row with label and control."""

    def __init__(self, label: str, **kwargs):
        kwargs.setdefault("orientation", "horizontal")
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", 50)
        kwargs.setdefault("padding", [10, 5, 10, 5])
        super().__init__(**kwargs)

        self.label = Label(
            text=label,
            font_size="14sp",
            size_hint=(0.5, 1),
            halign="left",
            valign="middle",
        )
        self.label.bind(size=self.label.setter("text_size"))
        self.add_widget(self.label)


class SwitchSetting(SettingRow):
    """Toggle switch setting."""

    def __init__(
        self,
        label: str,
        initial_value: bool = False,
        on_change: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(label, **kwargs)

        self.on_change = on_change
        self.switch = Switch(active=initial_value, size_hint=(0.5, 1))
        self.switch.bind(active=self._on_active)
        self.add_widget(self.switch)

    def _on_active(self, instance, value):
        if self.on_change:
            self.on_change(value)

    @property
    def value(self) -> bool:
        return self.switch.active


class SliderSetting(SettingRow):
    """Numeric slider setting."""

    def __init__(
        self,
        label: str,
        min_value: float = 0,
        max_value: float = 100,
        initial_value: float = 50,
        step: float = 1,
        on_change: Callable[[float], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("height", 70)
        super().__init__(label, **kwargs)

        self.on_change = on_change

        # Value display and slider container
        control_layout = BoxLayout(orientation="vertical", size_hint=(0.5, 1))

        # Value label
        self.value_label = Label(
            text=f"{initial_value:.0f}",
            font_size="12sp",
            size_hint_y=0.3,
        )
        control_layout.add_widget(self.value_label)

        # Slider
        self.slider = Slider(
            min=min_value,
            max=max_value,
            value=initial_value,
            step=step,
            size_hint_y=0.7,
        )
        self.slider.bind(value=self._on_value)
        control_layout.add_widget(self.slider)

        self.add_widget(control_layout)

    def _on_value(self, instance, value):
        self.value_label.text = f"{value:.0f}"
        if self.on_change:
            self.on_change(value)

    @property
    def value(self) -> float:
        return self.slider.value


class SettingsScreen(BoxLayout):
    """
    Settings screen with configuration options.

    Sections:
    - Camera settings
    - Recording settings
    - Color calibration
    - LLM settings
    """

    def __init__(
        self,
        config: Config,
        on_close: Callable[[], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding", [10, 10, 10, 10])
        kwargs.setdefault("spacing", 10)
        super().__init__(**kwargs)

        self.config = config
        self.on_close = on_close

        self._create_ui()

    def _create_ui(self):
        """Create the settings UI."""
        # Header
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=50)

        title = Label(
            text="Settings",
            font_size="20sp",
            bold=True,
            size_hint=(0.7, 1),
            halign="left",
            valign="middle",
        )
        title.bind(size=title.setter("text_size"))
        header.add_widget(title)

        close_btn = Button(text="Close", size_hint=(0.3, 1), font_size="14sp")
        close_btn.bind(on_press=self._on_close)
        header.add_widget(close_btn)

        self.add_widget(header)

        # Scrollable settings
        scroll_view = ScrollView(size_hint=(1, 1))
        settings_layout = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=5,
            padding=[0, 10, 0, 10],
        )
        settings_layout.bind(minimum_height=settings_layout.setter("height"))

        # Camera section
        settings_layout.add_widget(self._create_section_header("Camera"))
        settings_layout.add_widget(
            SliderSetting(
                label="Camera Index",
                min_value=0,
                max_value=5,
                initial_value=self.config.get("camera.source", 0),
                step=1,
                on_change=self._on_camera_change,
            )
        )

        # Recording section
        settings_layout.add_widget(self._create_section_header("Recording"))
        settings_layout.add_widget(
            SwitchSetting(
                label="Enable Recording",
                initial_value=self.config.get("recording.enabled", True),
                on_change=self._on_recording_enabled_change,
            )
        )
        settings_layout.add_widget(
            SliderSetting(
                label="Recording FPS",
                min_value=10,
                max_value=30,
                initial_value=self.config.get("recording.fps", 15),
                step=5,
                on_change=self._on_recording_fps_change,
            )
        )

        # Detection section
        settings_layout.add_widget(self._create_section_header("Detection"))
        settings_layout.add_widget(
            SwitchSetting(
                label="Require Fastener",
                initial_value=self.config.get("fastener_detection.require_fastener", True),
                on_change=self._on_require_fastener_change,
            )
        )
        settings_layout.add_widget(
            SliderSetting(
                label="Confidence Threshold (%)",
                min_value=30,
                max_value=90,
                initial_value=self.config.get("fastener_detection.confidence_threshold", 0.6) * 100,
                step=5,
                on_change=self._on_confidence_change,
            )
        )

        # LLM section
        settings_layout.add_widget(self._create_section_header("LLM Validation"))
        settings_layout.add_widget(
            SwitchSetting(
                label="Enable Groq API",
                initial_value=self.config.get("llm.groq.enabled", True),
                on_change=self._on_groq_enabled_change,
            )
        )

        scroll_view.add_widget(settings_layout)
        self.add_widget(scroll_view)

    def _create_section_header(self, text: str) -> Label:
        """Create a section header label."""
        return Label(
            text=text,
            font_size="16sp",
            bold=True,
            color=(0.4, 0.7, 1.0, 1),
            size_hint_y=None,
            height=40,
            halign="left",
            valign="bottom",
        )

    def _on_camera_change(self, value: float):
        """Handle camera index change."""
        logger.info(f"Camera index changed to: {int(value)}")
        # Note: Actual camera change requires app restart

    def _on_recording_enabled_change(self, value: bool):
        """Handle recording enabled change."""
        logger.info(f"Recording enabled: {value}")

    def _on_recording_fps_change(self, value: float):
        """Handle recording FPS change."""
        logger.info(f"Recording FPS changed to: {int(value)}")

    def _on_require_fastener_change(self, value: bool):
        """Handle require fastener change."""
        logger.info(f"Require fastener: {value}")

    def _on_confidence_change(self, value: float):
        """Handle confidence threshold change."""
        logger.info(f"Confidence threshold: {value / 100:.2f}")

    def _on_groq_enabled_change(self, value: bool):
        """Handle Groq API enabled change."""
        logger.info(f"Groq API enabled: {value}")

    def _on_close(self, instance):
        """Handle close button press."""
        if self.on_close:
            self.on_close()
