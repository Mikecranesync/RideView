"""
Detection tuning screen for RideView.

Real-time adjustment of detection parameters: preprocessing, thresholds, morphology.
"""

import logging
from typing import Callable

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider

from ...core.detector import TorqueStripeDetector

logger = logging.getLogger(__name__)


class TuningSlider(BoxLayout):
    """A tuning slider with label and value display."""

    def __init__(
        self,
        label: str,
        min_value: float,
        max_value: float,
        initial_value: float,
        step: float = 1,
        value_format: str = "{:.0f}",
        on_change: Callable[[float], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("orientation", "horizontal")
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", 60)
        kwargs.setdefault("padding", [10, 5, 10, 5])
        super().__init__(**kwargs)

        self.on_change = on_change
        self.value_format = value_format

        # Label
        self.label = Label(
            text=label,
            font_size="13sp",
            size_hint=(0.4, 1),
            halign="left",
            valign="middle",
        )
        self.label.bind(size=self.label.setter("text_size"))
        self.add_widget(self.label)

        # Slider + value container
        control_layout = BoxLayout(orientation="vertical", size_hint=(0.6, 1))

        # Value label
        self.value_label = Label(
            text=value_format.format(initial_value),
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
        self.value_label.text = self.value_format.format(value)
        if self.on_change:
            self.on_change(value)

    @property
    def value(self) -> float:
        return self.slider.value

    def set_value(self, value: float) -> None:
        """Set slider value without triggering callback."""
        self.slider.value = value
        self.value_label.text = self.value_format.format(value)


class TuningScreen(BoxLayout):
    """
    Detection tuning screen with real-time parameter adjustment.

    Sections:
    - Preprocessing (CLAHE, blur)
    - Thresholds (pass/warning coverage, gaps, confidence)
    - Morphology (kernel sizes)
    """

    def __init__(
        self,
        detector: TorqueStripeDetector,
        on_close: Callable[[], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding", [10, 10, 10, 10])
        kwargs.setdefault("spacing", 10)
        super().__init__(**kwargs)

        self.detector = detector
        self.on_close = on_close
        self.sliders: dict[str, TuningSlider] = {}

        self._create_ui()

    def _create_ui(self):
        """Create the tuning UI."""
        # Header
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=50)

        title = Label(
            text="Detection Tuning",
            font_size="20sp",
            bold=True,
            size_hint=(0.5, 1),
            halign="left",
            valign="middle",
        )
        title.bind(size=title.setter("text_size"))
        header.add_widget(title)

        reset_btn = Button(text="Reset", size_hint=(0.25, 1), font_size="14sp")
        reset_btn.bind(on_press=self._on_reset)
        header.add_widget(reset_btn)

        close_btn = Button(text="Close", size_hint=(0.25, 1), font_size="14sp")
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

        # Get current settings
        settings = self.detector.get_tuning_settings()
        preproc = settings["preprocessor"]
        validator = settings["validator"]
        morph = settings["morphology"]

        # Preprocessing section
        settings_layout.add_widget(self._create_section_header("Preprocessing"))

        self.sliders["clahe_clip"] = TuningSlider(
            label="CLAHE Clip Limit",
            min_value=1.0,
            max_value=5.0,
            initial_value=preproc["clahe_clip_limit"],
            step=0.5,
            value_format="{:.1f}",
            on_change=self._on_clahe_clip_change,
        )
        settings_layout.add_widget(self.sliders["clahe_clip"])

        self.sliders["blur_kernel"] = TuningSlider(
            label="Blur Kernel Size",
            min_value=3,
            max_value=15,
            initial_value=preproc["blur_kernel"],
            step=2,
            value_format="{:.0f}",
            on_change=self._on_blur_change,
        )
        settings_layout.add_widget(self.sliders["blur_kernel"])

        # Thresholds section
        settings_layout.add_widget(self._create_section_header("Thresholds"))

        self.sliders["pass_coverage"] = TuningSlider(
            label="Pass Coverage %",
            min_value=80,
            max_value=100,
            initial_value=validator["pass_coverage"] * 100,
            step=1,
            value_format="{:.0f}%",
            on_change=self._on_pass_coverage_change,
        )
        settings_layout.add_widget(self.sliders["pass_coverage"])

        self.sliders["warning_coverage"] = TuningSlider(
            label="Warning Coverage %",
            min_value=60,
            max_value=95,
            initial_value=validator["warning_coverage"] * 100,
            step=1,
            value_format="{:.0f}%",
            on_change=self._on_warning_coverage_change,
        )
        settings_layout.add_widget(self.sliders["warning_coverage"])

        self.sliders["pass_max_gaps"] = TuningSlider(
            label="Max Gaps (Pass)",
            min_value=0,
            max_value=5,
            initial_value=validator["pass_max_gaps"],
            step=1,
            value_format="{:.0f}",
            on_change=self._on_pass_gaps_change,
        )
        settings_layout.add_widget(self.sliders["pass_max_gaps"])

        self.sliders["min_confidence"] = TuningSlider(
            label="Min Confidence %",
            min_value=30,
            max_value=80,
            initial_value=validator["min_confidence"] * 100,
            step=5,
            value_format="{:.0f}%",
            on_change=self._on_min_confidence_change,
        )
        settings_layout.add_widget(self.sliders["min_confidence"])

        # Morphology section
        settings_layout.add_widget(self._create_section_header("Morphology"))

        self.sliders["close_kernel"] = TuningSlider(
            label="Close Kernel",
            min_value=3,
            max_value=9,
            initial_value=morph["close_kernel"],
            step=2,
            value_format="{:.0f}",
            on_change=self._on_close_kernel_change,
        )
        settings_layout.add_widget(self.sliders["close_kernel"])

        self.sliders["open_kernel"] = TuningSlider(
            label="Open Kernel",
            min_value=3,
            max_value=7,
            initial_value=morph["open_kernel"],
            step=2,
            value_format="{:.0f}",
            on_change=self._on_open_kernel_change,
        )
        settings_layout.add_widget(self.sliders["open_kernel"])

        scroll_view.add_widget(settings_layout)
        self.add_widget(scroll_view)

    def _create_section_header(self, text: str) -> Label:
        """Create a section header label."""
        label = Label(
            text=text,
            font_size="16sp",
            bold=True,
            color=(0.4, 0.7, 1.0, 1),
            size_hint_y=None,
            height=40,
            halign="left",
            valign="bottom",
        )
        label.bind(size=label.setter("text_size"))
        return label

    # Preprocessing callbacks
    def _on_clahe_clip_change(self, value: float):
        logger.info(f"CLAHE clip limit: {value:.1f}")
        self.detector.preprocessor.set_clahe_clip_limit(value)

    def _on_blur_change(self, value: float):
        size = int(value)
        logger.info(f"Blur kernel: {size}")
        self.detector.preprocessor.set_blur_kernel(size)

    # Threshold callbacks
    def _on_pass_coverage_change(self, value: float):
        coverage = value / 100.0
        logger.info(f"Pass coverage: {coverage:.2f}")
        self.detector.validator.set_pass_coverage(coverage)

    def _on_warning_coverage_change(self, value: float):
        coverage = value / 100.0
        logger.info(f"Warning coverage: {coverage:.2f}")
        self.detector.validator.set_warning_coverage(coverage)

    def _on_pass_gaps_change(self, value: float):
        gaps = int(value)
        logger.info(f"Pass max gaps: {gaps}")
        self.detector.validator.set_pass_max_gaps(gaps)

    def _on_min_confidence_change(self, value: float):
        confidence = value / 100.0
        logger.info(f"Min confidence: {confidence:.2f}")
        self.detector.validator.set_min_confidence(confidence)

    # Morphology callbacks
    def _on_close_kernel_change(self, value: float):
        size = int(value)
        logger.info(f"Close kernel: {size}")
        self.detector.set_close_kernel(size)

    def _on_open_kernel_change(self, value: float):
        size = int(value)
        logger.info(f"Open kernel: {size}")
        self.detector.set_open_kernel(size)

    def _on_reset(self, instance):
        """Reset all settings to defaults."""
        logger.info("Resetting tuning to defaults")

        # Reset preprocessor
        self.detector.preprocessor.set_clahe_clip_limit(2.0)
        self.detector.preprocessor.set_blur_kernel(5)

        # Reset thresholds
        self.detector.validator.set_pass_coverage(0.95)
        self.detector.validator.set_warning_coverage(0.80)
        self.detector.validator.set_pass_max_gaps(0)
        self.detector.validator.set_min_confidence(0.5)

        # Reset morphology
        self.detector.set_close_kernel(5)
        self.detector.set_open_kernel(3)

        # Update sliders
        self.sliders["clahe_clip"].set_value(2.0)
        self.sliders["blur_kernel"].set_value(5)
        self.sliders["pass_coverage"].set_value(95)
        self.sliders["warning_coverage"].set_value(80)
        self.sliders["pass_max_gaps"].set_value(0)
        self.sliders["min_confidence"].set_value(50)
        self.sliders["close_kernel"].set_value(5)
        self.sliders["open_kernel"].set_value(3)

    def _on_close(self, instance):
        """Handle close button press."""
        if self.on_close:
            self.on_close()
