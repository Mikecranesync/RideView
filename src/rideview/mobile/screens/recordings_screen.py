"""
Recordings screen for RideView.

Lists recorded videos with metadata, allows playback, export, and deletion.
"""

import logging
import os
import platform as sys_platform
import subprocess
from typing import Callable

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView

from ..recording_manager import RecordingManager

logger = logging.getLogger(__name__)


class RecordingItem(BoxLayout):
    """Single recording item in the list."""

    def __init__(
        self,
        recording: dict,
        on_play: Callable[[str], None] | None = None,
        on_delete: Callable[[str], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("orientation", "horizontal")
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", 80)
        kwargs.setdefault("padding", [10, 5, 10, 5])
        kwargs.setdefault("spacing", 10)
        super().__init__(**kwargs)

        self.recording = recording
        self.on_play = on_play
        self.on_delete = on_delete

        # Info section
        info_layout = BoxLayout(orientation="vertical", size_hint=(0.6, 1))

        # Filename
        filename_label = Label(
            text=recording.get("filename", "Unknown"),
            font_size="14sp",
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=0.5,
        )
        filename_label.bind(size=filename_label.setter("text_size"))
        info_layout.add_widget(filename_label)

        # Details
        duration = recording.get("duration_seconds", 0)
        size_mb = recording.get("size_mb", 0)
        summary = recording.get("summary", {})
        pass_count = summary.get("pass_count", 0)
        fail_count = summary.get("fail_count", 0)

        details_text = f"{duration:.0f}s | {size_mb:.1f}MB | PASS:{pass_count} FAIL:{fail_count}"
        details_label = Label(
            text=details_text,
            font_size="12sp",
            color=(0.7, 0.7, 0.7, 1),
            halign="left",
            valign="middle",
            size_hint_y=0.5,
        )
        details_label.bind(size=details_label.setter("text_size"))
        info_layout.add_widget(details_label)

        self.add_widget(info_layout)

        # Buttons section
        buttons_layout = BoxLayout(orientation="horizontal", size_hint=(0.4, 1), spacing=5)

        # Play button
        play_btn = Button(text="Play", size_hint=(0.5, 1), font_size="12sp")
        play_btn.bind(on_press=self._on_play)
        buttons_layout.add_widget(play_btn)

        # Delete button
        delete_btn = Button(
            text="Delete",
            size_hint=(0.5, 1),
            font_size="12sp",
            background_color=(0.8, 0.2, 0.2, 1),
        )
        delete_btn.bind(on_press=self._on_delete)
        buttons_layout.add_widget(delete_btn)

        self.add_widget(buttons_layout)

    def _on_play(self, instance):
        """Handle play button press."""
        if self.on_play:
            self.on_play(self.recording.get("filepath", ""))

    def _on_delete(self, instance):
        """Handle delete button press."""
        if self.on_delete:
            self.on_delete(self.recording.get("filepath", ""))


class RecordingsScreen(BoxLayout):
    """
    Screen showing list of recorded videos.

    Features:
    - List recordings sorted by date (newest first)
    - Show duration, size, pass/fail counts
    - Play recordings in native player
    - Delete recordings
    """

    def __init__(
        self,
        recording_manager: RecordingManager,
        on_close: Callable[[], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding", [10, 10, 10, 10])
        kwargs.setdefault("spacing", 10)
        super().__init__(**kwargs)

        self.recording_manager = recording_manager
        self.on_close = on_close

        self._create_ui()
        self.refresh_list()

    def _create_ui(self):
        """Create the UI components."""
        # Header
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=50)

        title = Label(
            text="Recordings",
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

        # Recordings list (scrollable)
        scroll_view = ScrollView(size_hint=(1, 1))
        self.list_layout = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=5,
        )
        self.list_layout.bind(minimum_height=self.list_layout.setter("height"))
        scroll_view.add_widget(self.list_layout)
        self.add_widget(scroll_view)

        # Footer with refresh button
        footer = BoxLayout(orientation="horizontal", size_hint_y=None, height=50)
        refresh_btn = Button(text="Refresh", size_hint=(1, 1), font_size="14sp")
        refresh_btn.bind(on_press=lambda x: self.refresh_list())
        footer.add_widget(refresh_btn)
        self.add_widget(footer)

    def refresh_list(self):
        """Refresh the recordings list."""
        self.list_layout.clear_widgets()

        recordings = self.recording_manager.get_recordings()

        if not recordings:
            no_recordings = Label(
                text="No recordings found.\nUse the REC button to record training data.",
                font_size="14sp",
                halign="center",
                valign="middle",
                size_hint_y=None,
                height=100,
            )
            no_recordings.bind(size=no_recordings.setter("text_size"))
            self.list_layout.add_widget(no_recordings)
            return

        for recording in recordings:
            item = RecordingItem(
                recording=recording,
                on_play=self._play_recording,
                on_delete=self._confirm_delete,
            )
            self.list_layout.add_widget(item)

        logger.info(f"Loaded {len(recordings)} recordings")

    def _play_recording(self, filepath: str):
        """Play a recording in the native video player."""
        if not os.path.exists(filepath):
            logger.error(f"Recording not found: {filepath}")
            return

        system = sys_platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["open", filepath], check=True)
            elif system == "Windows":
                os.startfile(filepath)
            else:  # Linux
                subprocess.run(["xdg-open", filepath], check=True)
            logger.info(f"Playing recording: {filepath}")
        except Exception as e:
            logger.error(f"Failed to play recording: {e}")

    def _confirm_delete(self, filepath: str):
        """Show confirmation dialog before deleting."""
        content = BoxLayout(orientation="vertical", padding=[20, 10, 20, 10], spacing=10)

        content.add_widget(
            Label(
                text=f"Delete this recording?\n{os.path.basename(filepath)}",
                font_size="14sp",
                halign="center",
                valign="middle",
                size_hint_y=0.6,
            )
        )

        buttons = BoxLayout(orientation="horizontal", size_hint_y=0.4, spacing=10)

        cancel_btn = Button(text="Cancel", font_size="14sp")
        delete_btn = Button(
            text="Delete",
            font_size="14sp",
            background_color=(0.8, 0.2, 0.2, 1),
        )

        buttons.add_widget(cancel_btn)
        buttons.add_widget(delete_btn)
        content.add_widget(buttons)

        popup = Popup(
            title="Confirm Delete",
            content=content,
            size_hint=(0.8, 0.4),
            auto_dismiss=False,
        )

        cancel_btn.bind(on_press=popup.dismiss)
        delete_btn.bind(
            on_press=lambda x: self._delete_recording(filepath, popup)
        )

        popup.open()

    def _delete_recording(self, filepath: str, popup: Popup):
        """Delete a recording and refresh list."""
        popup.dismiss()

        if self.recording_manager.delete_recording(filepath):
            logger.info(f"Deleted recording: {filepath}")
            self.refresh_list()
        else:
            logger.error(f"Failed to delete recording: {filepath}")

    def _on_close(self, instance):
        """Handle close button press."""
        if self.on_close:
            self.on_close()
