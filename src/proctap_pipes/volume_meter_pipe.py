"""Volume meter pipe for real-time audio level visualization.

This pipe displays a real-time volume meter in the terminal while passing
through audio data unchanged, allowing it to be used in the middle of pipelines.
"""

import sys
from typing import Any

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import AudioFormat, BasePipe


class VolumeMeterPipe(BasePipe):
    """Real-time volume meter pipe with passthrough.

    Displays audio levels in the terminal while passing PCM data through unchanged.
    This allows monitoring audio levels in the middle of processing pipelines.

    Example:
        # CLI usage
        proctap -pid 1234 --stdout | proctap-volume-meter | proctap-whisper

        # Python API usage
        pipe = VolumeMeterPipe(bar_width=50, update_interval=0.1)
        for chunk in pipe.run_stream(audio_stream):
            # chunk is the original audio data, unchanged
            process_audio(chunk)
    """

    def __init__(
        self,
        audio_format: AudioFormat | None = None,
        bar_width: int = 50,
        update_interval: float = 0.05,
        peak_hold_time: float = 1.0,
        show_db: bool = True,
        show_rms: bool = True,
        show_peak: bool = True,
        db_range: tuple[float, float] = (-60.0, 0.0),
    ):
        """Initialize volume meter pipe.

        Args:
            audio_format: Audio format configuration
            bar_width: Width of the volume bar in characters
            update_interval: Minimum time between display updates (seconds)
            peak_hold_time: How long to hold peak indicators (seconds)
            show_db: Show dB value
            show_rms: Show RMS level bar
            show_peak: Show peak level bar
            db_range: dB range for display (min, max)
        """
        super().__init__(audio_format)
        self.bar_width = bar_width
        self.update_interval = update_interval
        self.peak_hold_time = peak_hold_time
        self.show_db = show_db
        self.show_rms = show_rms
        self.show_peak = show_peak
        self.db_min, self.db_max = db_range

        # State tracking
        self.last_update_time = 0.0
        self.peak_value = 0.0
        self.peak_timestamp = 0.0
        self.chunk_count = 0

        # For timing
        import time

        self.time = time

    def _calculate_rms(self, audio_data: npt.NDArray[Any]) -> float:
        """Calculate RMS (Root Mean Square) value.

        Args:
            audio_data: Audio samples

        Returns:
            RMS value (0.0 to 1.0)
        """
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            samples = audio_data.astype(np.float32) / 32768.0
        else:
            samples = audio_data.astype(np.float32)

        # Calculate RMS
        rms = np.sqrt(np.mean(samples**2))
        return float(rms)

    def _calculate_peak(self, audio_data: npt.NDArray[Any]) -> float:
        """Calculate peak value.

        Args:
            audio_data: Audio samples

        Returns:
            Peak value (0.0 to 1.0)
        """
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            samples = audio_data.astype(np.float32) / 32768.0
        else:
            samples = audio_data.astype(np.float32)

        # Calculate peak
        peak = np.max(np.abs(samples))
        return float(peak)

    def _amplitude_to_db(self, amplitude: float) -> float:
        """Convert amplitude to dB.

        Args:
            amplitude: Amplitude value (0.0 to 1.0)

        Returns:
            dB value
        """
        if amplitude < 1e-10:  # Avoid log(0)
            return self.db_min
        db = 20 * np.log10(amplitude)
        return max(self.db_min, min(self.db_max, float(db)))

    def _create_bar(self, value: float, char: str = "█") -> str:
        """Create a visual bar representation.

        Args:
            value: Value from 0.0 to 1.0
            char: Character to use for the bar

        Returns:
            Bar string
        """
        # Map value to dB scale
        db = self._amplitude_to_db(value)

        # Normalize to bar width (0.0 to 1.0)
        normalized = (db - self.db_min) / (self.db_max - self.db_min)
        normalized = max(0.0, min(1.0, normalized))

        filled = int(normalized * self.bar_width)
        empty = self.bar_width - filled

        # Color coding based on level
        if normalized > 0.9:  # Red zone (clipping risk)
            bar_color = "\033[91m"  # Red
        elif normalized > 0.7:  # Yellow zone (loud)
            bar_color = "\033[93m"  # Yellow
        else:  # Green zone (normal)
            bar_color = "\033[92m"  # Green

        reset_color = "\033[0m"

        bar = f"{bar_color}{char * filled}{reset_color}{'·' * empty}"
        return bar

    def _display_meter(self, rms: float, peak: float) -> None:
        """Display the volume meter.

        Args:
            rms: RMS value
            peak: Peak value
        """
        lines = []

        if self.show_rms:
            rms_bar = self._create_bar(rms, "█")
            rms_db = self._amplitude_to_db(rms)
            if self.show_db:
                lines.append(f"RMS  [{rms_bar}] {rms_db:>6.1f} dB")
            else:
                lines.append(f"RMS  [{rms_bar}]")

        if self.show_peak:
            peak_bar = self._create_bar(peak, "▌")
            peak_db = self._amplitude_to_db(peak)
            if self.show_db:
                lines.append(f"Peak [{peak_bar}] {peak_db:>6.1f} dB")
            else:
                lines.append(f"Peak [{peak_bar}]")

        # Clear previous lines and write new ones
        if lines:
            # Move cursor up to overwrite previous output
            if self.chunk_count > 0:
                sys.stderr.write(f"\033[{len(lines)}A")

            for line in lines:
                sys.stderr.write(f"\r{line}\033[K\n")

            sys.stderr.flush()

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[Any] | None:
        """Process audio chunk and display volume meter.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            The same audio data unchanged (passthrough)
        """
        current_time = self.time.time()

        # Calculate levels
        rms = self._calculate_rms(audio_data)
        peak = self._calculate_peak(audio_data)

        # Update peak hold
        if peak > self.peak_value or (current_time - self.peak_timestamp) > self.peak_hold_time:
            self.peak_value = peak
            self.peak_timestamp = current_time

        # Update display at specified interval
        if (current_time - self.last_update_time) >= self.update_interval:
            self._display_meter(rms, self.peak_value)
            self.last_update_time = current_time
            self.chunk_count += 1

        # Return original audio data unchanged (passthrough)
        return audio_data

    def flush(self) -> npt.NDArray[Any] | None:
        """Flush any remaining data.

        Returns:
            None (no buffering in this pipe)
        """
        # Add a newline after the final meter display
        if self.chunk_count > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return None
