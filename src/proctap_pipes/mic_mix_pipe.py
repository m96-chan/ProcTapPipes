"""Microphone mix pipe for mixing mic input with ProcTap audio.

This pipe captures microphone input and mixes it with incoming ProcTap audio,
allowing you to combine process audio with your own voice for streaming,
recording, or further processing.
"""

import platform
from typing import Any

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import AudioFormat, BasePipe


class MicMixPipe(BasePipe):
    """Microphone mixer pipe that combines mic input with ProcTap audio.

    Captures system microphone input and mixes it with incoming audio stream
    using -6dB gain on both signals to prevent clipping. Supports platform-specific
    audio capture (WASAPI on Windows, CoreAudio on macOS, PulseAudio/PipeWire on Linux).

    Example:
        # CLI usage
        proctap -pid 1234 --stdout | proctap-mic-mix --gain 0.8 | proctap-whisper

        # Python API usage
        pipe = MicMixPipe(gain=0.8, mic_device="USB Microphone")
        for mixed_chunk in pipe.run_stream(audio_stream):
            # mixed_chunk contains both ProcTap and mic audio
            process_audio(mixed_chunk)
    """

    def __init__(
        self,
        audio_format: AudioFormat | None = None,
        gain: float = 1.0,
        mic_device: str | int | None = None,
        mic_sample_rate: int = 48000,
        mic_channels: int = 2,
        enable_mic: bool = True,
    ):
        """Initialize microphone mixer pipe.

        Args:
            audio_format: Audio format for output (default: 48kHz stereo 16-bit PCM)
            gain: Gain multiplier for microphone input (0.0-2.0, default: 1.0)
            mic_device: Microphone device name (str) or index (int), None = system default
            mic_sample_rate: Microphone sample rate (default: 48000)
            mic_channels: Microphone channel count (default: 2 for stereo)
            enable_mic: Enable microphone capture (default: True)
        """
        # Default to ProcTap standard format if not specified
        if audio_format is None:
            audio_format = AudioFormat(sample_rate=48000, channels=2, sample_width=2)

        super().__init__(audio_format)

        # Clamp gain to reasonable range
        self.gain = max(0.0, min(2.0, gain))
        self.mic_device_name = mic_device
        self.mic_sample_rate = mic_sample_rate
        self.mic_channels = mic_channels
        self.enable_mic = enable_mic

        # Microphone device handle
        self.mic_device: Any = None
        self.mic_buffer: npt.NDArray[Any] = np.array([], dtype=np.int16).reshape(0, 1)

        # Initialize microphone capture if enabled
        if self.enable_mic:
            try:
                self._init_mic_capture()
            except Exception as e:
                self.logger.warning(f"Failed to initialize microphone: {e}")
                self.logger.warning("Continuing in passthrough mode (no mic input)")
                self.enable_mic = False

    def _init_mic_capture(self) -> None:
        """Initialize platform-specific microphone capture.

        Raises:
            RuntimeError: If microphone initialization fails
        """
        system = platform.system()

        try:
            import sounddevice as sd

            # Get default input device if none specified
            if self.mic_device_name is None:
                device_info = sd.query_devices(kind="input")
                self.logger.info(f"Using default microphone: {device_info['name']}")
            else:
                # Support both device name (str) and device index (int)
                if isinstance(self.mic_device_name, int):
                    device_info = sd.query_devices(self.mic_device_name)
                    self.logger.info(
                        f"Using microphone [{self.mic_device_name}]: {device_info['name']}"
                    )
                else:
                    self.logger.info(f"Using microphone: {self.mic_device_name}")

            # Create audio stream
            self.mic_device = sd.InputStream(
                device=self.mic_device_name,
                samplerate=self.mic_sample_rate,
                channels=self.mic_channels,
                dtype=np.int16,
                blocksize=1024,
            )

            # Start the stream
            self.mic_device.start()
            self.logger.info(
                f"Microphone initialized: {self.mic_sample_rate}Hz, "
                f"{self.mic_channels}ch on {system}"
            )

        except ImportError:
            raise RuntimeError(
                "sounddevice library not installed. " "Install with: pip install sounddevice"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize microphone on {system}: {e}")

    def _read_mic_chunk(self, num_samples: int) -> npt.NDArray[Any]:
        """Read audio chunk from microphone.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio data with shape (samples, channels)

        Raises:
            RuntimeError: If microphone read fails
        """
        if not self.enable_mic or self.mic_device is None:
            # Return silence if mic disabled
            return np.zeros((num_samples, self.audio_format.channels), dtype=np.int16)

        try:
            # Read from sounddevice stream
            audio_data, overflowed = self.mic_device.read(num_samples)

            if overflowed:
                self.logger.warning("Microphone buffer overflow detected")

            # Ensure correct shape
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)

            return audio_data

        except Exception as e:
            self.logger.error(f"Failed to read from microphone: {e}")
            # Return silence on error
            return np.zeros((num_samples, self.audio_format.channels), dtype=np.int16)

    def _resample_mic_audio(self, mic_audio: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Resample microphone audio to match ProcTap format.

        Args:
            mic_audio: Microphone audio at mic_sample_rate

        Returns:
            Resampled audio at audio_format.sample_rate
        """
        if self.mic_sample_rate == self.audio_format.sample_rate:
            return mic_audio

        # Simple linear resampling
        num_samples = len(mic_audio)
        target_samples = int(num_samples * self.audio_format.sample_rate / self.mic_sample_rate)

        # Create index array for interpolation
        indices = np.linspace(0, num_samples - 1, target_samples)

        # Resample each channel
        resampled_channels = []
        for ch in range(mic_audio.shape[1]):
            resampled = np.interp(indices, np.arange(num_samples), mic_audio[:, ch])
            resampled_channels.append(resampled)

        result = np.column_stack(resampled_channels).astype(np.int16)
        return result

    def _convert_to_mono(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Convert stereo or multi-channel audio to mono.

        Args:
            audio_data: Audio with shape (samples, channels)

        Returns:
            Mono audio with shape (samples, 1)
        """
        if audio_data.shape[1] == 1:
            return audio_data

        # Average all channels
        mono = np.mean(audio_data, axis=1, dtype=np.float32).astype(np.int16)
        return mono.reshape(-1, 1)

    def _mix_audio(
        self, proctap_audio: npt.NDArray[Any], mic_audio: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Mix ProcTap audio with microphone input.

        Uses -6dB gain (0.5 multiplier) on both signals to prevent clipping.
        Applies user-specified gain to mic input before mixing.

        Args:
            proctap_audio: Audio from ProcTap, shape (samples, channels)
            mic_audio: Audio from microphone, shape (samples, channels)

        Returns:
            Mixed audio with same shape as proctap_audio
        """
        # Ensure same length
        min_length = min(len(proctap_audio), len(mic_audio))
        proctap_audio = proctap_audio[:min_length]
        mic_audio = mic_audio[:min_length]

        # Convert to float for mixing
        proctap_float = proctap_audio.astype(np.float32)
        mic_float = mic_audio.astype(np.float32)

        # Apply user gain to mic input
        mic_float *= self.gain

        # Mix with -6dB on both signals to prevent clipping
        mixed = (proctap_float + mic_float) * 0.5

        # Clip and convert back to int16
        mixed = np.clip(mixed, -32768, 32767)
        return mixed.astype(np.int16)

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Process audio chunk by mixing with microphone input.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            Mixed audio with microphone input
        """
        # Passthrough mode if mic disabled
        if not self.enable_mic:
            return audio_data

        try:
            # Read matching amount of mic audio
            num_samples = len(audio_data)

            # Calculate how many samples we need from mic (accounting for sample rate difference)
            mic_samples_needed = int(
                num_samples * self.mic_sample_rate / self.audio_format.sample_rate
            )

            # Read from mic
            mic_audio = self._read_mic_chunk(mic_samples_needed)

            # Resample mic audio if needed
            if self.mic_sample_rate != self.audio_format.sample_rate:
                mic_audio = self._resample_mic_audio(mic_audio)

            # Convert to mono if needed
            if self.mic_channels != self.audio_format.channels:
                if self.audio_format.channels == 1:
                    mic_audio = self._convert_to_mono(mic_audio)

            # Ensure mic audio matches ProcTap audio length
            if len(mic_audio) != len(audio_data):
                # Pad or truncate
                if len(mic_audio) < len(audio_data):
                    padding = np.zeros(
                        (len(audio_data) - len(mic_audio), self.audio_format.channels),
                        dtype=np.int16,
                    )
                    mic_audio = np.vstack([mic_audio, padding])
                else:
                    mic_audio = mic_audio[: len(audio_data)]

            # Mix the audio
            mixed_audio = self._mix_audio(audio_data, mic_audio)

            return mixed_audio

        except Exception as e:
            self.logger.error(f"Error mixing audio: {e}")
            # Return original audio on error (passthrough)
            return audio_data

    def flush(self) -> npt.NDArray[Any] | None:
        """Flush any remaining data and close microphone device.

        Returns:
            None (no buffered data)
        """
        if self.mic_device is not None:
            try:
                self.mic_device.stop()
                self.mic_device.close()
                self.logger.info("Microphone device closed")
            except Exception as e:
                self.logger.warning(f"Error closing microphone: {e}")

        return None
