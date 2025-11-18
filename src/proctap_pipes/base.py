"""Base class for all ProcTapPipes processing modules."""

import sys
import struct
import wave
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Iterator, BinaryIO, Optional, Any
import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class AudioFormat:
    """Audio format configuration for ProcTap streams."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width: int = 2,  # 2 bytes = 16-bit
        dtype: npt.DTypeLike = np.int16,
    ):
        """Initialize audio format configuration.

        Args:
            sample_rate: Sample rate in Hz (default: 48000)
            channels: Number of audio channels (default: 2)
            sample_width: Bytes per sample (default: 2 for s16le)
            dtype: NumPy dtype for audio data (default: np.int16)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.dtype = dtype

    @property
    def frame_size(self) -> int:
        """Get size of one frame in bytes."""
        return self.sample_width * self.channels


class BasePipe(ABC):
    """Abstract base class for all ProcTapPipes processing modules.

    All pipes must implement process_chunk to transform audio data.
    This class provides standard methods for streaming processing and CLI integration.
    """

    def __init__(self, audio_format: Optional[AudioFormat] = None):
        """Initialize the pipe with audio format configuration.

        Args:
            audio_format: Audio format configuration, defaults to ProcTap standard
                         (48kHz, stereo, s16le)
        """
        self.audio_format = audio_format or AudioFormat()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def process_chunk(self, audio_data: npt.NDArray[Any]) -> Any:
        """Process a single chunk of audio data.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            Processed output (type varies by pipe implementation)
        """
        pass

    def read_wav_header(self, stream: BinaryIO) -> Optional[AudioFormat]:
        """Attempt to read WAV header from stream.

        Args:
            stream: Input binary stream

        Returns:
            AudioFormat if WAV header found, None otherwise
        """
        try:
            # Try to read WAV header
            header = stream.read(44)
            if len(header) < 44:
                return None

            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                # Not a WAV file, put bytes back if possible
                if hasattr(stream, "seek"):
                    stream.seek(0)
                return None

            # Parse WAV header
            channels = struct.unpack("<H", header[22:24])[0]
            sample_rate = struct.unpack("<I", header[24:28])[0]
            bits_per_sample = struct.unpack("<H", header[34:36])[0]

            dtype_map = {8: np.uint8, 16: np.int16, 24: np.int32, 32: np.int32}
            dtype = dtype_map.get(bits_per_sample, np.int16)

            return AudioFormat(
                sample_rate=sample_rate,
                channels=channels,
                sample_width=bits_per_sample // 8,
                dtype=dtype,
            )
        except Exception as e:
            self.logger.debug(f"Failed to parse WAV header: {e}")
            return None

    def read_pcm_stream(
        self, stream: BinaryIO, chunk_size: int = 4096
    ) -> Iterator[npt.NDArray[Any]]:
        """Read PCM audio data from a binary stream.

        Automatically detects WAV format or assumes raw PCM (s16le).

        Args:
            stream: Input binary stream (e.g., sys.stdin.buffer)
            chunk_size: Number of frames to read per chunk

        Yields:
            NumPy arrays of audio data with shape (samples, channels)
        """
        # Try to detect WAV format
        fmt = self.read_wav_header(stream)
        if fmt:
            self.logger.info(
                f"Detected WAV format: {fmt.sample_rate}Hz, "
                f"{fmt.channels}ch, {fmt.sample_width * 8}bit"
            )
            self.audio_format = fmt
        else:
            self.logger.info(
                f"Using raw PCM format: {self.audio_format.sample_rate}Hz, "
                f"{self.audio_format.channels}ch, "
                f"{self.audio_format.sample_width * 8}bit"
            )

        bytes_per_chunk = chunk_size * self.audio_format.frame_size

        while True:
            data = stream.read(bytes_per_chunk)
            if not data:
                break

            # Convert bytes to numpy array
            try:
                samples = np.frombuffer(data, dtype=self.audio_format.dtype)
                # Reshape to (samples, channels)
                if self.audio_format.channels > 1:
                    # Truncate to multiple of channels
                    samples = samples[: len(samples) // self.audio_format.channels * self.audio_format.channels]
                    samples = samples.reshape(-1, self.audio_format.channels)
                else:
                    samples = samples.reshape(-1, 1)

                if len(samples) > 0:
                    yield samples
            except Exception as e:
                self.logger.error(f"Error decoding audio data: {e}")
                break

    def run_stream(self, stream: BinaryIO, chunk_size: int = 4096) -> Iterator[Any]:
        """Process audio from a stream, yielding results.

        Args:
            stream: Input binary stream
            chunk_size: Number of frames to read per chunk

        Yields:
            Processed output from process_chunk
        """
        for audio_chunk in self.read_pcm_stream(stream, chunk_size):
            try:
                result = self.process_chunk(audio_chunk)
                if result is not None:
                    yield result
            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}", exc_info=True)

    def run_cli(
        self,
        input_stream: Optional[BinaryIO] = None,
        output_stream: Optional[BinaryIO] = None,
        chunk_size: int = 4096,
    ) -> None:
        """Run the pipe in CLI mode (stdin -> stdout).

        Args:
            input_stream: Input stream (defaults to sys.stdin.buffer)
            output_stream: Output stream (defaults to sys.stdout)
            chunk_size: Number of frames to read per chunk
        """
        input_stream = input_stream or sys.stdin.buffer
        output_stream = output_stream or sys.stdout

        try:
            for result in self.run_stream(input_stream, chunk_size):
                if isinstance(result, str):
                    output_stream.write(result)
                    if not result.endswith("\n"):
                        output_stream.write("\n")
                elif isinstance(result, bytes):
                    # For binary output, write to buffer
                    if hasattr(output_stream, "buffer"):
                        output_stream.buffer.write(result)
                    else:
                        output_stream.write(result)
                else:
                    # Convert to string representation
                    output_stream.write(str(result))
                    output_stream.write("\n")

                output_stream.flush()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)

    def write_wav(
        self, audio_data: npt.NDArray[Any], output_stream: BinaryIO
    ) -> None:
        """Write audio data as WAV format to stream.

        Args:
            audio_data: NumPy array of audio samples
            output_stream: Output binary stream
        """
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.audio_format.channels)
            wav_file.setsampwidth(self.audio_format.sample_width)
            wav_file.setframerate(self.audio_format.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        output_stream.write(buffer.getvalue())
