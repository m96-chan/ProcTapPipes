"""Whisper speech-to-text transcription pipe.

This module provides multiple Whisper implementations:
- WhisperPipe: Using faster-whisper (local, faster inference)
- OpenAIWhisperPipe: Using OpenAI API
"""

import io
import wave
from typing import Any, Optional
import logging

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import BasePipe, AudioFormat

logger = logging.getLogger(__name__)


class WhisperPipe(BasePipe):
    """Faster-Whisper-based speech-to-text transcription pipe.

    Uses faster-whisper for efficient local transcription with CTranslate2.
    This is the recommended implementation for production use.

    Example:
        pipe = WhisperPipe(model="base", language="en")
        for transcription in pipe.run_stream(audio_stream):
            print(transcription)
    """

    def __init__(
        self,
        model: str = "base",
        language: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        device: str = "auto",
        compute_type: str = "default",
        buffer_duration: float = 5.0,
        vad_filter: bool = True,
        beam_size: int = 5,
    ):
        """Initialize Faster-Whisper transcription pipe.

        Args:
            model: Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
            language: Language code (e.g., 'en', 'ja', 'es'). None for auto-detect.
            audio_format: Audio format configuration
            device: Device to use ('cpu', 'cuda', 'auto')
            compute_type: Compute type for inference ('default', 'int8', 'int8_float16', 'int16', 'float16')
            buffer_duration: Duration in seconds to buffer before transcribing
            vad_filter: Enable voice activity detection filter
            beam_size: Beam size for beam search decoding
        """
        super().__init__(audio_format)
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.buffer_duration = buffer_duration
        self.vad_filter = vad_filter
        self.beam_size = beam_size

        # Calculate buffer size in samples
        self.buffer_size = int(self.audio_format.sample_rate * buffer_duration)
        self.buffer: list[npt.NDArray[Any]] = []
        self.buffer_samples = 0

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            self.logger.info(f"Loading faster-whisper model: {self.model_name}")

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

            self.logger.info(f"Faster-whisper model loaded successfully")
        except ImportError:
            raise ImportError(
                "faster-whisper package required. "
                "Install with: pip install faster-whisper"
            )

    def _prepare_audio(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Prepare audio data for transcription.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Mono float32 audio normalized to [-1, 1]
        """
        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_data.flatten()

        # Convert to float32 and normalize to [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0

        return audio_float

    def _transcribe(self, audio_data: npt.NDArray[Any]) -> str:
        """Transcribe audio using faster-whisper.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Transcribed text
        """
        audio_float = self._prepare_audio(audio_data)

        # Transcribe
        segments, info = self.model.transcribe(
            audio_float,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        # Combine all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        result = " ".join(text_parts).strip()

        if info.language_probability:
            self.logger.debug(
                f"Detected language: {info.language} "
                f"(probability: {info.language_probability:.2f})"
            )

        return result

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> Optional[str]:
        """Process audio chunk and return transcription when buffer is full.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            Transcribed text when buffer is full, None otherwise
        """
        # Add to buffer
        self.buffer.append(audio_data)
        self.buffer_samples += len(audio_data)

        # Check if buffer is full
        if self.buffer_samples >= self.buffer_size:
            # Concatenate buffer
            full_buffer = np.vstack(self.buffer)

            # Transcribe
            try:
                text = self._transcribe(full_buffer)

                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0

                if text:
                    return text
            except Exception as e:
                self.logger.error(f"Transcription failed: {e}", exc_info=True)
                # Reset buffer on error
                self.buffer = []
                self.buffer_samples = 0

        return None

    def flush(self) -> Optional[str]:
        """Transcribe any remaining audio in buffer.

        Returns:
            Transcribed text if buffer is not empty, None otherwise
        """
        if not self.buffer:
            return None

        full_buffer = np.vstack(self.buffer)

        try:
            text = self._transcribe(full_buffer)

            # Reset buffer
            self.buffer = []
            self.buffer_samples = 0

            if text:
                return text
        except Exception as e:
            self.logger.error(f"Transcription failed during flush: {e}", exc_info=True)
            self.buffer = []
            self.buffer_samples = 0

        return None


class OpenAIWhisperPipe(BasePipe):
    """OpenAI API-based Whisper transcription pipe.

    Uses OpenAI's Whisper API for transcription. Requires API key and internet connection.

    Example:
        pipe = OpenAIWhisperPipe(api_key="sk-...", model="whisper-1")
        for transcription in pipe.run_stream(audio_stream):
            print(transcription)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        buffer_duration: float = 5.0,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize OpenAI Whisper API pipe.

        Args:
            api_key: OpenAI API key
            model: Model name (currently only "whisper-1" is available)
            language: Language code (e.g., 'en', 'ja'). None for auto-detect.
            audio_format: Audio format configuration
            buffer_duration: Duration in seconds to buffer before transcribing
            prompt: Optional text to guide the model's style
            temperature: Sampling temperature (0 to 1)
        """
        super().__init__(audio_format)
        self.api_key = api_key
        self.model_name = model
        self.language = language
        self.buffer_duration = buffer_duration
        self.prompt = prompt
        self.temperature = temperature

        # Calculate buffer size in samples
        self.buffer_size = int(self.audio_format.sample_rate * buffer_duration)
        self.buffer: list[npt.NDArray[Any]] = []
        self.buffer_samples = 0

        # Initialize API client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI Whisper API client with model {self.model_name}")
        except ImportError:
            raise ImportError(
                "openai package required for API usage. "
                "Install with: pip install openai"
            )

    def _buffer_to_wav(self, audio_data: npt.NDArray[Any]) -> bytes:
        """Convert audio buffer to WAV bytes.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            WAV file as bytes
        """
        buffer = io.BytesIO()

        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1).astype(self.audio_format.dtype)
        else:
            audio_data = audio_data.flatten()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(self.audio_format.sample_width)
            wav_file.setframerate(self.audio_format.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return buffer.getvalue()

    def _transcribe(self, audio_data: npt.NDArray[Any]) -> str:
        """Transcribe using OpenAI API.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Transcribed text
        """
        wav_bytes = self._buffer_to_wav(audio_data)

        # Create a file-like object
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"

        # Prepare API call parameters
        params = {
            "model": self.model_name,
            "file": audio_file,
        }

        if self.language:
            params["language"] = self.language

        if self.prompt:
            params["prompt"] = self.prompt

        if self.temperature != 0.0:
            params["temperature"] = self.temperature

        # Call API
        transcript = self.client.audio.transcriptions.create(**params)

        return transcript.text.strip()

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> Optional[str]:
        """Process audio chunk and return transcription when buffer is full.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            Transcribed text when buffer is full, None otherwise
        """
        # Add to buffer
        self.buffer.append(audio_data)
        self.buffer_samples += len(audio_data)

        # Check if buffer is full
        if self.buffer_samples >= self.buffer_size:
            # Concatenate buffer
            full_buffer = np.vstack(self.buffer)

            # Transcribe
            try:
                text = self._transcribe(full_buffer)

                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0

                if text:
                    return text
            except Exception as e:
                self.logger.error(f"API transcription failed: {e}", exc_info=True)
                # Reset buffer on error
                self.buffer = []
                self.buffer_samples = 0

        return None

    def flush(self) -> Optional[str]:
        """Transcribe any remaining audio in buffer.

        Returns:
            Transcribed text if buffer is not empty, None otherwise
        """
        if not self.buffer:
            return None

        full_buffer = np.vstack(self.buffer)

        try:
            text = self._transcribe(full_buffer)

            # Reset buffer
            self.buffer = []
            self.buffer_samples = 0

            if text:
                return text
        except Exception as e:
            self.logger.error(f"API transcription failed during flush: {e}", exc_info=True)
            self.buffer = []
            self.buffer_samples = 0

        return None
