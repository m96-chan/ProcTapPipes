"""Whisper speech-to-text transcription pipe."""

import io
import wave
from typing import Any, Optional
import logging

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import BasePipe, AudioFormat

logger = logging.getLogger(__name__)


class WhisperPipe(BasePipe):
    """Whisper-based speech-to-text transcription pipe.

    Transcribes audio chunks using OpenAI's Whisper model (local or API).
    """

    def __init__(
        self,
        model: str = "base",
        language: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        buffer_duration: float = 5.0,
    ):
        """Initialize Whisper transcription pipe.

        Args:
            model: Whisper model name (tiny, base, small, medium, large) or
                   OpenAI API model (whisper-1)
            language: Language code for transcription (e.g., 'en', 'es')
            audio_format: Audio format configuration
            use_api: Whether to use OpenAI API instead of local model
            api_key: OpenAI API key (required if use_api=True)
            buffer_duration: Duration in seconds to buffer before transcribing
        """
        super().__init__(audio_format)
        self.model_name = model
        self.language = language
        self.use_api = use_api
        self.api_key = api_key
        self.buffer_duration = buffer_duration

        # Calculate buffer size in samples
        self.buffer_size = int(self.audio_format.sample_rate * buffer_duration)
        self.buffer: list[npt.NDArray[Any]] = []
        self.buffer_samples = 0

        # Initialize model/client
        if use_api:
            self._init_api_client()
        else:
            self._init_local_model()

    def _init_api_client(self) -> None:
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("API key required when use_api=True")

            self.client = OpenAI(api_key=self.api_key)
            self.model = None
            self.logger.info(f"Initialized OpenAI Whisper API client with model {self.model_name}")
        except ImportError:
            raise ImportError("openai package required for API usage. Install with: pip install openai")

    def _init_local_model(self) -> None:
        """Initialize local Whisper model."""
        try:
            import whisper

            self.model = whisper.load_model(self.model_name)
            self.client = None
            self.logger.info(f"Loaded local Whisper model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "whisper package required for local models. "
                "Install with: pip install openai-whisper"
            )

    def _buffer_to_wav(self, audio_data: npt.NDArray[Any]) -> bytes:
        """Convert audio buffer to WAV bytes.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            WAV file as bytes
        """
        buffer = io.BytesIO()

        # Convert to mono if stereo (Whisper expects mono)
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

    def _transcribe_local(self, audio_data: npt.NDArray[Any]) -> str:
        """Transcribe using local Whisper model.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Transcribed text
        """
        # Convert to mono float32 in range [-1, 1]
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_data.flatten()

        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Transcribe
        result = self.model.transcribe(
            audio_float,
            language=self.language,
            fp16=False,
        )

        return result["text"].strip()

    def _transcribe_api(self, audio_data: npt.NDArray[Any]) -> str:
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

        # Call API
        transcript = self.client.audio.transcriptions.create(
            model=self.model_name,
            file=audio_file,
            language=self.language,
        )

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
                if self.use_api:
                    text = self._transcribe_api(full_buffer)
                else:
                    text = self._transcribe_local(full_buffer)

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
            if self.use_api:
                text = self._transcribe_api(full_buffer)
            else:
                text = self._transcribe_local(full_buffer)

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
