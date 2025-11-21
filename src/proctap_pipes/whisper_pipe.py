"""Whisper speech-to-text transcription pipe.

This module provides multiple Whisper implementations:
- WhisperPipe: Using faster-whisper (local, faster inference)
- OpenAIWhisperPipe: Using OpenAI API
"""

import io
import logging
import wave
from typing import Any

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import AudioFormat, BasePipe

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

    # Common Whisper hallucination patterns to filter out
    HALLUCINATION_PATTERNS = [
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "Thank you for watching",
        "Thanks for watching",
        "ご清聴ありがとうございました",
        "字幕作成",
        "Amara.org",
        "字幕",
        "subtitles",
        ".",
        "..",
        "...",
        "thank you",
        "ありがとうございました",
        "ありがとうございます",
    ]

    def __init__(
        self,
        model: str = "base",
        language: str | None = None,
        audio_format: AudioFormat | None = None,
        device: str = "auto",
        compute_type: str = "default",
        buffer_duration: float = 5.0,
        vad_filter: bool = True,
        beam_size: int = 5,
        initial_prompt: str | None = None,
        silence_threshold: float = 0.01,
        skip_repetitions: bool = True,
        skip_hallucinations: bool = True,
    ):
        """Initialize Faster-Whisper transcription pipe.

        Args:
            model: Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
            language: Language code (e.g., 'en', 'ja', 'es'). None for auto-detect.
            audio_format: Audio format configuration
            device: Device to use ('cpu', 'cuda', 'auto')
            compute_type: Compute type for inference
                ('default', 'int8', 'int8_float16', 'int16', 'float16')
            buffer_duration: Duration in seconds to buffer before transcribing
            vad_filter: Enable voice activity detection filter
            beam_size: Beam size for beam search decoding
            initial_prompt: Optional text to guide the model's style and vocabulary
                (e.g., proper nouns)
            silence_threshold: RMS threshold below which audio is considered silence
                (0.0-1.0)
            skip_repetitions: Skip repeated transcriptions to avoid hallucination loops
            skip_hallucinations: Skip common hallucination phrases
        """
        super().__init__(audio_format)
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.buffer_duration = buffer_duration
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.silence_threshold = silence_threshold
        self.skip_repetitions = skip_repetitions
        self.skip_hallucinations = skip_hallucinations

        # Calculate buffer size in samples
        self.buffer_size = int(self.audio_format.sample_rate * buffer_duration)
        self.buffer: list[npt.NDArray[Any]] = []
        self.buffer_samples = 0

        # Track recent transcriptions for repetition detection
        self.recent_transcriptions: list[str] = []
        self.max_recent = 3  # Keep last 3 transcriptions

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

            self.logger.info("Faster-whisper model loaded successfully")
        except ImportError:
            raise ImportError(
                "faster-whisper package required. " "Install with: pip install faster-whisper"
            )

    def _resample(
        self, audio: npt.NDArray[np.float32], orig_sr: int, target_sr: int
    ) -> npt.NDArray[np.float32]:
        """Resample audio to target sample rate using linear interpolation.

        Args:
            audio: Audio samples (mono, normalized float32)
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        # Calculate the ratio and new length
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        # Use linear interpolation for resampling
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio).astype(np.float32)

        msg = f"Resampled from {orig_sr}Hz to {target_sr}Hz: "
        msg += f"{len(audio)} -> {len(resampled)} samples"
        self.logger.debug(msg)

        return resampled

    def _is_silence(self, audio_data: npt.NDArray[Any]) -> bool:
        """Check if audio data is silence or near-silence.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            True if audio is below silence threshold
        """
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Calculate RMS
        rms = np.sqrt(np.mean(audio_float**2))

        is_silent = rms < self.silence_threshold

        if is_silent:
            msg = f"Audio is silent (RMS: {rms:.4f} < "
            msg += f"threshold: {self.silence_threshold})"
            self.logger.debug(msg)

        return is_silent

    def _is_hallucination(self, text: str) -> bool:
        """Check if text matches known hallucination patterns.

        Args:
            text: Transcribed text

        Returns:
            True if text is likely a hallucination
        """
        if not self.skip_hallucinations:
            return False

        text_lower = text.lower().strip()

        # Check exact matches (case insensitive)
        for pattern in self.HALLUCINATION_PATTERNS:
            if text_lower == pattern.lower():
                self.logger.debug(f"Filtered hallucination (exact match): '{text}'")
                return True

        # Check if entire text is just the pattern repeated
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern.lower() in text_lower:
                # Count occurrences
                count = text_lower.count(pattern.lower())
                # If pattern appears multiple times and makes up most of the text
                if count > 1 and len(pattern) * count > len(text) * 0.7:
                    self.logger.debug(f"Filtered hallucination (repetition): '{text}'")
                    return True

        return False

    def _is_repetition(self, text: str) -> bool:
        """Check if text is a repetition of recent transcriptions.

        Args:
            text: Transcribed text

        Returns:
            True if text is a repetition
        """
        if not self.skip_repetitions:
            return False

        # Check if this exact text appeared in recent transcriptions
        if text in self.recent_transcriptions:
            self.logger.debug(f"Filtered repetition: '{text}'")
            return True

        return False

    def _update_recent_transcriptions(self, text: str) -> None:
        """Update the list of recent transcriptions.

        Args:
            text: New transcribed text
        """
        self.recent_transcriptions.append(text)
        # Keep only the most recent transcriptions
        if len(self.recent_transcriptions) > self.max_recent:
            self.recent_transcriptions.pop(0)

    def _prepare_audio(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Prepare audio data for transcription.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Mono float32 audio normalized to [-1, 1], resampled to 16kHz
        """
        # Debug: Log incoming audio format
        self.logger.debug(f"Input audio dtype: {audio_data.dtype}, shape: {audio_data.shape}")
        self.logger.debug(f"Input audio min: {audio_data.min()}, max: {audio_data.max()}")

        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_data.flatten()

        # Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            # int16: divide by 32768.0
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype in (np.float32, np.float64):
            # Float types: check if already normalized
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                # Unnormalized float, normalize it
                msg = f"Float audio appears unnormalized (max: {max_val}), normalizing..."
                self.logger.debug(msg)
                audio_float = (audio_data / max_val).astype(np.float32)
            else:
                # Already normalized
                audio_float = audio_data.astype(np.float32)
        else:
            # Unknown dtype, try to convert
            msg = f"Unexpected audio dtype: {audio_data.dtype}, attempting conversion..."
            self.logger.warning(msg)
            audio_float = audio_data.astype(np.float32)
            max_val = np.abs(audio_float).max()
            if max_val > 1.0:
                audio_float = audio_float / max_val

        # Resample to 16kHz (Whisper's native sample rate)
        if self.audio_format.sample_rate != 16000:
            audio_float = self._resample(audio_float, self.audio_format.sample_rate, 16000)

        msg = f"Output audio min: {audio_float.min():.4f}, "
        msg += f"max: {audio_float.max():.4f}"
        self.logger.debug(msg)

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
            initial_prompt=self.initial_prompt,
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

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> str | None:
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

            # Check if buffer is mostly silence
            if self._is_silence(full_buffer):
                self.logger.debug("Skipping transcription: audio is silence")
                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0
                return None

            # Transcribe
            try:
                text = self._transcribe(full_buffer)

                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0

                if text:
                    # Check for hallucinations
                    if self._is_hallucination(text):
                        self.logger.info(f"Skipped hallucination: '{text}'")
                        return None

                    # Check for repetitions
                    if self._is_repetition(text):
                        self.logger.info(f"Skipped repetition: '{text}'")
                        return None

                    # Update recent transcriptions
                    self._update_recent_transcriptions(text)

                    return text
            except Exception as e:
                self.logger.error(f"Transcription failed: {e}", exc_info=True)
                # Reset buffer on error
                self.buffer = []
                self.buffer_samples = 0

        return None

    def flush(self) -> str | None:
        """Transcribe any remaining audio in buffer.

        Returns:
            Transcribed text if buffer is not empty, None otherwise
        """
        if not self.buffer:
            return None

        full_buffer = np.vstack(self.buffer)

        # Check if buffer is mostly silence
        if self._is_silence(full_buffer):
            self.logger.debug("Skipping flush transcription: audio is silence")
            self.buffer = []
            self.buffer_samples = 0
            return None

        try:
            text = self._transcribe(full_buffer)

            # Reset buffer
            self.buffer = []
            self.buffer_samples = 0

            if text:
                # Check for hallucinations
                if self._is_hallucination(text):
                    self.logger.info(f"Skipped hallucination during flush: '{text}'")
                    return None

                # Check for repetitions
                if self._is_repetition(text):
                    self.logger.info(f"Skipped repetition during flush: '{text}'")
                    return None

                # Update recent transcriptions
                self._update_recent_transcriptions(text)

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

    # Use same hallucination patterns as WhisperPipe
    HALLUCINATION_PATTERNS = WhisperPipe.HALLUCINATION_PATTERNS

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        language: str | None = None,
        audio_format: AudioFormat | None = None,
        buffer_duration: float = 5.0,
        prompt: str | None = None,
        temperature: float = 0.0,
        silence_threshold: float = 0.01,
        skip_repetitions: bool = True,
        skip_hallucinations: bool = True,
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
            silence_threshold: RMS threshold below which audio is considered silence (0.0-1.0)
            skip_repetitions: Skip repeated transcriptions to avoid hallucination loops
            skip_hallucinations: Skip common hallucination phrases
        """
        super().__init__(audio_format)
        self.api_key = api_key
        self.model_name = model
        self.language = language
        self.buffer_duration = buffer_duration
        self.prompt = prompt
        self.temperature = temperature
        self.silence_threshold = silence_threshold
        self.skip_repetitions = skip_repetitions
        self.skip_hallucinations = skip_hallucinations

        # Calculate buffer size in samples
        self.buffer_size = int(self.audio_format.sample_rate * buffer_duration)
        self.buffer: list[npt.NDArray[Any]] = []
        self.buffer_samples = 0

        # Track recent transcriptions for repetition detection
        self.recent_transcriptions: list[str] = []
        self.max_recent = 3  # Keep last 3 transcriptions

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
                "openai package required for API usage. " "Install with: pip install openai"
            )

    def _is_silence(self, audio_data: npt.NDArray[Any]) -> bool:
        """Check if audio data is silence or near-silence.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            True if audio is below silence threshold
        """
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Calculate RMS
        rms = np.sqrt(np.mean(audio_float**2))

        is_silent = rms < self.silence_threshold

        if is_silent:
            msg = f"Audio is silent (RMS: {rms:.4f} < "
            msg += f"threshold: {self.silence_threshold})"
            self.logger.debug(msg)

        return is_silent

    def _is_hallucination(self, text: str) -> bool:
        """Check if text matches known hallucination patterns.

        Args:
            text: Transcribed text

        Returns:
            True if text is likely a hallucination
        """
        if not self.skip_hallucinations:
            return False

        text_lower = text.lower().strip()

        # Check exact matches (case insensitive)
        for pattern in self.HALLUCINATION_PATTERNS:
            if text_lower == pattern.lower():
                self.logger.debug(f"Filtered hallucination (exact match): '{text}'")
                return True

        # Check if entire text is just the pattern repeated
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern.lower() in text_lower:
                # Count occurrences
                count = text_lower.count(pattern.lower())
                # If pattern appears multiple times and makes up most of the text
                if count > 1 and len(pattern) * count > len(text) * 0.7:
                    self.logger.debug(f"Filtered hallucination (repetition): '{text}'")
                    return True

        return False

    def _is_repetition(self, text: str) -> bool:
        """Check if text is a repetition of recent transcriptions.

        Args:
            text: Transcribed text

        Returns:
            True if text is a repetition
        """
        if not self.skip_repetitions:
            return False

        # Check if this exact text appeared in recent transcriptions
        if text in self.recent_transcriptions:
            self.logger.debug(f"Filtered repetition: '{text}'")
            return True

        return False

    def _update_recent_transcriptions(self, text: str) -> None:
        """Update the list of recent transcriptions.

        Args:
            text: New transcribed text
        """
        self.recent_transcriptions.append(text)
        # Keep only the most recent transcriptions
        if len(self.recent_transcriptions) > self.max_recent:
            self.recent_transcriptions.pop(0)

    def _resample(
        self, audio: npt.NDArray[np.float32], orig_sr: int, target_sr: int
    ) -> npt.NDArray[np.float32]:
        """Resample audio to target sample rate using linear interpolation.

        Args:
            audio: Audio samples (mono, normalized float32)
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        # Calculate the ratio and new length
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)

        # Use linear interpolation for resampling
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio).astype(np.float32)

        msg = f"Resampled from {orig_sr}Hz to {target_sr}Hz: "
        msg += f"{len(audio)} -> {len(resampled)} samples"
        self.logger.debug(msg)

        return resampled

    def _buffer_to_wav(self, audio_data: npt.NDArray[Any]) -> bytes:
        """Convert audio buffer to WAV bytes (16kHz mono int16).

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            WAV file as bytes (16kHz, mono, int16)
        """
        # Debug: Log incoming audio format
        self.logger.debug(f"Input audio dtype: {audio_data.dtype}, shape: {audio_data.shape}")
        self.logger.debug(f"Input audio min: {audio_data.min()}, max: {audio_data.max()}")

        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_mono = audio_data.mean(axis=1)
        else:
            audio_mono = audio_data.flatten()

        # Convert to float32 and normalize
        if audio_mono.dtype == np.int16:
            audio_float = audio_mono.astype(np.float32) / 32768.0
        elif audio_mono.dtype in (np.float32, np.float64):
            max_val = np.abs(audio_mono).max()
            if max_val > 1.0:
                audio_float = (audio_mono / max_val).astype(np.float32)
            else:
                audio_float = audio_mono.astype(np.float32)
        else:
            audio_float = audio_mono.astype(np.float32)

        # Resample to 16kHz
        if self.audio_format.sample_rate != 16000:
            audio_float = self._resample(audio_float, self.audio_format.sample_rate, 16000)

        # Convert to int16
        audio_int16 = (audio_float * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_int16.tobytes())

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

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> str | None:
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

            # Check if buffer is mostly silence
            if self._is_silence(full_buffer):
                self.logger.debug("Skipping transcription: audio is silence")
                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0
                return None

            # Transcribe
            try:
                text = self._transcribe(full_buffer)

                # Reset buffer
                self.buffer = []
                self.buffer_samples = 0

                if text:
                    # Check for hallucinations
                    if self._is_hallucination(text):
                        self.logger.info(f"Skipped hallucination: '{text}'")
                        return None

                    # Check for repetitions
                    if self._is_repetition(text):
                        self.logger.info(f"Skipped repetition: '{text}'")
                        return None

                    # Update recent transcriptions
                    self._update_recent_transcriptions(text)

                    return text
            except Exception as e:
                self.logger.error(f"API transcription failed: {e}", exc_info=True)
                # Reset buffer on error
                self.buffer = []
                self.buffer_samples = 0

        return None

    def flush(self) -> str | None:
        """Transcribe any remaining audio in buffer.

        Returns:
            Transcribed text if buffer is not empty, None otherwise
        """
        if not self.buffer:
            return None

        full_buffer = np.vstack(self.buffer)

        # Check if buffer is mostly silence
        if self._is_silence(full_buffer):
            self.logger.debug("Skipping flush transcription: audio is silence")
            self.buffer = []
            self.buffer_samples = 0
            return None

        try:
            text = self._transcribe(full_buffer)

            # Reset buffer
            self.buffer = []
            self.buffer_samples = 0

            if text:
                # Check for hallucinations
                if self._is_hallucination(text):
                    self.logger.info(f"Skipped hallucination during flush: '{text}'")
                    return None

                # Check for repetitions
                if self._is_repetition(text):
                    self.logger.info(f"Skipped repetition during flush: '{text}'")
                    return None

                # Update recent transcriptions
                self._update_recent_transcriptions(text)

                return text
        except Exception as e:
            self.logger.error(f"API transcription failed during flush: {e}", exc_info=True)
            self.buffer = []
            self.buffer_samples = 0

        return None
