"""Webhook and HTTP event delivery pipe."""

import json
import sys
from typing import Any, Optional, Dict
from io import BytesIO
import logging

import numpy.typing as npt
import requests

from proctap_pipes.base import BasePipe, AudioFormat

logger = logging.getLogger(__name__)


class WebhookPipe(BasePipe):
    """Webhook delivery pipe for sending events and data to HTTP endpoints.

    Can operate in two modes:
    1. Text mode: Send text data (e.g., transcriptions) as JSON payloads
    2. Audio mode: Send audio chunks as WAV files (multipart/form-data)
    """

    def __init__(
        self,
        webhook_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        text_mode: bool = True,
        audio_format: Optional[AudioFormat] = None,
        payload_template: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None,
        timeout: float = 10.0,
        batch_size: int = 1,
    ):
        """Initialize webhook delivery pipe.

        Args:
            webhook_url: Target webhook URL
            method: HTTP method (POST, PUT, PATCH)
            headers: Additional HTTP headers
            text_mode: If True, send text; if False, send audio
            audio_format: Audio format configuration
            payload_template: JSON template for text payloads
            auth_token: Bearer token for authentication
            timeout: Request timeout in seconds
            batch_size: Number of items to batch before sending
        """
        super().__init__(audio_format)
        self.webhook_url = webhook_url
        self.method = method.upper()
        self.headers = headers or {}
        self.text_mode = text_mode
        self.payload_template = payload_template or {}
        self.auth_token = auth_token
        self.timeout = timeout
        self.batch_size = batch_size

        # Batch buffer
        self.batch: list[Any] = []

        # Set up authentication
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"

        # Set default content type for text mode
        if self.text_mode and "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def _send_request(self, payload: Any, is_audio: bool = False) -> bool:
        """Send HTTP request to webhook.

        Args:
            payload: Data to send (dict for JSON, bytes for audio)
            is_audio: Whether payload is audio data

        Returns:
            True if request succeeded, False otherwise
        """
        try:
            if is_audio:
                # Send audio as multipart/form-data
                files = {"audio": ("audio.wav", payload, "audio/wav")}
                response = requests.request(
                    self.method,
                    self.webhook_url,
                    files=files,
                    headers={k: v for k, v in self.headers.items() if k != "Content-Type"},
                    timeout=self.timeout,
                )
            else:
                # Send JSON payload
                response = requests.request(
                    self.method,
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )

            response.raise_for_status()
            self.logger.debug(f"Webhook sent successfully: {response.status_code}")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Webhook request failed: {e}")
            return False

    def _build_payload(self, data: Any) -> Dict[str, Any]:
        """Build JSON payload from data and template.

        Args:
            data: Input data (text or structured data)

        Returns:
            JSON payload dict
        """
        payload = self.payload_template.copy()

        # If data is already a dict, merge it
        if isinstance(data, dict):
            payload.update(data)
        else:
            # Otherwise, set it as "data" field
            payload["data"] = data

        return payload

    def send_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send text data to webhook.

        Args:
            text: Text to send
            metadata: Additional metadata to include in payload

        Returns:
            True if request succeeded, False otherwise
        """
        payload = self._build_payload({"text": text})

        if metadata:
            payload.update(metadata)

        return self._send_request(payload, is_audio=False)

    def send_audio(self, audio_data: npt.NDArray[Any]) -> bool:
        """Send audio data to webhook.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            True if request succeeded, False otherwise
        """
        # Convert to WAV
        wav_bytes = self._audio_to_wav(audio_data)
        return self._send_request(wav_bytes, is_audio=True)

    def _audio_to_wav(self, audio_data: npt.NDArray[Any]) -> bytes:
        """Convert audio to WAV bytes.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            WAV file as bytes
        """
        import wave

        buffer = BytesIO()

        # Flatten if needed
        if audio_data.ndim > 1:
            channels = audio_data.shape[1]
            samples = audio_data
        else:
            channels = 1
            samples = audio_data.reshape(-1, 1)

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(self.audio_format.sample_width)
            wav_file.setframerate(self.audio_format.sample_rate)
            wav_file.writeframes(samples.tobytes())

        return buffer.getvalue()

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> Optional[str]:
        """Process audio chunk and send to webhook.

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            Status message
        """
        if self.text_mode:
            self.logger.warning("Audio processing not supported in text mode")
            return None

        success = self.send_audio(audio_data)
        return "sent" if success else "failed"

    def process_text_batch(self, text: str) -> Optional[str]:
        """Process text with batching support.

        Args:
            text: Text to process

        Returns:
            Status message when batch is sent, None otherwise
        """
        self.batch.append(text)

        if len(self.batch) >= self.batch_size:
            # Send batch
            if self.batch_size == 1:
                payload = self._build_payload({"text": self.batch[0]})
            else:
                payload = self._build_payload({"texts": self.batch})

            success = self._send_request(payload, is_audio=False)
            self.batch = []

            return "sent" if success else "failed"

        return None

    def flush(self) -> Optional[str]:
        """Send any remaining items in batch.

        Returns:
            Status message if batch sent, None otherwise
        """
        if not self.batch:
            return None

        if self.batch_size == 1:
            payload = self._build_payload({"text": self.batch[0]})
        else:
            payload = self._build_payload({"texts": self.batch})

        success = self._send_request(payload, is_audio=False)
        self.batch = []

        return "sent" if success else "failed"


class WebhookPipeText(WebhookPipe):
    """Convenience class for text-only webhook pipe."""

    def __init__(self, webhook_url: str, **kwargs: Any):
        """Initialize text webhook pipe.

        Args:
            webhook_url: Target webhook URL
            **kwargs: Additional arguments for WebhookPipe
        """
        kwargs["text_mode"] = True
        super().__init__(webhook_url, **kwargs)

    def send(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send text to webhook.

        Args:
            text: Text to send
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        return self.send_text(text, metadata)


class WebhookPipeAudio(WebhookPipe):
    """Convenience class for audio-only webhook pipe."""

    def __init__(
        self,
        webhook_url: str,
        audio_format: Optional[AudioFormat] = None,
        **kwargs: Any,
    ):
        """Initialize audio webhook pipe.

        Args:
            webhook_url: Target webhook URL
            audio_format: Audio format configuration
            **kwargs: Additional arguments for WebhookPipe
        """
        kwargs["text_mode"] = False
        super().__init__(webhook_url, audio_format=audio_format, **kwargs)

    def send(self, audio_data: npt.NDArray[Any]) -> bool:
        """Send audio to webhook.

        Args:
            audio_data: Audio samples

        Returns:
            True if successful, False otherwise
        """
        return self.send_audio(audio_data)
