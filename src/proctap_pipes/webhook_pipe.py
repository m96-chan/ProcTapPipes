"""Webhook and HTTP event delivery pipe.

This module provides a generic webhook base class and specific implementations
for popular services like Slack, Discord, Microsoft Teams, etc.
"""

import json
import sys
from typing import Any, Optional, Dict
from io import BytesIO
from abc import abstractmethod
import logging

import numpy.typing as npt
import requests

from proctap_pipes.base import BasePipe, AudioFormat

logger = logging.getLogger(__name__)


class BaseWebhookPipe(BasePipe):
    """Abstract base webhook delivery pipe.

    Subclasses should implement format_payload() to create service-specific payloads.

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
            auth_token: Bearer token for authentication
            timeout: Request timeout in seconds
            batch_size: Number of items to batch before sending
        """
        super().__init__(audio_format)
        self.webhook_url = webhook_url
        self.method = method.upper()
        self.headers = headers or {}
        self.text_mode = text_mode
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

    @abstractmethod
    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format text into service-specific payload.

        Args:
            text: Text content to send
            metadata: Additional metadata

        Returns:
            Formatted payload dict
        """
        pass

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

    def send_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send text data to webhook.

        Args:
            text: Text to send
            metadata: Additional metadata to include in payload

        Returns:
            True if request succeeded, False otherwise
        """
        payload = self.format_payload(text, metadata)
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
                success = self.send_text(self.batch[0])
            else:
                # For batch, send as list - subclasses should handle this
                payload = self.format_payload("\n".join(self.batch), {"batch": True, "count": len(self.batch)})
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
            success = self.send_text(self.batch[0])
        else:
            payload = self.format_payload("\n".join(self.batch), {"batch": True, "count": len(self.batch)})
            success = self._send_request(payload, is_audio=False)

        self.batch = []
        return "sent" if success else "failed"


# Concrete implementations for specific services


class WebhookPipe(BaseWebhookPipe):
    """Generic webhook pipe with customizable payload template.

    Use this for custom webhooks or when you need full control over the payload format.
    """

    def __init__(self, payload_template: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Initialize generic webhook pipe.

        Args:
            payload_template: Template for JSON payloads
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        super().__init__(**kwargs)
        self.payload_template = payload_template or {}

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format text into generic payload.

        Args:
            text: Text content
            metadata: Additional metadata

        Returns:
            Formatted payload
        """
        payload = self.payload_template.copy()
        payload["text"] = text

        if metadata:
            payload.update(metadata)

        return payload


class SlackWebhookPipe(BaseWebhookPipe):
    """Slack webhook pipe using Incoming Webhooks format.

    Example:
        pipe = SlackWebhookPipe(
            webhook_url="https://hooks.slack.com/services/...",
            channel="#transcriptions",
            username="ProcTap Bot"
        )
        pipe.send_text("Meeting transcription: ...")
    """

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "ProcTap Bot",
        icon_emoji: Optional[str] = ":microphone:",
        **kwargs: Any,
    ):
        """Initialize Slack webhook pipe.

        Args:
            webhook_url: Slack incoming webhook URL
            channel: Target channel (e.g., #general)
            username: Bot username
            icon_emoji: Bot icon emoji
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        super().__init__(webhook_url, **kwargs)
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format text into Slack message payload.

        Args:
            text: Message text
            metadata: Additional metadata (can include attachments, blocks, etc.)

        Returns:
            Slack-formatted payload
        """
        payload: Dict[str, Any] = {
            "text": text,
            "username": self.username,
        }

        if self.channel:
            payload["channel"] = self.channel

        if self.icon_emoji:
            payload["icon_emoji"] = self.icon_emoji

        if metadata:
            # Allow metadata to override or add fields
            payload.update(metadata)

        return payload


class DiscordWebhookPipe(BaseWebhookPipe):
    """Discord webhook pipe.

    Example:
        pipe = DiscordWebhookPipe(
            webhook_url="https://discord.com/api/webhooks/...",
            username="ProcTap Bot"
        )
        pipe.send_text("Transcription: ...")
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "ProcTap Bot",
        avatar_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Discord webhook pipe.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username
            avatar_url: Bot avatar URL
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        super().__init__(webhook_url, **kwargs)
        self.username = username
        self.avatar_url = avatar_url

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format text into Discord message payload.

        Args:
            text: Message content
            metadata: Additional metadata (can include embeds, etc.)

        Returns:
            Discord-formatted payload
        """
        payload: Dict[str, Any] = {
            "content": text,
            "username": self.username,
        }

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        if metadata:
            payload.update(metadata)

        return payload


class TeamsWebhookPipe(BaseWebhookPipe):
    """Microsoft Teams webhook pipe using Incoming Webhook connector.

    Example:
        pipe = TeamsWebhookPipe(
            webhook_url="https://outlook.office.com/webhook/...",
            title="Meeting Transcription"
        )
        pipe.send_text("Transcription content...")
    """

    def __init__(
        self,
        webhook_url: str,
        title: str = "ProcTap Notification",
        theme_color: str = "0078D4",
        **kwargs: Any,
    ):
        """Initialize Teams webhook pipe.

        Args:
            webhook_url: Teams incoming webhook URL
            title: Message card title
            theme_color: Hex color for the card accent (without #)
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        super().__init__(webhook_url, **kwargs)
        self.title = title
        self.theme_color = theme_color

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format text into Teams message card payload.

        Args:
            text: Message text
            metadata: Additional metadata

        Returns:
            Teams MessageCard-formatted payload
        """
        payload: Dict[str, Any] = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": self.title,
            "themeColor": self.theme_color,
            "title": self.title,
            "text": text,
        }

        if metadata:
            # Metadata can add sections, facts, potentialAction, etc.
            payload.update(metadata)

        return payload


class WebhookPipeText(BaseWebhookPipe):
    """Convenience class for generic text-only webhook pipe."""

    def __init__(self, webhook_url: str, **kwargs: Any):
        """Initialize text webhook pipe.

        Args:
            webhook_url: Target webhook URL
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        kwargs["text_mode"] = True
        super().__init__(webhook_url, **kwargs)

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format as simple text payload."""
        payload = {"text": text}
        if metadata:
            payload.update(metadata)
        return payload

    def send(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send text to webhook.

        Args:
            text: Text to send
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        return self.send_text(text, metadata)


class WebhookPipeAudio(BaseWebhookPipe):
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
            **kwargs: Additional arguments for BaseWebhookPipe
        """
        kwargs["text_mode"] = False
        super().__init__(webhook_url, audio_format=audio_format, **kwargs)

    def format_payload(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Not used for audio mode."""
        return {}

    def send(self, audio_data: npt.NDArray[Any]) -> bool:
        """Send audio to webhook.

        Args:
            audio_data: Audio samples

        Returns:
            True if successful, False otherwise
        """
        return self.send_audio(audio_data)
