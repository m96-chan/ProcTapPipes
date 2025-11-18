"""ProcTapPipes - Official companion toolkit for ProcTap.

Provides modular audio-processing utilities that work as both Unix-style CLI
pipeline tools and importable Python modules.
"""

from proctap_pipes.base import BasePipe
from proctap_pipes.whisper_pipe import WhisperPipe, OpenAIWhisperPipe
from proctap_pipes.llm_pipe import LLMPipe, LLMPipeWithContext, LLMIntent
from proctap_pipes.webhook_pipe import (
    BaseWebhookPipe,
    WebhookPipe,
    SlackWebhookPipe,
    DiscordWebhookPipe,
    TeamsWebhookPipe,
    WebhookPipeText,
    WebhookPipeAudio,
)

__version__ = "0.1.0"
__all__ = [
    "BasePipe",
    "WhisperPipe",
    "OpenAIWhisperPipe",
    "LLMPipe",
    "LLMPipeWithContext",
    "LLMIntent",
    "BaseWebhookPipe",
    "WebhookPipe",
    "SlackWebhookPipe",
    "DiscordWebhookPipe",
    "TeamsWebhookPipe",
    "WebhookPipeText",
    "WebhookPipeAudio",
]
