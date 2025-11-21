"""ProcTapPipes - Official companion toolkit for ProcTap.

Provides modular audio-processing utilities that work as both Unix-style CLI
pipeline tools and importable Python modules.
"""

from proctap_pipes.audio_effects_pipe import AudioEffectsPipe
from proctap_pipes.base import BasePipe
from proctap_pipes.llm_pipe import LLMIntent, LLMPipe, LLMPipeWithContext
from proctap_pipes.mic_mix_pipe import MicMixPipe
from proctap_pipes.volume_meter_pipe import VolumeMeterPipe
from proctap_pipes.webhook_pipe import (
    BaseWebhookPipe,
    DiscordWebhookPipe,
    SlackWebhookPipe,
    TeamsWebhookPipe,
    WebhookPipe,
    WebhookPipeAudio,
    WebhookPipeText,
)
from proctap_pipes.whisper_pipe import OpenAIWhisperPipe, WhisperPipe

__version__ = "0.2.2"
__all__ = [
    "BasePipe",
    "WhisperPipe",
    "OpenAIWhisperPipe",
    "LLMPipe",
    "LLMPipeWithContext",
    "LLMIntent",
    "VolumeMeterPipe",
    "AudioEffectsPipe",
    "MicMixPipe",
    "BaseWebhookPipe",
    "WebhookPipe",
    "SlackWebhookPipe",
    "DiscordWebhookPipe",
    "TeamsWebhookPipe",
    "WebhookPipeText",
    "WebhookPipeAudio",
]
