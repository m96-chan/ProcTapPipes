"""LLM processing pipe for audio transcriptions and text processing."""

import json
from typing import Any, Optional, Callable
import logging

import numpy.typing as npt

from proctap_pipes.base import BasePipe, AudioFormat

logger = logging.getLogger(__name__)


class LLMPipe(BasePipe):
    """LLM-based text processing pipe.

    Processes text (from transcriptions or other sources) through an LLM
    for tasks like summarization, question answering, or intent extraction.

    Note: This pipe works in text mode - it expects to receive text input,
    not audio. Typically used after WhisperPipe in a pipeline.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        text_mode: bool = True,
        audio_format: Optional[AudioFormat] = None,
    ):
        """Initialize LLM processing pipe.

        Args:
            model: LLM model name (e.g., gpt-3.5-turbo, gpt-4)
            api_key: OpenAI API key
            system_prompt: System prompt for the LLM
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            base_url: Custom API base URL (for compatible APIs)
            text_mode: If True, process text input; if False, process audio
            audio_format: Audio format (used only if text_mode=False)
        """
        super().__init__(audio_format)
        self.model_name = model
        self.api_key = api_key
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.text_mode = text_mode

        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = OpenAI(**client_kwargs)
            self.logger.info(f"Initialized LLM client with model {self.model_name}")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def process_text(self, text: str) -> str:
        """Process text through the LLM.

        Args:
            text: Input text to process

        Returns:
            LLM response text
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]

            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
            }

            if self.max_tokens:
                completion_kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**completion_kwargs)

            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}", exc_info=True)
            return f"[ERROR: {str(e)}]"

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> Optional[str]:
        """Process audio chunk (not supported in text mode).

        Args:
            audio_data: NumPy array of audio samples

        Returns:
            None (audio processing not implemented)
        """
        if self.text_mode:
            self.logger.warning("Audio processing not supported in text mode")
            return None

        # Future: Could implement audio -> transcription -> LLM pipeline here
        self.logger.warning("Audio processing not yet implemented for LLMPipe")
        return None

    def process_stream_text(self, text_iterator: Any) -> Any:
        """Process a stream of text chunks through the LLM.

        Args:
            text_iterator: Iterator yielding text strings

        Yields:
            LLM responses for each text chunk
        """
        for text in text_iterator:
            if isinstance(text, str) and text.strip():
                result = self.process_text(text)
                if result:
                    yield result


class LLMPipeWithContext(LLMPipe):
    """LLM pipe that maintains conversation context.

    Extends LLMPipe to keep track of conversation history for
    multi-turn interactions.
    """

    def __init__(self, max_context_messages: int = 10, **kwargs: Any):
        """Initialize LLM pipe with context tracking.

        Args:
            max_context_messages: Maximum number of messages to keep in context
            **kwargs: Arguments passed to LLMPipe
        """
        super().__init__(**kwargs)
        self.max_context_messages = max_context_messages
        self.context: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def process_text(self, text: str) -> str:
        """Process text through the LLM with context.

        Args:
            text: Input text to process

        Returns:
            LLM response text
        """
        try:
            # Add user message to context
            self.context.append({"role": "user", "content": text})

            # Trim context if needed (keep system message + last N messages)
            if len(self.context) > self.max_context_messages + 1:
                self.context = [self.context[0]] + self.context[-(self.max_context_messages):]

            completion_kwargs = {
                "model": self.model_name,
                "messages": self.context,
                "temperature": self.temperature,
            }

            if self.max_tokens:
                completion_kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**completion_kwargs)

            assistant_message = response.choices[0].message.content.strip()

            # Add assistant response to context
            self.context.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}", exc_info=True)
            return f"[ERROR: {str(e)}]"

    def reset_context(self) -> None:
        """Reset conversation context to initial state."""
        self.context = [{"role": "system", "content": self.system_prompt}]


class LLMIntent(LLMPipe):
    """Specialized LLM pipe for intent extraction.

    Processes text to extract structured intent information (e.g., commands,
    actions, entities).
    """

    def __init__(
        self,
        intents: Optional[list[str]] = None,
        output_format: str = "json",
        **kwargs: Any,
    ):
        """Initialize intent extraction pipe.

        Args:
            intents: List of possible intents to detect
            output_format: Output format ('json' or 'text')
            **kwargs: Arguments passed to LLMPipe
        """
        self.intents = intents or []
        self.output_format = output_format

        # Build system prompt for intent extraction
        intent_list = ", ".join(self.intents) if self.intents else "any relevant intents"
        system_prompt = (
            f"You are an intent extraction assistant. "
            f"Extract the user's intent from their message. "
            f"Identify: {intent_list}. "
        )

        if output_format == "json":
            system_prompt += (
                "Respond ONLY with valid JSON in this format: "
                '{"intent": "intent_name", "entities": {}, "confidence": 0.0}'
            )

        kwargs["system_prompt"] = system_prompt
        super().__init__(**kwargs)

    def process_text(self, text: str) -> str:
        """Extract intent from text.

        Args:
            text: Input text to analyze

        Returns:
            Intent information (JSON or text)
        """
        result = super().process_text(text)

        if self.output_format == "json":
            try:
                # Validate JSON
                json.loads(result)
            except json.JSONDecodeError:
                self.logger.warning("LLM did not return valid JSON")
                return json.dumps({
                    "intent": "unknown",
                    "entities": {},
                    "confidence": 0.0,
                    "raw_response": result,
                })

        return result
