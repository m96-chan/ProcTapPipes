"""Tests for LLM pipe."""

from unittest.mock import Mock, patch, MagicMock

import pytest

from proctap_pipes.llm_pipe import LLMPipe, LLMPipeWithContext, LLMIntent


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_pipe_initialization(mock_openai: Mock) -> None:
    """Test LLM pipe initialization."""
    pipe = LLMPipe(model="gpt-3.5-turbo", api_key="test-key")

    assert pipe.model_name == "gpt-3.5-turbo"
    assert pipe.api_key == "test-key"
    assert mock_openai.called


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_process_text(mock_openai_class: Mock) -> None:
    """Test text processing."""
    # Create mock response
    mock_message = Mock()
    mock_message.content = "This is a response"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    pipe = LLMPipe(model="gpt-3.5-turbo", api_key="test-key")
    result = pipe.process_text("Hello")

    assert result == "This is a response"
    assert mock_client.chat.completions.create.called


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_with_context(mock_openai_class: Mock) -> None:
    """Test LLM pipe with context tracking."""
    # Setup mock
    mock_message = Mock()
    mock_message.content = "Response"

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    pipe = LLMPipeWithContext(model="gpt-3.5-turbo", api_key="test-key")

    # Process multiple messages
    pipe.process_text("First message")
    pipe.process_text("Second message")

    # Context should have: system + 2 user + 2 assistant = 5 messages
    assert len(pipe.context) == 5


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_context_reset(mock_openai_class: Mock) -> None:
    """Test context reset."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    pipe = LLMPipeWithContext(model="gpt-3.5-turbo", api_key="test-key")

    # Add to context
    pipe.context.append({"role": "user", "content": "test"})
    assert len(pipe.context) > 1

    # Reset
    pipe.reset_context()
    assert len(pipe.context) == 1  # Only system message


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_intent_extraction(mock_openai_class: Mock) -> None:
    """Test intent extraction."""
    # Setup mock to return JSON
    mock_message = Mock()
    mock_message.content = '{"intent": "greeting", "entities": {}, "confidence": 0.9}'

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    pipe = LLMIntent(
        model="gpt-3.5-turbo",
        api_key="test-key",
        intents=["greeting", "question"],
        output_format="json",
    )

    result = pipe.process_text("Hello there")

    # Should return valid JSON
    import json
    parsed = json.loads(result)
    assert parsed["intent"] == "greeting"


@patch("proctap_pipes.llm_pipe.OpenAI")
def test_llm_custom_base_url(mock_openai_class: Mock) -> None:
    """Test custom base URL."""
    pipe = LLMPipe(
        model="gpt-3.5-turbo",
        api_key="test-key",
        base_url="https://custom.api.com",
    )

    # Verify OpenAI was called with base_url
    call_kwargs = mock_openai_class.call_args.kwargs
    assert call_kwargs["base_url"] == "https://custom.api.com"
