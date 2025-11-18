"""Tests for webhook pipe."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from proctap_pipes.webhook_pipe import WebhookPipe, WebhookPipeText


@patch("proctap_pipes.webhook_pipe.requests.request")
def test_webhook_send_text(mock_request: Mock) -> None:
    """Test sending text to webhook."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    pipe = WebhookPipe(webhook_url="https://example.com/hook", text_mode=True)

    result = pipe.send_text("Hello World")

    assert result is True
    assert mock_request.called
    call_kwargs = mock_request.call_args.kwargs
    assert call_kwargs["json"]["text"] == "Hello World"


@patch("proctap_pipes.webhook_pipe.requests.request")
def test_webhook_batch(mock_request: Mock) -> None:
    """Test batching functionality."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    pipe = WebhookPipe(
        webhook_url="https://example.com/hook",
        text_mode=True,
        batch_size=3,
    )

    # Send two items - should not trigger send
    result1 = pipe.process_text_batch("item1")
    result2 = pipe.process_text_batch("item2")
    assert result1 is None
    assert result2 is None
    assert not mock_request.called

    # Third item should trigger send
    result3 = pipe.process_text_batch("item3")
    assert result3 == "sent"
    assert mock_request.called


@patch("proctap_pipes.webhook_pipe.requests.request")
def test_webhook_flush(mock_request: Mock) -> None:
    """Test flush functionality."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    pipe = WebhookPipe(
        webhook_url="https://example.com/hook",
        text_mode=True,
        batch_size=5,
    )

    # Add some items
    pipe.process_text_batch("item1")
    pipe.process_text_batch("item2")
    assert not mock_request.called

    # Flush should send remaining items
    result = pipe.flush()
    assert result == "sent"
    assert mock_request.called


@patch("proctap_pipes.webhook_pipe.requests.request")
def test_webhook_auth(mock_request: Mock) -> None:
    """Test authentication header."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    pipe = WebhookPipe(
        webhook_url="https://example.com/hook",
        text_mode=True,
        auth_token="test-token",
    )

    pipe.send_text("test")

    call_kwargs = mock_request.call_args.kwargs
    assert "Authorization" in call_kwargs["headers"]
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"


@patch("proctap_pipes.webhook_pipe.requests.request")
def test_webhook_template(mock_request: Mock) -> None:
    """Test payload template."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_request.return_value = mock_response

    template = {"event": "transcription", "source": "proctap"}

    pipe = WebhookPipe(
        webhook_url="https://example.com/hook",
        text_mode=True,
        payload_template=template,
    )

    pipe.send_text("test message")

    call_kwargs = mock_request.call_args.kwargs
    payload = call_kwargs["json"]
    assert payload["event"] == "transcription"
    assert payload["source"] == "proctap"
    assert payload["text"] == "test message"


def test_webhook_text_convenience_class() -> None:
    """Test WebhookPipeText convenience class."""
    pipe = WebhookPipeText(webhook_url="https://example.com/hook")

    assert pipe.text_mode is True
    assert pipe.webhook_url == "https://example.com/hook"
