#!/usr/bin/env python3
"""Basic usage examples for ProcTapPipes."""

import sys
from proctap_pipes import WhisperPipe, LLMPipe, WebhookPipe

def example_whisper():
    """Example: Transcribe audio from stdin."""
    print("Example 1: Whisper Transcription", file=sys.stderr)
    
    pipe = WhisperPipe(
        model="base",
        language="en",
        buffer_duration=5.0
    )
    
    # Process audio from stdin
    for transcription in pipe.run_stream(sys.stdin.buffer):
        print(f"Transcribed: {transcription}")

def example_llm():
    """Example: Process text with LLM."""
    print("Example 2: LLM Processing", file=sys.stderr)
    
    pipe = LLMPipe(
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",
        system_prompt="Summarize the following in one sentence."
    )
    
    text = "The quick brown fox jumps over the lazy dog. This is a test."
    summary = pipe.process_text(text)
    print(f"Summary: {summary}")

def example_webhook():
    """Example: Send data to webhook."""
    print("Example 3: Webhook Delivery", file=sys.stderr)
    
    pipe = WebhookPipe(
        webhook_url="https://httpbin.org/post",
        batch_size=1
    )
    
    pipe.send_text("Hello from ProcTapPipes!", metadata={"source": "example"})
    print("Webhook sent!", file=sys.stderr)

def example_pipeline():
    """Example: Complete pipeline (audio -> transcription -> LLM -> webhook)."""
    print("Example 4: Complete Pipeline", file=sys.stderr)
    
    # Create components
    whisper = WhisperPipe(model="base", buffer_duration=5.0)
    llm = LLMPipe(
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",
        system_prompt="Extract action items from this text."
    )
    webhook = WebhookPipe(
        webhook_url="https://httpbin.org/post",
        batch_size=3
    )
    
    # Process pipeline
    for transcription in whisper.run_stream(sys.stdin.buffer):
        if transcription:
            action_items = llm.process_text(transcription)
            result = webhook.process_text_batch(action_items)
            if result:
                print(f"Batch sent: {result}", file=sys.stderr)
    
    # Flush remaining items
    webhook.flush()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <example_number>")
        print("  1 - Whisper transcription")
        print("  2 - LLM processing")
        print("  3 - Webhook delivery")
        print("  4 - Complete pipeline")
        sys.exit(1)
    
    example = sys.argv[1]
    
    if example == "1":
        example_whisper()
    elif example == "2":
        example_llm()
    elif example == "3":
        example_webhook()
    elif example == "4":
        example_pipeline()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
