# ProcTapPipes Quick Start Guide

Get up and running with ProcTapPipes in 5 minutes!

## Installation

```bash
# Basic installation
pip install proctap-pipes

# With local Whisper support
pip install proctap-pipes[whisper]

# For development
git clone https://github.com/proctap/proctap-pipes.git
cd proctap-pipes
pip install -e ".[dev]"
```

## Your First Pipeline

### 1. Simple Transcription

```bash
# Capture audio from a process and transcribe
proctap -pid 1234 --stdout | proctap-whisper

# With specific model and language
proctap -pid 1234 --stdout | proctap-whisper --model small --language en
```

### 2. Transcription + LLM Processing

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Transcribe and summarize
proctap -pid 1234 --stdout | \
  proctap-whisper --api | \
  proctap-llm -s "Summarize this in one sentence"
```

### 3. Complete Workflow

```bash
# Transcribe, process with LLM, and send to webhook
export OPENAI_API_KEY="sk-..."

proctap -pid 1234 --stdout | \
  proctap-whisper --api --model whisper-1 | \
  proctap-llm -s "Extract action items" | \
  proctap-webhook https://your-webhook.com/endpoint
```

## Python API Usage

### Basic Transcription

```python
from proctap_pipes import WhisperPipe
import sys

# Create pipe
pipe = WhisperPipe(model="base", language="en")

# Process audio from stdin
for transcription in pipe.run_stream(sys.stdin.buffer):
    print(f"📝 {transcription}")
```

### LLM Processing

```python
from proctap_pipes import LLMPipe

# Create LLM pipe
llm = LLMPipe(
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    system_prompt="You are a helpful assistant."
)

# Process text
response = llm.process_text("What is the capital of France?")
print(response)
```

### Webhook Delivery

```python
from proctap_pipes import WebhookPipe

# Create webhook pipe
webhook = WebhookPipe(
    webhook_url="https://example.com/hook",
    auth_token="your-token",
    batch_size=5
)

# Send data
webhook.send_text("Hello from ProcTapPipes!")
```

### Complete Pipeline in Python

```python
from proctap_pipes import WhisperPipe, LLMPipe, WebhookPipe
import sys

# Set up pipeline components
whisper = WhisperPipe(model="base", buffer_duration=5.0)
llm = LLMPipe(
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    system_prompt="Extract key points from this transcription."
)
webhook = WebhookPipe(
    webhook_url="https://example.com/hook",
    batch_size=3
)

# Process audio -> text -> LLM -> webhook
for transcription in whisper.run_stream(sys.stdin.buffer):
    if transcription:
        key_points = llm.process_text(transcription)
        webhook.send_text(key_points, metadata={"source": "meeting"})

# Don't forget to flush
webhook.flush()
```

## Common Use Cases

### 1. Meeting Transcription

```bash
#!/bin/bash
# transcribe_meeting.sh

ZOOM_PID=$(pgrep zoom)
OUTPUT_FILE="meeting_transcript_$(date +%Y%m%d_%H%M%S).txt"

proctap -pid $ZOOM_PID --stdout | \
  proctap-whisper --model base | \
  tee $OUTPUT_FILE
```

### 2. Real-time Translation

```bash
# Transcribe in one language, translate with LLM
proctap -pid 1234 --stdout | \
  proctap-whisper --language es | \
  proctap-llm -s "Translate this Spanish text to English"
```

### 3. Action Item Extraction

```bash
# Extract and send action items to Slack
export SLACK_WEBHOOK="https://hooks.slack.com/..."

proctap -pid 1234 --stdout | \
  proctap-whisper | \
  proctap-llm -s "Extract action items as a bullet list" | \
  proctap-webhook $SLACK_WEBHOOK
```

### 4. Sentiment Analysis

```bash
# Analyze sentiment of transcribed audio
proctap -pid 1234 --stdout | \
  proctap-whisper | \
  proctap-llm -s "Analyze the sentiment: positive, negative, or neutral"
```

## Configuration

### Environment Variables

```bash
# OpenAI API key (for Whisper API and LLM)
export OPENAI_API_KEY="sk-..."

# Webhook authentication
export WEBHOOK_AUTH_TOKEN="your-token"
```

### Audio Format Options

```bash
# Custom sample rate and channels
proctap-whisper --rate 44100 --channels 1

# Adjust buffer duration
proctap-whisper --buffer 10.0  # 10 seconds
```

### LLM Options

```bash
# Use different model
proctap-llm --model gpt-4

# Adjust temperature
proctap-llm --temperature 0.3  # More focused

# Limit response length
proctap-llm --max-tokens 100
```

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the right environment
pip install -e .
```

### "API key not found" errors
```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or pass directly
proctap-whisper --api-key "your-key-here"
```

### Audio format issues
```bash
# Try specifying format explicitly
proctap-whisper --rate 48000 --channels 2
```

### Verbose logging
```bash
# Enable verbose mode for debugging
proctap-whisper --verbose
proctap-llm --verbose
proctap-webhook --verbose
```

## Next Steps

- Read the [full README](README.md) for detailed API reference
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to add your own pipes
- Explore [examples/](examples/) for more use cases
- Join the community and share your pipelines!

## Get Help

- GitHub Issues: https://github.com/proctap/proctap-pipes/issues
- Documentation: https://github.com/proctap/proctap-pipes#readme

Happy piping! 🎵➜📝➜🤖
