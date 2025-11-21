# ProcTapPipes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ProcTapPipes** is an official companion toolkit for [ProcTap](https://github.com/proctap/proctap), providing modular audio-processing utilities that work as both Unix-style CLI pipeline tools and importable Python modules.

It enables real-time workflows such as **Whisper transcription**, **LLM processing**, **webhook automation**, and more — all built on top of ProcTap's per-process audio streams.

## Features

- **Modular & Composable**: Each tool does one thing well and can be chained together
- **Dual Interface**: Use as CLI tools in pipelines OR import as Python modules
- **Real-time Processing**: Stream audio and text through processing pipelines
- **Fully Typed**: Complete type hints for better IDE support and type safety
- **Extensible**: Easy to add new pipes by extending `BasePipe`

## Available Pipes

### 1. WhisperPipe
Speech-to-text transcription using OpenAI's Whisper (local or API).

**CLI**: `proctap-whisper`
**Module**: `proctap_pipes.WhisperPipe`

### 2. LLMPipe
Process text through LLMs for summarization, Q&A, intent extraction, etc.

**CLI**: `proctap-llm`
**Module**: `proctap_pipes.LLMPipe`

### 3. WebhookPipe
Send events and data to HTTP endpoints (webhooks, APIs).

**CLI**: `proctap-webhook`
**Module**: `proctap_pipes.WebhookPipe`

### 4. VolumeMeterPipe
Real-time volume meter with audio passthrough for pipeline monitoring.

**CLI**: `proctap-volume-meter`
**Module**: `proctap_pipes.VolumeMeterPipe`

### 5. AudioEffectsPipe
Real-time audio effects processing (noise reduction, normalization, EQ) to improve audio quality.

**CLI**: `proctap-effects`
**Module**: `proctap_pipes.AudioEffectsPipe`

### 6. MicMixPipe
Mix microphone input with ProcTap audio streams for combined process + voice recording/streaming.

**CLI**: `proctap-mic-mix`
**Module**: `proctap_pipes.MicMixPipe`

## Installation

```bash
# Basic installation
pip install proctap-pipes

# With Whisper support (local models)
pip install proctap-pipes[whisper]

# With microphone mixing support
pip install proctap-pipes[mic-mix]

# Development installation
pip install proctap-pipes[dev]

# All features
pip install proctap-pipes[all]
```

## Quick Start

### CLI Usage

```bash
# Transcribe audio from a process
proctap -pid 1234 --stdout | proctap-whisper

# Monitor volume levels while transcribing
proctap -pid 1234 --stdout | proctap-volume-meter | proctap-whisper

# Transcribe and process with LLM
proctap -pid 1234 --stdout | proctap-whisper | proctap-llm

# Full pipeline: monitor volume, transcribe, and send to webhook
proctap -pid 1234 --stdout | proctap-volume-meter | proctap-whisper | proctap-webhook https://example.com/hook

# Use audio effects to improve transcription accuracy
proctap -pid 1234 --stdout | proctap-effects --denoise --normalize | proctap-whisper

# Full enhancement pipeline with filters
proctap -pid 1234 --stdout | proctap-effects --denoise --normalize --highpass 80 --lowpass 8000 | proctap-whisper

# Use OpenAI API for Whisper
export OPENAI_API_KEY="sk-..."
proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1

# Mix microphone input with process audio
proctap --pid 1234 --stdout | proctap-mic-mix | proctap-whisper

# Mix with custom gain and record to MP3
proctap --pid 1234 --stdout | proctap-mic-mix --gain 0.8 | ffmpeg -f s16le -ar 48000 -ac 2 -i pipe:0 output.mp3

# Use specific microphone device (list devices first)
proctap-mic-mix --list-devices
proctap --pid 1234 --stdout | proctap-mic-mix --device 0 | proctap-whisper

# Record process + mic audio to MP3 with high quality
proctap --pid 1234 --stdout | proctap-mic-mix --gain 0.9 | ffmpeg -f s16le -ar 48000 -ac 2 -i pipe:0 -c:a libmp3lame -b:a 192k output.mp3
```

**Windows PowerShell users**: PowerShell has issues with binary piping. Use `.exe` directly from your virtual environment:

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Use .exe directly for binary pipelines
.\venv\Scripts\proctap.exe --pid 54856 --stdout | .\venv\Scripts\proctap-volume-meter.exe | .\venv\Scripts\proctap-whisper.exe

# Or save to file
.\venv\Scripts\proctap.exe --pid 54856 --stdout | .\venv\Scripts\proctap-volume-meter.exe > audio.pcm
```

**Alternative for Windows**: Use Command Prompt (`cmd.exe`) for seamless binary piping:

```cmd
REM Works perfectly in cmd.exe
proctap --pid 54856 --stdout | proctap-volume-meter | proctap-whisper
```

**Note on Windows Output**: When piping on Windows, text output (like transcriptions) automatically redirects to stderr instead of stdout due to Windows pipe limitations. This is transparent to the user - you'll still see all output in your terminal. If you need to redirect transcription output to a file, use stderr redirection:

```cmd
REM Capture transcriptions to file
proctap --pid 54856 --stdout | proctap-whisper 2> transcriptions.txt
```

## Troubleshooting

### Poor Transcription Accuracy

If you're getting poor or repetitive transcriptions, try these steps:

1. **Enable verbose logging to diagnose audio format issues**:
   ```bash
   proctap --pid 1234 --stdout | proctap-whisper -v
   ```

   Check the debug output for:
   - Input audio dtype (should be int16)
   - Min/max values (int16 range: -32768 to 32767)
   - If values are outside this range, there may be a format mismatch

2. **Use a larger model for better accuracy**:
   ```bash
   # Small model (faster, good accuracy)
   proctap --pid 1234 --stdout | proctap-whisper -m small

   # Medium model (balanced)
   proctap --pid 1234 --stdout | proctap-whisper -m medium

   # Large v3 model (best accuracy, slowest)
   proctap --pid 1234 --stdout | proctap-whisper -m large-v3
   ```

3. **Specify language explicitly** (especially for non-English):
   ```bash
   proctap --pid 1234 --stdout | proctap-whisper -l ja  # Japanese
   proctap --pid 1234 --stdout | proctap-whisper -l es  # Spanish
   ```

4. **Adjust buffer duration** (default 5.0 seconds):
   ```bash
   # Longer buffer for more context
   proctap --pid 1234 --stdout | proctap-whisper -b 10.0
   ```

5. **Use OpenAI API for best accuracy** (requires API key):
   ```bash
   export OPENAI_API_KEY="sk-..."
   proctap --pid 1234 --stdout | proctap-whisper --api --model whisper-1
   ```

6. **Check if audio format conversion is needed**:

   If verbose logs show unexpected dtypes or values, you may need to convert the audio format. Contact ProcTap support or check the audio capture settings.

### Python Module Usage

```python
from proctap_pipes import WhisperPipe
import sys

# Create a Whisper transcription pipe
pipe = WhisperPipe(model="base", language="en")

# Process audio from stdin
for transcription in pipe.run_stream(sys.stdin.buffer):
    print(f"Transcribed: {transcription}")
```

```python
from proctap_pipes import LLMPipe

# Create an LLM processing pipe
llm = LLMPipe(
    model="gpt-3.5-turbo",
    api_key="sk-...",
    system_prompt="Summarize the following text concisely."
)

# Process text
text = "Long article text here..."
summary = llm.process_text(text)
print(summary)
```

```python
from proctap_pipes import WebhookPipe

# Create a webhook pipe
webhook = WebhookPipe(
    webhook_url="https://example.com/hook",
    auth_token="your-token",
    batch_size=5
)

# Send text data
webhook.send_text("Hello from ProcTapPipes!")
```

```python
from proctap_pipes import VolumeMeterPipe
import sys

# Create a volume meter pipe with passthrough
meter = VolumeMeterPipe(
    bar_width=50,
    update_interval=0.05,
    show_db=True
)

# Monitor volume while passing audio through
for audio_chunk in meter.run_stream(sys.stdin.buffer):
    # Audio chunk is unchanged and can be piped to other tools
    sys.stdout.buffer.write(audio_chunk.tobytes())
```

## CLI Tools Reference

### proctap-whisper

Transcribe audio from stdin using Whisper.

```bash
proctap-whisper [OPTIONS]

Options:
  -m, --model TEXT        Whisper model (tiny/base/small/medium/large) [default: base]
  -l, --language TEXT     Language code (e.g., en, es, fr)
  --api                   Use OpenAI API instead of local model
  --api-key TEXT          OpenAI API key (or set OPENAI_API_KEY)
  -b, --buffer FLOAT      Buffer duration in seconds [default: 5.0]
  -r, --rate INTEGER      Sample rate in Hz [default: 48000]
  -c, --channels INTEGER  Number of channels [default: 2]
  -v, --verbose           Enable verbose logging
```

**Examples:**

```bash
# Use local model
proctap -pid 1234 --stdout | proctap-whisper -m small -l en

# Use OpenAI API
proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1

# Adjust buffer size
proctap -pid 1234 --stdout | proctap-whisper -b 10.0
```

### proctap-llm

Process text through an LLM.

```bash
proctap-llm [OPTIONS]

Options:
  -m, --model TEXT          LLM model name [default: gpt-3.5-turbo]
  --api-key TEXT            OpenAI API key (or set OPENAI_API_KEY)
  -s, --system-prompt TEXT  System prompt for the LLM
  -t, --temperature FLOAT   Sampling temperature [default: 0.7]
  --max-tokens INTEGER      Maximum tokens in response
  --base-url TEXT           Custom API base URL
  --context                 Maintain conversation context
  --max-context INTEGER     Max context messages [default: 10]
  -v, --verbose             Enable verbose logging
```

**Examples:**

```bash
# Simple text processing
echo "Explain quantum computing" | proctap-llm

# Chain after transcription
proctap -pid 1234 --stdout | proctap-whisper | proctap-llm

# Custom system prompt
echo "Hello!" | proctap-llm -s "You are a pirate assistant"

# Maintain context
cat conversation.txt | proctap-llm --context
```

### proctap-webhook

Send text to a webhook URL.

```bash
proctap-webhook URL [OPTIONS]

Options:
  -m, --method TEXT       HTTP method (POST/PUT/PATCH) [default: POST]
  -H, --header TEXT       Additional HTTP header (format: 'Key: Value')
  -a, --auth-token TEXT   Bearer token for authentication
  -t, --timeout FLOAT     Request timeout in seconds [default: 10.0]
  -b, --batch INTEGER     Batch size before sending [default: 1]
  --template TEXT         JSON template for payload
  -v, --verbose           Enable verbose logging
```

**Examples:**

```bash
# Simple webhook
echo "test" | proctap-webhook https://example.com/hook

# With authentication
echo "data" | proctap-webhook https://example.com/hook -a "token123"

# Custom headers
echo "test" | proctap-webhook https://example.com/hook -H "X-Custom: value"

# Batch multiple items
cat texts.txt | proctap-webhook https://example.com/hook --batch 5
```

### proctap-volume-meter

Display real-time volume levels with audio passthrough.

```bash
proctap-volume-meter [OPTIONS]

Options:
  -r, --sample-rate INTEGER   Sample rate in Hz [default: 48000]
  -c, --channels INTEGER      Number of channels [default: 2]
  -w, --sample-width INTEGER  Sample width in bytes [default: 2]
  -b, --bar-width INTEGER     Width of volume bar [default: 50]
  -u, --update-interval FLOAT Display update interval [default: 0.05]
  -p, --peak-hold FLOAT       Peak hold time in seconds [default: 1.0]
  --no-db                     Hide dB values
  --no-rms                    Hide RMS meter
  --no-peak                   Hide peak meter
  --db-min FLOAT              Minimum dB for display [default: -60.0]
  --db-max FLOAT              Maximum dB for display [default: 0.0]
  --chunk-size INTEGER        Audio chunk size [default: 4096]
  -v, --verbose               Enable verbose logging
```

**Examples:**

```bash
# Monitor volume while transcribing
proctap -pid 1234 --stdout | proctap-volume-meter | proctap-whisper

# Just monitor volume (no further processing)
proctap -pid 1234 --stdout | proctap-volume-meter > /dev/null

# Custom bar width and update rate
proctap -pid 1234 --stdout | proctap-volume-meter -b 80 -u 0.1

# Show only RMS, hide peak
proctap -pid 1234 --stdout | proctap-volume-meter --no-peak
```

### proctap-effects

Apply real-time audio effects to improve audio quality.

```bash
proctap-effects [OPTIONS]

Options:
  -r, --sample-rate INTEGER   Sample rate in Hz [default: 48000]
  -c, --channels INTEGER      Number of channels [default: 2]
  -w, --sample-width INTEGER  Sample width in bytes [default: 2]
  -d, --denoise               Enable noise reduction
  --noise-mode TEXT           Noise reduction algorithm: simple (fast, 2200x real-time), spectral_gate (balanced, 145x), wiener (best quality, 550x) [default: simple]
  -t, --noise-threshold FLOAT Noise reduction threshold, 0.0-1.0 [default: 0.02]
  -n, --normalize             Enable volume normalization
  -l, --target-level FLOAT    Target volume level, 0.0-1.0 [default: 0.7]
  -hp, --highpass FLOAT       High-pass filter cutoff frequency in Hz
  -lp, --lowpass FLOAT        Low-pass filter cutoff frequency in Hz
  --chunk-size INTEGER        Audio chunk size [default: 4096]
  -v, --verbose               Enable verbose logging
```

**Examples:**

```bash
# Apply noise reduction (fast mode, default)
proctap -pid 1234 --stdout | proctap-effects --denoise | proctap-whisper

# Use spectral gate for better quality
proctap -pid 1234 --stdout | proctap-effects --denoise --noise-mode spectral_gate | proctap-whisper

# Best quality with Wiener filter
proctap -pid 1234 --stdout | proctap-effects --denoise --noise-mode wiener | proctap-whisper

# Normalize volume for consistent transcription
proctap -pid 1234 --stdout | proctap-effects --normalize | proctap-whisper

# Remove low-frequency rumble and high-frequency hiss
proctap -pid 1234 --stdout | proctap-effects --highpass 80 --lowpass 8000 | proctap-whisper

# Full audio enhancement pipeline (fast mode)
proctap -pid 1234 --stdout | proctap-effects --denoise --normalize --highpass 80 --lowpass 8000 | proctap-whisper

# Aggressive noise reduction
proctap -pid 1234 --stdout | proctap-effects --denoise --noise-threshold 0.001 | proctap-whisper
```

## Python API Reference

### BasePipe

Abstract base class for all pipes.

```python
from proctap_pipes.base import BasePipe, AudioFormat
import numpy as np

class CustomPipe(BasePipe):
    def process_chunk(self, audio_data: np.ndarray) -> str:
        # Process audio chunk
        return f"Processed {len(audio_data)} samples"

# Use the pipe
pipe = CustomPipe()
for result in pipe.run_stream(input_stream):
    print(result)
```

### WhisperPipe

```python
WhisperPipe(
    model: str = "base",
    language: Optional[str] = None,
    audio_format: Optional[AudioFormat] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    buffer_duration: float = 5.0
)
```

**Methods:**
- `process_chunk(audio_data)`: Process a single audio chunk
- `run_stream(stream)`: Process audio from a stream
- `flush()`: Transcribe any remaining buffered audio

### LLMPipe

```python
LLMPipe(
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None
)
```

**Methods:**
- `process_text(text)`: Process text through the LLM
- `process_stream_text(text_iterator)`: Process a stream of text

**Variants:**
- `LLMPipeWithContext`: Maintains conversation context
- `LLMIntent`: Specialized for intent extraction

### WebhookPipe

```python
WebhookPipe(
    webhook_url: str,
    method: str = "POST",
    headers: Optional[Dict[str, str]] = None,
    text_mode: bool = True,
    auth_token: Optional[str] = None,
    timeout: float = 10.0,
    batch_size: int = 1
)
```

**Methods:**
- `send_text(text, metadata)`: Send text to webhook
- `send_audio(audio_data)`: Send audio to webhook
- `process_text_batch(text)`: Process with batching
- `flush()`: Send remaining batch

## Creating Custom Pipes

Extend `BasePipe` to create your own processing modules:

```python
from proctap_pipes.base import BasePipe
import numpy as np
import numpy.typing as npt

class MyCustomPipe(BasePipe):
    """Custom audio processing pipe."""

    def __init__(self, param1: str, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1

    def process_chunk(self, audio_data: npt.NDArray) -> str:
        """Process audio chunk.

        Args:
            audio_data: NumPy array with shape (samples, channels)

        Returns:
            Processed result
        """
        # Your processing logic here
        volume = np.abs(audio_data).mean()
        return f"Volume: {volume:.2f}"

# Use it
pipe = MyCustomPipe(param1="value")
pipe.run_cli()  # Read from stdin, write to stdout
```

## Advanced Examples

### Multi-stage Pipeline

```python
from proctap_pipes import WhisperPipe, LLMPipe, WebhookPipe
import sys

# Create pipeline components
whisper = WhisperPipe(model="base", buffer_duration=5.0)
llm = LLMPipe(
    model="gpt-3.5-turbo",
    api_key="sk-...",
    system_prompt="Extract action items from this transcription."
)
webhook = WebhookPipe(
    webhook_url="https://example.com/actions",
    batch_size=3
)

# Process audio -> transcription -> LLM -> webhook
for transcription in whisper.run_stream(sys.stdin.buffer):
    if transcription:
        action_items = llm.process_text(transcription)
        webhook.send_text(action_items, metadata={"source": "meeting"})

webhook.flush()
```

### Real-time Meeting Transcription

```bash
#!/bin/bash
# Transcribe Zoom audio and send to Slack

export OPENAI_API_KEY="sk-..."
export SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# Find Zoom process and transcribe
ZOOM_PID=$(pgrep zoom)
proctap -pid $ZOOM_PID --stdout \
    | proctap-whisper --api --model whisper-1 \
    | proctap-webhook $SLACK_WEBHOOK \
        -H "Content-Type: application/json" \
        --template '{"channel": "#meetings"}'
```

## Architecture

### Audio Format

ProcTapPipes supports ProcTap's default audio format:
- **Sample Rate**: 48000 Hz
- **Channels**: 2 (stereo)
- **Format**: s16le (16-bit signed little-endian PCM)

Both raw PCM and WAV formats are automatically detected and supported.

### Design Principles

1. **Unix Philosophy**: Each tool does one thing well
2. **Composability**: Tools can be chained via pipes
3. **Dual Interface**: Same functionality via CLI and Python API
4. **OS-agnostic**: Pure processing, no platform-specific audio capture
5. **Type Safety**: Fully typed with mypy support
6. **Logging**: Diagnostics always go to stderr, never stdout
7. **Windows Compatibility**: Automatic fallback to stderr for text output in binary pipe contexts

## Development

### Setup

```bash
git clone https://github.com/proctap/proctap-pipes.git
cd proctap-pipes
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
pytest --cov=proctap_pipes --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Requirements

- Python 3.10+
- NumPy
- Click
- Requests
- OpenAI (for API features)
- openai-whisper (optional, for local Whisper models)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit a pull request

## Related Projects

- [ProcTap](https://github.com/proctap/proctap) - Per-process audio capture tool

## Support

- Issues: https://github.com/proctap/proctap-pipes/issues
- Documentation: https://github.com/proctap/proctap-pipes#readme

## Roadmap

Future pipes under consideration:

- **VisualizerPipe**: FFT/waveform visualization
- **WebRTCPipe**: Real-time WebRTC audio sending
- **VADPipe**: Voice activity detection
- **AudioFilterPipe**: Noise reduction, equalization
- **WebSocketPipe**: Real-time WebSocket delivery
- **StoragePipe**: Save audio chunks to disk/cloud

## Acknowledgments

Built with support from the ProcTap community.
