# Whisper Implementation Migration

## Changes

### Old Implementation (openai-whisper)
- Single `WhisperPipe` class with `use_api` flag
- Used `openai-whisper` package for local inference
- Slower inference speed

### New Implementation (faster-whisper)

**WhisperPipe** (Default, Recommended)
- Uses `faster-whisper` (CTranslate2-based)
- 4x faster inference than openai-whisper
- Lower memory usage
- GPU acceleration support
- Voice Activity Detection (VAD) built-in
- Streaming-optimized

**OpenAIWhisperPipe** (API-based)
- Separate class for OpenAI API
- Requires API key and internet
- Pay-per-use pricing
- No local resources needed

## Installation

```bash
# For local faster-whisper (recommended)
pip install proctap-pipes[whisper]

# For OpenAI API only
pip install proctap-pipes[whisper-openai]

# For both
pip install proctap-pipes[all]
```

## Usage

### Python API

**Local (faster-whisper)**
```python
from proctap_pipes import WhisperPipe

# Basic usage
pipe = WhisperPipe(model="base", language="en")

# Advanced options
pipe = WhisperPipe(
    model="large-v3",
    language="ja",
    device="cuda",           # Use GPU
    compute_type="int8",     # Faster inference
    vad_filter=True,         # Filter silence
    beam_size=5,
)

for transcription in pipe.run_stream(audio_stream):
    print(transcription)
```

**OpenAI API**
```python
from proctap_pipes import OpenAIWhisperPipe

pipe = OpenAIWhisperPipe(
    api_key="sk-...",
    model="whisper-1",
    language="en",
    prompt="Technical discussion about AI",  # Guide style
)

for transcription in pipe.run_stream(audio_stream):
    print(transcription)
```

### CLI

**Local (faster-whisper)**
```bash
# Default (base model, CPU)
proctap -pid 1234 --stdout | proctap-whisper

# Specify model and language
proctap -pid 1234 --stdout | proctap-whisper -m small -l en

# Use GPU acceleration
proctap -pid 1234 --stdout | proctap-whisper --device cuda

# Optimize for speed (int8 quantization)
proctap -pid 1234 --stdout | proctap-whisper --compute-type int8

# Disable VAD (transcribe everything including silence)
proctap -pid 1234 --stdout | proctap-whisper --no-vad
```

**OpenAI API**
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Use API
proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1

# With language
proctap -pid 1234 --stdout | proctap-whisper --api -l en
```

## Performance Comparison

| Implementation | Speed | Memory | GPU | Cost |
|---------------|-------|--------|-----|------|
| faster-whisper | 4x faster | Low | ✅ | Free |
| OpenAI API | Varies | None (cloud) | N/A | $0.006/min |

## Model Sizes (faster-whisper)

| Model | Parameters | VRAM (GPU) | Speed | Quality |
|-------|-----------|------------|-------|---------|
| tiny | 39M | ~1GB | Fastest | Basic |
| base | 74M | ~1GB | Very Fast | Good |
| small | 244M | ~2GB | Fast | Better |
| medium | 769M | ~5GB | Moderate | Great |
| large-v3 | 1550M | ~10GB | Slower | Best |

### Compute Types

- `default`: Standard FP32/FP16 (highest quality)
- `int8`: 8-bit quantization (faster, good quality)
- `int8_float16`: Mixed precision (balanced)
- `float16`: FP16 (faster on GPU, good quality)

## Migration Guide

### Code Changes

**Before:**
```python
from proctap_pipes import WhisperPipe

# Local model
pipe = WhisperPipe(model="base", use_api=False)

# API
pipe = WhisperPipe(model="whisper-1", use_api=True, api_key="sk-...")
```

**After:**
```python
from proctap_pipes import WhisperPipe, OpenAIWhisperPipe

# Local model (faster-whisper)
pipe = WhisperPipe(model="base")

# API
pipe = OpenAIWhisperPipe(api_key="sk-...", model="whisper-1")
```

### CLI Changes

**Before:**
```bash
# Local
proctap ... | proctap-whisper -m base

# API
proctap ... | proctap-whisper --api --model whisper-1
```

**After:**
```bash
# Local (now uses faster-whisper)
proctap ... | proctap-whisper -m base

# API (same, but now uses OpenAIWhisperPipe internally)
proctap ... | proctap-whisper --api --model whisper-1
```

## New Features

### 1. GPU Acceleration
```bash
proctap -pid 1234 --stdout | proctap-whisper --device cuda
```

### 2. Compute Type Optimization
```bash
# 4x faster with minimal quality loss
proctap -pid 1234 --stdout | proctap-whisper --compute-type int8
```

### 3. Voice Activity Detection
```bash
# Automatically filters silence (default: enabled)
proctap -pid 1234 --stdout | proctap-whisper

# Disable to transcribe everything
proctap -pid 1234 --stdout | proctap-whisper --no-vad
```

### 4. Language Detection Info
```python
pipe = WhisperPipe(model="base")
# Logs: "Detected language: en (probability: 0.95)"
```

## Troubleshooting

### "faster_whisper not found"
```bash
pip install faster-whisper
```

### CUDA out of memory
```bash
# Use smaller model
proctap -pid 1234 --stdout | proctap-whisper -m small

# Or use int8 quantization
proctap -pid 1234 --stdout | proctap-whisper --compute-type int8

# Or use CPU
proctap -pid 1234 --stdout | proctap-whisper --device cpu
```

### Slow transcription on CPU
```bash
# Use int8 quantization for 4x speedup
proctap -pid 1234 --stdout | proctap-whisper --compute-type int8

# Or use smaller model
proctap -pid 1234 --stdout | proctap-whisper -m tiny
```

## Recommendations

### For Real-time Transcription
- Model: `tiny` or `base`
- Compute: `int8`
- Device: `cuda` if available
- VAD: enabled (default)

```bash
proctap -pid 1234 --stdout | proctap-whisper -m tiny --compute-type int8 --device cuda
```

### For High Accuracy
- Model: `large-v3`
- Compute: `default` or `float16`
- Device: `cuda`
- VAD: enabled

```bash
proctap -pid 1234 --stdout | proctap-whisper -m large-v3 --device cuda
```

### For Low Resources
- Model: `tiny`
- Compute: `int8`
- Device: `cpu`

```bash
proctap -pid 1234 --stdout | proctap-whisper -m tiny --compute-type int8 --device cpu
```

## References

- [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
