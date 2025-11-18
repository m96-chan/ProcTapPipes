"""Tests for base pipe functionality."""

import io
import wave
import struct

import numpy as np
import pytest

from proctap_pipes.base import BasePipe, AudioFormat


class TestPipe(BasePipe):
    """Test implementation of BasePipe."""

    def process_chunk(self, audio_data: np.ndarray) -> str:
        """Return chunk info as string."""
        return f"Processed {len(audio_data)} samples"


def test_audio_format_defaults() -> None:
    """Test AudioFormat default values."""
    fmt = AudioFormat()
    assert fmt.sample_rate == 48000
    assert fmt.channels == 2
    assert fmt.sample_width == 2
    assert fmt.frame_size == 4  # 2 bytes * 2 channels


def test_audio_format_custom() -> None:
    """Test AudioFormat with custom values."""
    fmt = AudioFormat(sample_rate=44100, channels=1, sample_width=2)
    assert fmt.sample_rate == 44100
    assert fmt.channels == 1
    assert fmt.sample_width == 2
    assert fmt.frame_size == 2  # 2 bytes * 1 channel


def test_base_pipe_initialization() -> None:
    """Test BasePipe initialization."""
    pipe = TestPipe()
    assert pipe.audio_format.sample_rate == 48000
    assert pipe.audio_format.channels == 2


def test_read_pcm_stream_raw() -> None:
    """Test reading raw PCM data."""
    pipe = TestPipe()

    # Create raw PCM data (100 stereo samples)
    samples = np.random.randint(-32768, 32767, size=(100, 2), dtype=np.int16)
    pcm_bytes = samples.tobytes()

    stream = io.BytesIO(pcm_bytes)

    chunks = list(pipe.read_pcm_stream(stream, chunk_size=50))
    assert len(chunks) == 2
    assert chunks[0].shape == (50, 2)
    assert chunks[1].shape == (50, 2)


def test_read_pcm_stream_wav() -> None:
    """Test reading WAV formatted data."""
    pipe = TestPipe()

    # Create WAV data
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)

        samples = np.random.randint(-32768, 32767, size=(100, 2), dtype=np.int16)
        wav_file.writeframes(samples.tobytes())

    buffer.seek(0)
    chunks = list(pipe.read_pcm_stream(buffer, chunk_size=50))

    assert len(chunks) == 2
    assert chunks[0].shape == (50, 2)


def test_run_stream() -> None:
    """Test run_stream processing."""
    pipe = TestPipe()

    # Create test data
    samples = np.random.randint(-32768, 32767, size=(100, 2), dtype=np.int16)
    stream = io.BytesIO(samples.tobytes())

    results = list(pipe.run_stream(stream, chunk_size=50))

    assert len(results) == 2
    assert all("Processed" in r for r in results)


def test_write_wav() -> None:
    """Test WAV file writing."""
    pipe = TestPipe()

    samples = np.random.randint(-32768, 32767, size=(100, 2), dtype=np.int16)
    output = io.BytesIO()

    pipe.write_wav(samples, output)

    # Verify WAV header
    output.seek(0)
    assert output.read(4) == b"RIFF"
    output.seek(8)
    assert output.read(4) == b"WAVE"
