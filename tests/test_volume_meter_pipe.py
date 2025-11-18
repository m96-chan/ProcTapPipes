"""Tests for VolumeMeterPipe."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from proctap_pipes.volume_meter_pipe import VolumeMeterPipe
from proctap_pipes.base import AudioFormat


@pytest.fixture
def audio_format():
    """Create test audio format."""
    return AudioFormat(sample_rate=48000, channels=2, sample_width=2)


@pytest.fixture
def volume_meter(audio_format):
    """Create VolumeMeterPipe instance."""
    return VolumeMeterPipe(
        audio_format=audio_format,
        bar_width=50,
        update_interval=0.0,  # Update every chunk for testing
    )


def test_volume_meter_initialization(audio_format):
    """Test VolumeMeterPipe initialization."""
    pipe = VolumeMeterPipe(audio_format=audio_format)
    assert pipe.bar_width == 50
    assert pipe.update_interval == 0.05
    assert pipe.show_db is True
    assert pipe.show_rms is True
    assert pipe.show_peak is True


def test_volume_meter_custom_settings(audio_format):
    """Test VolumeMeterPipe with custom settings."""
    pipe = VolumeMeterPipe(
        audio_format=audio_format,
        bar_width=30,
        update_interval=0.1,
        show_db=False,
        show_rms=False,
        db_range=(-80.0, 0.0),
    )
    assert pipe.bar_width == 30
    assert pipe.update_interval == 0.1
    assert pipe.show_db is False
    assert pipe.show_rms is False
    assert pipe.db_min == -80.0
    assert pipe.db_max == 0.0


def test_calculate_rms(volume_meter):
    """Test RMS calculation."""
    # Create test signal: full scale sine wave
    samples = 1000
    audio_data = (np.sin(np.linspace(0, 2 * np.pi, samples)) * 32767).astype(np.int16)
    audio_data = audio_data.reshape(-1, 2)  # Stereo

    rms = volume_meter._calculate_rms(audio_data)

    # RMS of a sine wave should be ~0.707 (1/sqrt(2))
    assert 0.6 < rms < 0.8


def test_calculate_peak(volume_meter):
    """Test peak calculation."""
    # Create test signal with known peak
    audio_data = np.array([0, 16384, -32768, 8192], dtype=np.int16).reshape(-1, 2)

    peak = volume_meter._calculate_peak(audio_data)

    # Peak should be 32768/32768 = 1.0
    assert peak == pytest.approx(1.0, rel=0.01)


def test_amplitude_to_db(volume_meter):
    """Test amplitude to dB conversion."""
    # Test known conversions
    assert volume_meter._amplitude_to_db(1.0) == pytest.approx(0.0, abs=0.01)
    assert volume_meter._amplitude_to_db(0.5) == pytest.approx(-6.02, abs=0.1)
    assert volume_meter._amplitude_to_db(0.1) == pytest.approx(-20.0, abs=0.1)
    assert volume_meter._amplitude_to_db(0.0) == volume_meter.db_min


def test_create_bar(volume_meter):
    """Test bar creation."""
    # Test different levels
    bar_full = volume_meter._create_bar(1.0)
    bar_half = volume_meter._create_bar(0.1)  # -20 dB
    bar_quiet = volume_meter._create_bar(0.001)  # -60 dB

    # Full volume should have more filled characters
    assert "█" in bar_full
    assert bar_full.count("█") > bar_half.count("█")
    assert bar_half.count("█") > bar_quiet.count("█")


@patch("sys.stderr", new_callable=io.StringIO)
def test_process_chunk_passthrough(mock_stderr, volume_meter):
    """Test that process_chunk returns audio unchanged."""
    # Create test audio
    audio_data = np.random.randint(-32768, 32767, size=(1000, 2), dtype=np.int16)

    # Process chunk
    result = volume_meter.process_chunk(audio_data)

    # Should return the same audio data
    assert result is not None
    np.testing.assert_array_equal(result, audio_data)


@patch("sys.stderr", new_callable=io.StringIO)
def test_display_meter(mock_stderr, volume_meter):
    """Test that display_meter writes to stderr."""
    volume_meter._display_meter(rms=0.5, peak=0.8)

    output = mock_stderr.getvalue()
    assert "RMS" in output or "Peak" in output


@patch("sys.stderr", new_callable=io.StringIO)
def test_flush(mock_stderr, volume_meter):
    """Test flush adds newline."""
    # Process at least one chunk first
    audio_data = np.random.randint(-32768, 32767, size=(1000, 2), dtype=np.int16)
    volume_meter.process_chunk(audio_data)

    # Call flush
    result = volume_meter.flush()

    # Should return None (no buffered data)
    assert result is None

    # Should have written to stderr
    output = mock_stderr.getvalue()
    assert "\n" in output


def test_peak_hold(audio_format):
    """Test peak hold functionality."""
    pipe = VolumeMeterPipe(audio_format=audio_format, peak_hold_time=1.0, update_interval=0.0)

    # Create loud audio
    loud_audio = (np.ones((1000, 2)) * 32000).astype(np.int16)
    pipe.process_chunk(loud_audio)
    first_peak = pipe.peak_value

    # Create quiet audio
    quiet_audio = (np.ones((1000, 2)) * 1000).astype(np.int16)
    pipe.process_chunk(quiet_audio)

    # Peak should still be high due to hold
    assert pipe.peak_value == first_peak


@patch("sys.stderr", new_callable=io.StringIO)
def test_run_stream(mock_stderr, volume_meter):
    """Test run_stream processes audio correctly."""
    # Create test WAV data
    audio_data = np.random.randint(-32768, 32767, size=(5000, 2), dtype=np.int16)
    input_stream = io.BytesIO(audio_data.tobytes())

    results = list(volume_meter.run_stream(input_stream, chunk_size=1000))

    # Should return all chunks unchanged
    assert len(results) == 5
    for result in results:
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2  # Stereo


def test_meter_with_silence(volume_meter):
    """Test meter with silent audio."""
    silence = np.zeros((1000, 2), dtype=np.int16)

    result = volume_meter.process_chunk(silence)

    # Should still return the audio
    assert result is not None
    np.testing.assert_array_equal(result, silence)


def test_meter_with_clipping(volume_meter):
    """Test meter with clipping audio."""
    # Create audio that clips
    clipping = (np.ones((1000, 2)) * 32767).astype(np.int16)

    result = volume_meter.process_chunk(clipping)

    # Should still return the audio
    assert result is not None
    np.testing.assert_array_equal(result, clipping)
