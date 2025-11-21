"""Tests for MicMixPipe."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from proctap_pipes.base import AudioFormat
from proctap_pipes.mic_mix_pipe import MicMixPipe


@pytest.fixture
def audio_format() -> AudioFormat:
    """Create test audio format (ProcTap standard: 16-bit PCM, stereo, 48kHz)."""
    return AudioFormat(sample_rate=48000, channels=2, sample_width=2)


@pytest.fixture
def mock_mic_device() -> Mock:
    """Create a mock microphone device."""
    mock_device = Mock()
    mock_device.read = Mock(return_value=np.zeros((1024, 1), dtype=np.int16))
    return mock_device


def test_mic_mix_initialization(audio_format: AudioFormat) -> None:
    """Test MicMixPipe initialization."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(audio_format=audio_format, gain=0.8)
        assert pipe.gain == 0.8
        assert pipe.audio_format.sample_rate == 48000
        assert pipe.audio_format.channels == 2


def test_mic_mix_default_gain(audio_format: AudioFormat) -> None:
    """Test default gain is 1.0."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(audio_format=audio_format)
        assert pipe.gain == 1.0


def test_mic_mix_invalid_gain(audio_format: AudioFormat) -> None:
    """Test that invalid gain values are handled."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        # Negative gain should be clamped to 0
        pipe = MicMixPipe(audio_format=audio_format, gain=-0.5)
        assert pipe.gain >= 0.0

        # Gain > 2.0 should be clamped
        pipe = MicMixPipe(audio_format=audio_format, gain=5.0)
        assert pipe.gain <= 2.0


def test_mix_audio_equal_gain() -> None:
    """Test mixing audio with equal gain."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(gain=1.0)

        # Create test signals
        proctap_audio = np.array([[1000], [2000], [3000]], dtype=np.int16)
        mic_audio = np.array([[500], [1000], [1500]], dtype=np.int16)

        # Mix should apply -6dB gain (0.5) to both and sum
        result = pipe._mix_audio(proctap_audio, mic_audio)

        # Expected: (proctap + mic) * 0.5
        expected = ((proctap_audio.astype(np.float32) + mic_audio.astype(np.float32)) * 0.5).astype(
            np.int16
        )

        np.testing.assert_array_almost_equal(result, expected, decimal=0)


def test_mix_audio_with_gain() -> None:
    """Test mixing audio with custom gain on mic input."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(gain=0.5)

        proctap_audio = np.array([[1000], [2000]], dtype=np.int16)
        mic_audio = np.array([[1000], [2000]], dtype=np.int16)

        result = pipe._mix_audio(proctap_audio, mic_audio)

        # Mic should be at 0.5 gain before mixing
        # Then both signals mixed at -6dB (0.5)
        assert result.shape == proctap_audio.shape


def test_mix_audio_prevents_clipping() -> None:
    """Test that mixing prevents integer overflow/clipping."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(gain=1.0)

        # Create signals that would clip if not handled
        proctap_audio = np.array([[30000], [32000]], dtype=np.int16)
        mic_audio = np.array([[30000], [32000]], dtype=np.int16)

        result = pipe._mix_audio(proctap_audio, mic_audio)

        # Should not exceed int16 range
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)


def test_resample_mic_audio() -> None:
    """Test resampling microphone audio to match ProcTap format."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        # Mic at 16kHz, ProcTap at 48kHz
        pipe = MicMixPipe(
            audio_format=AudioFormat(sample_rate=48000, channels=2, sample_width=2),
            mic_sample_rate=16000,
            mic_channels=2,
        )

        # Create 160 samples at 16kHz (10ms)
        mic_audio_16k = np.random.randint(-1000, 1000, size=(160, 2), dtype=np.int16)

        # Resample to 48kHz should give ~480 samples
        result = pipe._resample_mic_audio(mic_audio_16k)

        expected_samples = int(160 * 48000 / 16000)
        assert result.shape[0] == expected_samples
        assert result.shape[1] == 2


def test_convert_stereo_to_mono() -> None:
    """Test converting stereo mic input to mono (for custom use case)."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        # Test case where output is mono but mic is stereo
        pipe = MicMixPipe(
            audio_format=AudioFormat(sample_rate=48000, channels=1, sample_width=2),
            mic_sample_rate=48000,
            mic_channels=2,
        )

        # Create stereo audio
        stereo_audio = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.int16)

        result = pipe._convert_to_mono(stereo_audio)

        # Should average the channels
        expected = np.array([[150], [350], [550]], dtype=np.int16)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)


def test_process_chunk_with_mic_input(mock_mic_device: Mock) -> None:
    """Test processing audio chunk with microphone input."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._read_mic_chunk") as mock_read:
            pipe = MicMixPipe(gain=1.0)

            # Mock microphone read
            proctap_audio = np.random.randint(-1000, 1000, size=(1024, 1), dtype=np.int16)
            mic_audio = np.random.randint(-500, 500, size=(1024, 1), dtype=np.int16)
            mock_read.return_value = mic_audio

            result = pipe.process_chunk(proctap_audio)

            # Should return mixed audio
            assert result is not None
            assert result.shape == proctap_audio.shape
            assert result.dtype == np.int16


def test_process_chunk_passthrough_on_mic_error() -> None:
    """Test that process_chunk passes through audio if mic read fails."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        with patch(
            "proctap_pipes.mic_mix_pipe.MicMixPipe._read_mic_chunk",
            side_effect=Exception("Mic error"),
        ):
            pipe = MicMixPipe(gain=1.0)

            proctap_audio = np.random.randint(-1000, 1000, size=(1024, 1), dtype=np.int16)

            # Should pass through original audio on error
            result = pipe.process_chunk(proctap_audio)

            np.testing.assert_array_equal(result, proctap_audio)


def test_flush_closes_mic_device() -> None:
    """Test that flush properly closes microphone device."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(gain=1.0)
        pipe.mic_device = Mock()
        pipe.mic_device.close = Mock()

        result = pipe.flush()

        # Should close the device
        pipe.mic_device.close.assert_called_once()
        assert result is None


def test_mic_device_selection() -> None:
    """Test specifying a specific microphone device."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture") as mock_init:
        # Test with device name (string)
        pipe = MicMixPipe(mic_device="USB Microphone")
        mock_init.assert_called_once()
        assert pipe.mic_device_name == "USB Microphone"

        # Test with device index (int)
        mock_init.reset_mock()
        pipe_idx = MicMixPipe(mic_device=0)
        mock_init.assert_called_once()
        assert pipe_idx.mic_device_name == 0


def test_passthrough_mode() -> None:
    """Test passthrough mode when mic is disabled."""
    with patch("proctap_pipes.mic_mix_pipe.MicMixPipe._init_mic_capture"):
        pipe = MicMixPipe(enable_mic=False)

        proctap_audio = np.random.randint(-1000, 1000, size=(1024, 1), dtype=np.int16)

        result = pipe.process_chunk(proctap_audio)

        # Should return audio unchanged
        np.testing.assert_array_equal(result, proctap_audio)
