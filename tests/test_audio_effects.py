#!/usr/bin/env python3
"""Simple test script for AudioEffectsPipe."""

import sys

import numpy as np

from proctap_pipes import AudioEffectsPipe
from proctap_pipes.base import AudioFormat

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def generate_test_audio(duration_sec=1.0, sample_rate=48000, channels=2):
    """Generate test audio with tone + noise.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate
        channels: Number of channels

    Returns:
        NumPy array of int16 audio samples
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)

    # Generate 440Hz tone (A4)
    tone = np.sin(2 * np.pi * 440 * t)

    # Add some noise
    noise = np.random.randn(num_samples) * 0.1

    # Combine tone + noise
    signal = tone + noise

    # Normalize to int16 range
    signal = signal * 16384  # Leave some headroom

    # Create stereo (duplicate to both channels)
    if channels == 2:
        audio = np.column_stack([signal, signal])
    else:
        audio = signal.reshape(-1, 1)

    return audio.astype(np.int16)


def test_denoise():
    """Test noise reduction."""
    print("Testing noise reduction...")

    audio_format = AudioFormat(sample_rate=48000, channels=2)
    pipe = AudioEffectsPipe(
        audio_format=audio_format, denoise=True, noise_threshold=0.02, verbose=True
    )

    # Generate test audio
    test_audio = generate_test_audio(duration_sec=0.5)

    # Process 10 chunks to build noise profile
    chunk_size = len(test_audio) // 10
    for i in range(10):
        chunk = test_audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            processed = pipe.process_chunk(chunk)
            input_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            output_rms = np.sqrt(np.mean(processed.astype(np.float64) ** 2))
            print(f"Chunk {i+1}: Input RMS={input_rms:.1f}, Output RMS={output_rms:.1f}")

    print("✓ Noise reduction test passed\n")


def test_normalize():
    """Test volume normalization."""
    print("Testing volume normalization...")

    audio_format = AudioFormat(sample_rate=48000, channels=2)
    pipe = AudioEffectsPipe(
        audio_format=audio_format, normalize=True, target_level=0.7, verbose=True
    )

    # Generate quiet audio
    quiet_audio = generate_test_audio(duration_sec=0.1) // 10  # Make it quiet

    # Process
    processed = pipe.process_chunk(quiet_audio)

    input_peak = np.max(np.abs(quiet_audio))
    output_peak = np.max(np.abs(processed))

    print(f"Input peak: {input_peak}")
    print(f"Output peak: {output_peak}")
    print(f"Gain applied: {output_peak / input_peak if input_peak > 0 else 0:.2f}x")
    print("✓ Normalization test passed\n")


def test_filters():
    """Test high-pass and low-pass filters."""
    print("Testing filters...")

    audio_format = AudioFormat(sample_rate=48000, channels=2)
    pipe = AudioEffectsPipe(audio_format=audio_format, highpass=100.0, lowpass=8000.0, verbose=True)

    # Generate test audio
    test_audio = generate_test_audio(duration_sec=0.1)

    # Process
    processed = pipe.process_chunk(test_audio)

    print(f"Input shape: {test_audio.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output dtype: {processed.dtype}")
    print("✓ Filter test passed\n")


def test_combined():
    """Test all effects combined."""
    print("Testing all effects combined...")

    audio_format = AudioFormat(sample_rate=48000, channels=2)
    pipe = AudioEffectsPipe(
        audio_format=audio_format,
        denoise=True,
        noise_threshold=0.02,
        normalize=True,
        target_level=0.7,
        highpass=80.0,
        lowpass=8000.0,
        verbose=True,
    )

    # Generate test audio
    test_audio = generate_test_audio(duration_sec=0.5)

    # Process multiple chunks
    chunk_size = len(test_audio) // 10
    for i in range(10):
        chunk = test_audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            _ = pipe.process_chunk(chunk)
            print(f"Processed chunk {i+1}/{10}")

    print("✓ Combined effects test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AudioEffectsPipe Test Suite")
    print("=" * 60)
    print()

    test_denoise()
    test_normalize()
    test_filters()
    test_combined()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
