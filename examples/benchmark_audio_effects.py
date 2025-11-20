#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark script for audio effects performance."""

import sys
import time

import numpy as np

from proctap_pipes import AudioEffectsPipe
from proctap_pipes.base import AudioFormat

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def benchmark_mode(mode, audio_format, test_audio, num_iterations=10):
    """Benchmark a specific noise reduction mode.

    Args:
        mode: Noise reduction mode
        audio_format: Audio format
        test_audio: Test audio data
        num_iterations: Number of iterations to average
    """
    pipe = AudioEffectsPipe(
        audio_format=audio_format,
        denoise=True,
        noise_reduction_mode=mode,
        verbose=False,
    )

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        processed = pipe.process_chunk(test_audio)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    return avg_time, std_time, min_time, max_time


def main():
    """Run benchmarks."""
    print("\n" + "=" * 70)
    print("Audio Effects Performance Benchmark")
    print("=" * 70)
    print()

    # Test parameters
    audio_format = AudioFormat(sample_rate=48000, channels=2)
    duration = 0.1  # 0.1 seconds (4800 samples at 48kHz)
    num_samples = int(duration * audio_format.sample_rate)

    # Generate test audio (noise + tone)
    print(f"Generating test audio: {duration}s ({num_samples} samples, stereo)")
    t = np.linspace(0, duration, num_samples)
    tone = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    noise = np.random.randn(num_samples) * 0.1
    signal = tone + noise
    test_audio = np.column_stack([signal, signal]).astype(np.int16) * 8000

    print(f"Audio shape: {test_audio.shape}")
    print()

    # Benchmark each mode
    modes = ["simple", "spectral_gate", "wiener"]
    results = {}

    for mode in modes:
        print(f"Benchmarking {mode} mode...")
        avg, std, min_t, max_t = benchmark_mode(mode, audio_format, test_audio)
        results[mode] = {
            "avg": avg,
            "std": std,
            "min": min_t,
            "max": max_t,
        }
        print(f"  Average: {avg:.2f}ms (±{std:.2f}ms)")
        print(f"  Range: {min_t:.2f}ms - {max_t:.2f}ms")
        print()

    # Summary
    print("=" * 70)
    print("Summary (processing 0.1s of audio)")
    print("=" * 70)
    print()
    print(f"{'Mode':<20} {'Avg Time':<15} {'Real-time Factor':<20}")
    print("-" * 70)

    for mode in modes:
        avg = results[mode]["avg"]
        # Real-time factor: how much faster than real-time
        # e.g., if it takes 10ms to process 100ms of audio, factor is 10x
        rt_factor = (duration * 1000) / avg
        speed_desc = "FAST" if rt_factor > 50 else "OK" if rt_factor > 20 else "SLOW"

        print(f"{mode:<20} {avg:>6.2f}ms        {rt_factor:>6.1f}x ({speed_desc})")

    print()
    print("Recommendations:")
    print("  - For real-time processing: Use 'simple' mode (fastest)")
    print("  - For balanced quality/speed: Use 'spectral_gate' mode")
    print("  - For best quality: Use 'wiener' mode (slowest)")
    print()

    # Calculate throughput
    simple_throughput = (duration * 1000) / results["simple"]["avg"]
    print(f"With 'simple' mode, you can process audio {simple_throughput:.1f}x faster than real-time")
    print(
        f"This means processing 1 hour of audio would take {3600 / simple_throughput / 60:.1f} minutes"
    )
    print()


if __name__ == "__main__":
    main()
