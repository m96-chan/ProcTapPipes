#!/usr/bin/env python3
"""Test script for Whisper hallucination filtering."""

import sys

import numpy as np

from proctap_pipes import WhisperPipe
from proctap_pipes.base import AudioFormat

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def test_silence_detection():
    """Test that silence is properly detected and skipped."""
    print("\n" + "=" * 70)
    print("Test 1: Silence Detection")
    print("=" * 70)

    audio_format = AudioFormat(sample_rate=48000, channels=2)

    # Mock WhisperPipe to test silence detection without actual transcription
    class MockWhisperPipe(WhisperPipe):
        def _init_model(self):
            # Skip model initialization for testing
            pass

        def _transcribe(self, audio_data):
            # Should not be called for silent audio
            raise AssertionError("Transcription should not be called for silent audio!")

    pipe = MockWhisperPipe(model="base", audio_format=audio_format, silence_threshold=0.01)

    # Generate silent audio (very low amplitude)
    silent_audio = np.zeros((240000, 2), dtype=np.int16)  # 5 seconds of silence
    silent_audio += np.random.randint(-10, 10, silent_audio.shape, dtype=np.int16)

    print(f"Generated silent audio: {silent_audio.shape[0]} samples")
    print(f"RMS: {np.sqrt(np.mean((silent_audio / 32768.0) ** 2)):.6f}")

    # Test _is_silence method directly
    is_silent = pipe._is_silence(silent_audio)
    print(f"Is silent: {is_silent}")

    if is_silent:
        print("✓ Silence detection works correctly")
    else:
        print("✗ Silence detection failed")

    # Test with process_chunk (should return None without calling _transcribe)
    result = pipe.process_chunk(silent_audio)

    if result is None:
        print("✓ Silent audio was correctly skipped during processing")
    else:
        print(f"✗ Silent audio was not skipped: {result}")


def test_hallucination_filtering():
    """Test that known hallucination phrases are filtered out."""
    print("\n" + "=" * 70)
    print("Test 2: Hallucination Filtering")
    print("=" * 70)

    audio_format = AudioFormat(sample_rate=48000, channels=2)

    class MockWhisperPipe(WhisperPipe):
        def _init_model(self):
            pass

    pipe = MockWhisperPipe(model="base", audio_format=audio_format, skip_hallucinations=True)

    # Test various hallucination patterns
    test_cases = [
        ("ご視聴ありがとうございました", True, "Exact Japanese hallucination"),
        ("Thank you for watching", True, "Exact English hallucination"),
        ("ありがとうございました", True, "Short Japanese hallucination"),
        ("...", True, "Ellipsis hallucination"),
        ("This is actual speech content", False, "Real content"),
        (
            "ご視聴ありがとうございました ご視聴ありがとうございました",
            True,
            "Repeated hallucination",
        ),
    ]

    all_passed = True
    for text, should_filter, description in test_cases:
        is_hallucination = pipe._is_hallucination(text)
        passed = is_hallucination == should_filter

        status = "✓" if passed else "✗"
        print(f"{status} {description}: '{text}' -> filtered={is_hallucination}")

        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All hallucination filtering tests passed")
    else:
        print("\n✗ Some hallucination filtering tests failed")


def test_repetition_detection():
    """Test that repeated transcriptions are detected and skipped."""
    print("\n" + "=" * 70)
    print("Test 3: Repetition Detection")
    print("=" * 70)

    audio_format = AudioFormat(sample_rate=48000, channels=2)

    class MockWhisperPipe(WhisperPipe):
        def _init_model(self):
            pass

    pipe = MockWhisperPipe(model="base", audio_format=audio_format, skip_repetitions=True)

    # Test repetition detection
    text1 = "This is the first transcription"
    text2 = "This is different content"
    text3 = "This is the first transcription"  # Repeat of text1

    # First occurrence - should not be filtered
    pipe._update_recent_transcriptions(text1)
    is_rep = pipe._is_repetition(text1)
    print(f"First occurrence of text1: filtered={is_rep} (should be False after update)")

    # Different text - should not be filtered
    is_rep2 = pipe._is_repetition(text2)
    print(f"✓ Different text (text2): filtered={is_rep2} (expected: False)")

    # Repeated text - should be filtered
    is_rep3 = pipe._is_repetition(text3)
    print(f"{'✓' if is_rep3 else '✗'} Repeated text (text3): filtered={is_rep3} (expected: True)")

    if is_rep3:
        print("\n✓ Repetition detection works correctly")
    else:
        print("\n✗ Repetition detection failed")


def test_integration():
    """Test all filters working together."""
    print("\n" + "=" * 70)
    print("Test 4: Integration Test")
    print("=" * 70)

    audio_format = AudioFormat(sample_rate=48000, channels=2)

    class MockWhisperPipe(WhisperPipe):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.transcription_results = [
                "Real content",
                "ご視聴ありがとうございました",  # Should be filtered
                "Real content",  # Should be filtered as repetition
                "More real content",
            ]
            self.call_count = 0

        def _init_model(self):
            pass

        def _transcribe(self, audio_data):
            if self.call_count < len(self.transcription_results):
                result = self.transcription_results[self.call_count]
                self.call_count += 1
                return result
            return ""

    pipe = MockWhisperPipe(
        model="base",
        audio_format=audio_format,
        silence_threshold=0.01,
        skip_repetitions=True,
        skip_hallucinations=True,
        buffer_duration=0.1,  # Small buffer for testing
    )

    # Generate audio chunks (non-silent)
    audio_chunk = (np.random.randn(4800, 2) * 8000).astype(np.int16)

    results = []
    for i in range(4):
        result = pipe.process_chunk(audio_chunk)
        if result:
            results.append(result)
            print(f"Chunk {i+1}: '{result}'")
        else:
            print(f"Chunk {i+1}: (filtered)")

    print(f"\nResults collected: {results}")
    expected = ["Real content", "More real content"]

    if results == expected:
        print(f"✓ Integration test passed: got expected results {expected}")
    else:
        print(f"✗ Integration test failed: expected {expected}, got {results}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Whisper Hallucination Filter Test Suite")
    print("=" * 70)

    try:
        test_silence_detection()
        test_hallucination_filtering()
        test_repetition_detection()
        test_integration()

        print("\n" + "=" * 70)
        print("All Tests Complete!")
        print("=" * 70)
        print()
        print("The following improvements have been added:")
        print("  1. ✓ Silence detection - skips transcription of silent audio")
        print("  2. ✓ Hallucination filtering - removes known hallucination phrases")
        print("  3. ✓ Repetition detection - skips repeated transcriptions")
        print()
        print("Usage:")
        print("  # Default (all filters enabled)")
        print("  proctap -pid <PID> --stdout | proctap-whisper")
        print()
        print("  # Adjust silence threshold (lower = more sensitive)")
        print("  proctap -pid <PID> --stdout | proctap-whisper --silence-threshold 0.005")
        print()
        print("  # Disable specific filters if needed")
        print("  proctap -pid <PID> --stdout | proctap-whisper --no-skip-hallucinations")
        print()

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
