"""CLI tool for mixing microphone input with ProcTap audio.

This tool captures system microphone input and mixes it with incoming ProcTap audio,
allowing you to combine process audio with your own voice for streaming, recording,
or further processing.

Usage:
    # Mix microphone with ProcTap audio for Whisper transcription
    proctap --pid 1234 --stdout | proctap-mic-mix | proctap-whisper

    # Mix with custom gain on microphone
    proctap --pid 1234 --stdout | proctap-mic-mix --gain 0.8 | proctap-webhook

    # Use specific microphone device
    proctap --pid 1234 --stdout | proctap-mic-mix --device "USB Microphone" | proctap-whisper

    # Use device index
    proctap --pid 1234 --stdout | proctap-mic-mix --device 0 | proctap-whisper

    # Record to MP3 file using FFmpeg
    proctap --pid 1234 --stdout | proctap-mic-mix | \\
        ffmpeg -f s16le -ar 48000 -ac 2 -i pipe:0 output.mp3

    # Stream to FFmpeg with real-time encoding
    proctap --pid 1234 --stdout | proctap-mic-mix --gain 0.9 | \\
        ffmpeg -f s16le -ar 48000 -ac 2 -i pipe:0 -c:a libmp3lame -b:a 192k output.mp3

    # List available microphone devices
    proctap-mic-mix --list-devices
"""

import logging
import sys

import click

from proctap_pipes.base import AudioFormat
from proctap_pipes.mic_mix_pipe import MicMixPipe


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def list_audio_devices() -> None:
    """List available audio input devices."""
    try:
        import sounddevice as sd

        # Use stdout instead of stderr to avoid Windows console errors
        print("Available audio input devices:")
        print("=" * 60)

        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                name = device["name"]
                channels = device["max_input_channels"]
                sample_rate = device["default_samplerate"]
                default_marker = " (default)" if idx == sd.default.device[0] else ""
                print(
                    f"  [{idx}] {name}{default_marker}\n"
                    f"       Channels: {channels}, Sample Rate: {sample_rate} Hz"
                )

    except ImportError:
        print("Error: sounddevice library not installed.")
        print("Install with: pip install sounddevice")
        sys.exit(1)
    except Exception as e:
        print(f"Error listing devices: {e}")
        sys.exit(1)


@click.command()
@click.option(
    "--sample-rate",
    "-r",
    type=int,
    default=48000,
    help="Output sample rate in Hz (default: 48000)",
)
@click.option(
    "--channels",
    "-c",
    type=int,
    default=2,
    help="Output number of channels (default: 2 for stereo)",
)
@click.option(
    "--sample-width",
    "-w",
    type=int,
    default=2,
    help="Sample width in bytes (default: 2 for 16-bit)",
)
@click.option(
    "--gain",
    "-g",
    type=float,
    default=1.0,
    help="Microphone gain multiplier (0.0-2.0, default: 1.0)",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default=None,
    help="Microphone device name or index (e.g., '0' or 'USB Microphone')",
)
@click.option(
    "--mic-sample-rate",
    type=int,
    default=48000,
    help="Microphone sample rate in Hz (default: 48000)",
)
@click.option(
    "--mic-channels",
    type=int,
    default=2,
    help="Microphone channels (default: 2 for stereo)",
)
@click.option(
    "--no-mic",
    is_flag=True,
    help="Disable microphone (passthrough mode)",
)
@click.option(
    "--list-devices",
    is_flag=True,
    help="List available audio input devices and exit",
)
@click.option(
    "--chunk-size",
    type=int,
    default=4096,
    help="Audio chunk size in frames (default: 4096)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    sample_rate: int,
    channels: int,
    sample_width: int,
    gain: float,
    device: str | None,
    mic_sample_rate: int,
    mic_channels: int,
    no_mic: bool,
    list_devices: bool,
    chunk_size: int,
    verbose: bool,
) -> None:
    """Microphone mixer for ProcTap audio streams.

    Reads PCM audio from stdin, captures microphone input, mixes them together,
    and writes the mixed audio to stdout.

    The mixer applies -6dB gain to both signals to prevent clipping.
    You can adjust the microphone level with the --gain option.
    """
    setup_logging(verbose)

    # List devices and exit if requested
    if list_devices:
        list_audio_devices()
        sys.exit(0)

    # Validate options
    if gain < 0.0 or gain > 2.0:
        print("Error: gain must be between 0.0 and 2.0", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse device - convert to int if it's a numeric string
        mic_device_parsed: str | int | None = device
        if device is not None and device.isdigit():
            mic_device_parsed = int(device)

        # Create audio format
        audio_format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

        # Create mic mix pipe
        pipe = MicMixPipe(
            audio_format=audio_format,
            gain=gain,
            mic_device=mic_device_parsed,
            mic_sample_rate=mic_sample_rate,
            mic_channels=mic_channels,
            enable_mic=not no_mic,
        )

        # Show status to stderr using print to avoid Windows console errors
        if no_mic:
            print("Mic Mix (Passthrough mode - no mic)", file=sys.stderr)
        else:
            print(
                f"Mic Mix (Gain: {gain:.1f}, "
                f"Format: {sample_rate}Hz {channels}ch, "
                f"Ctrl+C to stop)",
                file=sys.stderr,
            )

        # Run CLI with mixing
        pipe.run_cli(
            input_stream=sys.stdin.buffer,
            output_stream=sys.stdout.buffer,
            chunk_size=chunk_size,
        )

    except KeyboardInterrupt:
        print("\nStopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
