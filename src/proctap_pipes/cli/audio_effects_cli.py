"""CLI tool for real-time audio effects processing.

This tool applies various audio effects (noise reduction, normalization, EQ)
to improve audio quality for transcription or other processing.

Usage:
    # Apply noise reduction
    proctap -pid 1234 --stdout | proctap-effects --denoise | proctap-whisper

    # Apply multiple effects
    proctap -pid 1234 --stdout | proctap-effects --denoise --normalize \\
        --highpass 80 | proctap-whisper

    # Just normalize volume
    proctap -pid 1234 --stdout | proctap-effects --normalize | proctap-whisper

    # Full enhancement pipeline
    proctap -pid 1234 --stdout | proctap-effects --denoise --normalize \\
        --highpass 80 --lowpass 8000 | proctap-whisper
"""

import logging
import sys

import click

from proctap_pipes.audio_effects_pipe import AudioEffectsPipe, NoiseReductionMode
from proctap_pipes.base import AudioFormat


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


@click.command()
@click.option(
    "--sample-rate",
    "-r",
    type=int,
    default=48000,
    help="Sample rate in Hz (default: 48000)",
)
@click.option(
    "--channels",
    "-c",
    type=int,
    default=2,
    help="Number of channels (default: 2)",
)
@click.option(
    "--sample-width",
    "-w",
    type=int,
    default=2,
    help="Sample width in bytes (default: 2 for 16-bit)",
)
@click.option(
    "--denoise",
    "-d",
    is_flag=True,
    help="Enable noise reduction",
)
@click.option(
    "--noise-mode",
    type=click.Choice(["simple", "spectral_gate", "wiener"], case_sensitive=False),
    default="simple",
    help=(
        "Noise reduction algorithm: simple (fast), spectral_gate (balanced), "
        "wiener (slow, best quality). Default: simple"
    ),
)
@click.option(
    "--noise-threshold",
    "-t",
    type=float,
    default=0.02,
    help="Noise reduction threshold, 0.0-1.0, lower = more aggressive (default: 0.02)",
)
@click.option(
    "--normalize",
    "-n",
    is_flag=True,
    help="Enable volume normalization",
)
@click.option(
    "--target-level",
    "-l",
    type=float,
    default=0.7,
    help="Target volume level for normalization, 0.0-1.0 (default: 0.7)",
)
@click.option(
    "--highpass",
    "-hp",
    type=float,
    default=None,
    help="High-pass filter cutoff frequency in Hz (removes low frequencies/rumble)",
)
@click.option(
    "--lowpass",
    "-lp",
    type=float,
    default=None,
    help="Low-pass filter cutoff frequency in Hz (removes high frequencies/hiss)",
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
    denoise: bool,
    noise_mode: str,
    noise_threshold: float,
    normalize: bool,
    target_level: float,
    highpass: float | None,
    lowpass: float | None,
    chunk_size: int,
    verbose: bool,
) -> None:
    """Real-time audio effects processing.

    Reads PCM audio from stdin, applies effects, and writes processed audio to stdout.

    This tool is designed to improve audio quality before transcription or other
    downstream processing.

    Effects are applied in this order:
    1. High-pass filter (if enabled)
    2. Low-pass filter (if enabled)
    3. Noise reduction (if enabled)
    4. Normalization (if enabled)
    """
    setup_logging(verbose)

    # Validate options
    if noise_threshold < 0.0 or noise_threshold > 1.0:
        click.echo("Error: noise-threshold must be between 0.0 and 1.0", err=True)
        sys.exit(1)

    if target_level < 0.0 or target_level > 1.0:
        click.echo("Error: target-level must be between 0.0 and 1.0", err=True)
        sys.exit(1)

    if highpass is not None and highpass <= 0:
        click.echo("Error: highpass frequency must be positive", err=True)
        sys.exit(1)

    if lowpass is not None and lowpass <= 0:
        click.echo("Error: lowpass frequency must be positive", err=True)
        sys.exit(1)

    if highpass is not None and lowpass is not None and highpass >= lowpass:
        click.echo("Error: highpass frequency must be less than lowpass frequency", err=True)
        sys.exit(1)

    # Check if any effects are enabled
    if not any([denoise, normalize, highpass is not None, lowpass is not None]):
        click.echo("Warning: No effects enabled. Audio will pass through unchanged.", err=True)
        click.echo(
            "Use --denoise, --normalize, --highpass, or --lowpass to enable effects.", err=True
        )

    try:
        # Create audio format
        audio_format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

        # Create audio effects pipe
        pipe = AudioEffectsPipe(
            audio_format=audio_format,
            denoise=denoise,
            noise_reduction_mode=NoiseReductionMode(noise_mode),
            noise_threshold=noise_threshold,
            normalize=normalize,
            target_level=target_level,
            highpass=highpass,
            lowpass=lowpass,
            verbose=verbose,
        )

        # Display active effects
        effects_list = []
        if denoise:
            effects_list.append(f"Noise Reduction ({noise_mode})")
        if normalize:
            effects_list.append(f"Normalization (target={target_level})")
        if highpass:
            effects_list.append(f"High-pass ({highpass}Hz)")
        if lowpass:
            effects_list.append(f"Low-pass ({lowpass}Hz)")

        if effects_list:
            click.echo("Audio Effects Processor (Ctrl+C to stop)", err=True)
            click.echo("Active effects: " + ", ".join(effects_list), err=True)
            click.echo("=" * 60, err=True)
        else:
            click.echo("Audio Passthrough (no effects enabled)", err=True)

        # Run CLI with passthrough
        pipe.run_cli(
            input_stream=sys.stdin.buffer,
            output_stream=sys.stdout.buffer,
            chunk_size=chunk_size,
        )

    except KeyboardInterrupt:
        click.echo("\nStopped by user", err=True)
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
