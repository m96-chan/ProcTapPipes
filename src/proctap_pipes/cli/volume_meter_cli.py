"""CLI tool for real-time volume meter with passthrough.

This tool displays audio levels in the terminal while passing PCM data through
unchanged, allowing it to be used in the middle of processing pipelines.

Usage:
    # Monitor volume while transcribing
    proctap -pid 1234 --stdout | proctap-volume-meter | proctap-whisper

    # Monitor volume while sending to webhook
    proctap -pid 1234 --stdout | proctap-volume-meter --bar-width 60 | proctap-webhook

    # Just monitor without further processing
    proctap -pid 1234 --stdout | proctap-volume-meter > /dev/null
"""

import logging
import sys

import click

from proctap_pipes.base import AudioFormat
from proctap_pipes.volume_meter_pipe import VolumeMeterPipe


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
    "--bar-width",
    "-b",
    type=int,
    default=50,
    help="Width of the volume bar in characters (default: 50)",
)
@click.option(
    "--update-interval",
    "-u",
    type=float,
    default=0.05,
    help="Display update interval in seconds (default: 0.05)",
)
@click.option(
    "--peak-hold",
    "-p",
    type=float,
    default=1.0,
    help="Peak hold time in seconds (default: 1.0)",
)
@click.option(
    "--no-db",
    is_flag=True,
    help="Hide dB values",
)
@click.option(
    "--no-rms",
    is_flag=True,
    help="Hide RMS meter",
)
@click.option(
    "--no-peak",
    is_flag=True,
    help="Hide peak meter",
)
@click.option(
    "--db-min",
    type=float,
    default=-60.0,
    help="Minimum dB value for display range (default: -60.0)",
)
@click.option(
    "--db-max",
    type=float,
    default=0.0,
    help="Maximum dB value for display range (default: 0.0)",
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
    bar_width: int,
    update_interval: float,
    peak_hold: float,
    no_db: bool,
    no_rms: bool,
    no_peak: bool,
    db_min: float,
    db_max: float,
    chunk_size: int,
    verbose: bool,
) -> None:
    """Real-time volume meter with audio passthrough.

    Reads PCM audio from stdin, displays volume levels to stderr,
    and writes the same audio unchanged to stdout.

    This allows inserting volume monitoring into audio processing pipelines.
    """
    setup_logging(verbose)

    # Validate options
    if no_rms and no_peak:
        click.echo("Error: Cannot hide both RMS and peak meters", err=True)
        sys.exit(1)

    if db_min >= db_max:
        click.echo("Error: db-min must be less than db-max", err=True)
        sys.exit(1)

    try:
        # Create audio format
        audio_format = AudioFormat(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

        # Create volume meter pipe
        pipe = VolumeMeterPipe(
            audio_format=audio_format,
            bar_width=bar_width,
            update_interval=update_interval,
            peak_hold_time=peak_hold,
            show_db=not no_db,
            show_rms=not no_rms,
            show_peak=not no_peak,
            db_range=(db_min, db_max),
        )

        # Run CLI with passthrough
        click.echo("Volume Meter (Ctrl+C to stop)", err=True)
        click.echo("=" * (bar_width + 20), err=True)

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
