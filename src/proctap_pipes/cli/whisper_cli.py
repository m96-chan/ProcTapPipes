#!/usr/bin/env python3
"""CLI tool for Whisper transcription.

Usage:
    proctap -pid 1234 --stdout | proctap-whisper
    proctap -pid 1234 --stdout | proctap-whisper --model small --language en
"""

import sys
import logging
import os
from typing import Optional

import click

from proctap_pipes.whisper_pipe import WhisperPipe
from proctap_pipes.base import AudioFormat


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@click.command()
@click.option(
    "--model",
    "-m",
    default="base",
    help="Whisper model (tiny, base, small, medium, large) or whisper-1 for API",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code (e.g., en, es, fr). Auto-detect if not specified.",
)
@click.option(
    "--api",
    is_flag=True,
    help="Use OpenAI API instead of local model",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--buffer",
    "-b",
    default=5.0,
    type=float,
    help="Buffer duration in seconds before transcribing (default: 5.0)",
)
@click.option(
    "--rate",
    "-r",
    default=48000,
    type=int,
    help="Sample rate in Hz (default: 48000)",
)
@click.option(
    "--channels",
    "-c",
    default=2,
    type=int,
    help="Number of audio channels (default: 2)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    model: str,
    language: Optional[str],
    api: bool,
    api_key: Optional[str],
    buffer: float,
    rate: int,
    channels: int,
    verbose: bool,
) -> None:
    """Transcribe audio from stdin using Whisper.

    Reads audio data from stdin (raw PCM or WAV) and outputs transcribed text
    to stdout. Diagnostics are logged to stderr.

    Examples:

        # Using local Whisper model
        proctap -pid 1234 --stdout | proctap-whisper

        # Using OpenAI API
        proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1

        # Specify language and model
        proctap -pid 1234 --stdout | proctap-whisper -m small -l en

        # Chain with LLM processing
        proctap -pid 1234 --stdout | proctap-whisper | proctap-llm
    """
    setup_logging(verbose)

    if api and not api_key:
        click.echo("Error: API key required when using --api", err=True)
        click.echo("Set OPENAI_API_KEY environment variable or use --api-key", err=True)
        sys.exit(1)

    try:
        # Create audio format
        audio_format = AudioFormat(sample_rate=rate, channels=channels)

        # Create Whisper pipe
        pipe = WhisperPipe(
            model=model,
            language=language,
            audio_format=audio_format,
            use_api=api,
            api_key=api_key,
            buffer_duration=buffer,
        )

        # Run CLI mode
        pipe.run_cli()

        # Flush any remaining audio
        if hasattr(pipe, "flush"):
            result = pipe.flush()
            if result:
                print(result)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
