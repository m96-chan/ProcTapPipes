#!/usr/bin/env python3
"""CLI tool for Whisper transcription.

Usage:
    proctap -pid 1234 --stdout | proctap-whisper
    proctap -pid 1234 --stdout | proctap-whisper --model small --language en
    proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1
"""

import sys
import logging
import os
from typing import Optional

import click

from proctap_pipes.whisper_pipe import WhisperPipe, OpenAIWhisperPipe
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
    help="Whisper model (tiny, base, small, medium, large-v3) or whisper-1 for API",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code (e.g., en, ja, es). Auto-detect if not specified.",
)
@click.option(
    "--api",
    is_flag=True,
    help="Use OpenAI API instead of local faster-whisper",
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
    "--device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    help="Device for local model (auto, cpu, cuda). Default: auto",
)
@click.option(
    "--compute-type",
    default="default",
    type=click.Choice(["default", "int8", "int8_float16", "float16"], case_sensitive=False),
    help="Compute type for local model. Default: default",
)
@click.option(
    "--no-vad",
    is_flag=True,
    help="Disable voice activity detection filter",
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
    device: str,
    compute_type: str,
    no_vad: bool,
    rate: int,
    channels: int,
    verbose: bool,
) -> None:
    """Transcribe audio from stdin using Whisper.

    Reads audio data from stdin (raw PCM or WAV) and outputs transcribed text
    to stdout. Diagnostics are logged to stderr.

    By default uses faster-whisper for local transcription. Use --api for OpenAI API.

    Examples:

        # Using local faster-whisper (default)
        proctap -pid 1234 --stdout | proctap-whisper

        # Specify model size and language
        proctap -pid 1234 --stdout | proctap-whisper -m small -l en

        # Use GPU acceleration
        proctap -pid 1234 --stdout | proctap-whisper --device cuda

        # Use OpenAI API
        proctap -pid 1234 --stdout | proctap-whisper --api --model whisper-1

        # Chain with LLM processing
        proctap -pid 1234 --stdout | proctap-whisper | proctap-llm

        # Optimize for speed (int8 quantization)
        proctap -pid 1234 --stdout | proctap-whisper --compute-type int8
    """
    setup_logging(verbose)

    if api and not api_key:
        click.echo("Error: API key required when using --api", err=True)
        click.echo("Set OPENAI_API_KEY environment variable or use --api-key", err=True)
        sys.exit(1)

    try:
        # Create audio format
        audio_format = AudioFormat(sample_rate=rate, channels=channels)

        # Create appropriate Whisper pipe
        if api:
            # Use OpenAI API
            pipe = OpenAIWhisperPipe(
                api_key=api_key,  # type: ignore
                model=model,
                language=language,
                audio_format=audio_format,
                buffer_duration=buffer,
            )
        else:
            # Use local faster-whisper
            pipe = WhisperPipe(
                model=model,
                language=language,
                audio_format=audio_format,
                device=device,
                compute_type=compute_type,
                buffer_duration=buffer,
                vad_filter=not no_vad,
            )

        # Run CLI mode
        pipe.run_cli()

        # Flush any remaining audio
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
