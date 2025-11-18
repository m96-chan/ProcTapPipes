#!/usr/bin/env python3
"""CLI tool for LLM text processing.

Usage:
    echo "What is the meaning of life?" | proctap-llm
    proctap -pid 1234 --stdout | proctap-whisper | proctap-llm
"""

import sys
import logging
from typing import Optional

import click

from proctap_pipes.llm_pipe import LLMPipe, LLMPipeWithContext


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
    default="gpt-3.5-turbo",
    help="LLM model name (default: gpt-3.5-turbo)",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--system-prompt",
    "-s",
    default=None,
    help="System prompt for the LLM",
)
@click.option(
    "--temperature",
    "-t",
    default=0.7,
    type=float,
    help="Sampling temperature (0.0 to 2.0, default: 0.7)",
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="Maximum tokens in response",
)
@click.option(
    "--base-url",
    default=None,
    help="Custom API base URL",
)
@click.option(
    "--context",
    is_flag=True,
    help="Maintain conversation context across inputs",
)
@click.option(
    "--max-context",
    default=10,
    type=int,
    help="Maximum context messages to keep (default: 10)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    base_url: Optional[str],
    context: bool,
    max_context: int,
    verbose: bool,
) -> None:
    """Process text from stdin through an LLM.

    Reads text from stdin (line by line or from pipes) and processes each
    input through an LLM. Outputs responses to stdout, logs to stderr.

    Examples:

        # Simple text processing
        echo "Summarize: The quick brown fox..." | proctap-llm

        # Chain after Whisper transcription
        proctap -pid 1234 --stdout | proctap-whisper | proctap-llm

        # Use custom system prompt
        echo "Hello!" | proctap-llm -s "You are a pirate assistant"

        # Maintain context across inputs
        cat conversation.txt | proctap-llm --context

        # Use different model
        echo "Explain quantum computing" | proctap-llm -m gpt-4
    """
    setup_logging(verbose)

    if not api_key:
        click.echo("Error: API key required", err=True)
        click.echo("Set OPENAI_API_KEY environment variable or use --api-key", err=True)
        sys.exit(1)

    try:
        # Create LLM pipe
        if context:
            pipe = LLMPipeWithContext(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                max_context_messages=max_context,
            )
        else:
            pipe = LLMPipe(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
            )

        # Process text from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                result = pipe.process_text(line)
                print(result)
                sys.stdout.flush()
            except Exception as e:
                logging.error(f"Error processing line: {e}", exc_info=verbose)

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
