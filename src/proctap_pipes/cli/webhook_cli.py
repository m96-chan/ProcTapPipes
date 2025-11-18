#!/usr/bin/env python3
"""CLI tool for webhook delivery.

Usage:
    echo "Hello World" | proctap-webhook https://example.com/webhook
    proctap -pid 1234 --stdout | proctap-whisper | proctap-webhook https://example.com/hook
"""

import sys
import logging
import json
from typing import Optional

import click

from proctap_pipes.webhook_pipe import WebhookPipe


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
@click.argument("url")
@click.option(
    "--method",
    "-m",
    default="POST",
    type=click.Choice(["POST", "PUT", "PATCH"], case_sensitive=False),
    help="HTTP method (default: POST)",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    help="Additional HTTP header (format: 'Key: Value')",
)
@click.option(
    "--auth-token",
    "-a",
    envvar="WEBHOOK_AUTH_TOKEN",
    help="Bearer token for authentication",
)
@click.option(
    "--timeout",
    "-t",
    default=10.0,
    type=float,
    help="Request timeout in seconds (default: 10.0)",
)
@click.option(
    "--batch",
    "-b",
    default=1,
    type=int,
    help="Batch size before sending (default: 1)",
)
@click.option(
    "--template",
    default=None,
    help="JSON template for payload (as JSON string)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    url: str,
    method: str,
    header: tuple[str, ...],
    auth_token: Optional[str],
    timeout: float,
    batch: int,
    template: Optional[str],
    verbose: bool,
) -> None:
    """Send text from stdin to a webhook URL.

    Reads text from stdin (line by line or from pipes) and sends each
    line as a JSON payload to the specified webhook URL.

    Examples:

        # Send simple text
        echo "Hello World" | proctap-webhook https://example.com/webhook

        # Chain after Whisper transcription
        proctap -pid 1234 --stdout | proctap-whisper | proctap-webhook https://example.com/hook

        # Use custom headers
        echo "test" | proctap-webhook https://example.com/hook -H "X-Custom: value"

        # Batch multiple inputs
        cat texts.txt | proctap-webhook https://example.com/hook --batch 5

        # Use authentication
        echo "data" | proctap-webhook https://example.com/hook -a "your-token-here"

        # Custom payload template
        echo "test" | proctap-webhook https://example.com/hook --template '{"event": "data"}'
    """
    setup_logging(verbose)

    try:
        # Parse headers
        headers = {}
        for h in header:
            if ":" not in h:
                click.echo(f"Warning: Invalid header format: {h}", err=True)
                continue
            key, value = h.split(":", 1)
            headers[key.strip()] = value.strip()

        # Parse template
        payload_template = {}
        if template:
            try:
                payload_template = json.loads(template)
            except json.JSONDecodeError as e:
                click.echo(f"Error: Invalid JSON template: {e}", err=True)
                sys.exit(1)

        # Create webhook pipe
        pipe = WebhookPipe(
            webhook_url=url,
            method=method,
            headers=headers,
            text_mode=True,
            auth_token=auth_token,
            timeout=timeout,
            batch_size=batch,
            payload_template=payload_template,
        )

        # Process text from stdin
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                result = pipe.process_text_batch(line)
                if result:
                    logging.info(f"Webhook {result}")

            # Flush any remaining batch
            result = pipe.flush()
            if result:
                logging.info(f"Webhook {result} (flush)")

        except KeyboardInterrupt:
            # Flush on interrupt
            result = pipe.flush()
            if result:
                logging.info(f"Webhook {result} (interrupted)")
            sys.exit(0)

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
