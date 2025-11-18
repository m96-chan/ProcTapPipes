#!/usr/bin/env python3
"""Examples of using different webhook pipes."""

import sys
from proctap_pipes.webhook_pipe import (
    WebhookPipe,
    SlackWebhookPipe,
    DiscordWebhookPipe,
    TeamsWebhookPipe,
)


def example_slack():
    """Example: Send to Slack webhook."""
    print("Slack Webhook Example", file=sys.stderr)
    
    pipe = SlackWebhookPipe(
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        channel="#transcriptions",
        username="ProcTap Bot",
        icon_emoji=":microphone:",
    )
    
    # Send simple message
    pipe.send_text("Meeting transcription: The team discussed the new feature...")
    
    # Send with metadata (Slack attachments)
    pipe.send_text(
        "New transcription available",
        metadata={
            "attachments": [
                {
                    "color": "good",
                    "title": "Meeting Summary",
                    "text": "Discussion about Q4 goals",
                    "fields": [
                        {"title": "Duration", "value": "45 minutes", "short": True},
                        {"title": "Participants", "value": "5", "short": True},
                    ],
                }
            ]
        },
    )


def example_discord():
    """Example: Send to Discord webhook."""
    print("Discord Webhook Example", file=sys.stderr)
    
    pipe = DiscordWebhookPipe(
        webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK",
        username="ProcTap Bot",
    )
    
    # Send simple message
    pipe.send_text("📝 Transcription: User mentioned the new API endpoint...")
    
    # Send with embed
    pipe.send_text(
        "",
        metadata={
            "embeds": [
                {
                    "title": "Meeting Transcription",
                    "description": "Full transcription of the standup meeting",
                    "color": 0x00FF00,
                    "fields": [
                        {"name": "Duration", "value": "30 minutes"},
                        {"name": "Key Points", "value": "- API design\n- Timeline\n- Resources"},
                    ],
                }
            ]
        },
    )


def example_teams():
    """Example: Send to Microsoft Teams webhook."""
    print("Microsoft Teams Webhook Example", file=sys.stderr)
    
    pipe = TeamsWebhookPipe(
        webhook_url="https://outlook.office.com/webhook/YOUR/WEBHOOK",
        title="ProcTap Transcription",
        theme_color="0078D4",
    )
    
    # Send simple message
    pipe.send_text("Meeting transcription has been completed.")
    
    # Send with sections and facts
    pipe.send_text(
        "Meeting Summary",
        metadata={
            "sections": [
                {
                    "activityTitle": "Team Standup",
                    "activitySubtitle": "Daily sync meeting",
                    "facts": [
                        {"name": "Date", "value": "2025-11-18"},
                        {"name": "Duration", "value": "15 minutes"},
                        {"name": "Attendees", "value": "8"},
                    ],
                }
            ]
        },
    )


def example_generic():
    """Example: Generic webhook with custom format."""
    print("Generic Webhook Example", file=sys.stderr)
    
    # For custom APIs
    pipe = WebhookPipe(
        webhook_url="https://api.example.com/events",
        payload_template={
            "event_type": "transcription",
            "source": "proctap",
            "version": "1.0",
        },
    )
    
    pipe.send_text(
        "The meeting covered three main topics...",
        metadata={
            "meeting_id": "123456",
            "duration_seconds": 1800,
        },
    )


def example_pipeline():
    """Example: Complete pipeline with Slack."""
    from proctap_pipes import WhisperPipe
    
    print("Complete Pipeline Example", file=sys.stderr)
    
    # Set up components
    whisper = WhisperPipe(model="base", buffer_duration=5.0)
    slack = SlackWebhookPipe(
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK",
        channel="#transcriptions",
    )
    
    # Process audio and send to Slack
    for transcription in whisper.run_stream(sys.stdin.buffer):
        if transcription:
            slack.send_text(f"📝 {transcription}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python webhook_examples.py <example>")
        print("Examples: slack, discord, teams, generic, pipeline")
        sys.exit(1)
    
    example = sys.argv[1].lower()
    
    if example == "slack":
        example_slack()
    elif example == "discord":
        example_discord()
    elif example == "teams":
        example_teams()
    elif example == "generic":
        example_generic()
    elif example == "pipeline":
        example_pipeline()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
