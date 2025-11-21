# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-21

### Fixed
- Resolved all CI/CD workflow test failures
- Fixed test issues to ensure stable builds
- Replaced click.echo with print to avoid Windows stderr errors
- Aligned MicMixPipe defaults with ProcTap standard format

### Added
- Integrated mic-mix pipe into ProcTapPipes package
- Added CLI tool for microphone mixer (`proctap-mic-mix`)
- Added microphone mix pipe for combining mic input with ProcTap audio
- Auto-detect microphone channel count to prevent initialization errors
- Added mono to stereo conversion for microphone mixing

### Documentation
- Added FFmpeg MP3 recording examples and usage documentation

## [0.2.2] - 2025-01-XX

### Changed
- Synced version to 0.2.2 across all files

## [0.2.1] - 2025-01-XX

### Changed
- Version bump to 0.2.1

## [0.2.0] - 2025-01-XX

### Added
- Real-time audio effects processing
- Enhanced Whisper transcription capabilities

## [0.1.1] - 2025-01-XX

### Added
- Volume meter pipe functionality

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- WhisperPipe for audio transcription
- LLMPipe for language model integration
- SlackWebhookPipe for Slack notifications
- CLI tools: proctap-whisper, proctap-llm, proctap-webhook

[0.3.0]: https://github.com/proctap/proctap-pipes/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/proctap/proctap-pipes/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/proctap/proctap-pipes/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/proctap/proctap-pipes/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/proctap/proctap-pipes/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/proctap/proctap-pipes/releases/tag/v0.1.0
