# Contributing to ProcTapPipes

Thank you for your interest in contributing to ProcTapPipes! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/proctap-pipes.git
   cd proctap-pipes
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **mypy**: Type checking

Before committing, run:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Testing

We use pytest for testing. All new features must include tests.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=proctap_pipes --cov-report=html

# Run specific test file
pytest tests/test_base.py
```

### Adding a New Pipe

To add a new processing pipe:

1. Create a new file in `src/proctap_pipes/` (e.g., `my_pipe.py`)
2. Extend `BasePipe`:

```python
from proctap_pipes.base import BasePipe
import numpy.typing as npt

class MyPipe(BasePipe):
    """My custom pipe.
    
    Detailed description of what this pipe does.
    """
    
    def __init__(self, param1: str, **kwargs):
        """Initialize the pipe.
        
        Args:
            param1: Description of param1
            **kwargs: Additional arguments for BasePipe
        """
        super().__init__(**kwargs)
        self.param1 = param1
    
    def process_chunk(self, audio_data: npt.NDArray) -> str:
        """Process a chunk of audio.
        
        Args:
            audio_data: Audio samples with shape (samples, channels)
            
        Returns:
            Processed result
        """
        # Your implementation here
        return "result"
```

3. Add CLI tool in `src/proctap_pipes/cli/my_cli.py`:

```python
#!/usr/bin/env python3
import click
from proctap_pipes.my_pipe import MyPipe

@click.command()
@click.option("--param1", default="default", help="Parameter 1")
def main(param1: str) -> None:
    """CLI tool description."""
    pipe = MyPipe(param1=param1)
    pipe.run_cli()

if __name__ == "__main__":
    main()
```

4. Register CLI entry point in `pyproject.toml`:

```toml
[project.scripts]
proctap-mypipe = "proctap_pipes.cli.my_cli:main"
```

5. Add to `src/proctap_pipes/__init__.py`:

```python
from proctap_pipes.my_pipe import MyPipe

__all__ = [..., "MyPipe"]
```

6. Write tests in `tests/test_my_pipe.py`

7. Update README.md with usage examples

### Commit Guidelines

- Use clear, descriptive commit messages
- Follow conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test additions/changes
  - `refactor:` for code refactoring
  - `chore:` for maintenance tasks

Examples:
```
feat: add FFT visualization pipe
fix: handle empty audio chunks in WhisperPipe
docs: update README with WebRTC examples
test: add tests for batching in WebhookPipe
```

### Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add my new feature"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

4. Open a Pull Request on GitHub

5. Ensure:
   - All tests pass
   - Code is formatted with Black
   - No linting errors from Ruff
   - Type checking passes with mypy
   - Documentation is updated

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Documentation

- All public APIs must have Google-style docstrings
- Update README.md for user-facing changes
- Add examples for new features

## Questions?

Feel free to open an issue for questions or discussions!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
