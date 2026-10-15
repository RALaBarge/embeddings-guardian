# Contributing to embeddings-guardian

Thank you for your interest in contributing. This document provides guidelines
and information for contributors.

## Development Setup

```bash
git clone https://github.com/beigebox-ai/embeddings-guardian.git
cd embeddings-guardian
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # all tests
pytest -m unit            # fast unit tests only
pytest -xvs              # stop on first failure, verbose
pytest --cov             # with coverage report
```

## Code Style

This project uses:
- **ruff** for linting
- **black** for formatting
- **mypy** for type checking

```bash
ruff check .
black .
mypy embeddings_guardian/
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Update documentation if your change affects the public API.
5. Submit a pull request with a clear description of the change.

## Reporting Issues

Open an issue on GitHub with:
- A clear title describing the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. Be respectful and constructive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
