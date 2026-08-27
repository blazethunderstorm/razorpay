"""Khata -- settlement reconciliation with measured, held-out accuracy."""

from pathlib import Path

from dotenv import load_dotenv

# Tier 3 reads ANTHROPIC_API_KEY off the environment at construction time, so
# the .env file has to be loaded before any entry point builds an Engine.
# Anchored to the repo root rather than the cwd: `python -m khata.cli` and
# `uvicorn khata.api:app` must both find it. An already-exported variable wins.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

__version__ = "1.0.0"
