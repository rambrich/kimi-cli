"""Command-line interface for kimi-cli.

Provides the main entry point for interacting with the Kimi AI API
from the terminal, supporting single-shot queries and interactive sessions.
"""

import os
import sys
import argparse
import textwrap
from typing import Optional

from .client import KimiClient, KimiClientError


# Using 128k as default since I frequently work with large codebases and long documents
DEFAULT_MODEL = "moonshot-v1-128k"
AVAILABLE_MODELS = ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="kimi",
        description="Chat with Kimi AI from your terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              kimi "What is the capital of France?"
              kimi --model moonshot-v1-32k "Summarize this long document..."
              kimi --interactive
              echo "Explain this" | kimi
        """),
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to Kimi. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="Moonshot API key. Defaults to MOONSHOT_API_KEY env variable.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive chat session.",
    )
    parser.add_argument(
        "--system", "-s",
        default=None,
        help="System prompt to set assistant behaviour.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and print the full response at once.",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return parser


def resolve_api_key(cli_key: Optional[str]) -> str:
    """Resolve the API key from CLI arg or environment variable."""
    key = cli_key or os.environ.get("MOONSHOT_API_KEY", "")
    if not key:
        print(
            "Error: No API key provided. Set MOONSHOT_API_KEY or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def run_interactive(client: KimiClient, stream: bool) -> None:
    """Run an interactive multi-turn chat session."""
    print("Kimi interactive session. Type 'exit' or press Ctrl+C to quit.\n")
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            pri
