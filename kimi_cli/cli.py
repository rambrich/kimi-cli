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


DEFAULT_MODEL = "moonshot-v1-8k"
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
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            print("Kimi: ", end="", flush=True)
            response = client.chat(history, stream=stream)
            print(response)
            history.append({"role": "assistant", "content": response})
        except KimiClientError as exc:
            print(f"\nError: {exc}", file=sys.stderr)


def run_single(client: KimiClient, prompt: str, system: Optional[str], stream: bool) -> None:
    """Send a single prompt and print the response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(messages, stream=stream)
        print(response)
    except KimiClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the kimi CLI."""
    parser = build_parser()
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    client = KimiClient(api_key=api_key, model=args.model)
    stream = not args.no_stream

    if args.interactive:
        if args.system:
            client.set_system_prompt(args.system)
        run_interactive(client, stream=stream)
        return

    # Resolve prompt from positional arg or stdin
    if args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(0)

    if not prompt:
        print("Error: Empty prompt.", file=sys.stderr)
        sys.exit(1)

    run_single(client, prompt, system=args.system, stream=stream)


if __name__ == "__main__":
    main()
