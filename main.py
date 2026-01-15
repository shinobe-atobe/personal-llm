#!/usr/bin/env python3
"""Entry point for the Personal LLM with RAG."""

from src.chat import PersonalLLMChat


def main():
    """Run the personal LLM chat."""
    chat = PersonalLLMChat()
    chat.interactive_chat()


if __name__ == "__main__":
    main()
