"""
LLM Provider Abstraction Layer (simplified version).

Wraps Claude API calls behind one function, so the rest of the code
never touches the Anthropic SDK directly.
"""

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
MODEL = "claude-sonnet-4-6"


def generate(system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """
    Send a system prompt + conversation to Claude, return the text reply.

    Args:
        system: Instructions for the agent.
        messages: [{"role": "user", "content": "..."}]
        max_tokens: Max tokens to generate.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text


if __name__ == "__main__":
    reply = generate(
        system="You are a concise assistant.",
        messages=[{"role": "user", "content": "Say hello in 5 words or less."}],
    )
    print("Response:", reply)