from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic() 
MODEL = "claude-sonnet-4-6"


def generate(system: str, messages: list[dict], max_tokens: int = 1024) -> str:
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