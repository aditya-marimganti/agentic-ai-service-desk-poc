import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

def test_connection():
    print("Sending test request to Claude API...\n")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Respond ONLY with valid JSON, no other text. "
                    "Return an object with three fields: "
                    "'status' (string, say 'connected'), "
                    "'model_confirmed' (string, name the model you are), "
                    "'ready_for_poc' (boolean, true)."
                )
            }
        ]
    )

    raw_text = response.content[0].text
    print("Raw response from Claude:")
    print(raw_text)
    print()

    try:
        parsed = json.loads(raw_text)
        print("✅ Successfully parsed structured JSON output:")
        print(json.dumps(parsed, indent=2))
        print("\nSetup confirmed working. API key, SDK, and structured output all functional.")
    except json.JSONDecodeError:
        print("⚠️ Response was not valid JSON. The API call worked, but the model")
        print("didn't follow the structured-output instruction exactly. Try again")
        print("or check the prompt engineering guide for stricter formatting techniques.")

if __name__ == "__main__":
    test_connection()