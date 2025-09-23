import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in .env file")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def ask_llm(question: str) -> str:
    """Send a question to Grok LLM and return the answer."""
    response = client.chat.completions.create(
        model="x-ai/grok-4-fast:free",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content
