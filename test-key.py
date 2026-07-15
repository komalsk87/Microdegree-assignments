import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"Loaded API Key: {api_key[:10] if api_key else 'None Found'}...")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello, reply with the word 'Success'"
    )
    print(f"\n✅ API Response: {response.text}")
except Exception as e:
    print(f"\n❌ THE EXACT ERROR IS:\n{e}")