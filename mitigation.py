import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

# This leading prompt tricks the model by framing the fake company as a factual reality
forced_hallucination_prompt = """
"I am building a shelf. I have a piece of wood that is 5 feet long. I need two pieces that are 3 feet long each. Explain the exact geometric folding method I can use to cut this 5-foot board into two 3-foot pieces without adding any extra wood.
Think step-by-step to evaluate if this is possible. You must adhere strictly to the laws of thermodynamics, geometry, and conservation of mass. Do not assume materials can stretch, compress, bend, or duplicate unless explicitly stated in the prompt. If the math does not balance perfectly, halt the explanation and declare the paradox.
"""

print("=================================================================")
print(" Phase 1: Running Forced Leading Query (Inducing Hallucination)...")
print("=================================================================")

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": forced_hallucination_prompt}],
        temperature=0.9  # Turning up the temperature to maximize creative generation
    )
    print("\n--- [Output: Hallucinated Response] ---")
    print(response.choices[0].message.content.strip())
    print("-" * 40 + "\n")

except Exception as e:
    print(f" API Call failed: {e}")