import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

# Recommended LLM Parameter Configuration Presets for Common Enterprise Use Cases
LLM_CONFIG_PRESETS = {
    # USE CASE A: CUSTOMER SUPPORT CHATBOT
    # Priority: Deterministic, factual, brand-safe, and highly predictable.
    "customer_support": {
        # Temperature is set to 0.0 to eliminate random sampling. The model will 
        # always choose the token with the absolute highest probability. This ensures 
        # identical, safe, and vetted responses for repeated user inquiries.
        "temperature": 0.0,
        
        # When temperature is 0.0, top_p is mathematically neutralized. However, 
        # we explicitly set it to a very low value (0.1) as a defensive configuration 
        # layer, ensuring only the most logical/highest-confidence tokens are evaluated.
        "top_p": 0.1,
        
        # Customer service interactions should be punchy, clear, and direct. 
        # Capping at 250 tokens prevents the model from rambling, reducing both 
        # latency and potential brand safety drifts.
        "max_tokens": 250,
        
        # System instructions must enforce strict boundaries.
        "system_instruction_fallback": "Answer concisely, strictly using verified internal documentation. If unsure, escalate."
    },

    # USE CASE B: CREATIVE MARKETING COPY GENERATOR
    # Priority: High linguistic diversity, unique metaphors, and expressive language.
    "marketing_creative": {
        # Elevated temperature (1.0) flattens the probability curve of next-token 
        # selections, making creative, unexpected, and varied word associations 
        # far more likely to occur.
        "temperature": 1.0,
        
        # A high top_p (0.95) allows the model to draw from almost the entire 
        # vocabulary spectrum. It provides maximum expressive range (diverse synonyms, 
        # creative adjectives) while throwing away the bottom 5% of erratic/gibberish tokens.
        "top_p": 0.95,
        
        # Marketing copy (such as blog posts, newsletters, or social ad options) 
        # requires significant headroom to complete full structured layouts.
        "max_tokens": 1000,
        
        "system_instruction_fallback": "You are an award-winning copywriter. Use vivid, persuasive, and punchy hooks."
    },

    # USE CASE C: CODE GENERATION ASSISTANT
    # Priority: Syntactic precision, accurate logic, and zero hallucinated libraries.
    "code_generation": {
        # Set to an extremely low but non-zero value (0.1). This introduces just enough 
        # fluidity for the model to choose valid syntactic alternatives (e.g., choosing 
        # between a 'for' loop and a list comprehension) while strictly avoiding logic errors.
        "temperature": 0.1,
        
        # A low top_p (0.2) acts as a structural filter. It discards 80% of the vocabulary 
        # list, forcing the model to select strictly from standard API keywords, actual 
        # syntax blocks, and confirmed library functions, preventing hallucinated libraries.
        "top_p": 0.2,
        
        # Writing complex classes, unit tests, or full boilerplate setups demands 
        # substantial generation headroom. Truncation halfway through a code block breaks execution.
        "max_tokens": 1500,
        
        "system_instruction_fallback": "Write strictly functional, robust, and commented code. Avoid conversational filler."
    }
}

def run_presets_demo():
    # Load environment variables (such as GROQ_API_KEY) from a .env file if available
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[-] Error: GROQ_API_KEY environment variable not found.")
        print("Please set your Groq API key in your terminal or .env file.")
        sys.exit(1)

    # Initialize the Groq client
    client = Groq(api_key=api_key)
    model_name = "llama-3.1-8b-instant"

    # Define appropriate test prompts corresponding to each specific configuration preset
    prompts = {
        "customer_support": "What is your refund policy for broken items? (Context: We offer a 30-day money-back guarantee with proof of damage).",
        "marketing_creative": "Write a catchy 1-sentence slogan for an organic, caffeine-free energy drink.",
        "code_generation": "Write a highly optimized python function to find the nth Fibonacci number."
    }

    print("=" * 80)
    print("RUNNING LLM PARAMETER CONFIGURATION PRESETS")
    print(f"Model: {model_name}")
    print("=" * 80)

    for use_case, config in LLM_CONFIG_PRESETS.items():
        print(f"\n Use Case: {use_case.upper()}")
        print(f"   Parameters: Temp={config['temperature']}, Top_P={config['top_p']}, Max_Tokens={config['max_tokens']}")
        
        prompt = prompts[use_case]
        print(f"   Prompt: '{prompt}'")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": config["system_instruction_fallback"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"]
            )
            
            output = response.choices[0].message.content.strip()
            print("\n--- [Output Response] ---")
            print(output)
            print("-" * 80)
            
        except Exception as e:
            print(f"[-] API Error executing {use_case}: {e}")
        
        # Pacing step to respect rate boundaries smoothly
        time.sleep(1.0)

    print("\nPresets Demo Run Complete!")

if __name__ == "__main__":
    run_presets_demo()