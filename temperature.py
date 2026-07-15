import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq

def run_experiment():
    # Load environment variables from a .env file if present
    load_dotenv()
    
    # Ensure your Groq API key is configured
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("[-] Error: GROQ_API_KEY environment variable not found.")
        print("Please set your Groq API key in your terminal or .env file before running this script.")
        print("\nHow to set it in VS Code terminal:")
        print("  Windows (PowerShell): $env:GROQ_API_KEY=\"your_api_key\"")
        print("  macOS/Linux/Git Bash: export GROQ_API_KEY=\"your_api_key\"")
        sys.exit(1)

    # Initialize the Groq client
    client = Groq(api_key=api_key)

    #prompt = "Write a short story opening about a robot"
    prompt =  "Write a story on mickey mouse"
    temperatures = [0.0, 0.5, 1.0, 1.5]
    model_name = "llama-3.1-8b-instant"

    print("=" * 60)
    print(f"Starting Temperature Experiment using {model_name}")
    print(f"Prompt: '{prompt}'")
    print("=" * 60)

    for temp in temperatures:
        print(f"\n Running API Call with Temperature: {temp}...")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=temp,
                max_tokens=70,  # Limits token cost while giving enough text for an opening
            )
            
            content = response.choices[0].message.content.strip()
            
            # Print directly to the terminal
            print("-" * 40)
            print(f"Results for Temperature {temp}:")
            print(content)
            print("-" * 40)
            
        except Exception as e:
            print(f"[-] API Error at temperature {temp}: {e}")
        
        # Pacing pause to respect rate limits
        time.sleep(1.0)


    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_experiment()