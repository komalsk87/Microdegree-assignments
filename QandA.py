import os
import sys
import time
import textwrap
from dotenv import load_dotenv
from groq import Groq

def run_factual_table_experiment():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("[-] Error: GROQ_API_KEY environment variable not found.")
        print("Please set your Groq API key in your terminal or .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    model_name = "llama-3.1-8b-instant"
    prompt = "What is the capital of France? List three facts about it"
    
    runs = 5
    temp_0_results = []
    temp_1_results = []

    print("=" * 100)
    print("STARTING FACTUAL Q&A EXPERIMENT TO GENERATE DATA TABLE")
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    print("=" * 100)

    print("\n[+] Gathering 5 runs at Temperature 0.0 (Deterministic)...")
    for i in range(runs):
        print(f"    - Run {i+1}/{runs}...", end="", flush=True)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,  # Keep it concise so it fits comfortably in the table cells
            )
            content = response.choices[0].message.content.strip().replace("\n", " ")
            temp_0_results.append(content)
            print(" Done.")
        except Exception as e:
            print(f" Error: {e}")
            temp_0_results.append(f"Error fetching: {e}")
        time.sleep(1.0)  # Pacing pause

    print("\n[+] Gathering 5 runs at Temperature 1.0 (Varied/Explorer)...")
    for i in range(runs):
        print(f"    - Run {i+1}/{runs}...", end="", flush=True)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip().replace("\n", " ")
            temp_1_results.append(content)
            print(" Done.")
        except Exception as e:
            print(f" Error: {e}")
            temp_1_results.append(f"Error fetching: {e}")
        time.sleep(1.0)

    print("\n" + "=" * 110)
    print("COMPARISON TABLE: TEMPERATURE 0.0 VS TEMPERATURE 1.0")
    print("=" * 110)

    # Column configuration widths
    col_run_w = 6
    col_t0_w = 48
    col_t1_w = 48

    # Table Header
    header_run = "Run #"
    header_t0 = "Temperature 0.0 (The Train on Tracks)"
    header_t1 = "Temperature 1.0 (The Free Explorer)"

    # Print Header line
    border_line = f"+{'-' * (col_run_w+2)}+{'-' * (col_t0_w+2)}+{'-' * (col_t1_w+2)}+"
    print(border_line)
    print(f"| {header_run:<{col_run_w}} | {header_t0:<{col_t0_w}} | {header_t1:<{col_t1_w}} |")
    print(border_line)

    for run_idx in range(runs):
        t0_text = temp_0_results[run_idx]
        t1_text = temp_1_results[run_idx]

        # Wrap text in cell limits to prevent console overflow and maintain table structure
        t0_wrapped = textwrap.wrap(t0_text, width=col_t0_w)
        t1_wrapped = textwrap.wrap(t1_text, width=col_t1_w)

        # Handle rows with differing line counts
        max_lines = max(len(t0_wrapped), len(t1_wrapped))

        for line_idx in range(max_lines):
            run_label = f"Run {run_idx+1}" if line_idx == 0 else ""
            t0_line = t0_wrapped[line_idx] if line_idx < len(t0_wrapped) else ""
            t1_line = t1_wrapped[line_idx] if line_idx < len(t1_wrapped) else ""

            print(f"| {run_label:<{col_run_w}} | {t0_line:<{col_t0_w}} | {t1_line:<{col_t1_w}} |")
        
        print(border_line)

    print("\n[✔] Experiment completed successfully! Observe the structural replication in Temp 0.0 vs the variations in Temp 1.0.")

if __name__ == "__main__":
    run_factual_table_experiment()