import os
import time
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Start total tracking timer
start_time = time.time()
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

# 1. The Dataset
complaints = [
    "I was charged twice for my subscription this month. Please refund.",
    "The app keeps crashing every time I try to open the camera roll.",
    "My package was supposed to arrive yesterday but tracking says it's delayed.",
    "I cannot log into my account, it says password incorrect but I reset it.",
    "There is an unknown $45 fee on my latest invoice. What is this?",
    "The box arrived completely damaged and the item inside is broken.",
    "Can I change my delivery address to my office instead of my home?",
    "The website is throwing a 500 internal server error during checkout.",
    "I applied a 20% discount code but it didn't apply to my final total.",
    "The courier left my parcel in the rain and now it is ruined."
]

# 2. The Ground Truth (Answer Key) to measure accuracy against
ground_truth = [
    "Billing",     # Item 0
    "Technical",   # Item 1
    "Shipping",    # Item 2
    "Technical",   # Item 3
    "Billing",     # Item 4
    "Shipping",    # Item 5
    "Shipping",    # Item 6
    "Technical",   # Item 7
    "Billing",     # Item 8
    "Shipping"     # Item 9
]

# 3. Define the Prompting Formats
def get_zero_shot(text):
    return f'Classify into exactly one: Billing, Technical, Shipping. Return ONLY the word.\nComplaint: "{text}"\nCategory:'

def get_few_shot(text):
    return f'Classify into exactly one: Billing, Technical, Shipping. Return ONLY the word.\nExamples:\n"My screen is frozen." -> Technical\n"I need a receipt." -> Billing\n"Where is my order?" -> Shipping\n\nComplaint: "{text}" ->'

def get_role_based(text):
    return f'You are an expert senior customer support classifier. Output exactly "Billing", "Technical", or "Shipping". Never write full sentences.\nTicket: "{text}"\nBucket:'

def get_cot(text):
    return f'Classify this complaint into Billing, Technical, or Shipping. Think step-by-step.\nAt the very end of your response, write "Final Category: " followed by the category name.\nComplaint: "{text}"\nThinking Process:'

# Helper function to send requests cleanly to Groq
def call_groq(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Error"

# Storage for results
results = {"Zero-Shot": [], "Few-Shot": [], "Role-Based": [], "CoT": []}

print("Starting Complete Benchmark Evaluation across all 4 methods...")

# Loop through each text block and execute all 4 variations
for idx, text in enumerate(complaints, 1):
    print(f"Processing sentence {idx}/10...")
    
    # Run Zero-Shot
    zs_res = call_groq(get_zero_shot(text)).capitalize()
    results["Zero-Shot"].append(zs_res)
    
    # Run Few-Shot
    fs_res = call_groq(get_few_shot(text)).capitalize()
    results["Few-Shot"].append(fs_res)
    
    # Run Role-Based
    rb_res = call_groq(get_role_based(text)).capitalize()
    results["Role-Based"].append(rb_res)
    
    # Run Chain of Thought (and parse out final category)
    cot_raw = call_groq(get_cot(text))
    match = re.search(r"Final Category:\s*(\w+)", cot_raw, re.IGNORECASE)
    if match:
        cot_res = match.group(1).strip().capitalize()
    else:
        if "Billing" in cot_raw: cot_res = "Billing"
        elif "Technical" in cot_raw: cot_res = "Technical"
        elif "Shipping" in cot_raw: cot_res = "Shipping"
        else: cot_res = "Unknown"
    results["CoT"].append(cot_res)
    
    # Small pacing sleep to keep Groq API perfectly healthy
    time.sleep(1.0)

# 4. Calculate Accuracy Metrics
def compute_accuracy(predictions, truth):
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    return (correct / len(truth)) * 100

zs_acc = compute_accuracy(results["Zero-Shot"], ground_truth)
fs_acc = compute_accuracy(results["Few-Shot"], ground_truth)
rb_acc = compute_accuracy(results["Role-Based"], ground_truth)
cot_acc = compute_accuracy(results["CoT"], ground_truth)

total_time = time.time() - start_time

# 5. Build and Display the Raw Prediction Table
df_detailed = pd.DataFrame({
    "Complaint Snippet": [c[:40] + "..." for c in complaints],
    "Actual (Truth)": ground_truth,
    "Zero-Shot": results["Zero-Shot"],
    "Few-Shot": results["Few-Shot"],
    "Role-Based": results["Role-Based"],
    "CoT": results["CoT"]
})

print("\n" + "="*30 + " DETAILED BREAKDOWN TABLE " + "="*30)
print(df_detailed.to_string(index=False))

# 6. Build and Display the Accuracy Report Card
df_summary = pd.DataFrame({
    "Prompting Technique": ["Zero-Shot", "Few-Shot", "Role-Based", "Chain-of-Thought (CoT)"],
    "Accuracy Score": [f"{zs_acc}%", f"{fs_acc}%", f"{rb_acc}%", f"{cot_acc}%"]
})

print("\n" + "="*25 + " FINAL ACCURACY LEADERBOARD " + "="*25)
print(df_summary.to_string(index=False))
print("="*76)
print(f"Total Benchmark Evaluation completed in {total_time:.2f} seconds.\n")