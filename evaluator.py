import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# 1. Load Environment Variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print(" ")
    #raise ValueError("Error: GEMINI_API_KEY not found. Please check your .env file setup.")

# 2. Initialize Gemini Client
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# 3. Establish the 10 Test Customer Complaints
data = {
    "complaint": [
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
    ],
    "true_label": [
        "Billing", "Technical", "Shipping", "Technical", "Billing", 
        "Shipping", "Shipping", "Technical", "Billing", "Shipping"
    ]
}
df = pd.DataFrame(data)

# 4. Prompt Generator Function
def get_prompt(complaint, style):
    if style == "zero-shot":
        return f"""
Classify the following customer complaint into exactly one of these categories: Billing, Technical, or Shipping.
Return ONLY the category name. Do not include any other text, punctuation, or explanation.

Complaint: "{complaint}"
Category:"""

    elif style == "few-shot":
        return f"""
Classify the customer complaint into exactly one of these categories: Billing, Technical, or Shipping.
Return ONLY the category name.

Examples:
Complaint: "My screen is frozen and won't respond to touch."
Category: Technical

Complaint: "I need an itemized receipt for my taxes."
Category: Billing

Complaint: "Where is my order? It's been two weeks."
Category: Shipping

Complaint: "{complaint}"
Category:"""

    elif style == "role-based":
        return f"""
You are a senior customer support classifier automated agent. Your sole job is to route incoming customer tickets efficiently and flawlessly into one of three buckets: Billing, Technical, or Shipping. 
You must output exactly the word "Billing", "Technical", or "Shipping". Never write full sentences or explanations.

Ticket to route: "{complaint}"
Bucket:"""

# 5. Upgraded API Wrapper Function with Precise Error Catching & Backoff
def query_gemini(prompt):
    max_retries = 5
    delay = 4  # Start with a solid 4-second delay if limited

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text.strip().capitalize()
        
        except Exception as e:
            error_str = str(e)
            # Catching standard rate limits (429, RESOURCE_EXHAUSTED)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    #print(f"⚠️ Hit Free-Tier limit. Backing off for {delay} seconds...")
                    time.sleep(delay)
                    delay *= 0.5  # Gradually scale up wait time
                    continue
            
            # If it's an unrelated system error, return it explicitly
            return #

# 6. Run the Experiment Loop with Structured Gaps
print("Running classifications across 10 inputs with 3 prompt styles.")
#print("Note: Intentional pacing delays added to remain safely under Free Tier limits...\n")

zero_shot_preds = []
few_shot_preds = []
role_preds = []

for index, row in df.iterrows():
    text = row['complaint']
    print(f"Processing row {index + 1}/10...")
    
    # Run and append each variant with gaps between prompts
    zero_shot_preds.append(query_gemini(get_prompt(text, "zero-shot")))
    time.sleep(0.5) 
    
    few_shot_preds.append(query_gemini(get_prompt(text, "few-shot")))
    time.sleep(0.5) 
    
    role_preds.append(query_gemini(get_prompt(text, "role-based")))
    
    # Cooling-off window before starting the next row
    time.sleep(1.0)

df['Zero-Shot'] = zero_shot_preds
df['Few-Shot'] = few_shot_preds
df['Role-Based'] = role_preds

# 7. Evaluate Performance Metrics
zero_shot_acc = (df['Zero-Shot'] == df['true_label']).mean() * 100
few_shot_acc = (df['Few-Shot'] == df['true_label']).mean() * 100
role_based_acc = (df['Role-Based'] == df['true_label']).mean() * 100

print("\n================== ACCURACY RESULTS ==================")
print(f"Zero-Shot Accuracy:  {zero_shot_acc:.1f}%")
print(f"Few-Shot Accuracy:   {few_shot_acc:.1f}%")
print(f"Role-Based Accuracy: {role_based_acc:.1f}%")
print("======================================================\n")

# Display complete comparison table in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df[['complaint', 'true_label', 'Zero-Shot', 'Few-Shot', 'Role-Based']])