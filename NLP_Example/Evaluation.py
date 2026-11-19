# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # 1. Load the generated predictions output CSV file
# FILE_PATH = 'test_predictions_output.csv'
# df = pd.read_csv(FILE_PATH)

# # 2. Map original ticket types to standard target categories (Technical, Billing, Other)
# def map_category(ticket_type):
#     tt = str(ticket_type).lower()
#     if 'technical' in tt:
#         return 'Technical'
#     elif 'billing' in tt or 'refund' in tt or 'cancellation' in tt:
#         return 'Billing'
#     else:
#         return 'Other'

# # Ensure ground truth category column exists
# if 'Ticket Type' in df.columns:
#     df['Actual_Category'] = df['Ticket Type'].apply(map_category)
# elif 'Actual_Category' not in df.columns:
#     raise KeyError(
#         "Ground truth category column ('Ticket Type' or 'Actual_Category') not found in CSV."
#     )

# # 3. Create a comparison column to check if prediction is right or wrong
# df['Is_Correct'] = df['Actual_Category'] == df['Predicted_Category']

# # 4. Calculate Overall Accuracy
# accuracy = accuracy_score(df['Actual_Category'], df['Predicted_Category'])

# # ---------------------------------------------------------------------
# # DISPLAY EVALUATION RESULTS
# # ---------------------------------------------------------------------

# print("=" * 60)
# print(f" CATEGORY 1 MODEL ACCURACY: {accuracy * 100:.2f}%")
# print("=" * 60)

# # Display sample row-by-row comparisons (Right vs. Wrong)
# print("\n--- Row-by-Row Prediction Check (First 10 Rows) ---")
# print(
#     df[['combined_text', 'Actual_Category', 'Predicted_Category', 'Is_Correct']].head(10)
# )

# # Display correctness summary counts
# print("\n--- Prediction Summary Counts ---")
# print(df['Is_Correct'].value_counts().rename({True: 'Correct', False: 'Incorrect'}))

# # Display Detailed Classification Metrics (Precision, Recall, F1-Score)
# print("\n--- Classification Report ---")
# print(classification_report(df['Actual_Category'], df['Predicted_Category']))

# # Display Confusion Matrix
# print("--- Confusion Matrix ---")
# cm = pd.crosstab(
#     df['Actual_Category'],
#     df['Predicted_Category'],
#     rownames=['Actual'],
#     colnames=['Predicted']
# )
# print(cm)

import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------
# 1. LOAD PREDICTIONS DATASET
# ---------------------------------------------------------------------
FILE_PATH = 'test_predictions_output.csv'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the file '{FILE_PATH}'. Ensure the path is correct.")

# Ensure input text exists
if 'combined_text' not in df.columns:
    if 'Ticket Subject' in df.columns and 'Ticket Description' in df.columns:
        df['combined_text'] = df['Ticket Subject'].fillna('') + " " + df['Ticket Description'].fillna('')
    else:
        df['combined_text'] = df.iloc[:, 0].astype(str)

# ---------------------------------------------------------------------
# 2. MULTI-KEYWORD SCORING GROUND TRUTH GENERATOR
# ---------------------------------------------------------------------
def derive_ground_truth_scored(text):
    t = str(text).lower()
    
    # Critical technical fault indicators (higher priority weight)
    tech_critical_patterns = [
        r'\bcrash(ed)?\b', r'\b500\b', r'\b404\b', r'\bserver\b', 
        r'\bbroken\b', r'\btimeout\b', r'\bnot working\b'
    ]
    
    # Standard technical keywords
    tech_patterns = [
        r'\berror\b', r'\bbug\b', r'\blogin\b', r'\bapi\b', 
        r'\bpassword\b', r'\bfail(ed)?\b', r'\bapp\b', r'\bportal\b', r'\bissue\b'
    ]
    
    # Financial / Billing keywords
    billing_patterns = [
        r'\brefund\b', r'\bbilling\b', r'\bcharg(e|ed|ing)\b', r'\binvoice\b',
        r'\bpayment\b', r'\bsubscription\b', r'\breceipt\b', r'\bcredit card\b',
        r'\bcancel(lation)?\b', r'\bpricing\b', r'\bovercharged?\b', r'\bbalance\b', r'\bcost\b'
    ]
    
    # Compute keyword occurrences
    tech_score = sum(1 for kw in tech_patterns if re.search(kw, t))
    critical_tech_score = sum(1.5 for kw in tech_critical_patterns if re.search(kw, t))
    total_tech_score = tech_score + critical_tech_score
    
    billing_score = sum(1 for kw in billing_patterns if re.search(kw, t))
    
    # Decision Hierarchy & Tie-Breaking
    if total_tech_score > billing_score:
        return 'Technical'
    elif billing_score > total_tech_score:
        return 'Billing'
    elif total_tech_score > 0 and total_tech_score == billing_score:
        # If tied (e.g. app crashed during invoice payment), prioritize technical fault
        if any(re.search(kw, t) for kw in tech_critical_patterns):
            return 'Technical'
        return 'Billing'
    else:
        return 'Other'

# Apply score-based algorithm to construct ground truth column
df['Actual_Category'] = df['combined_text'].apply(derive_ground_truth_scored)

# ---------------------------------------------------------------------
# 3. EVALUATION & ACCURACY COMPUTATION
# ---------------------------------------------------------------------

# Match prediction against dynamic ground truth
df['Is_Correct'] = df['Actual_Category'] == df['Predicted_Category']

# Compute overall accuracy
accuracy = accuracy_score(df['Actual_Category'], df['Predicted_Category'])

# ---------------------------------------------------------------------
# 4. DISPLAY RESULTS
# ---------------------------------------------------------------------

print("=" * 65)
print(f" CATEGORY 1 MODEL ACCURACY (vs Scored Ground Truth): {accuracy * 100:.2f}%")
print("=" * 65)

# First 10 rows comparison
print("\n--- Sample Row-by-Row Comparison (First 10 Rows) ---")
print(
    df[['combined_text', 'Actual_Category', 'Predicted_Category', 'Is_Correct']].head(10)
)

# Correctness summary
print("\n--- Summary Counts ---")
print(df['Is_Correct'].value_counts().rename({True: 'Correct', False: 'Incorrect'}))

# Detailed Metrics
print("\n--- Classification Report ---")
print(classification_report(df['Actual_Category'], df['Predicted_Category']))

# Confusion Matrix
print("--- Confusion Matrix ---")
cm = pd.crosstab(
    df['Actual_Category'],
    df['Predicted_Category'],
    rownames=['Actual (Scored)'],
    colnames=['Predicted (ML)']
)
print(cm)