# backend.py
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama

app = FastAPI(title="Instacart Retail Chatbot Engine")

# ==========================================================
# 1. LOAD INSTACART GROCERY DATASETS
# ==========================================================
print("Loading Instacart Datasets...")
try:
    products_df = pd.read_csv('products.csv')
    aisles_df = pd.read_csv('aisles.csv')
    
    # Merge them so we have human-readable aisles linked to products instantly
    grocery_db = pd.merge(products_df, aisles_df, on='aisle_id')
    
    # Mock some dynamic inventory/stock levels since Instacart data is historical orders
    # We will generate static mock quantities to simulate a real store system
    grocery_db['stock_level'] = grocery_db['product_id'] % 25  # Deterministic mock stock
    
    print("Instacart Database fully initialized!")
except Exception as e:
    print(f"File Loading Error: {e}. Ensure products.csv and aisles.csv are in this folder.")

# ==========================================================
# 2. INTERNAL PYTHON SEARCH TOOLS
# ==========================================================
def find_product_location(search_query: str) -> str:
    """Finds the location (aisle) of a grocery item based on its name."""
    query = search_query.lower().strip()
    match = grocery_db[grocery_db['product_name'].str.contains(query, case=False, na=False)]
    
    if match.empty:
        return f"Sorry, we couldn't find any grocery items matching '{search_query}' in our catalog."
    
    # Return the top 3 closest matching products to avoid flooding the context window
    results = []
    for _, row in match.head(3).iterrows():
        results.append(f"• {row['product_name']}: Located in the [{row['aisle']}] aisle.")
        
    return "Here are the matching items I found:\n" + "\n".join(results)

def check_stock_availability(search_query: str) -> str:
    """Checks the stock inventory levels for a grocery item by name."""
    query = search_query.lower().strip()
    match = grocery_db[grocery_db['product_name'].str.contains(query, case=False, na=False)]
    
    if match.empty:
        return f"We don't carry '{search_query}' at this store location."
        
    results = []
    for _, row in match.head(3).iterrows():
        status = f"IN STOCK ({row['stock_level']} units left)" if row['stock_level'] > 0 else "OUT OF STOCK"
        results.append(f"• {row['product_name']} -> {status}.")
        
    return "Inventory status check:\n" + "\n".join(results)

# ==========================================================
# 3. CORE ROUTING CONTEXT
# ==========================================================
llm = Ollama(model="llama3", temperature=0.9)

class ChatRequest(BaseModel):
    message: str

# ==========================================================
# 4. ROBUST API ENDPOINT
# ==========================================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # Step A: Direct the LLM to extract the product term and the user intent
        routing_prompt = (
            f"You are an AI router for a grocery store. Extract the grocery product name the user wants, and their intent.\n"
            f"Allowed intents: 'LOCATION' (if asking where an item is, what aisle) or 'STOCK' (if asking if it is available/in stock).\n"
            f"Respond ONLY with a valid JSON block containing keys 'product_name' (string) and 'intent' ('LOCATION', 'STOCK', or 'NONE').\n"
            f"User message: \"{user_msg}\"\n"
            f"JSON Output:"
        )
        
        routing_output = llm.invoke(routing_prompt).strip()
        
        # Strip code markdown block wrappers if local LLM includes them
        if "```json" in routing_output:
            routing_output = routing_output.split("```json")[1].split("```")[0].strip()
        elif "```" in routing_output:
            routing_output = routing_output.split("```")[1].split("```")[0].strip()
            
        try:
            route_data = json.loads(routing_output)
            product_name = route_data.get("product_name")
            intent = route_data.get("intent", "NONE")
        except:
            # Code fallback if the local LLM fails structural JSON formatting
            product_name = user_msg.replace("where is", "").replace("do you have", "").replace("in stock", "").strip()
            intent = "LOCATION" if "where" in user_msg.lower() or "aisle" in user_msg.lower() else "STOCK"

        # Step B: Route to the respective programmatic python function
        data_context = "Could not identify product reference details."
        if product_name and product_name != "":
            if intent == "LOCATION":
                data_context = find_product_location(product_name)
            elif intent == "STOCK":
                data_context = check_stock_availability(product_name)
        
        # Step C: Generate friendly response with the real data
        final_generation_prompt = (
            f"You are a helpful Instacart grocery store assistant. Translate the data context below into a friendly, helpful reply to the customer.\n"
            f"Data Context:\n{data_context}\n"
            f"Original Customer Question: {user_msg}\n"
            f"Polished Answer:"
        )
        final_answer = llm.invoke(final_generation_prompt).strip()
        
        return {"response": final_answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")