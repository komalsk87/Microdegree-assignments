# prepare_data.py
import pandas as pd

print("Step 1: Reading Instacart CSV files...")
try:
    # We only need these three small-to-medium files
    products = pd.read_csv('products.csv')
    aisles = pd.read_csv('aisles.csv')
    order_products = pd.read_csv('order_products__prior.csv', nrows=2000000) # Load a 2M row sample to find true popularity
    
    print("Step 2: Merging data relations...")
    # Combine products with their human-readable aisle names
    df = pd.merge(products, aisles, on='aisle_id')
    
    print("Step 3: Calculating top popular products...")
    # Count how many times each product_id appears in transactions
    popularity = order_products['product_id'].value_counts().reset_index()
    popularity.columns = ['product_id', 'count']
    
    # Keep only the top 50 most popular products
    top_50 = popularity.head(50)
    
    # Filter our main database to keep ONLY these top 50 rows
    popular_grocery_db = df[df['product_id'].isin(top_50['product_id'])]
    
    print(f"Filtered database down to {len(popular_grocery_db)} highly popular items.")
    
    print("Step 4: Compiling text corpus for custom LLM training...")
    corpus = ""
    # We repeat the rows a few times so the neural network has multiple epochs to see patterns
    for repeat in range(30): 
        for _, row in popular_grocery_db.iterrows():
            # Formatting as structured text tokens the LLM can easily learn
            corpus += f"Product: {row['product_name']} | Aisle: {row['aisle']}.\n"
            
    # Save to a clean text file
    with open('popular_groceries_training.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
        
    print("Success! Generated 'popular_groceries_training.txt' for your LLM.")
    print("Sample text structure:")
    print("\n".join(corpus.split('\n')[:5]))

except FileNotFoundError as e:
    print(f"Error: Missing CSV files in directory. Details: {e}")