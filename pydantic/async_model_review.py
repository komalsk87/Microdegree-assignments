import asyncio
import time
from ollama import AsyncClient


# 1. Define an "async" function to analyze ONE review
async def analyze_review(client: AsyncClient, review_id: int, review_text: str) -> str:
    print(f" Sending Review #{review_id} to Llama...")

    prompt = (
        "Classify the sentiment of this product review as Positive, Negative, or Neutral, "
        "and state the main issue or praise in 5 words or less.\n\n"
        f"Review: \"{review_text}\""
    )

    # `await` tells Python: "Pause THIS task while waiting for Llama,
    # and go work on other reviews in the meantime!"
    response = await client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )

    result = response["message"]["content"].strip()
    print(f" Finished Review #{review_id}!")
    return f"Review #{review_id}: {result}"


# 2. Main coordinator function
async def main():
    # Real-world sample data: 5 customer reviews
    reviews = [
        "The battery lasts two full days, best purchase this year!",
        "Arrived with a cracked screen and customer support never replied.",
        "It works okay for the price, nothing special.",
        "Super lightweight laptop, fits right in my backpack.",
        "The app crashes every time I try to export a PDF.",
    ]

    # Create an async Ollama client
    client = AsyncClient()

    start_time = time.time()

    # Create a list of tasks (nothing runs yet—we are just queuing them up)
    tasks = [
        analyze_review(client, i + 1, text)
        for i, text in enumerate(reviews)
    ]

    # `asyncio.gather` launches ALL tasks at the same time and waits for all to finish
    results = await asyncio.gather(*tasks)

    elapsed = round(time.time() - start_time, 2)

    print("\n--- Final Summary Report ---")
    for line in results:
        print(line)

    print(f"\n Total time taken: {elapsed} seconds")


# 3. Start the async event loop
if __name__ == "__main__":
    asyncio.run(main())