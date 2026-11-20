import asyncio
import time


async def call_llm(prompt: str, delay: float = 0.2) -> str:
    """Pretend to process a ticket using an LLM API."""
    print(f"Processing: {prompt}")
    await asyncio.sleep(delay)
    print(f"Completed: {prompt}")
    return f"Response to: {prompt}"


def collect_validated_ticket(category: str) -> tuple[str, str] | None:
    description = input(f"Describe your {category} issue: ").strip()
    if not description:
        print("Ticket skipped: a description is required.")
        return None

    while True:
        answer = input(
            f"Submit this {category} ticket? (yes/no): "
        ).strip().lower()
        if answer in {"yes", "y"}:
            return category, description
        if answer in {"no", "n"}:
            print(f"{category.capitalize()} ticket not submitted.")
            return None
        print("Please answer yes or no.")


async def handle_ticket(category: str, description: str) -> str:
    prompt = f"Handle {category} ticket: {description}"
    return await call_llm(prompt)


async def main() -> None:
    tickets = []
    for category in ("billing", "technical", "shipping"):
        ticket = collect_validated_ticket(category)
        if ticket is not None:
            tickets.append(ticket)

    if not tickets:
        print("No tickets were approved.")
        return

    start = time.perf_counter()
    results = await asyncio.gather(
        *(handle_ticket(category, description) for category, description in tickets)
    )
    elapsed = time.perf_counter() - start

    for (category, _), result in zip(tickets, results):
        print(f"{category.capitalize()} response: {result}")
    print(f"Processed {len(tickets)} approved ticket(s) in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
