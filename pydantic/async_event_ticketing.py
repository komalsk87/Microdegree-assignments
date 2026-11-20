import asyncio
import time


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


async def handle_ticket(
    category: str, description: str, tickets_ready: asyncio.Event
) -> str:
    print(f"{category.capitalize()} ticket is queued and waiting for release.")
    await tickets_ready.wait()

    prompt = f"Handle {category} ticket: {description}"
    print(f"Processing: {prompt}")
    await asyncio.sleep(0.3)
    print(f"Completed: {prompt}")
    return f"Response to: {prompt}"


async def run_ticketing_agent() -> None:
    tickets = []
    for category in ("billing", "technical", "shipping"):
        ticket = collect_validated_ticket(category)
        if ticket is not None:
            tickets.append(ticket)

    if not tickets:
        print("No tickets were approved.")
        return

    tickets_ready = asyncio.Event()
    tasks = [
        asyncio.create_task(
            handle_ticket(category, description, tickets_ready),
            name=f"{category}-ticket",
        )
        for category, description in tickets
    ]

    await asyncio.sleep(0)
    print("All approved tickets are ready; releasing the workers.")
    tickets_ready.set()

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    for (category, _), result in zip(tickets, results):
        print(f"{category.capitalize()} response: {result}")
    print(f"Processed {len(tickets)} approved ticket(s) in {elapsed:.2f} seconds.")


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_ticketing_agent())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


if __name__ == "__main__":
    main()
