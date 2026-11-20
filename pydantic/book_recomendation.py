from typing import Literal
import ollama
from pydantic import BaseModel, ConfigDict, Field, computed_field

MODEL = "llama3.1:8b"  # Uses your local Ollama Llama model


class BookEntry(BaseModel):
    # frozen=True makes the created book object read-only after validation
    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, description="Book title")
    author: str = Field(min_length=1, description="Author name")
    pages: int = Field(gt=0, le=3000, description="Total number of pages")
    genre: Literal["tech", "fiction", "science", "biography", "other"]

    # pattern enforces a 3-digit shelf code like "A01" or "T12"
    shelf_code: str = Field(
        default="G01",
        pattern=r"^[A-Z]\d{2}$",
        description="Shelf code like 'T04' (one capital letter + two digits)",
    )

    # exclude=True keeps personal notes on the Python object, but omits them in model_dump_json()
    private_note: str | None = Field(
        default=None,
        exclude=True,
        description="Optional personal note not exported in public JSON",
    )

    @computed_field  # type: ignore[misc]
    @property
    def estimated_hours(self) -> float:
        """Derived automatically at ~40 pages per hour."""
        return round(self.pages / 40, 1)


def extract_book(raw_text: str) -> BookEntry:
    response = ollama.chat(
        model=MODEL,
        format=BookEntry.model_json_schema(),  # Enforces structured JSON output
        options={"temperature": 0},  # Deterministic output
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract book details as JSON matching the schema. "
                    "Use a realistic page count and a shelf_code like 'T01' or 'F02'."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    )
    return BookEntry.model_validate_json(response.message.content)  #


def add_book_to_library(raw_text: str, library: list[BookEntry]) -> None:
    book = extract_book(raw_text)
    library.append(book)
    print(
        f"Added '{book.title}' by {book.author} "
        f"({book.pages}p, ~{book.estimated_hours}h) -> Shelf {book.shelf_code}"
    )
    if book.private_note:
        print(f"  Private note (kept locally): {book.private_note}")
    print(f"  Exported JSON (without private_note): {book.model_dump_json()}\n")


if __name__ == "__main__":
    reading_list: list[BookEntry] = []
    add_book_to_library(
        "Add 'Designing Data-Intensive Applications' (~600 pages, tech). Note: read chapter 5 first.",
        reading_list,
    )
    print(f"Total books in reading list: {len(reading_list)}")
