from typing import Literal
import ollama
from pydantic import BaseModel, Field

MODEL = "llama3.1:8b"


class StudyTask(BaseModel):
    topic: str = Field(description="Main topic to study")
    minutes: int = Field(gt=0, le=180, description="Duration in minutes (1 to 180)")
    priority: Literal["low", "medium", "high"] = Field(description="Study priority")


def plan_study_task(raw_input: str) -> StudyTask:
    response = ollama.chat(
        model=MODEL,
        format=StudyTask.model_json_schema(),
        options={"temperature": 0.1},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a study task matching the required JSON schema. "
                    "Keep topic concise and choose realistic minutes (1-180)."
                ),
            },
            {"role": "user", "content": raw_input},
        ],
    )
    return StudyTask.model_validate_json(response.message.content)


def run_agent(notes: list[str]) -> None:
    for note in notes:
        task = plan_study_task(note)
        print(
            f"[Added] {task.topic} | {task.minutes} mins | Priority: {task.priority}"
        )


if __name__ == "__main__":
    sample_notes = [
        "Spend 25 minutes reviewing Python dictionaries before tomorrow's quiz",
        "Quickly skim SQL joins for 15 mins when I have spare time",
    ]
    run_agent(sample_notes)