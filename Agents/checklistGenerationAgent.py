import os
import json

from pydantic import BaseModel
from templates.prompt import build_checklist_generation_prompt
from templates.state import ReviewChecklist, prState
from utils.models import get_structured_google_model


class checklistGenerationAgent:
    def __init__(self, model_name: str | None = None):
        self.model = get_structured_google_model(
            ReviewChecklist,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pr_number is None:
            raise ValueError("PR number is missing from the workflow state.")
        if state.generated_review is None:
            raise ValueError("Generated review must be available before checklist generation.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        prompt = build_checklist_generation_prompt(
            generated_review=json.dumps(self.serialize_value(state.generated_review), indent=2),
        )

        print("Generating review checklist with Gemini...")
        checklist = self.model.invoke(prompt)

        state.checklist = checklist
        self.save_checklist(folder_path, checklist)
        return state.model_dump()

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    def save_checklist(self, folder_path: str, checklist: ReviewChecklist) -> None:
        output_path = os.path.join(folder_path, "review_checklist.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(checklist.model_dump(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=5131)
    agent = checklistGenerationAgent()
    result = agent.run(state)
    print(state.checklist)
    print("Done generating review checklist.")
