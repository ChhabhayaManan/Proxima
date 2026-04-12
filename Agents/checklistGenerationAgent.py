import os
import json

from pydantic import BaseModel
from templates.prompt import build_checklist_generation_prompt
from templates.state import PseudoSolution, ReviewChecklist, prState
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
        if state.pseudo_solution is None:
            raise ValueError("Pseudo solution must be available before checklist generation.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        data_bundle = self.load_pr_data(folder_path)
        prompt = build_checklist_generation_prompt(
            pseudo_solution=json.dumps(self.serialize_value(state.pseudo_solution), indent=2, ensure_ascii=False),
            merged_code=json.dumps(data_bundle.get("merged_code", {}), indent=2, ensure_ascii=False),
            changed_code=json.dumps(data_bundle.get("changed_code", {}), indent=2, ensure_ascii=False),
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

    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "merged_code": "merged_code.json",
            "changed_code": "changed_code.json",
        }
        optional_files = {
            "pseudo_solution": "pseudo_solution.json",
        }

        data_bundle = {}
        for key, filename in required_files.items():
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing required file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data_bundle[key] = json.load(file)

        for key, filename in optional_files.items():
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                data_bundle[key] = json.load(file)

        return data_bundle

    def hydrate_state_from_saved_data(self, state: prState, data_bundle: dict) -> None:
        if state.pseudo_solution is None and "pseudo_solution" in data_bundle:
            state.pseudo_solution = PseudoSolution.model_validate(data_bundle["pseudo_solution"])

    def save_checklist(self, folder_path: str, checklist: ReviewChecklist) -> None:
        output_path = os.path.join(folder_path, "review_checklist.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(checklist.model_dump(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=5131)
    agent = checklistGenerationAgent()
    folder_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        f"{state.owner}_{state.repo}_pr{state.pr_number}",
    )
    saved_data = agent.load_pr_data(folder_path)
    agent.hydrate_state_from_saved_data(state, saved_data)
    result = agent.run(state)
    print(state.checklist)
    print("Done generating review checklist.")
