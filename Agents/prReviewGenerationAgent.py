import os
import json

from pydantic import BaseModel
from templates.prompt import build_pr_review_generation_prompt
from templates.state import GeneratedPRReview, prState
from utils.models import get_structured_google_model


class prReviewGenerationAgent:
    def __init__(self, model_name: str | None = None):
        self.model = get_structured_google_model(
            GeneratedPRReview,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pr_number is None:
            raise ValueError("PR number is missing from the workflow state.")
        if state.pseudo_solution is None:
            raise ValueError("Pseudo solution must be generated before the review generation step.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        data_bundle = self.load_pr_data(folder_path)

        prompt = build_pr_review_generation_prompt(
            pseudo_solution=json.dumps(self.serialize_value(state.pseudo_solution), indent=2),
            merged_code=json.dumps(data_bundle.get("merged_code", {}), indent=2),
        )

        print("Generating PR review with Gemini...")
        generated_review = self.model.invoke(prompt)

        state.generated_review = generated_review
        self.save_generated_review(folder_path, generated_review)
        return state.model_dump()

    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "merged_code": "merged_code.json",
        }

        data_bundle = {}
        for key, filename in required_files.items():
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing required file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data_bundle[key] = json.load(file)

        return data_bundle

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    def save_generated_review(self, folder_path: str, generated_review: GeneratedPRReview) -> None:
        output_path = os.path.join(folder_path, "generated_review.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(generated_review.model_dump(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=5131)
    agent = prReviewGenerationAgent()
    result = agent.run(state)
    print(state.generated_review)
    print("Done generating PR review.")
