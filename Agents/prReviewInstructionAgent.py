import os
import json
from templates.prompt import build_review_instruction_generation_prompt
from templates.state import ReviewInstruction, prState
from utils.models import get_structured_provider_model

class prReviewInstructionAgent:
    def __init__(self, provider: str = "Google Gemini", model_name: str | None = None):
        self.model = get_structured_provider_model(
            provider=provider,
            output_schema=ReviewInstruction,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pr_number is None:
            raise ValueError("PR number is missing from the workflow state.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        data_bundle = self.load_pr_data(folder_path)

        prompt = build_review_instruction_generation_prompt(
            ps_metadata=json.dumps(data_bundle.get("pr_details", {}), indent=2),
            gt_code_diff=json.dumps(data_bundle.get("changed_code", {}), indent=2),
            issue_info=json.dumps(data_bundle.get("issue", {}), indent=2),
        )

        print("Generating review instructions with Gemini...")
        review_instruct = self.model.invoke(prompt)

        state.review_instruct = review_instruct
        self.save_review_instruction(folder_path, review_instruct)
        return state.model_dump()

    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            # "base_code": "base_code.json",
            # "merged_code": "merged_code.json",
            "changed_code": "changed_code.json",
            "pr_details": "pr_details.json",
        }
        optional_files = {"issue": "issue.json"}

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
                data_bundle[key] = {}
                continue

            with open(file_path, "r", encoding="utf-8") as file:
                data_bundle[key] = json.load(file)

        return data_bundle

    # def build_prompt(self, data_bundle: dict) -> str:
    #     return build_review_instruction_generation_prompt(
    #         ps_metadata=json.dumps(data_bundle.get("pr_details", {}), indent=2),
    #         gt_code_diff=json.dumps(data_bundle.get("changed_code", {}), indent=2),
    #         issue_info=json.dumps(data_bundle.get("issue", {}), indent=2),
    #     )

    def save_review_instruction(self, folder_path: str, review_instruct: ReviewInstruction) -> None:
        output_path = os.path.join(folder_path, "review_instruction.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(review_instruct.model_dump(), file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=5131)
    agent = prReviewInstructionAgent()
    result = agent.run(state)
    print(state.review_instruct)
    print("Done generating review instructions.")
