import os
import json
from pydantic import BaseModel
from templates.prompt import build_pseudo_solution_generation_prompt
from templates.state import PseudoSolution, prState
from utils.models import get_structured_google_model


class pseudoSolutionAgent:
    def __init__(self, model_name: str | None = None):
        self.model = get_structured_google_model(
            PseudoSolution,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pr_number is None:
            raise ValueError("PR number is missing from the workflow state.")
        if state.review_instruct is None:
            raise ValueError("Review instructions must be generated before the pseudo solution step.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )
        
        data_bundle = self.load_pr_data(folder_path)
        
        prompt = build_pseudo_solution_generation_prompt(
            base_code=json.dumps(data_bundle.get("base_code", {}), indent=2),
            review_instruct=json.dumps(self.serialize_value(state.review_instruct), indent=2),
        )
    
        print("Generating pseudo solution with Gemini...")
        pseudo_solution = self.model.invoke(prompt)

        state.pseudo_solution = pseudo_solution
        self.save_pseudo_solution(folder_path, pseudo_solution)
        return state.model_dump()
    
    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "base_code": "base_code.json",
        }

        data_bundle = {}
        for key, filename in required_files.items():
            file_path = os.path.join(folder_path,filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(F"Missing required file: {file_path}")
            with open(file_path, "r",encoding="utf-8") as file:
                data_bundle[key] = json.load(file)
        
        return data_bundle

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    def save_pseudo_solution(self, folder_path: str, pseudo_solution: PseudoSolution) -> None:
        output_path = os.path.join(folder_path, "pseudo_solution.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(pseudo_solution.model_dump(), file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs",repo="anything-llm",pr_number=5131)
    agent = pseudoSolutionAgent()
    result = agent.run(state)
    print(state.pseudo_solution)
    print("Done generating pseudo solution.")
