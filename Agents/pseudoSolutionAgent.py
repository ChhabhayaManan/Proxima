import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from templates.prompt import build_pseudo_solution_generation_prompt
from templates.state import PseudoSolution, prState

load_dotenv()

class pseudoSolutionAgent:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env before running this agent."
            )
        
        self.model = ChatGoogleGenerativeAI(
            model = model,
            temperature=0.2,
            api_key=resolved_api_key,
        ).with_structured_output(PseudoSolution)
    def run(self, state: prState) -> dict:
        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )
        
        data_bundle = self.load_pr_data(folder_path)
        
        prompt = build_pseudo_solution_generation_prompt(
            pr_metadata=json.dumps(data_bundle.get("pr_details", {}), indent=2),
            base_code=json.dumps(data_bundle.get("base_code", {}), indent=2),
            changed_code=json.dumps(data_bundle.get("changed_code", {}), indent=2),
            review_instruct=json.dumps(state.review_instruct, indent=2),
        )
    
        print("Generating pseudo solution with Gemini...")
        pseudo_solution = self.model.invoke(prompt)

        state.pseudo_solution = pseudo_solution
        self.save_pseudo_solution(folder_path, pseudo_solution)
        return state.model_dump()
    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "base_code": "base_code.json",
            "changed_code": "changed_code.json",
            "pr_details": "pr_details.json",
            "review_instruct": "review_instruction.json",
        }

        data_bundle = {}
        for key, filename in required_files.items():
            file_path = os.path.join(folder_path,filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(F"Missing required file: {file_path}")
            with open(file_path, "r",encoding="utf-8") as file:
                data_bundle[key] = json.load(file)
        
        return data_bundle

    def save_pseudo_solution(self, folder_path: str, pseudo_solution: PseudoSolution) -> None:
            output_path = os.path.join(folder_path, "pseudo_solution.json")
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(pseudo_solution.model_dump(), file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs",repo="anything-llm",pr_number=5131)
    api_key = os.getenv("GEMINI_API_KEY")
    agent = pseudoSolutionAgent(api_key=api_key)
    result = agent.run(state)
    print(state.pseudo_solution)
    print("Done generating pseudo solution.")
