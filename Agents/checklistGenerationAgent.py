import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from templates.prompt import build_checklist_generation_prompt
from templates.state import ReviewChecklist, prState

load_dotenv()


class checklistGenerationAgent:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env before running this agent."
            )

        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.2,
            api_key=resolved_api_key,
        ).with_structured_output(ReviewChecklist)

    def run(self, state: prState) -> dict:
        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        data_bundle = self.load_pr_data(folder_path)

        prompt = build_checklist_generation_prompt(
            generated_review=json.dumps(data_bundle.get("generated_review", {}), indent=2),
        )

        print("Generating review checklist with Gemini...")
        checklist = self.model.invoke(prompt)

        state.checklist = checklist
        self.save_checklist(folder_path, checklist)
        return state.model_dump()

    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "generated_review": "generated_review.json",
        }

        data_bundle = {}
        for key, filename in required_files.items():
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing required file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data_bundle[key] = json.load(file)

        return data_bundle

    def save_checklist(self, folder_path: str, checklist: ReviewChecklist) -> None:
        output_path = os.path.join(folder_path, "review_checklist.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(checklist.model_dump(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    state = prState(owner="Mintplex-Labs", repo="anything-llm", pr_number=5131)
    api_key = os.getenv("GEMINI_API_KEY")
    agent = checklistGenerationAgent(api_key=api_key)
    result = agent.run(state)
    print(state.checklist)
    print("Done generating review checklist.")
