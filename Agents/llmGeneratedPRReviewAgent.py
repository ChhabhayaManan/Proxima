import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from templates.prompt import build_llm_generated_pr_review_prompt
from templates.state import GeneratedPRReview, PseudoSolution, prState
from utils.models import (
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_NAME,
    configure_groq_model,
    configure_ollama_model,
    get_provider_display_name,
    get_structured_model,
    normalize_provider,
)


load_dotenv()


class llmGeneratedPRReviewAgent:
    def __init__(self, model_name: str | None = DEFAULT_GROQ_MODEL_NAME, provider: str = "groq"):
        self.provider = normalize_provider(provider, default="groq")
        self.model = get_structured_model(
            self.provider,
            GeneratedPRReview,
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
        self.hydrate_state_from_saved_data(state, data_bundle)
        if state.pseudo_solution is None:
            raise ValueError("Pseudo solution must be available before LLM-generated PR review generation.")

        prompt = build_llm_generated_pr_review_prompt(
            pr_metadata=json.dumps(data_bundle.get("pr_details", {}), indent=2, ensure_ascii=False),
            original_code=json.dumps(data_bundle.get("base_code", {}), indent=2, ensure_ascii=False),
            generated_code=json.dumps(self.serialize_value(state.pseudo_solution), indent=2, ensure_ascii=False),
        )

        print(
            "Generating LLM-generated PR review with "
            f"{get_provider_display_name(self.provider)} using PR metadata, original code, pseudo solution, and merged code..."
        )
        llm_generated_pr_review = self.model.invoke(prompt)

        state.llm_generated_pr_review = llm_generated_pr_review
        self.save_llm_generated_pr_review(folder_path, llm_generated_pr_review)
        return state.model_dump()

    def load_pr_data(self, folder_path: str) -> dict:
        required_files = {
            "pr_details": "pr_details.json",
            "base_code": "base_code.json",
            "pseudo_solution": "pseudo_solution.json"
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

    def hydrate_state_from_saved_data(self, state: prState, data_bundle: dict) -> None:
        if state.pseudo_solution is None:
            state.pseudo_solution = PseudoSolution.model_validate(data_bundle["pseudo_solution"])

    def save_llm_generated_pr_review(
        self,
        folder_path: str,
        llm_generated_pr_review: GeneratedPRReview,
    ) -> None:
        output_path = os.path.join(folder_path, "llm_generated_pr_review.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(llm_generated_pr_review.model_dump(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    ollama_model_name = input(
        "Enter Ollama model name for offline review generation (leave blank to use Groq): "
    ).strip()
    if ollama_model_name:
        provider = "ollama"
        model_name = ollama_model_name or DEFAULT_OLLAMA_MODEL_NAME
        configure_ollama_model(model_name=model_name)
    else:
        provider = "groq"
        groq_api_key = os.getenv("GROQ_API_KEY") or input("Enter Groq API key: ").strip()
        if not groq_api_key:
            raise ValueError("A Groq API key is required. Set GROQ_API_KEY or enter it when prompted.")

        model_name = input(
            f"Enter Groq model name (press Enter for {DEFAULT_GROQ_MODEL_NAME}): "
        ).strip() or DEFAULT_GROQ_MODEL_NAME
        configure_groq_model(api_key=groq_api_key, model_name=model_name)

    owner = input("Enter repository owner: ").strip()
    repo = input("Enter repository name: ").strip()
    pr_number = int(input("Enter PR number: ").strip())

    state = prState(owner=owner, repo=repo, pr_number=pr_number)
    agent = llmGeneratedPRReviewAgent(model_name=model_name, provider=provider)
    result = agent.run(state)
    print(result)
    print("Done generating LLM-generated PR review.")
