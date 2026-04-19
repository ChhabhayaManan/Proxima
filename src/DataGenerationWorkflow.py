import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Agents import (
    checklistGenerationAgent,
    gitPRRetriever,
    prReviewInstructionAgent,
    pseudoSolutionAgent,
)
from templates.state import prState
from utils.models import DEFAULT_MODEL_NAME, configure_model_provider, normalize_provider


load_dotenv()


def initialize_data_generation_runtime() -> tuple[str, str | None]:
    github_token = os.getenv("GITHUB_TOKEN") or input("Enter GitHub token: ").strip()
    if not github_token:
        raise ValueError("A GitHub token is required. Set GITHUB_TOKEN or enter it when prompted.")

    ollama_model_name = input(
        "Enter Ollama model name for offline data generation (leave blank to use Google): "
    ).strip()
    if ollama_model_name:
        os.environ["GITHUB_TOKEN"] = github_token
        configure_model_provider("ollama", model_name=ollama_model_name)
        return "ollama", ollama_model_name

    google_api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or input("Enter Gemini/Google API key: ").strip()
    )
    if not google_api_key:
        raise ValueError(
            "A Gemini/Google API key is required. Set GEMINI_API_KEY / GOOGLE_API_KEY or enter it when prompted."
        )

    model_name = input(
        f"Enter Google model name (press Enter for {DEFAULT_MODEL_NAME}): "
    ).strip() or DEFAULT_MODEL_NAME

    os.environ["GITHUB_TOKEN"] = github_token
    configure_model_provider("google", api_key=google_api_key, model_name=model_name)
    return "google", model_name


class DataGenerationWorkflow:
    def __init__(self, model_provider: str = "google", model_name: str | None = None):
        resolved_github_token = (os.getenv("GITHUB_TOKEN") or "").strip()
        if not resolved_github_token:
            raise ValueError(
                "A GitHub token is required. Set GITHUB_TOKEN or call initialize_data_generation_runtime() first."
            )

        self.model_provider = normalize_provider(model_provider, default="google")
        if self.model_provider not in {"google", "ollama"}:
            raise ValueError("DataGenerationWorkflow supports only Google or Ollama providers.")

        configure_model_provider(self.model_provider, model_name=model_name)

        self.pr_retriever = gitPRRetriever(resolved_github_token)
        self.review_instruction_agent = prReviewInstructionAgent(
            model_name=model_name,
            provider=self.model_provider,
        )
        self.pseudo_solution_agent = pseudoSolutionAgent(
            model_name=model_name,
            provider=self.model_provider,
        )
        self.checklist_generation_agent = checklistGenerationAgent(
            model_name=model_name,
            provider=self.model_provider,
        )

        proxima_workflow = StateGraph(prState)
        proxima_workflow.add_node("retrieve_pr", self.retrieve_pr_node)
        proxima_workflow.add_node("generate_review_instruction", self.generate_instruction_node)
        proxima_workflow.add_node("generate_pseudo_solution", self.generate_pseudo_solution_node)
        proxima_workflow.add_node("generate_checklist", self.generate_checklist_node)

        proxima_workflow.add_edge(START, "retrieve_pr")
        proxima_workflow.add_edge("retrieve_pr", "generate_review_instruction")
        proxima_workflow.add_edge("generate_review_instruction", "generate_pseudo_solution")
        proxima_workflow.add_edge("generate_pseudo_solution", "generate_checklist")
        proxima_workflow.add_edge("generate_checklist", END)

        self.graph = proxima_workflow.compile()

    def retrieve_pr_node(self, state: prState):
        print("--- Node 1: PR Retriever ---")
        return self.pr_retriever.run(state)

    def generate_instruction_node(self, state: prState):
        print("--- Node 2: Review Instruction Generation ---")
        return self.review_instruction_agent.run(state)

    def generate_pseudo_solution_node(self, state: prState):
        print("--- Node 3: Pseudo Solution Generation ---")
        return self.pseudo_solution_agent.run(state)

    def generate_checklist_node(self, state: prState):
        print("--- Node 4: Checklist Generation ---")
        return self.checklist_generation_agent.run(state)


def main() -> None:
    model_provider, model_name = initialize_data_generation_runtime()

    owner = "pathwaycom"
    repo = "llm-app"
    pr_number_text = input("Enter PR number (leave blank to choose interactively): ").strip()

    initial_state = prState(
        owner=owner,
        repo=repo,
        pr_number=int(pr_number_text) if pr_number_text else None,
    )

    workflow = DataGenerationWorkflow(model_provider=model_provider, model_name=model_name)
    print("Starting data generation workflow...")
    final_state = workflow.graph.invoke(initial_state)

    print("\n==================\nData Generation Finished!\n==================")
    if final_state.get("pr_number") is not None:
        print(f"Reviewed PR: {final_state['pr_number']}")
    if final_state.get("review_instruct"):
        print("Review instructions generated successfully.")
    if final_state.get("pseudo_solution"):
        print("Pseudo solution generated successfully.")
    if final_state.get("checklist"):
        print("Checklist generated successfully.")


if __name__ == "__main__":
    main()
