import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.DataGenerationWorkflow import DataGenerationWorkflow
from src.ModelEvaluationWorkflow import ModelEvaluationWorkflow
from templates.state import prState
from utils.models import (
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    configure_google_model,
    configure_groq_model,
)


load_dotenv()


class _SequentialWorkflowGraph:
    def __init__(
        self,
        data_generation_workflow: DataGenerationWorkflow,
        model_evaluation_workflow: ModelEvaluationWorkflow,
    ):
        self.data_generation_workflow = data_generation_workflow
        self.model_evaluation_workflow = model_evaluation_workflow

    def invoke(self, state: prState | dict) -> dict:
        data_generation_result = self.data_generation_workflow.graph.invoke(state)
        evaluation_input = prState.model_validate(data_generation_result)
        return self.model_evaluation_workflow.graph.invoke(evaluation_input)


class Workflow:
    def __init__(self, data_provider: str = "Google Gemini", data_model: str | None = None, eval_provider: str = "Groq", eval_model: str | None = None):
        self.data_generation_workflow = DataGenerationWorkflow(provider=data_provider, model_name=data_model)
        self.model_evaluation_workflow = ModelEvaluationWorkflow(provider=eval_provider, model_name=eval_model)
        self.graph = _SequentialWorkflowGraph(
            data_generation_workflow=self.data_generation_workflow,
            model_evaluation_workflow=self.model_evaluation_workflow,
        )


def initialize_workflow_runtime() -> None:
    github_token = os.getenv("GITHUB_TOKEN") or input("Enter GitHub token: ").strip()
    if not github_token:
        raise ValueError("A GitHub token is required. Set GITHUB_TOKEN or enter it when prompted.")

    google_api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or input("Enter Gemini/Google API key: ").strip()
    )
    if not google_api_key:
        raise ValueError(
            "A Gemini/Google API key is required. Set GEMINI_API_KEY / GOOGLE_API_KEY or enter it when prompted."
        )

    groq_api_key = os.getenv("GROQ_API_KEY") or input("Enter Groq API key: ").strip()
    if not groq_api_key:
        raise ValueError("A Groq API key is required. Set GROQ_API_KEY or enter it when prompted.")

    google_model_name = input(
        f"Enter Google model name (press Enter for {DEFAULT_MODEL_NAME}): "
    ).strip() or DEFAULT_MODEL_NAME
    groq_model_name = input(
        f"Enter Groq model name (press Enter for {DEFAULT_GROQ_MODEL_NAME}): "
    ).strip() or DEFAULT_GROQ_MODEL_NAME

    os.environ["GITHUB_TOKEN"] = github_token
    configure_google_model(api_key=google_api_key, model_name=google_model_name)
    configure_groq_model(api_key=groq_api_key, model_name=groq_model_name)


if __name__ == "__main__":
    initialize_workflow_runtime()
    workflow = Workflow()

    owner = input("Enter repository owner: ").strip()
    repo = input("Enter repository name: ").strip()

    initial_state = prState(owner=owner, repo=repo)

    print("Starting LangGraph workflow...")
    final_state = workflow.graph.invoke(initial_state)

    print("\n==================\nWorkflow Finished!\n==================")
    if final_state.get("pr_number") is not None:
        print(f"Reviewed PR: {final_state['pr_number']}")
    if final_state.get("review_instruct"):
        print("Review instructions generated successfully.")
    if final_state.get("pseudo_solution"):
        print("Pseudo solution generated successfully.")
    if final_state.get("checklist"):
        print("Checklist generated successfully.")
    if final_state.get("llm_generated_pr_review"):
        print("LLM-generated PR review generated successfully.")
    if final_state.get("score"):
        print("Score generated successfully.")
