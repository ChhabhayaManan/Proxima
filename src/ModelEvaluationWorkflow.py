import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Agents import (
    llmGeneratedPRReviewAgent,
    scoreGenerationAgent,
)
from templates.state import prState
from utils.models import (
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    configure_model_provider,
    normalize_provider,
)


load_dotenv()


def initialize_model_evaluation_runtime() -> dict[str, str | None]:
    ollama_model_name = input(
        "Enter Ollama model name for offline evaluation (leave blank to use Groq + Google): "
    ).strip()
    if ollama_model_name:
        configure_model_provider("ollama", model_name=ollama_model_name)
        return {
            "review_provider": "ollama",
            "review_model_name": ollama_model_name,
            "score_provider": "ollama",
            "score_model_name": ollama_model_name,
        }

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

    configure_model_provider("google", api_key=google_api_key, model_name=google_model_name)
    configure_model_provider("groq", api_key=groq_api_key, model_name=groq_model_name)
    return {
        "review_provider": "groq",
        "review_model_name": groq_model_name,
        "score_provider": "google",
        "score_model_name": google_model_name,
    }


class ModelEvaluationWorkflow:
    def __init__(
        self,
        review_provider: str = "groq",
        review_model_name: str | None = None,
        score_provider: str = "google",
        score_model_name: str | None = None,
    ):
        self.review_provider = normalize_provider(review_provider, default="groq")
        self.score_provider = normalize_provider(score_provider, default="google")

        if self.review_provider not in {"groq", "ollama"}:
            raise ValueError("ModelEvaluationWorkflow review generation supports only Groq or Ollama providers.")
        if self.score_provider not in {"google", "ollama"}:
            raise ValueError("ModelEvaluationWorkflow scoring supports only Google or Ollama providers.")

        configure_model_provider(self.review_provider, model_name=review_model_name)
        configure_model_provider(self.score_provider, model_name=score_model_name)

        self.llm_generated_pr_review_agent = llmGeneratedPRReviewAgent(
            model_name=review_model_name,
            provider=self.review_provider,
        )
        self.score_generation_agent = scoreGenerationAgent(
            model_name=score_model_name,
            provider=self.score_provider,
        )

        proxima_workflow = StateGraph(prState)
        proxima_workflow.add_node("generate_llm_pr_review", self.generate_llm_pr_review_node)
        proxima_workflow.add_node("generate_score", self.generate_score_node)

        proxima_workflow.add_edge(START, "generate_llm_pr_review")
        proxima_workflow.add_edge("generate_llm_pr_review", "generate_score")
        proxima_workflow.add_edge("generate_score", END)

        self.graph = proxima_workflow.compile()

    def generate_llm_pr_review_node(self, state: prState):
        print("--- Node 5: LLM-Generated PR Review ---")
        return self.llm_generated_pr_review_agent.run(state)

    def generate_score_node(self, state: prState):
        print("--- Node 6: Score Generation ---")
        return self.score_generation_agent.run(state)


def main() -> None:
    runtime_config = initialize_model_evaluation_runtime()

    # owner = input("Enter repository owner: ").strip()
    # repo = input("Enter repository name: ").strip()
    owner = "pathwaycom"
    repo = "llm-app"
    pr_number = int(input("Enter PR number: ").strip())

    workflow = ModelEvaluationWorkflow(
        review_provider=runtime_config["review_provider"] or "groq",
        review_model_name=runtime_config["review_model_name"],
        score_provider=runtime_config["score_provider"] or "google",
        score_model_name=runtime_config["score_model_name"],
    )
    initial_state = prState(owner=owner, repo=repo, pr_number=pr_number)

    print("Starting model evaluation workflow...")
    final_state = workflow.graph.invoke(initial_state)

    print("\n==================\nModel Evaluation Finished!\n==================")
    if final_state.get("pr_number") is not None:
        print(f"Reviewed PR: {final_state['pr_number']}")
    if final_state.get("llm_generated_pr_review"):
        print("LLM-generated PR review generated successfully.")
    if final_state.get("score"):
        print("Score generated successfully.")


if __name__ == "__main__":
    main()
