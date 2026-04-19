import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.DataGenerationWorkflow import DataGenerationWorkflow, initialize_data_generation_runtime
from src.ModelEvaluationWorkflow import ModelEvaluationWorkflow, initialize_model_evaluation_runtime
from templates.state import prState

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
        import concurrent.futures

        def run_with_timeout(func, args, timeout=300):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args)
                return future.result(timeout=timeout)

        data_generation_result = run_with_timeout(self.data_generation_workflow.graph.invoke, (state,))
        evaluation_input = prState.model_validate(data_generation_result)
        
        assert evaluation_input.review_instruct is not None, "Review instruction lost during state handoff!"
        assert evaluation_input.pseudo_solution is not None, "Pseudo solution lost during state handoff!"
        assert evaluation_input.checklist is not None, "Checklist lost during state handoff!"

        return run_with_timeout(self.model_evaluation_workflow.graph.invoke, (evaluation_input,))


class Workflow:
    def __init__(self):
        self.data_generation_workflow = DataGenerationWorkflow()
        self.model_evaluation_workflow = ModelEvaluationWorkflow()
        self.graph = _SequentialWorkflowGraph(
            data_generation_workflow=self.data_generation_workflow,
            model_evaluation_workflow=self.model_evaluation_workflow,
        )


def initialize_workflow_runtime() -> None:
    initialize_data_generation_runtime()
    initialize_model_evaluation_runtime()


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
