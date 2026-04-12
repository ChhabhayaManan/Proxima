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
    llmGeneratedPRReviewAgent,
    prReviewGenerationAgent,
    prReviewInstructionAgent,
    pseudoSolutionAgent,
    scoreGenerationAgent,
)
from templates.state import prState
from utils.models import DEFAULT_MODEL_NAME, configure_google_model


load_dotenv()


class Workflow:
    def __init__(
        self,
        github_token: str,
        api_key: str | None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        resolved_github_token = github_token.strip() if github_token else os.getenv("GITHUB_TOKEN")
        if not resolved_github_token:
            raise ValueError("A GitHub token is required. Set GITHUB_TOKEN or provide it when starting the workflow.")

        configure_google_model(api_key=api_key, model_name=model_name)

        self.pr_retriever = gitPRRetriever(resolved_github_token)
        self.review_instruction_agent = prReviewInstructionAgent()
        self.pseudo_solution_agent = pseudoSolutionAgent()
        self.review_generation_agent = prReviewGenerationAgent()
        self.checklist_generation_agent = checklistGenerationAgent()
        self.llm_generated_pr_review_agent = llmGeneratedPRReviewAgent()
        self.score_generation_agent = scoreGenerationAgent()

        proxima_workflow = StateGraph(prState)
        proxima_workflow.add_node("retrieve_pr", self.retrieve_pr_node)
        proxima_workflow.add_node("generate_review_instruction", self.generate_instruction_node)
        proxima_workflow.add_node("generate_pseudo_solution", self.generate_pseudo_solution_node)
        proxima_workflow.add_node("generate_pr_review", self.generate_review_node)
        proxima_workflow.add_node("generate_checklist", self.generate_checklist_node)
        proxima_workflow.add_node("generate_llm_pr_review", self.generate_llm_pr_review_node)
        proxima_workflow.add_node("generate_score", self.generate_score_node)

        proxima_workflow.add_edge(START, "retrieve_pr")
        proxima_workflow.add_edge("retrieve_pr", "generate_review_instruction")
        proxima_workflow.add_edge("generate_review_instruction", "generate_pseudo_solution")
        proxima_workflow.add_edge("generate_pseudo_solution", "generate_pr_review")
        proxima_workflow.add_edge("generate_pr_review", "generate_checklist")
        proxima_workflow.add_edge("generate_checklist", "generate_llm_pr_review")
        proxima_workflow.add_edge("generate_llm_pr_review", "generate_score")
        proxima_workflow.add_edge("generate_score", END)

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

    def generate_review_node(self, state: prState):
        print("--- Node 4: Review Generation ---")
        return self.review_generation_agent.run(state)

    def generate_checklist_node(self, state: prState):
        print("--- Node 5: Checklist Generation ---")
        return self.checklist_generation_agent.run(state)

    def generate_llm_pr_review_node(self, state: prState):
        print("--- Node 6: LLM-Generated PR Review ---")
        return self.llm_generated_pr_review_agent.run(state)

    def generate_score_node(self, state: prState):
        print("--- Node 7: Score Generation ---")
        return self.score_generation_agent.run(state)


if __name__ == "__main__":
    github_token = os.getenv("GITHUB_TOKEN") or input("Enter GitHub token: ").strip()
    api_key = input("Enter Gemini/Google API key (leave blank to use env): ").strip() or None
    model_name = input(
        f"Enter model name (press Enter for {DEFAULT_MODEL_NAME}): "
    ).strip() or DEFAULT_MODEL_NAME

    workflow = Workflow(
        github_token=github_token,
        api_key=api_key,
        model_name=model_name,
    )

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
    if final_state.get("generated_review"):
        print("PR review generated successfully.")
    if final_state.get("checklist"):
        print("Checklist generated successfully.")
    if final_state.get("llm_generated_pr_review"):
        print("LLM-generated PR review generated successfully.")
    if final_state.get("score"):
        print("Score generated successfully.")
