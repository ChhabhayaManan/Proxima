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
from utils.models import get_data_generation_provider

load_dotenv()


class DataGenerationWorkflow:
    def __init__(self, github_token: str | None = None, provider: str = "ollama"):
        resolved_github_token = (github_token or os.getenv("GITHUB_TOKEN") or "").strip()
        if not resolved_github_token:
            raise ValueError(
                "A GitHub token is required. Set GITHUB_TOKEN."
            )

        self.provider = provider
        self.pr_retriever = gitPRRetriever(resolved_github_token)
        self.review_instruction_agent = prReviewInstructionAgent(provider=self.provider)
        self.pseudo_solution_agent = pseudoSolutionAgent(provider=self.provider)
        self.checklist_generation_agent = checklistGenerationAgent(provider=self.provider)

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
