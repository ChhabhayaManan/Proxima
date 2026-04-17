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
from utils.models import get_scoring_provider


load_dotenv()


class ModelEvaluationWorkflow:
    def __init__(self, provider: str = "ollama"):
        self.scoring_provider = provider

        self.llm_generated_pr_review_agent = llmGeneratedPRReviewAgent()
        self.score_generation_agent = scoreGenerationAgent(provider=self.scoring_provider)

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
