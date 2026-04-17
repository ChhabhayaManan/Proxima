import json

from pydantic import BaseModel

from templates.prompt import build_llm_generated_pr_review_prompt
from templates.state import GeneratedPRReview, prState
from utils.models import get_provider_display_name, get_structured_model

class llmGeneratedPRReviewAgent:
    def __init__(self, model_name: str | None = None, provider: str = "ollama"):
        self.provider_display_name = get_provider_display_name(provider)
        self.model = get_structured_model(
            GeneratedPRReview,
            provider=provider,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pseudo_solution is None:
            raise ValueError("Pseudo solution must be available before LLM-generated PR review generation.")

        pr_details = state.pr_details or {}
        base_code = state.base_code or {}

        prompt = build_llm_generated_pr_review_prompt(
            pr_metadata=json.dumps(pr_details, indent=2, ensure_ascii=False),
            original_code=json.dumps(base_code, indent=2, ensure_ascii=False),
            generated_code=json.dumps(self.serialize_value(state.pseudo_solution), indent=2, ensure_ascii=False),
        )

        print(f"Generating LLM-generated PR review with {self.provider_display_name}...")
        llm_generated_pr_review = self.model.invoke(prompt)

        state.llm_generated_pr_review = llm_generated_pr_review
        return state.model_dump()

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value
