import json
from templates.prompt import build_review_instruction_generation_prompt
from templates.state import ReviewInstruction, prState
from utils.models import get_provider_display_name, get_structured_model

class prReviewInstructionAgent:
    def __init__(self, model_name: str | None = None, provider: str = "ollama"):
        self.provider_display_name = get_provider_display_name(provider)
        self.model = get_structured_model(
            ReviewInstruction,
            provider=provider,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        pr_details = state.pr_details or {}
        changed_code = state.changed_code or {}
        issue_info = state.issue_details or {}

        prompt = build_review_instruction_generation_prompt(
            ps_metadata=json.dumps(pr_details, indent=2),
            gt_code_diff=json.dumps(changed_code, indent=2),
            issue_info=json.dumps(issue_info, indent=2),
        )

        print(f"Generating review instructions with {self.provider_display_name}...")
        review_instruct = self.model.invoke(prompt)

        state.review_instruct = review_instruct
        return state.model_dump()
