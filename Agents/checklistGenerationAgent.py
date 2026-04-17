import json

from pydantic import BaseModel
from templates.prompt import build_checklist_generation_prompt
from templates.state import ReviewChecklist, prState
from utils.models import get_provider_display_name, get_structured_model


class checklistGenerationAgent:
    def __init__(self, model_name: str | None = None, provider: str = "ollama"):
        self.provider_display_name = get_provider_display_name(provider)
        self.model = get_structured_model(
            ReviewChecklist,
            provider=provider,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.pseudo_solution is None:
            raise ValueError("Pseudo solution must be available before checklist generation.")

        merged_code = state.merged_code or {}
        changed_code = state.changed_code or {}

        prompt = build_checklist_generation_prompt(
            pseudo_solution=json.dumps(self.serialize_value(state.pseudo_solution), indent=2, ensure_ascii=False),
            merged_code=json.dumps(merged_code, indent=2, ensure_ascii=False),
            changed_code=json.dumps(changed_code, indent=2, ensure_ascii=False),
        )

        print(f"Generating review checklist with {self.provider_display_name}...")
        checklist = self.model.invoke(prompt)

        state.checklist = checklist
        return state.model_dump()

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value
