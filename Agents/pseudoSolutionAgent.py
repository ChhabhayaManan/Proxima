import json
from pydantic import BaseModel
from templates.prompt import build_pseudo_solution_generation_prompt
from templates.state import PseudoSolution, prState
from utils.models import get_provider_display_name, get_structured_model


class pseudoSolutionAgent:
    def __init__(self, model_name: str | None = None, provider: str = "ollama"):
        self.provider_display_name = get_provider_display_name(provider)
        self.model = get_structured_model(
            PseudoSolution,
            provider=provider,
            model_name=model_name,
        )

    def run(self, state: prState) -> dict:
        if state.review_instruct is None:
            raise ValueError("Review instructions must be generated before the pseudo solution step.")

        base_code = state.base_code or {}
        
        prompt = build_pseudo_solution_generation_prompt(
            base_code=json.dumps(base_code, indent=2),
            review_instruct=json.dumps(self.serialize_value(state.review_instruct), indent=2),
        )
    
        print(f"Generating pseudo solution with {self.provider_display_name}...")
        pseudo_solution = self.model.invoke(prompt)

        state.pseudo_solution = pseudo_solution
        return state.model_dump()
    
    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value
