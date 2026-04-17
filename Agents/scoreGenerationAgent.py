import json
import re
from typing import Any

from pydantic import BaseModel

from templates.prompt import build_checklist_score_prompt
from templates.state import ChecklistScore, ReviewChecklist, prState
from utils.models import get_model_for_provider, get_provider_display_name


class scoreGenerationAgent:
    def __init__(self, model_name: str | None = None, provider: str = "ollama"):
        self.provider_display_name = get_provider_display_name(provider)
        self.model = get_model_for_provider(
            provider=provider,
            model_name=model_name,
            temperature=0,
        )

    def run(self, state: prState) -> dict:
        if state.llm_generated_pr_review is None:
            raise ValueError("LLM-generated PR review must be available before scoring.")

        checklist = state.checklist

        review_text = json.dumps(self.serialize_value(state.llm_generated_pr_review), indent=2, ensure_ascii=False)

        if checklist is None:
            checklist_payload, total_items, evaluation_mode = "No checklist", 1, "no_checklist"
        else:
            checklist_payload, total_items, evaluation_mode = self.build_checklist_payload(checklist)
            
        prompt = build_checklist_score_prompt(
            checklist=checklist_payload,
            code_review=review_text,
        )

        print(f"Scoring LLM-generated PR review against checklist with {self.provider_display_name}...")
        response = self.model.invoke(prompt)
        response_text = self.extract_response_text(response)
        matched_items = self.parse_score_response(response_text, total_items, evaluation_mode)
        coverage_score = self.compute_coverage(matched_items, total_items)

        score = ChecklistScore(
            matched_items=matched_items,
            total_items=total_items,
            coverage_score=coverage_score,
            evaluation_mode=evaluation_mode,
        )

        state.score = score
        return state.model_dump()

    def build_checklist_payload(self, checklist: ReviewChecklist) -> tuple[str, int, str]:
        if not getattr(checklist, "items", None):
            return "No checklist", 1, "no_checklist"

        checklist_strings = [
            f"{item.file_path}: {item.target} - {item.verification_point} Expected: {item.expected_outcome}"
            for item in checklist.items
        ]
        return json.dumps(checklist_strings, indent=2, ensure_ascii=False), len(checklist.items), "checklist"

    def serialize_value(self, value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    def extract_response_text(self, response: Any) -> str:
        # Native ollama client via get_model_for_provider returns a plain string
        if isinstance(response, str):
            return response
        content = getattr(response, "content", response)
        return self.normalize_response_content(content)

    def normalize_response_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, list):
            parts = []
            for item in content:
                normalized_item = self.normalize_response_content(item)
                if normalized_item:
                    parts.append(normalized_item)
            return "\n".join(parts)
        if isinstance(content, dict):
            for key in ("text", "content", "value"):
                value = content.get(key)
                if value is not None:
                    return self.normalize_response_content(value)
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def parse_score_response(self, response_text: str, total_items: int, evaluation_mode: str) -> int:
        match = re.search(r"-?\d+", response_text or "")
        if not match:
            raise ValueError(f"Unable to parse numeric score from model response: {response_text!r}")

        score = int(match.group(0))
        if evaluation_mode == "no_checklist":
            return max(0, min(score, 1))
        return max(0, min(score, total_items))

    def compute_coverage(self, matched_items: int, total_items: int) -> float:
        if total_items <= 0:
            return 0.0
        return matched_items / total_items
