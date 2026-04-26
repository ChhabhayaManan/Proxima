import json
import os
import re
from typing import Any

from pydantic import BaseModel

from templates.prompt import build_checklist_score_prompt
from templates.state import ChecklistScore, ReviewChecklist, prState
from utils.models import get_provider_model


class scoreGenerationAgent:
    def __init__(self, provider: str = "Google Gemini", model_name: str | None = None):
        self.model = get_provider_model(provider=provider, model_name=model_name, temperature=0)

    def run(self, state: prState) -> dict:
        if state.pr_number is None:
            raise ValueError("PR number is missing from the workflow state.")
        if state.llm_generated_pr_review is None:
            raise ValueError("LLM-generated PR review must be available before scoring.")

        folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{state.owner}_{state.repo}_pr{state.pr_number}",
        )

        checklist = self.load_checklist(folder_path, state)
        review_text = json.dumps(self.serialize_value(state.llm_generated_pr_review), indent=2, ensure_ascii=False)

        checklist_payload, total_items, evaluation_mode = self.build_checklist_payload(checklist)
        prompt = build_checklist_score_prompt(
            checklist=checklist_payload,
            code_review=review_text,
        )

        print("Scoring LLM-generated PR review against checklist...")
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
        self.save_score(folder_path, score)
        return state.model_dump()

    def load_checklist(self, folder_path: str, state: prState) -> ReviewChecklist:
        if state.checklist is not None:
            return state.checklist

        file_path = os.path.join(folder_path, "review_checklist.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return ReviewChecklist.model_validate(json.load(file))

    def build_checklist_payload(self, checklist: ReviewChecklist) -> tuple[str, int, str]:
        if not checklist.items:
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

    def save_score(self, folder_path: str, score: ChecklistScore) -> None:
        output_path = os.path.join(folder_path, "score.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(score.model_dump(), file, indent=4, ensure_ascii=False)
