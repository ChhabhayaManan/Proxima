import json
from .state import ReviewInstruction


def build_review_instruction_generation_prompt(
    ps_metadata: str,
    gt_code_diff: str,
    issue_info: str = "",
) -> str:
    normalized_issue_info = issue_info.strip() or "No related issues provided."
    return f"""You are generating PR review instructions for a coding agent.

Task:
1. Write a concise `problem_definition` based on the inputs.
2. Generate `code_editing_requirement` as a list of specific code-level edits.

Rules:
- Focus only on code modifications.
- Ignore non-code file updates (Markdown, docs, config files, lockfiles, assets).
- Extract and use exact identifiers from the diff when possible (functions, classes, methods, variables, constants, modules).
- Every item in `code_editing_requirement` must include:
  - `modification_target`: exact symbol/location to update.
  - `modification_logic`: precise implementation logic to apply.
  - `input_output_requirement`: behavior contract, edge-case handling, or `null` if not applicable.
- Keep instructions technical and implementation-focused.
- Exclude team discussion, business context, release notes, and non-code tasks.
- Do not output markdown. Return valid JSON only.

Input:
Problem Statement:
{ps_metadata.strip()}

Related Issues:
{normalized_issue_info}

Diff Format Code Changes:
{gt_code_diff.strip()}

Output JSON schema:
{json.dumps(ReviewInstruction.model_json_schema(), indent=2)}
"""