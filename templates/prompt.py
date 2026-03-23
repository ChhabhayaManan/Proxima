import json
from .state import (
    ReviewInstruction,
    PseudoSolution,
    GeneratedPRReview,
    ReviewChecklist,
)

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

def build_pseudo_solution_generation_prompt(
    pr_metadata: str,
    base_code: str,
    changed_code: str,
    review_instruct: str,
) -> str:
    return f"""You are generating an LLM pseudo solution for a pull request task.

Goal:
Produce a plausible generated implementation candidate, not a polished final solution.

Task:
1. Read the PR metadata, original code, changed diff, and review instruction.
2. Infer what code changes should be attempted.
3. Produce a structured pseudo solution that describes the intended implementation at a moderate level of detail.

Important rules:
- Do not output final polished code.
- Do not try to exactly reproduce the merged implementation.
- Do not include long exact code snippets unless absolutely necessary.
- Focus on intended modifications, affected files, changed behaviors, and approximate implementation steps.
- Keep the pseudo solution realistic as an LLM-generated implementation candidate that may still miss some details.
- Use exact file paths and important identifiers when possible.
- Mention changed UI states, labels, strings, imports, and behavior, but avoid turning the output into a near-copy of the final code.
- The review instruction is the source of truth for intended changes.
- Return valid JSON only.

Output requirements:
- `summary` should describe the intended implementation at a high level.
- `files` should list the files likely to be modified.
- `solution_summary` should briefly explain the intended change in that file.
- `pseudo_code` should describe the implementation plan in concise steps, not full final code.

Input:
PR Metadata:
{pr_metadata.strip()}

Original Code:
{base_code.strip()}

Changed Code Diff:
{changed_code.strip()}

Review Instruction:
{review_instruct.strip()}

Output JSON schema:
{json.dumps(PseudoSolution.model_json_schema(), indent=2)}
"""




def build_pr_review_generation_prompt(
    pseudo_solution: str,
    merged_code: str,
) -> str:
    return f"""You are reviewing a generated implementation against a known correct reference implementation.

You are presented with two versions of code:

Generated Version:
A pseudo or automatically generated implementation.

Reference Version:
A known correct or ground-truth implementation.

Your task:
Compare the Generated Version against the Reference Version and produce pull-request-style review comments on the Generated Version.

Primary objective:
Identify meaningful review-worthy differences between the Generated Version and the Reference Version, including:
- missing logic
- incorrect assumptions
- incomplete implementation
- missing conditions or state handling
- incorrect strings, labels, or UI behavior
- missing imports or required supporting code
- maintainability or clarity problems caused by the generated version being less precise than the reference version

Important review behavior:
- Do not restrict comments only to obvious bugs.
- If the Generated Version is broadly correct but omits important implementation detail, leave a review comment about that omission.
- If the Generated Version is mostly aligned but underspecified, leave comments on the missing specificity.
- Prefer several high-signal comments over a single vague summary.
- A strong review should capture important implementation deltas and reviewer verification points, not just fatal errors.

Guidelines:
- Frame feedback as PR comments directed at the Generated Version.
- Be specific and objective.
- Use exact file paths, labels, strings, identifiers, UI states, conditions, and patterns when possible.
- Reference the Reference Version only to explain why the Generated Version is incomplete, less correct, or less robust.
- Do not explain the full logic of the Reference Version.
- If both versions are functionally equivalent, you may still comment on meaningful trade-offs or missing specificity if they would matter in review.
- Only return "No comment." when there are truly no meaningful differences worth flagging.

Comment quality rules:
- Each comment should correspond to one concrete issue or one tightly related set of issues.
- Avoid generic praise.
- Avoid vague comments like "consider improving clarity".
- Prefer implementation-grounded comments such as missing conditional behavior, missing label changes, missing string changes, missing supporting UI elements, or incomplete handling of state transitions.

Generated Version:
{pseudo_solution.strip()}

Reference Version:
{merged_code.strip()}

Severity guidance:
- high: core functional bug, missing behavior, or incorrect implementation
- medium: important omission, incomplete logic, regression-sensitive behavior, or major clarity gap
- low: minor precision, maintainability, or reviewer-facing clarification

Return valid JSON only.

Output JSON schema:
{json.dumps(GeneratedPRReview.model_json_schema(), indent=2)}
"""




def build_checklist_generation_prompt(
    generated_review: str,
) -> str:
    return f"""You are generating a structured PR review checklist from PR review comments.

Task:
Based on the following generated PR review, create a comprehensive and highly detailed checklist that captures all concrete issues, suggestions, and verification points mentioned in the review.

Checklist Requirements:
- Extract all meaningful review concerns from the PR review.
- Break each concern into the smallest actionable verification step possible.
- Keep checklist items specific and implementation-focused.
- Avoid vague or generic advice.
- Do not invent checklist items that are not supported by the PR review.
- Avoid duplicates and near-duplicate items.
- Prefer precise, scoring-friendly verification points.

Field requirements:
- `file_path`: repository-relative file path if available
- `target`: exact symbol, code region, string, condition, import, UI element, or pattern to verify
- `verification_point`: what should be checked
- `expected_outcome`: what should be true if the implementation is correct
- `severity`: one of `low`, `medium`, or `high`

Severity guidance:
- high: core functional issue, missing behavior, or incorrect implementation
- medium: important implementation detail, regression-sensitive logic, or maintainability issue
- low: secondary clarity, consistency, or polish issue

Generated PR Review:
{generated_review.strip()}

Return valid JSON only.

Output JSON schema:
{json.dumps(ReviewChecklist.model_json_schema(), indent=2)}
"""

