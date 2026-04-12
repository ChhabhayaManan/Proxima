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
- Infer the programming language from the code and file extensions, and tailor the instruction to that language's syntax, APIs, idioms, and common review concerns.
- When relevant, use language-appropriate terminology such as function vs method, module vs package, header vs source file, async patterns, typing conventions, memory/resource management, or exception/error handling style.
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
    base_code: str,
    review_instruct: str,
) -> str:
    return f"""You are generating an LLM pseudo solution for a pull request task.

Goal:
Produce a plausible generated implementation candidate, not a polished final solution.

Task:
1. Read the original code and review instruction.
2. Infer what code changes should be attempted.
3. Produce a structured pseudo solution that describes the intended implementation at a moderate level of detail.

Important rules:
- Do not output final polished code.
- Do not try to exactly reproduce the merged implementation.
- Do not include long exact code snippets unless absolutely necessary.
- Focus on intended modifications, affected files, changed behaviors, and approximate implementation steps.
- Keep the pseudo solution realistic as an LLM-generated implementation candidate that may still miss some details.
- Use exact file paths and important identifiers when possible.
- Infer the programming language from the code and file extensions, and make the pseudo solution consistent with that language's syntax, imports/includes, control flow, naming patterns, type system, and project conventions.
- Mention language-specific implementation details when they matter, such as exception handling, async control flow, state management, type annotations, generics/templates, memory ownership, or test expectations.
- Mention changed UI states, labels, strings, imports, and behavior, but avoid turning the output into a near-copy of the final code.
- The review instruction is the source of truth for intended changes.
- Return valid JSON only.

Output requirements:
- `summary` should describe the intended implementation at a high level.
- `files` should list the files likely to be modified.
- `solution_summary` should briefly explain the intended change in that file.
- `pseudo_code` should describe the implementation plan in concise steps, not full final code.

Input:
Original Code:
{base_code.strip()}

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
- Infer the programming language from the code and evaluate the implementation using language-appropriate expectations for correctness, style, resource handling, typing, concurrency, error handling, and API usage.

Guidelines:
- Frame feedback as PR comments directed at the Generated Version.
- Be specific and objective.
- Use exact file paths, labels, strings, identifiers, UI states, conditions, and patterns when possible.
- Reference the Reference Version only to explain why the Generated Version is incomplete, less correct, or less robust.
- Do not explain the full logic of the Reference Version.
- If both versions are functionally equivalent, you may still comment on meaningful trade-offs or missing specificity if they would matter in review.
- Only return "No comment." when there are truly no meaningful differences worth flagging.

Comment quality rules:
- Consistency: The review MUST accurately describe all functional changes and address all major code modifications.
- Mapping: Each review comment MUST be mapped to the correct line(s) or section(s) of code precisely.
- Actionability: Suggestions or critiques MUST be actionable given the current code. Focus on practical helpfulness that would help a developer improve or merge the PR efficiently.
- Each comment should correspond to one concrete issue or one tightly related set of issues.
- Avoid generic praise or vague comments like "consider improving clarity".
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
- Completeness: Ensure the checklist covers ALL critical aspects that should be reviewed for this PR.
- Accuracy: Strictly NO irrelevant items. Checklist items MUST align perfectly with the issues actually present in the differences between the Generated Review and reference behavior.
- Clarity: The wording of checklist items must be crystal clear and unambiguous.
- Practical Helpfulness: The checklist MUST reflect real-world developer concerns when performing code reviews, focusing on items that add value over a manual review alone.
- Infer the programming language from the review content and preserve language-specific verification concerns such as exception handling, nullability, typing, state transitions, memory/resource cleanup, API contract adherence, UI behavior, or test expectations.
- Extract all meaningful review concerns from the PR review.
- Break each concern into the smallest actionable verification step possible.
- Keep checklist items specific and implementation-focused. Avoid vague or generic advice.
- Do not invent checklist items that are not supported by the PR review.
- Avoid duplicates and near-duplicate items. Prefer precise, scoring-friendly verification points.

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


def build_llm_generated_pr_review_prompt(
    pr_metadata: str,
    original_code: str,
    generated_code: str,
) -> str:
    return f"""You are an expert software engineer performing a code review on a pull request.

You are given:
- The pull request metadata, including title and description
- The full original version of the modified file(s) before changes
- The generated code introduced in the pull request

Your job is to provide precise, objective, and actionable review comments based on the changes introduced in the pull request.
Use the full original source for understanding context and identifying problems.

Rules:
- Infer the programming language from the file paths and code, and apply language-appropriate expectations for syntax, semantics, typing, resource handling, async behavior, error handling, and framework conventions.
- Focus on substantive code-review issues in the generated implementation. Ignore markdown linting, documentation-only polish, cosmetic formatting, lockfiles, and other non-code concerns unless the PR metadata explicitly makes them part of the requested behavior change.
- Each comment should capture one concrete issue or one tightly related set of issues.
- Reference exact file paths, identifiers, conditions, strings, behaviors, or code patterns when possible.
- Use language-appropriate terminology in the review comments.
- Keep comments actionable and phrased like real PR review feedback.
- Prefer correctness, robustness, edge cases, API misuse, incomplete behavior, and regression-sensitive logic over stylistic nits.
- If you find the pull request is 100% correct, return a review with summary `No comment.` and an empty `comments` list.
- Return valid JSON only.

PR metadata:
{pr_metadata.strip()}

Original code:
{original_code.strip()}

LLM-generated code:
{generated_code.strip()}

Output JSON schema:
{json.dumps(GeneratedPRReview.model_json_schema(), indent=2)}
"""


def build_checklist_score_prompt(
    checklist: str,
    code_review: str,
) -> str:
    return f"""You are given a checklist and a code review.

Your task is to determine how many checklist items are mentioned or addressed in the code review.

Rules:
- Return only a single number: the count of checklist items matched in the review.
- If the checklist is `No checklist`, determine whether the code review truly suggests something.
- If the code review contains any meaningful suggestion or feedback, return 0.
- If the code review implies there is nothing to check, return 1.
- Ignore the purpose, change, and other background information. Focus only on the suggestions.
- If there is no suggestion or only blank suggestion content, return 1.

Checklist:
{checklist.strip()}

Code review:
{code_review.strip()}
"""

