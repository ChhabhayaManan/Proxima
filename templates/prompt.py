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

def build_checklist_generation_prompt(
    pseudo_solution: str,
    merged_code: str,
    changed_code: str,
) -> str:
    return f"""You are generating a structured PR review checklist by comparing a pseudo solution against the merged implementation.

Task:
Compare the pseudo solution with the merged code and create a comprehensive and highly detailed checklist that captures all concrete issues, missing behaviors, implementation gaps, and reviewer verification points revealed by that comparison.

Checklist Requirements:
- Completeness: Ensure the checklist covers ALL critical aspects that should be reviewed for this PR.
- Accuracy: Strictly NO irrelevant items. Checklist items MUST align perfectly with the issues actually present in the differences between the pseudo solution and the merged code.
- Clarity: The wording of checklist items must be crystal clear and unambiguous.
- Practical Helpfulness: The checklist MUST reflect real-world developer concerns when performing code reviews, focusing on items that add value over a manual review alone.
- Infer the programming language from the pseudo solution and merged code, and preserve language-specific verification concerns such as exception handling, nullability, typing, state transitions, memory/resource cleanup, API contract adherence, UI behavior, or test expectations.
- Extract all meaningful review concerns from the differences between the pseudo solution and the merged code.
- Focus especially on missing logic, incorrect assumptions, incomplete implementation, missing conditions or state handling, incorrect strings or UI behavior, missing imports or support code, and important precision gaps.
- Break each concern into the smallest actionable verification step possible.
- Keep checklist items specific and implementation-focused. Avoid vague or generic advice.
- Do not invent checklist items that are not supported by the pseudo solution and merged code.
- Restrict checklist items to the changed code context only. Use the changed-code diff to determine which files, code regions, and behaviors are actually in scope.
- Do not create checklist items for untouched code, pre-existing helpers, or surrounding implementation details unless they are directly affected by the changed code.
- Avoid duplicates and near-duplicate items. Prefer precise, scoring-friendly verification points.
- The number of checklist items must depend on the size and complexity of the PR:
  - Small PRs: 3 to 4 checklist items is enough.
  - Medium PRs: 5 to 7 checklist items.
  - Larger PRs: at most 8 to 10 checklist items.
- Even for large PRs, include only the most important checklist items. Prioritize changed behavior, changed logic, and regression-sensitive code paths over secondary polish.

Field requirements:
- `file_path`: repository-relative file path if available
- `target`: exact symbol, code region, string, condition, import, UI element, or pattern to verify
- `verification_point`: what should be checked
- `expected_outcome`: what should be true if the implementation is correct

Pseudo Solution:
{pseudo_solution.strip()}

Changed Code Diff:
{changed_code.strip()}

Merged Code:
{merged_code.strip()}

Return valid JSON only.

Output JSON schema:
{json.dumps(ReviewChecklist.model_json_schema(), indent=2)}
"""


def build_llm_generated_pr_review_prompt(
    pr_metadata: str,
    merged_code: str,
    pseudo_solution: str,
) -> str:
    return f"""You are an expert, meticulous software engineer performing a precise code review on a pull request.

You are given:
- PR Metadata: The title and description of the intended changes.
- Merged Code: The merged code after the changes.
- Generated Code: The generated pseudo solution from the pr review instructions.

Your task is to provide objective, actionable, and specific review comments analyzing the Generated Code.
Compare the Generated Code against the Merged Code and the requirements in the PR Metadata. You must aggressively identify logical errors, incomplete implementations, missing edge cases, syntax issues, or poor practices.

Rules:
- Be highly specific. Vague suggestions do not count.
- Point out exactly what is wrong and how to fix it.
- Your output MUST absolutely be a JSON object containing a `summary` field and a `comments` array matching the schema. NEVER return just a plain array.
- If no issues are found, return a JSON object with your `summary` and an empty `comments` array.
- STRICTLY return valid JSON matching the explicit schema parameters natively. Do not wrap your JSON in markdown backticks or formatting.

PR metadata:
{pr_metadata.strip()}

Merged code:
{merged_code.strip()}

Pseudo solution:
{pseudo_solution.strip()}

Output JSON schema:
{json.dumps(GeneratedPRReview.model_json_schema(), indent=2)}
"""


def build_checklist_score_prompt(
    checklist: str,
    code_review: str,
) -> str:
    return f"""You are an automated code evaluation engine calculating benchmark coverage for a pull request review.

You are given:
1. Checklist items (The ground-truth bugs that existed)
2. Generated code review comments (The AI's attempt to find those bugs)

Task:
Evaluate EACH checklist item one by one silently. 
Count an item as matched ONLY if the review clearly and specifically calls out the exact same core issue.

Matching Rules:
- Semantic equivalence is highly allowed.
- Generic, vague advice does NOT count.

- Partial overlap counts ONLY if the main concern is clearly addressed.

Procedure:
1. For each checklist item, mentally decide MATCHED or NOT MATCHED.
2. Count the total matched items.
3. OUTPUT ONLY THE FINAL TOTAL INTEGER.

CRITICAL: Your entire response must consist of exactly ONE character/number. Do not include step-by-step reasoning, bullet points, explanations, or any other words. If the score is 3, your entire output must be "3".

Checklist:
{checklist}

Code Review:
{code_review}
"""

