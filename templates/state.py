from pydantic import BaseModel, Field


class CodeEditingRequirement(BaseModel):
    modification_target: str = Field(
        ...,
        description="Exact function/class/method/variable or module that must be modified.",
    )
    modification_logic: str = Field(
        ...,
        description="Precise change logic that should be implemented for the target.",
    )
    input_output_requirement: str | None = Field(
        default=None,
        description="Expected input/output behavior constraints for the change when applicable.",
    )


class ReviewInstruction(BaseModel):
    problem_definition: str = Field(
        ...,
        description="Concise technical statement of the issue or requirement.",
    )
    code_editing_requirement: list[CodeEditingRequirement] = Field(
        default_factory=list,
        description="Detailed, actionable list of code modifications required.",
    )

class PseudoSolutionFile(BaseModel):
    file_path: str = Field(
        ...,
        description="Repository-relative file path that should be modified. " \
        "",
    )
    solution_summary: str = Field(
        ...,
        description="Concise description of the intended implementation in this file.",
    )
    pseudo_code: str = Field(
        ...,
        description="Structured pseudo-code or patch-like description of the changes."
    )

class PseudoSolution(BaseModel):
    summary: str = Field(
        ...,
        description="High-level summary of the intended implementation.",
    )
    files: list[PseudoSolutionFile] = Field(
        default_factory=list,
        description="Per-file pseudo solution plan.",
    )

class ReviewComment(BaseModel):
    file_path: str = Field(
        ...,
        description="Repository-relative file path the review comment refers to.",
    )
    line_reference: str | None = Field(
        default=None,
        description="Approximate line, block, or symbol reference if available.",
    )
    severity: str = Field(
        ...,
        description="Severity of the review comment: low, medium, or high.",
    )
    category: str = Field(
        ...,
        description="Category of the review comment such as correctness, UX, maintainability, error_handling, or consistency.",
    )
    comment: str = Field(
        ...,
        description="Concrete pull request review comment written in reviewer style.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Optional fix suggestion or requested change.",
    )


class GeneratedPRReview(BaseModel):
    summary: str = Field(
        ...,
        description="Short overall summary of the generated PR review.",
    )
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description="List of concrete PR review comments.",
    )

class ReviewChecklistItem(BaseModel):
    file_path: str = Field(
        ...,
        description="Repository-relative file path associated with this checklist item.",
    )
    target: str = Field(
        ...,
        description="Exact symbol, UI element, logic branch, string, or code region to verify.",
    )
    verification_point: str = Field(
        ...,
        description="Concrete thing a reviewer should check in the implementation.",

    )
    expected_outcome: str = Field(
        ...,
        description="Expected behavior or code outcome if the implementation is correct.",
    )
    severity: str = Field(
        ...,
        description="Importance of the checklist item, e.g. low, medium, high.",
    )

class ReviewChecklist(BaseModel):
    summary: str = Field(
        ...,
        description="High-level summary of what the review checklist covers.",
    )
    items: list[ReviewChecklistItem] = Field(
        default_factory=list,
        description="Concrete checklist items used to evaluate PR review quality.",
    )


class ChecklistScore(BaseModel):
    matched_items: int = Field(
        ...,
        description="Number of checklist items addressed by the evaluated PR review.",
    )
    total_items: int = Field(
        ...,
        description="Total number of checklist items considered during evaluation.",
    )
    coverage_score: float = Field(
        ...,
        description="Checklist coverage score computed as matched_items divided by total_items.",
    )
    evaluation_mode: str = Field(
        ...,
        description="Evaluation mode used for scoring, such as checklist or no_checklist.",
    )

class prState(BaseModel):
    owner: str
    repo: str
    pr_number: int | None = None
    review_instruct: ReviewInstruction | None = None
    pseudo_solution: PseudoSolution | None = None
    generated_review: GeneratedPRReview | None = None
    checklist: ReviewChecklist | None = None
    llm_generated_pr_review: GeneratedPRReview | None = None
    score: ChecklistScore | None = None
