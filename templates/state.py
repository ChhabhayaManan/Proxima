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


class prState(BaseModel):
    owner: str
    repo: str
    pr_number: int
    review_instruct: ReviewInstruction | None = None
    
