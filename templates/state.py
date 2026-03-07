from pydantic import BaseModel
# from types import Field, List, Dict, Optional

class prState(BaseModel):
    owner: str
    repo: str
    pr_number: int
    