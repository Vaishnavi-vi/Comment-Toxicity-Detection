from pydantic import BaseModel,Field
from typing import Annotated

class user_input(BaseModel):
    Comment_text:Annotated[str,Field(...,description="Enter the comment")]
