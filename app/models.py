from pydantic import BaseModel
from typing import Optional

class DatasetInfo(BaseModel):
    name: str
    num_instances: int
    num_attributes: int
    lang_count: int
