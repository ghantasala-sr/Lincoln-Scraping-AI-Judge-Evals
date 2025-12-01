from typing import Optional
from pydantic import BaseModel

class Document(BaseModel):
    id: str
    title: str
    reference: str
    document_type: str  # "Letter" | "Speech" | "Note" | etc.
    date: str
    place: Optional[str] = None
    from_: Optional[str] = None  # 'from' is a reserved keyword
    to: Optional[str] = None
    content: str

    class Config:
        populate_by_name = True
        fields = {
            'from_': 'from'
        }
