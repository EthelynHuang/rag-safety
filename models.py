from pydantic import BaseModel


class Source(BaseModel):
    title: str
    url: str
    authors: list[str] | None = None
    date_published: str | None = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
