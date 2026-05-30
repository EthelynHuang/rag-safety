from models import QueryResponse, Source
from retrieval.search import hybrid_search
from generation.generate import generate_answer


def answer_query(query: str, top_k: int = 10) -> QueryResponse:
    chunks = hybrid_search(query, top_k=top_k)
    answer = generate_answer(query, chunks)
    sources = [
        Source(
            title=c["title"],
            url=c["url"],
            authors=c.get("authors"),
            date_published=c.get("date_published"),
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)
