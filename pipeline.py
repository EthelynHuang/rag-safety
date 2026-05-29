from retrieval.search import hybrid_search
from generation.generate import generate_answer


def answer_query(query: str, top_k: int = 10) -> str:
    chunks = hybrid_search(query, top_k=top_k)
    return generate_answer(query, chunks)
