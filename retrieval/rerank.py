"""
Cross-encoder reranking for the AI safety RAG pipeline.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score each chunk against the query with a cross-encoder and return the
    top_k highest-scoring chunks, sorted descending by cross-encoder score.
    Adds a 'rerank_score' key to each returned dict.
    """
    model = _get_model()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs).tolist()
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
