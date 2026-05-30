"""
Hybrid retrieval for the AI safety RAG pipeline.
Dense (bge-base-en-v1.5) + sparse (SPLADE) fused with RRF via Qdrant Query API.
"""

import os

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer

BGE_MODEL = "BAAI/bge-base-en-v1.5"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
COLLECTION_NAME = "ai_safety"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

_dense_model: SentenceTransformer | None = None
_sparse_model: SparseTextEmbedding | None = None
_client: QdrantClient | None = None


def _get_resources() -> tuple[SentenceTransformer, SparseTextEmbedding, QdrantClient]:
    global _dense_model, _sparse_model, _client
    if _dense_model is None:
        _dense_model = SentenceTransformer(BGE_MODEL)
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPLADE_MODEL)
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _dense_model, _sparse_model, _client


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search with dense + sparse vectors fused via RRF.
    Returns top_k results, each a dict with 'score' plus all payload fields.
    {"score": ..., "title": ..., "text": ...}
    """
    dense_model, sparse_model, client = _get_resources()

    dense_vec = dense_model.encode(query, normalize_embeddings=True).tolist()

    sparse_result = next(sparse_model.embed([query]))
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense_vec", limit=top_k * 2),
            Prefetch(query=sparse_vec, using="sparse_vec", limit=top_k * 2),
        ],
        query=Fusion.RRF,
        limit=top_k,
        with_payload=True,
    )

    return [{"score": pt.score, **pt.payload} for pt in response.points]
