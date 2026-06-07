"""
Isolation test for reranking.
Pulls 10 chunks from Qdrant via hybrid_search, reranks to top 5,
and prints each chunk with its cross-encoder score.
"""
import os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.search import hybrid_search
from retrieval.rerank import rerank

QUERY = "What is inner alignment?"

print(f"Query: {QUERY}")
print(f"Fetching 10 chunks from Qdrant...")
chunks = hybrid_search(QUERY, top_k=10)
print(f"Retrieved {len(chunks)} chunks. Reranking to top 5...")
print()

reranked = rerank(QUERY, chunks, top_k=5)

SEP = "─" * 72
for i, chunk in enumerate(reranked, start=1):
    print(SEP)
    print(f"Rank {i}  |  cross-encoder score: {chunk['rerank_score']:.4f}  |  RRF score: {chunk.get('score', 'n/a')}")
    print(f"Title : {chunk.get('title', '')}")
    print(f"Source: {chunk.get('source', '')}/{chunk.get('subsource', '')}")
    print(f"Text  : {chunk['text']}")
    print()
print(SEP)
