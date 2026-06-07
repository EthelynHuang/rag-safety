# Temporary debug file to figure out improving context retrieval

import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))

from eval.questions import questions
from retrieval.search import hybrid_search

FIRST_5 = questions[:5]

for i, q in enumerate(FIRST_5, start=1):
    print(f"QUESTION {i}: {q['question']}")
    print("-" * 40)
    try:
        results = hybrid_search(q["question"], top_k=10)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for j, chunk in enumerate(results, start=1):
        title = chunk.get("title", "<no title>")
        source = chunk.get("source", "<no source>")
        subsource = chunk.get("subsource", "")
        source_str = f"{source} / {subsource}" if subsource else source
        text = chunk.get("text", "<no text>")
        print(f"Chunk {j}:")
        print(f"  Title: {title}")
        print(f"  Source: {source_str}")
        print(f"  Text: {text}")
        print()

    print()
