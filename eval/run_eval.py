"""
Evaluate the hybrid RAG pipeline with RAGAS.

Run from project root:
    python3 eval/run_eval.py
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness
from ragas.run_config import RunConfig

from retrieval.search import hybrid_search
from generation.generate import generate_answer
from eval.questions import questions as EVAL_QUESTIONS

GENERATE_MODEL = "claude-sonnet-4-6"
EVAL_MODEL = "claude-haiku-4-5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-only", action="store_true")
    args = parser.parse_args()

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")

        chunks = hybrid_search(question, top_k=10, dense_only=args.dense_only)
        answer = generate_answer(question, chunks)
        contexts = [c["text"] for c in chunks if "text" in c]

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(rows)

    eval_llm = ChatAnthropic(model=EVAL_MODEL, temperature=0, max_retries=3, timeout=60)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    run_config = RunConfig(max_workers=1, max_wait=60)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision],
        llm=eval_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    print("\n=== RAGAS scores ===")
    scores = result.scores
    print("\n=== RAGAS scores ===")
    for metric in ["faithfulness", "context_precision"]:
        vals = [s[metric] for s in scores if metric in s]
        print(f"{metric}: {np.mean(vals):.3f} (n={len(vals)})")

    df = result.to_pandas()
    fname = "results_dense.csv" if args.dense_only else "results_hybrid.csv"
    out_path = os.path.join(os.path.dirname(__file__), fname)
    df.to_csv(out_path, index=False)
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
